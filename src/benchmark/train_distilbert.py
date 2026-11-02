"""
Model 2: fine-tuned DistilBERT (Sanh et al., 2019).

    python -m src.benchmark.train_distilbert                          # all 3 datasets, in-domain test
    python -m src.benchmark.train_distilbert --holdout sms            # cross-dataset generalisation test
    python -m src.benchmark.train_distilbert --max-train 6000         # faster run on a laptop

DistilBERT is a 6-layer transformer distilled from BERT, pretrained on English
Wikipedia + BookCorpus. We add a classification head on the [CLS] token and
fine-tune the whole network on spam/ham with cross-entropy loss.

Generalisation measures
  * transfer learning: the model starts from general language knowledge, so it
    needs far fewer spam examples and copes better with unseen wording
  * small learning rate (2e-5) + linear warm-up/decay: avoids destroying the
    pretrained weights ("catastrophic forgetting")
  * weight decay (AdamW) and the built-in dropout (0.1) as regularisation
  * few epochs + early stopping on the validation F1

Speed: an NVIDIA GPU (e.g. free Google Colab T4) does one epoch on the full data
in ~10-15 min. Apple Silicon (MPS) is slower, and CPU only is very slow; use
--max-train there.
"""
import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from .common import (OUT_DIR, SEED, SOURCES, compute_metrics, evaluate_by_source, load_data,
                    make_splits, print_results, run_name, save_json,
                    source_balanced_probs, val_score)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def subsample(df, n, seed=SEED):
    """Stratified (by source and label) random subset, so every dataset and
    both classes stay represented."""
    if n is None or n >= len(df):
        return df
    frac = n / len(df)
    return df.groupby(["source", "label"], group_keys=False).sample(frac=frac, random_state=seed)


def make_loader(df, tokenizer, max_len, batch_size, shuffle, balance=False):
    texts, labels = df.text.tolist(), df.label.tolist()

    def collate(idx):
        enc = tokenizer([texts[i] for i in idx], truncation=True, max_length=max_len,
                        padding=True, return_tensors="pt")  # dynamic padding per batch
        enc["labels"] = torch.tensor([labels[i] for i in idx])
        return enc

    sampler = None
    if balance:  # every dataset contributes equally per epoch (see common.py)
        g = torch.Generator().manual_seed(SEED)
        sampler = WeightedRandomSampler(source_balanced_probs(df.source), len(df), generator=g)
        shuffle = False
    return DataLoader(list(range(len(texts))), batch_size=batch_size, shuffle=shuffle,
                      sampler=sampler, collate_fn=collate)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        logits = model(**batch).logits
        probs.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", choices=SOURCES, default=None)
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=256, help="max tokens per message")
    ap.add_argument("--max-train", type=int, default=None, help="subsample training set")
    ap.add_argument("--max-val", type=int, default=3000, help="subsample validation set")
    ap.add_argument("--patience", type=int, default=1)
    ap.add_argument("--no-balance", action="store_true",
                    help="do not balance datasets during training")
    args = ap.parse_args()

    set_seed(SEED)
    device = pick_device()
    print(f"device: {device}")

    df = load_data()
    train, val, test = make_splits(df, args.holdout)
    train, val = subsample(train, args.max_train), subsample(val, args.max_val)
    print(f"train {len(train)}  val {len(val)}  test {len(test)}  "
          f"(holdout = {args.holdout or 'none, in-domain'})")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(device)

    tr_loader = make_loader(train, tokenizer, args.max_len, args.batch_size, shuffle=True,
                            balance=not args.no_balance)
    va_loader = make_loader(val, tokenizer, args.max_len, args.batch_size * 2, shuffle=False)
    te_loader = make_loader(test, tokenizer, args.max_len, args.batch_size * 2, shuffle=False)

    # AdamW; no weight decay on biases and LayerNorm weights (standard BERT recipe)
    no_decay = ("bias", "LayerNorm.weight")
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    total_steps = len(tr_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    # mixed precision on NVIDIA GPUs: ~2x faster, same accuracy
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out = os.path.join(OUT_DIR, run_name("distilbert", args.holdout))
    best_dir = os.path.join(out, "best_model")
    best_f1, bad, history = -1.0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for step, batch in enumerate(tr_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = model(**batch).loss  # cross-entropy on the [CLS] head
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.item()
            if step % 50 == 0 or step == len(tr_loader):
                print(f"  epoch {epoch} step {step}/{len(tr_loader)}  loss {running / step:.4f}"
                      f"  ({time.time() - t0:.0f}s)", flush=True)

        val_prob = predict(model, va_loader, device)
        val_m = compute_metrics(val.label, val_prob)
        val_f1 = val_score(val.label.values, val.source.values, val_prob)  # mean F1 over datasets
        history.append({"epoch": epoch, "train_loss": round(running / len(tr_loader), 5),
                        "val_f1": round(val_f1, 5), "val_acc": val_m["accuracy"]})
        print(f"epoch {epoch}: train loss {running / len(tr_loader):.4f}  "
              f"val F1 {val_f1:.4f}  val acc {val_m['accuracy']:.4f}")
        if val_f1 > best_f1:
            best_f1, bad = val_f1, 0
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop")
                break

    model = AutoModelForSequenceClassification.from_pretrained(best_dir).to(device)
    res = evaluate_by_source(test, predict(model, te_loader, device))
    print("\nTEST results:")
    print_results(res)

    pd.DataFrame(history).to_csv(os.path.join(out, "history.csv"), index=False)
    save_json({"model": "distilbert", "holdout": args.holdout, "base": args.model,
               "train_size": len(train), "max_len": args.max_len, "lr": args.lr,
               "balanced": not args.no_balance, "val_f1": round(best_f1, 4), "test": res}, os.path.join(out, "metrics.json"))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
