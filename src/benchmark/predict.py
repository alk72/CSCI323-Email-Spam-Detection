"""
Classify your own messages with a trained model (handy for the demo).

    python -m src.benchmark.predict --model mlp "WINNER!! Claim your free prize now"
    python -m src.benchmark.predict --model distilbert "Hi, are we still meeting at 3?"
    python -m src.benchmark.predict --model mlp            # interactive: type messages, empty line quits
"""
import argparse
import os
import sys

from . import common
from .common import OUT_DIR, clean_text, distilbert_max_len, run_name


def load_predictor(model, holdout):
    run = os.path.join(OUT_DIR, run_name(model, holdout))
    if model == "mlp":
        import joblib
        sys.modules.setdefault("common", common)  # models saved before the move to src/benchmark
        bundle = joblib.load(os.path.join(run, "model.joblib"))
        return lambda texts: bundle["mlp"].predict_proba(bundle["features"].transform(texts))[:, 1]

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from .train_distilbert import pick_device
    device = pick_device()
    path = os.path.join(run, "best_model")
    tok = AutoTokenizer.from_pretrained(path)
    net = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()
    max_len = distilbert_max_len(path)

    @torch.no_grad()
    def f(texts):
        enc = tok(texts, truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(device)
        return torch.softmax(net(**enc).logits.float(), dim=-1)[:, 1].cpu().numpy()
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mlp", "distilbert"], default="mlp")
    ap.add_argument("--holdout", default=None, help="load the model trained with this holdout")
    ap.add_argument("texts", nargs="*")
    args = ap.parse_args()
    predict = load_predictor(args.model, args.holdout)

    def show(t):
        p = float(predict([clean_text(t)])[0])
        print(f"{'SPAM' if p >= 0.5 else 'ham ':4}  p(spam)={p:.3f}  | {t[:80]}")

    if args.texts:
        for t in args.texts:
            show(t)
        return
    print("Type a message (empty line to quit):")
    for line in sys.stdin:
        if not line.strip():
            break
        show(line.strip())


if __name__ == "__main__":
    main()
