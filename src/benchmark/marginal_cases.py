"""
Marginal (edge) cases for all four models on the in-domain test set.

    python -m src.benchmark.marginal_cases      # -> results/benchmark/marginal/*.csv

Adapts the margin and attack experiments (src/experiments.py, src/adversarial.py)
so the same tests run on NB, SVM, MLP and DistilBERT:

1. Margin: every model's output is turned into a spam log-odds score s
   (NB: log P(spam|x) - log P(ham|x); SVM: w.x + b; MLP and DistilBERT: logit).
   A message is "marginal" when |s| < 1, i.e. 0.27 < p(spam) < 0.73 (for the SVM
   this is exactly the inside of its margin). We compare error rates inside and
   outside that band, and by message length.
2. Attacks on test spam: obfuscating spam words ("free" -> "fr33" / "f.r.e.e")
   and appending ham words (a good-word attack). The word lists come from the
   SVM's largest positive / negative weights and are used for every model.
3. Hand-written edge cases scored by all four models.
"""
import os

import numpy as np
import pandas as pd

from ..adversarial import good_word_attack, obfuscate_text, top_ham_words, top_spam_words
from ..models import build_pipeline
from .common import (OUT_DIR, SEED, TfidfFeatures, clean_text, distilbert_max_len, load_data,
                     make_splits, source_balanced_probs)
from .train_classical import classical_clean
from .train_mlp import train_mlp

DISTILBERT_DIR = os.environ.get("DISTILBERT_DIR", os.path.join(OUT_DIR, "distilbert_indomain", "best_model"))
OUT = os.path.join(OUT_DIR, "marginal")
N_ATTACK = 400  # spam messages attacked (DistilBERT on CPU is slow)

EDGE_CASES = [
    ("Very short ham", 0, "ok see u there"),
    ("Very short ham", 0, "call me when you get this"),
    ("Very short spam", 1, "FREE entry! Txt WIN to 80086 now"),
    ("Legitimate newsletter", 0, "Our autumn sale ends Sunday: 30% off all jackets for members. Shop now at https://shop.example.com"),
    ("Phishing in business tone", 1, "Hi, please review the attached invoice and confirm your account details by end of day so that payment is not delayed. Regards, Accounts"),
    ("Negation", 0, "This is not a prize draw or a sales offer, just a reminder that the team meeting has moved to Friday at 10."),
    ("Obfuscated spam", 1, "C1aim y0ur fr33 pr1ze n0w!! Reply Y3S to w1n c4sh"),
    ("Non-English spam (Malay)", 1, "Tahniah! Anda telah memenangi hadiah wang tunai RM5000. Hubungi kami sekarang untuk menuntut hadiah anda."),
    ("Non-English ham (Malay)", 0, "Jom makan malam pukul 8? Aku dah sampai rumah."),
]
_INTRO = ("Hi team, thanks for joining the planning meeting yesterday. As discussed, the quarterly report is due on "
          "the 14th and each section owner should send a draft to me by Wednesday. Please also check the shared "
          "calendar for the review sessions and let me know if any of the times do not work for you. ")
EDGE_CASES.append(("Spam after a long harmless intro", 1,
                   _INTRO * 6 + "WINNER!! You have won a FREE iPhone, click https://win.example.com to claim your prize now"))


def load_distilbert():
    """The trained DistilBERT as a function texts -> spam log-odds. Loaded before the
    other models are trained, so a missing checkpoint fails fast."""
    if not os.path.isdir(DISTILBERT_DIR):
        raise SystemExit(f"{DISTILBERT_DIR} not found - run `python -m src.benchmark.train_distilbert` "
                         "first, or point DISTILBERT_DIR at a trained best_model folder")
    max_len = distilbert_max_len(DISTILBERT_DIR)
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:  # no PyTorch: fall back to a NumPy forward pass
        try:
            from .np_distilbert import NpDistilBert
        except ImportError as e:
            raise SystemExit(f"DistilBERT needs PyTorch + Transformers (pip install -r requirements-neural.txt) "
                             f"or, for the NumPy fallback, safetensors + tokenizers ({e})")
        return NpDistilBert(DISTILBERT_DIR, max_len).logit

    from .train_distilbert import pick_device
    device = pick_device()
    print(f"DistilBERT scoring on {device}")
    tok = AutoTokenizer.from_pretrained(DISTILBERT_DIR)
    net = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_DIR).to(device).eval()

    @torch.no_grad()
    def db_logit(t, batch=32):
        out = []
        t = list(t)
        for s in range(0, len(t), batch):
            enc = tok(t[s:s + batch], truncation=True, max_length=max_len, padding=True,
                      return_tensors="pt").to(device)
            lg = net(**enc).logits.float()
            out.append((lg[:, 1] - lg[:, 0]).cpu().numpy())
        return np.concatenate(out)
    return db_logit


def build_models(train, val):
    """Each model as a function texts -> spam log-odds."""
    db_logit = load_distilbert()
    models = {}
    tr_clean = train.text.map(classical_clean)
    for m in ("nb", "svm"):
        pipe = build_pipeline(m, "tfidf").fit(tr_clean, train.label)
        if m == "nb":
            models["NB"] = lambda t, p=pipe: np.diff(p.predict_log_proba(pd.Series(t).map(classical_clean)), axis=1)[:, 0]
        else:
            models["SVM"] = lambda t, p=pipe: p.decision_function(pd.Series(t).map(classical_clean))
            svm = pipe

    feats = TfidfFeatures()
    X_tr = feats.fit_transform(train.text)
    mlp, _, _ = train_mlp(X_tr, train.label.values, feats.transform(val.text), val.label.values,
                          val.source.values, (128,), 1e-4, 1e-3, 10, 2, 256,
                          source_balanced_probs(train.source))
    def mlp_logit(t):
        p = np.clip(mlp.predict_proba(feats.transform(list(t)))[:, 1], 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p))
    models["MLP"] = mlp_logit
    models["DistilBERT"] = db_logit
    return models, svm


def main():
    os.makedirs(OUT, exist_ok=True)
    df = load_data()
    train, val, test = make_splits(df)
    models, svm = build_models(train, val)
    y = test.label.values
    words = test.text.str.split().str.len().values
    length_bins = pd.cut(words, [0, 10, 30, 100, np.inf], labels=["1-10", "11-30", "31-100", ">100"])

    margin_rows, length_rows, scores = [], [], {}
    for name, f in models.items():
        print("scoring test set:", name, flush=True)
        s = scores[name] = f(test.text.values)
        pred, near = (s > 0).astype(int), np.abs(s) < 1
        err = pred != y
        margin_rows.append({"model": name, "test": len(y), "share_marginal": near.mean(),
                            "error_rate_marginal": err[near].mean() if near.any() else np.nan,
                            "error_rate_rest": err[~near].mean(), "errors": int(err.sum()),
                            "share_of_errors_marginal": err[near].sum() / max(err.sum(), 1)})
        for b in length_bins.categories:
            m = np.asarray(length_bins == b)
            length_rows.append({"model": name, "words": b, "messages": int(m.sum()),
                                "error_rate": err[m].mean(), "share_marginal": near[m].mean()})
    pd.DataFrame(scores).assign(label=y, source=test.source.values, words=words).to_csv(
        os.path.join(OUT, "test_scores.csv"), index=False)

    # attacks, using the SVM's most spammy / most hammy words for every model
    spam_words, ham_words = set(top_spam_words(svm)), np.array(top_ham_words(svm))
    spam = test[test.label == 1].sample(N_ATTACK, random_state=SEED).text.tolist()
    attacks = [("none", 0)] + [("obfuscation", r) for r in (0.5, 1.0)] + [("good words", n) for n in (25, 100)]
    attack_rows = []
    for kind, strength in attacks:
        rng = np.random.default_rng(SEED)
        texts = [obfuscate_text(t, strength, rng, spam_words) if kind == "obfuscation" else
                 good_word_attack(t, ham_words, strength, rng) if kind == "good words" else t for t in spam]
        for name, f in models.items():
            print("attack", kind, strength, name, flush=True)
            attack_rows.append({"attack": kind, "strength": strength, "model": name,
                                "spam_caught": float((f(texts) > 0).mean())})

    edge_rows = []
    edge_texts = [clean_text(t) for _, _, t in EDGE_CASES]
    edge_scores = {name: f(edge_texts) for name, f in models.items()}
    for i, (case, label, text) in enumerate(EDGE_CASES):
        edge_rows.append({"case": case, "true": "spam" if label else "ham", "text": text[:90],
                          **{n: round(float(1 / (1 + np.exp(-edge_scores[n][i]))), 3) for n in models}})

    for name, rows in [("margin", margin_rows), ("by_length", length_rows),
                       ("attacks", attack_rows), ("edge_cases", edge_rows)]:
        t = pd.DataFrame(rows)
        t.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
        print(f"\n== {name}\n" + (t.pivot(index=["attack", "strength"], columns="model", values="spam_caught")
                                  if name == "attacks" else t).round(3).to_string())


if __name__ == "__main__":
    main()
