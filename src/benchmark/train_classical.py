"""
Baselines: Multinomial Naive Bayes and linear SVM from src/models.py, run on the
SAME cleaned data and the SAME train/val/test split as the MLP and DistilBERT,
so all four models can go in one table.

    python -m src.benchmark.train_classical                   # in-domain, all 4 combinations
    python -m src.benchmark.train_classical --holdout sms     # cross-dataset (also: enron, spamassassin)
    python -m src.benchmark.train_classical --balance         # weight sources equally, like the MLP/DistilBERT

Model settings come from src/config.py (word unigrams, max 9,000 features, min_df 3,
MultinomialNB alpha=1, LinearSVC C=1). Text cleaning matches src/preprocess.py, see
classical_clean below.
"""
import argparse
import os

from scipy.special import expit

from ..config import FEATURES, MAX_FEATURES, MODELS
from ..models import build_pipeline
from ..preprocess import ENGLISH_STOP_WORDS, NON_LETTERS_RE
from .common import (OUT_DIR, SOURCES, evaluate_by_source, load_data, make_splits,
                     print_results, run_name, save_json, source_balanced_probs)


def classical_clean(text):
    """src/preprocess.clean_text without its URL / e-mail regexes: common.clean_text has
    already turned those into the words URL / EMAIL / PHONENUMBER, which should survive.
    (Running the full clean_text also drops a few tokens like "msg@", which moves some
    report numbers by up to 0.01.)"""
    text = NON_LETTERS_RE.sub(" ", text.lower())
    return " ".join(t for t in text.split() if len(t) > 1 and t not in ENGLISH_STOP_WORDS)


def spam_prob(pipe, texts):
    """P(spam) for NB; for the SVM the margin w.x+b squashed by a sigmoid, so that
    the 0.5 threshold in compute_metrics equals the SVM's own w.x+b = 0 boundary."""
    if hasattr(pipe.named_steps["clf"], "predict_proba"):
        return pipe.predict_proba(texts)[:, 1]
    return expit(pipe.decision_function(texts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", choices=SOURCES, default=None)
    ap.add_argument("--balance", action="store_true",
                    help="source-balanced sample weights (src/experiments.py uses none)")
    args = ap.parse_args()

    df = load_data()
    df["clean"] = df.text.map(classical_clean)
    train, _, test = make_splits(df, args.holdout)  # no tuning, so no validation set needed
    print(f"train {len(train)}  test {len(test)}  (holdout = {args.holdout or 'none, in-domain'})")

    fit_kw = {}
    if args.balance:
        fit_kw["clf__sample_weight"] = source_balanced_probs(train.source) * len(train)

    for model in MODELS:
        for feats in FEATURES:
            pipe = build_pipeline(model, feats).fit(train.clean, train.label, **fit_kw)
            res = evaluate_by_source(test, spam_prob(pipe, test.clean))
            name = f"{model}-{feats}" + ("-balanced" if args.balance else "")
            print(f"\n{name}:")
            print_results(res)
            save_json({"model": name, "holdout": args.holdout, "max_features": MAX_FEATURES, "balanced": args.balance,
                       "test": res}, os.path.join(OUT_DIR, run_name(name, args.holdout), "metrics.json"))


if __name__ == "__main__":
    main()
