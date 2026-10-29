"""
Baselines: Multinomial Naive Bayes and linear SVM (teammate's models), run on the
SAME cleaned data and the SAME train/val/test split as the MLP and DistilBERT,
so all four models can go in one table.

    python train_classical.py                   # in-domain, all 4 combinations
    python train_classical.py --holdout sms     # cross-dataset (also: enron, spamassassin)
    python train_classical.py --balance         # weight sources equally, like the MLP/DistilBERT

Settings copied from the teammate's repo (CSCI323-Email-Spam-Detection, src/config.py
and src/models.py): lower-case, keep letters only, drop English stop words and
1-letter tokens; word unigrams, max 9,000 features, min_df 3; Bag of Words or
TF-IDF (sublinear tf); MultinomialNB(alpha=1.0) or LinearSVC(C=1.0).
"""
import argparse
import os
import re

import numpy as np
from sklearn.feature_extraction.text import (ENGLISH_STOP_WORDS, CountVectorizer,
                                             TfidfVectorizer)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from common import (OUT_DIR, SEED, evaluate_by_source, load_data, make_splits,
                    print_results, run_name, save_json, source_balanced_probs)

MAX_FEATURES, MIN_DF, NB_ALPHA, SVM_C = 9000, 3, 1.0, 1.0
_NON_LETTERS = re.compile(r"[^a-z\s]")


def teammate_clean(text):
    """Same as the teammate's preprocess.clean_text. Our cleaning has already turned
    URLs / e-mails / numbers into the words URL / EMAIL / PHONENUMBER, which survive."""
    text = _NON_LETTERS.sub(" ", text.lower())
    return " ".join(t for t in text.split() if len(t) > 1 and t not in ENGLISH_STOP_WORDS)


def build(model, feats):
    vec = (CountVectorizer if feats == "bow" else TfidfVectorizer)(
        max_features=MAX_FEATURES, min_df=MIN_DF, **({"sublinear_tf": True} if feats == "tfidf" else {}))
    clf = MultinomialNB(alpha=NB_ALPHA) if model == "nb" else \
        LinearSVC(C=SVM_C, dual="auto", max_iter=20000, random_state=SEED)
    return Pipeline([("vec", vec), ("clf", clf)])


def spam_prob(pipe, texts):
    """P(spam) for NB; for the SVM the margin w.x+b squashed by a sigmoid, so that
    the 0.5 threshold in compute_metrics equals the SVM's own w.x+b = 0 boundary."""
    if hasattr(pipe.named_steps["clf"], "predict_proba"):
        return pipe.predict_proba(texts)[:, 1]
    return 1 / (1 + np.exp(-pipe.decision_function(texts)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", choices=["sms", "enron", "spamassassin"], default=None)
    ap.add_argument("--balance", action="store_true",
                    help="source-balanced sample weights (the teammate's original runs had none)")
    args = ap.parse_args()

    df = load_data()
    df["clean"] = df.text.map(teammate_clean)
    train, _, test = make_splits(df, args.holdout)  # no tuning, so no validation set needed
    print(f"train {len(train)}  test {len(test)}  (holdout = {args.holdout or 'none, in-domain'})")

    fit_kw = {}
    if args.balance:
        fit_kw["clf__sample_weight"] = source_balanced_probs(train.source) * len(train)

    for model in ("nb", "svm"):
        for feats in ("bow", "tfidf"):
            pipe = build(model, feats).fit(train.clean, train.label, **fit_kw)
            res = evaluate_by_source(test, spam_prob(pipe, test.clean))
            name = f"{model}-{feats}" + ("-balanced" if args.balance else "")
            print(f"\n{name}:")
            print_results(res)
            save_json({"model": name, "holdout": args.holdout, "max_features": MAX_FEATURES, "balanced": args.balance,
                       "test": res}, os.path.join(OUT_DIR, run_name(name, args.holdout), "metrics.json"))


if __name__ == "__main__":
    main()
