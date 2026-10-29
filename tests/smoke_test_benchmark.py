"""Runs the shared-split benchmark (NB, SVM, MLP) on small fake data.
Usage: python -m tests.smoke_test_benchmark   (DistilBERT is not covered: it needs a download)"""
import pandas as pd

from src.benchmark.common import (SOURCES, TfidfFeatures, clean_text, evaluate_by_source,
                                  make_splits, source_balanced_probs)
from src.benchmark.train_classical import classical_clean, spam_prob
from src.benchmark.train_mlp import train_mlp
from src.models import build_pipeline
from tests.smoke_test import fake_dataset


def fake_benchmark_data():
    frames = []
    for i, name in enumerate(SOURCES):
        df = fake_dataset(name, seed=i)[["text", "label", "source"]]
        df["split"] = ["train"] * 210 + ["val"] * 45 + ["test"] * (len(df) - 255)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["text"] = df["text"].map(clean_text)
    return df


def main():
    df = fake_benchmark_data()
    for holdout in (None, SOURCES[0]):
        train, val, test = make_splits(df, holdout)

        feats = TfidfFeatures(max_word=500, max_char=500)
        X_tr, X_va, X_te = feats.fit_transform(train.text), feats.transform(val.text), feats.transform(test.text)
        clf, val_f1, history = train_mlp(X_tr, train.label.values, X_va, val.label.values,
                                         val.source.values, hidden=(16,), alpha=1e-4, lr=1e-3,
                                         max_epochs=3, patience=2, batch_size=64,
                                         sample_p=source_balanced_probs(train.source))
        res = evaluate_by_source(test, clf.predict_proba(X_te)[:, 1])
        assert res["overall"]["roc_auc"] > 0.9, res["overall"]

        for model in ("nb", "svm"):
            pipe = build_pipeline(model, "tfidf").fit(train.text.map(classical_clean), train.label)
            res = evaluate_by_source(test, spam_prob(pipe, test.text.map(classical_clean)))
            assert res["overall"]["roc_auc"] > 0.9, (model, res["overall"])
    print("\nBenchmark smoke test passed")


if __name__ == "__main__":
    main()
