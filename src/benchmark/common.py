"""Shared helpers: text cleaning, train/val/test selection and metrics."""
import html
import json
import os
import re

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

from ..config import DATA_PROCESSED, DATASETS, RANDOM_STATE, RESULTS

DATA_CSV = os.path.join(DATA_PROCESSED, "spam_all.csv")
OUT_DIR = os.path.join(RESULTS, "benchmark")
SOURCES = DATASETS
SEED = RANDOM_STATE
MAX_CHARS = 3000  # long emails are cut; the start of a message carries most signal

_TAG_RE = re.compile(r"<[^>]+>")
_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.S | re.I)
_LINK_RE = re.compile(r"<a\s[^>]*href[^>]*>", re.I)
_URL_RE = re.compile(r"(https?://|www\.)\S+", re.I)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_LONGNUM_RE = re.compile(r"\+?\d[\d\s-]{6,}\d")  # phone numbers, codes, ids
_WS_RE = re.compile(r"\s+")


def html_to_text(s):
    s = _STYLE_RE.sub(" ", s)
    s = _LINK_RE.sub(" URL ", s)  # keep the fact that there was a link
    return html.unescape(_TAG_RE.sub(" ", s))


def clean_text(s):
    """Light normalisation used for BOTH models.

    URLs, e-mail addresses and long numbers are replaced by placeholder words:
    the exact URL/phone number of one spam campaign never appears again, but
    "this message contains a URL / phone number" is a signal that transfers to
    new spam (better generalisation, less memorising)."""
    if "<" in s and ">" in s and re.search(r"</?(html|body|div|p|br|font|a|table)\b", s, re.I):
        s = html_to_text(s)
    s = _URL_RE.sub(" URL ", s)
    s = _EMAIL_RE.sub(" EMAIL ", s)
    s = _LONGNUM_RE.sub(" PHONENUMBER ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s[:MAX_CHARS]


class TfidfFeatures:
    """Word + character TF-IDF, concatenated into one sparse matrix."""

    def __init__(self, max_word=25000, max_char=25000):
        self.word = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_df=0.95,
                                    max_features=max_word, sublinear_tf=True,
                                    token_pattern=r"(?u)\b\w[\w']*\b")
        self.char = TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3, 5),
                                    min_df=3, max_features=max_char, sublinear_tf=True)

    def fit_transform(self, texts):
        return hstack([self.word.fit_transform(texts), self.char.fit_transform(texts)]).tocsr()

    def transform(self, texts):
        return hstack([self.word.transform(texts), self.char.transform(texts)]).tocsr()


def load_data(path=DATA_CSV):
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found - run `python -m src.benchmark.prepare_data` first")
    df = pd.read_csv(path, keep_default_na=False)
    return df


def make_splits(df, holdout=None):
    """Return (train, val, test) DataFrames.

    holdout=None  -> standard in-domain experiment: all sources, 70/15/15.
    holdout='sms' -> cross-dataset experiment: train/val on the OTHER sources
                     only, test on every message of the held-out source, which
                     the model has never seen. This is the generalisation test.
    """
    if holdout is None:
        return (df[df.split == "train"], df[df.split == "val"], df[df.split == "test"])
    if holdout not in df.source.unique():
        raise SystemExit(f"holdout '{holdout}' not in data (have {sorted(df.source.unique())})")
    seen = df[df.source != holdout]
    return seen[seen.split == "train"], seen[seen.split == "val"], df[df.source == holdout]


def source_balanced_probs(sources):
    """Sampling probability per training row so that every dataset contributes
    the same number of examples per epoch. Without this, Enron (~60% of all
    messages) dominates and the model under-fits the short SMS style."""
    counts = sources.map(sources.value_counts())
    p = 1.0 / counts.values
    return p / p.sum()


def val_score(y_va, src_va, prob):
    """Mean F1 over the datasets in the validation set (each dataset counts equally)."""
    return float(np.mean([f1_score(y_va[src_va == s], prob[src_va == s] >= 0.5)
                          for s in np.unique(src_va)]))


def compute_metrics(y_true, prob_spam, threshold=0.5):
    y_true = np.asarray(y_true)
    prob_spam = np.asarray(prob_spam)
    y_pred = (prob_spam >= threshold).astype(int)
    m = {
        "n": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y_true, prob_spam) if len(set(y_true)) > 1 else float("nan"),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    m.update(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    return {k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in m.items()}


def evaluate_by_source(test_df, prob_spam):
    """Overall metrics plus a breakdown per dataset."""
    res = {"overall": compute_metrics(test_df.label, prob_spam)}
    prob_spam = np.asarray(prob_spam)
    for src in sorted(test_df.source.unique()):
        mask = (test_df.source == src).values
        res[src] = compute_metrics(test_df.label.values[mask], prob_spam[mask])
    return res


def print_results(res):
    print(f"\n{'subset':<14}{'n':>7}{'acc':>8}{'prec':>8}{'rec':>8}{'f1':>8}{'auc':>8}   fp/fn")
    for k, m in res.items():
        print(f"{k:<14}{m['n']:>7}{m['accuracy']:>8.4f}{m['precision']:>8.4f}"
              f"{m['recall']:>8.4f}{m['f1']:>8.4f}{m['roc_auc']:>8.4f}   {m['fp']}/{m['fn']}")


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def run_name(model, holdout):
    return f"{model}_{'indomain' if holdout is None else 'holdout-' + holdout}"


def distilbert_max_len(model_dir, default=256):
    """Token limit a DistilBERT checkpoint (<run>/best_model) was trained with, read
    from <run>/metrics.json, so inference truncates exactly like training did."""
    path = os.path.join(os.path.dirname(os.path.normpath(model_dir)), "metrics.json")
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return int(json.load(f).get("max_len", default))
