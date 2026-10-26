from typing import Literal
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .config import MAX_FEATURES, MIN_DF, NB_ALPHA, RANDOM_STATE, SVM_C

FeatureKind = Literal["bow", "tfidf"]
ModelKind = Literal["nb", "svm", "lr"]


def make_vectorizer(kind: FeatureKind):
    if kind == "bow":
        return CountVectorizer(max_features=MAX_FEATURES, min_df=MIN_DF)
    if kind == "tfidf":
        return TfidfVectorizer(max_features=MAX_FEATURES, min_df=MIN_DF, sublinear_tf=True)
    raise ValueError(f"Unsupported feature type: '{kind}'")


def make_classifier(kind: ModelKind):
    if kind == "nb":
        return MultinomialNB(alpha=NB_ALPHA)
    if kind == "svm":
        return LinearSVC(C=SVM_C, dual="auto", max_iter=20000, random_state=RANDOM_STATE)
    if kind == "lr":
        return LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    raise ValueError(f"Unsupported model type: '{kind}'")


def build_pipeline(model: ModelKind, features: FeatureKind) -> Pipeline:
    """Builds an isolated scikit-learn preprocessing and classification pipeline."""
    return Pipeline([
        ("vec", make_vectorizer(features)),
        ("clf", make_classifier(model))
    ])


def spam_score(pipe: Pipeline, texts):
    """Computes real-valued decision boundary score (> 0 indicates spam)."""
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "predict_log_proba"):
        logp = pipe.predict_log_proba(texts)
        return logp[:, 1] - logp[:, 0]
    return pipe.decision_function(texts)
