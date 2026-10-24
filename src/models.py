from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .config import MAX_FEATURES, MIN_DF, NB_ALPHA, RANDOM_STATE, SVM_C


def make_vectorizer(kind: str):
    if kind == "bow":
        return CountVectorizer(max_features=MAX_FEATURES, min_df=MIN_DF)
    if kind == "tfidf":
        return TfidfVectorizer(max_features=MAX_FEATURES, min_df=MIN_DF)
    raise ValueError(kind)


def make_classifier(kind: str):
    if kind == "nb":
        return MultinomialNB(alpha=NB_ALPHA)
    if kind == "svm":
        return LinearSVC(C=SVM_C, max_iter=20000, random_state=RANDOM_STATE)
    if kind == "lr":
        return LogisticRegression(max_iter=2000)
    raise ValueError(kind)


def build_pipeline(model: str, features: str) -> Pipeline:
    # vectorizer is inside the pipeline so it only sees training data
    return Pipeline([("vec", make_vectorizer(features)), ("clf", make_classifier(model))])


def spam_score(pipe: Pipeline, texts):
    """> 0 means spam. NB: log P(spam|x) - log P(ham|x). SVM: w.x + b."""
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "predict_log_proba"):
        logp = pipe.predict_log_proba(texts)
        return logp[:, 1] - logp[:, 0]
    return pipe.decision_function(texts)
