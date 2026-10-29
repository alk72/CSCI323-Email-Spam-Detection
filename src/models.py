import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .config import (
    DISTILBERT_BATCH_SIZE,
    DISTILBERT_EPOCHS,
    DISTILBERT_LR,
    MAX_FEATURES,
    MIN_DF,
    MLP_BATCH_SIZE,
    MLP_EPOCHS,
    MLP_LR,
    NB_ALPHA,
    RANDOM_STATE,
    SVM_C,
)


class PyTorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """Multi-Layer Perceptron (Neural Network) for Spam Detection."""
    def __init__(self, hidden_dim1=128, hidden_dim2=64, epochs=MLP_EPOCHS, batch_size=MLP_BATCH_SIZE, lr=MLP_LR, dropout=0.3, random_state=RANDOM_STATE):
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.dropout = dropout
        self.random_state = random_state

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        input_dim = X.shape[1]

        self.model_ = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim1),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim1, self.hidden_dim2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim2, 1),
            nn.Sigmoid()
        )

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model_.train()
        for epoch in range(self.epochs):
            for bx, by in dataloader:
                optimizer.zero_grad()
                out = self.model_(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        X_t = torch.tensor(X, dtype=torch.float32)

        self.model_.eval()
        with torch.no_grad():
            probs1 = self.model_(X_t).numpy().flatten()
        probs0 = 1.0 - probs1
        return np.column_stack((probs0, probs1))

    def predict_log_proba(self, X):
        proba = self.predict_proba(X)
        return np.log(np.clip(proba, 1e-12, 1.0))

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)


class DistilBERTClassifier(BaseEstimator, ClassifierMixin):
    """DistilBERT Fine-Tuned Transformer Classifier for Spam Detection."""
    def __init__(self, model_name="distilbert-base-uncased", epochs=DISTILBERT_EPOCHS, batch_size=DISTILBERT_BATCH_SIZE, lr=DISTILBERT_LR, max_length=128, random_state=RANDOM_STATE):
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.max_length = max_length
        self.random_state = random_state

    def fit(self, X, y):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.tokenizer_ = AutoTokenizer.from_pretrained(self.model_name)
        self.model_ = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)

        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_.to(self.device_)

        texts = list(X)
        labels = list(y)

        encodings = self.tokenizer_(texts, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt")
        dataset = torch.utils.data.TensorDataset(encodings["input_ids"], encodings["attention_mask"], torch.tensor(labels, dtype=torch.long))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = AdamW(self.model_.parameters(), lr=self.lr)

        self.model_.train()
        for epoch in range(self.epochs):
            for batch_ids, batch_mask, batch_y in dataloader:
                batch_ids, batch_mask, batch_y = batch_ids.to(self.device_), batch_mask.to(self.device_), batch_y.to(self.device_)
                optimizer.zero_grad()
                outputs = self.model_(input_ids=batch_ids, attention_mask=batch_mask, labels=batch_y)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        texts = list(X)
        encodings = self.tokenizer_(texts, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt")

        dataset = torch.utils.data.TensorDataset(encodings["input_ids"], encodings["attention_mask"])
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.model_.eval()
        probs_list = []
        with torch.no_grad():
            for batch_ids, batch_mask in dataloader:
                batch_ids, batch_mask = batch_ids.to(self.device_), batch_mask.to(self.device_)
                logits = self.model_(input_ids=batch_ids, attention_mask=batch_mask).logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                probs_list.append(probs)

        return np.vstack(probs_list)

    def predict_log_proba(self, X):
        proba = self.predict_proba(X)
        return np.log(np.clip(proba, 1e-12, 1.0))

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


def make_vectorizer(kind: str):
    if kind == "bow":
        return CountVectorizer(max_features=MAX_FEATURES, min_df=MIN_DF)
    if kind == "tfidf":
        return TfidfVectorizer(max_features=MAX_FEATURES, min_df=MIN_DF)
    if kind == "passthrough":
        return "passthrough"
    raise ValueError(kind)


def make_classifier(kind: str):
    if kind == "nb":
        return MultinomialNB(alpha=NB_ALPHA)
    if kind == "svm":
        return LinearSVC(C=SVM_C, max_iter=20000, random_state=RANDOM_STATE)
    if kind == "mlp":
        return PyTorchMLPClassifier(epochs=MLP_EPOCHS, batch_size=MLP_BATCH_SIZE, lr=MLP_LR, random_state=RANDOM_STATE)
    if kind == "distilbert":
        return DistilBERTClassifier(epochs=DISTILBERT_EPOCHS, batch_size=DISTILBERT_BATCH_SIZE, lr=DISTILBERT_LR, random_state=RANDOM_STATE)
    if kind == "lr":
        return LogisticRegression(max_iter=2000)
    raise ValueError(kind)


def build_pipeline(model: str, features: str) -> Pipeline:
    # DistilBERT performs tokenization directly on raw text
    if model == "distilbert":
        return Pipeline([("vec", "passthrough"), ("clf", make_classifier("distilbert"))])
    return Pipeline([("vec", make_vectorizer(features)), ("clf", make_classifier(model))])


def spam_score(pipe: Pipeline, texts):
    """Returns classification confidence score."""
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "predict_log_proba"):
        logp = pipe.predict_log_proba(texts)
        return logp[:, 1] - logp[:, 0]
    return pipe.decision_function(texts)
