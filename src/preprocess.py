import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\S+@\S+")
NON_LETTERS_RE = re.compile(r"[^a-z\s]")


def clean_text(text: str) -> str:
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = NON_LETTERS_RE.sub(" ", text)
    tokens = [t for t in text.split() if len(t) > 1 and t not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)


def add_clean_column(df):
    df = df.copy()
    df["clean"] = df["text"].map(clean_text)
    return df[df["clean"] != ""].reset_index(drop=True)
