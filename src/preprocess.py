import re
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\S+@\S+")
NON_LETTERS_RE = re.compile(r"[^a-z\s]")


def clean_text(text: str) -> str:
    """Normalize and clean raw input text by removing URLs, emails, and special characters."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = NON_LETTERS_RE.sub(" ", text)
    tokens = [t for t in text.split() if len(t) > 1 and t not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)


def add_clean_column(df: pd.DataFrame) -> pd.DataFrame:
    """Applies cleaning to dataframe text column and removes uninformative empty rows."""
    df = df.copy()
    df["clean"] = df["text"].astype(str).map(clean_text)
    return df[df["clean"].str.strip() != ""].reset_index(drop=True)
