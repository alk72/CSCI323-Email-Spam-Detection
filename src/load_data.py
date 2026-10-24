import email
import html
import re
from email.header import decode_header, make_header

import pandas as pd

from .config import DATA_PROCESSED, DATA_RAW


def _strip_html(text: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return html.unescape(text)


def _decode_subject(msg) -> str:
    raw = msg.get("Subject", "") or ""
    try:
        return str(make_header(decode_header(raw)))
    except Exception:
        return str(raw)


def email_to_text(raw: bytes) -> str:
    msg = email.message_from_bytes(raw)
    parts = [_decode_subject(msg)]
    for part in msg.walk():
        ctype = part.get_content_type()
        if part.is_multipart() or ctype not in ("text/plain", "text/html"):
            continue
        payload = part.get_payload(decode=True)
        if payload is None:
            continue
        charset = part.get_content_charset() or "latin-1"
        try:
            text = payload.decode(charset, errors="replace")
        except LookupError:
            text = payload.decode("latin-1", errors="replace")
        parts.append(_strip_html(text) if ctype == "text/html" else text)
    return "\n".join(parts)


def load_spamassassin() -> pd.DataFrame:
    base = DATA_RAW / "spamassassin"
    folders = {"easy_ham": 0, "easy_ham_2": 0, "hard_ham": 0, "spam": 1, "spam_2": 1}
    rows = []
    for folder, label in folders.items():
        for f in sorted((base / folder).glob("*")):
            if f.is_file() and f.name != "cmds":
                rows.append((email_to_text(f.read_bytes()), label))
    return pd.DataFrame(rows, columns=["text", "label"])


def load_enron() -> pd.DataFrame:
    df = pd.read_csv(DATA_RAW / "enron" / "enron_spam_data.csv")
    text = df["Subject"].fillna("") + "\n" + df["Message"].fillna("")
    label = (df["Spam/Ham"].str.lower() == "spam").astype(int)
    return pd.DataFrame({"text": text, "label": label})


def load_sms() -> pd.DataFrame:
    df = pd.read_csv(DATA_RAW / "sms" / "SMSSpamCollection", sep="\t",
                     header=None, names=["tag", "text"], quoting=3)
    return pd.DataFrame({"text": df["text"], "label": (df["tag"] == "spam").astype(int)})


LOADERS = {"spamassassin": load_spamassassin, "enron": load_enron, "sms": load_sms}


def load_dataset(name: str, use_cache: bool = True) -> pd.DataFrame:
    """Returns columns text, label (1 = spam, 0 = ham), source."""
    cache = DATA_PROCESSED / f"{name}.pkl"
    if use_cache and cache.exists():
        return pd.read_pickle(cache)

    df = LOADERS[name]()
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.strip() != ""]
    # duplicates could end up in both train and test
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    df["source"] = name

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_pickle(cache)
    return df
