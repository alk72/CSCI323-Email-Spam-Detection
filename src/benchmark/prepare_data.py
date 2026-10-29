"""
Download the three public spam datasets and merge them into one clean CSV.

    python download_data.py

Output: data/spam_all.csv with columns
    text    - cleaned message (subject + body for emails)
    label   - 1 = spam, 0 = ham (not spam)
    source  - sms | enron | spamassassin
    split   - train | val | test  (70 / 15 / 15, stratified per source, fixed seed)

Datasets
  * SMS Spam Collection (Almeida et al., 2011)        ~5.5k SMS,     ~13% spam
  * Enron-Spam (Metsis et al., 2006)                   ~33k emails,   ~51% spam
  * SpamAssassin public corpus (Apache, 2002-2005)     ~6k emails,    ~31% spam
"""
import argparse
import email
import os
import tarfile
import urllib.request
import zipfile
from email import policy

import pandas as pd
from sklearn.model_selection import train_test_split

from common import DATA_DIR, SEED, clean_text, html_to_text

RAW_DIR = os.path.join(DATA_DIR, "raw")

SMS_URLS = [
    "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
    # mirror of the same file, used if UCI is down
    "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv",
]
ENRON_URL = "https://raw.githubusercontent.com/MWiechmann/enron_spam_data/master/enron_spam_data.zip"
SA_BASE = "https://spamassassin.apache.org/old/publiccorpus/"
SA_FILES = {  # file -> label
    "20030228_easy_ham.tar.bz2": 0,
    "20030228_easy_ham_2.tar.bz2": 0,
    "20030228_hard_ham.tar.bz2": 0,
    "20030228_spam.tar.bz2": 1,
    "20050311_spam_2.tar.bz2": 1,
}


def download(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest + ".part", "wb") as f:
        f.write(r.read())
    os.replace(dest + ".part", dest)
    return dest


# ---------------------------------------------------------------- SMS
def load_sms():
    for url in SMS_URLS:
        dest = os.path.join(RAW_DIR, os.path.basename(url).replace("+", "_"))
        try:
            download(url, dest)
        except Exception as e:  # try the mirror
            print(f"  failed ({e}), trying next source")
            continue
        if dest.endswith(".zip"):
            with zipfile.ZipFile(dest) as z:
                raw = z.read("SMSSpamCollection").decode("utf-8", errors="replace")
        else:
            with open(dest, encoding="utf-8", errors="replace") as f:
                raw = f.read()
        rows = []
        for line in raw.splitlines():
            if "\t" not in line:
                continue
            lab, txt = line.split("\t", 1)
            rows.append({"text": txt, "label": int(lab.strip() == "spam")})
        return pd.DataFrame(rows)
    raise RuntimeError("Could not download the SMS Spam Collection")


# ---------------------------------------------------------------- Enron
def load_enron():
    dest = download(ENRON_URL, os.path.join(RAW_DIR, "enron_spam_data.zip"))
    df = pd.read_csv(dest, compression="zip")
    text = df["Subject"].fillna("") + "\n" + df["Message"].fillna("")
    return pd.DataFrame({"text": text, "label": (df["Spam/Ham"] == "spam").astype(int)})


# ---------------------------------------------------------- SpamAssassin
def parse_raw_email(raw_bytes):
    """Keep only Subject + readable body. All other headers are dropped so the
    model cannot learn shortcuts from server names, dates, etc."""
    msg = email.message_from_bytes(raw_bytes, policy=policy.default)
    subject = str(msg.get("Subject", "") or "")
    parts = []
    for part in msg.walk():
        if part.is_multipart() or part.get_content_maintype() != "text":
            continue
        try:
            payload = part.get_content()
        except Exception:
            payload = (part.get_payload(decode=True) or b"").decode("latin-1", errors="replace")
        if part.get_content_subtype() == "html":
            payload = html_to_text(payload)
        parts.append(payload)
    return subject + "\n" + "\n".join(parts)


def load_spamassassin():
    rows = []
    for fname, label in SA_FILES.items():
        dest = download(SA_BASE + fname, os.path.join(RAW_DIR, fname))
        with tarfile.open(dest, "r:bz2") as tar:
            for m in tar.getmembers():
                if not m.isfile() or os.path.basename(m.name) == "cmds":
                    continue
                rows.append({"text": parse_raw_email(tar.extractfile(m).read()), "label": label})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", nargs="*", default=[], help="sources to skip, e.g. --skip spamassassin")
    args = ap.parse_args()
    os.makedirs(RAW_DIR, exist_ok=True)

    loaders = {"sms": load_sms, "enron": load_enron, "spamassassin": load_spamassassin}
    frames = []
    for name, fn in loaders.items():
        if name in args.skip:
            continue
        print(f"[{name}]")
        try:
            d = fn()
        except Exception as e:
            print(f"  ERROR loading {name}: {e}\n  (continuing without it)")
            continue
        d["source"] = name
        print(f"  {len(d)} messages, {d.label.mean():.1%} spam")
        frames.append(d)

    df = pd.concat(frames, ignore_index=True)
    df["text"] = df["text"].astype(str).map(clean_text)
    df = df[df["text"].str.len() > 0]

    # Remove exact duplicates (after cleaning). Duplicates that land in both
    # train and test would inflate test scores, i.e. fake generalisation.
    # Texts that appear with BOTH labels are ambiguous and dropped entirely.
    n0 = len(df)
    conflict = df.groupby("text")["label"].transform("nunique") > 1
    df = df[~conflict].drop_duplicates("text").reset_index(drop=True)
    print(f"\nRemoved {n0 - len(df)} duplicate/conflicting messages")

    # Stratified 70/15/15 split inside each source (fixed seed = reproducible)
    df["split"] = ""
    for src, g in df.groupby("source"):
        tr, rest = train_test_split(g.index, test_size=0.30, stratify=g.label, random_state=SEED)
        va, te = train_test_split(rest, test_size=0.50, stratify=g.loc[rest, "label"], random_state=SEED)
        df.loc[tr, "split"], df.loc[va, "split"], df.loc[te, "split"] = "train", "val", "test"

    out = os.path.join(DATA_DIR, "spam_all.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} messages -> {out}")
    print(pd.crosstab([df.source, df.split], df.label.map({0: "ham", 1: "spam"}), margins=True))


if __name__ == "__main__":
    main()
