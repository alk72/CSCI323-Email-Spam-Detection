import shutil
import ssl
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import certifi

from .config import DATA_RAW

SA = "https://spamassassin.apache.org/old/publiccorpus/"

SOURCES = {
    "spamassassin": [
        SA + "20030228_easy_ham.tar.bz2",
        SA + "20030228_easy_ham_2.tar.bz2",
        SA + "20030228_hard_ham.tar.bz2",
        SA + "20030228_spam.tar.bz2",
        SA + "20050311_spam_2.tar.bz2",
    ],
    "enron": [
        "https://github.com/MWiechmann/enron_spam_data/raw/master/enron_spam_data.zip",
    ],
    "sms": [
        "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
    ],
}


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  already have {dest.name}")
        return
    print(f"  downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(url, context=ctx, timeout=60) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.rename(dest)


def _inside(base: Path, target: Path) -> bool:
    return target.resolve().is_relative_to(base.resolve())


def _extract(archive: Path, out_dir: Path) -> None:
    name = archive.name
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            for member in z.namelist():
                if not _inside(out_dir, out_dir / member):
                    raise ValueError(f"unsafe path in {name}: {member}")
            z.extractall(out_dir)
    elif ".tar" in name:
        with tarfile.open(archive) as t:
            members = [m for m in t.getmembers()
                       if (m.isfile() or m.isdir()) and _inside(out_dir, out_dir / m.name)]
            t.extractall(out_dir, members=members)


def download(names=None) -> None:
    for name in names or SOURCES:
        print(f"[{name}]")
        out_dir = DATA_RAW / name
        out_dir.mkdir(parents=True, exist_ok=True)
        for url in SOURCES[name]:
            archive = out_dir / url.rsplit("/", 1)[-1].replace("+", "_")
            _download(url, archive)
            marker = out_dir / (archive.name + ".extracted")
            if not marker.exists():
                _extract(archive, out_dir)
                marker.touch()
    print("Done. Data is in", DATA_RAW)
