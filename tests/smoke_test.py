"""Runs every experiment on small fake data. Usage: python -m tests.smoke_test"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments import run_attack, run_cross, run_cv, run_explore, run_margin
from src.preprocess import add_clean_column

SPAM = "free winner cash prize offer click claim money urgent viagra discount".split()
HAM = "meeting project report schedule lunch team review attached thanks agenda".split()
FILL = "please today week time update note call message".split()


def fake_dataset(name, n=300, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        label = int(rng.random() < 0.4)
        words = rng.choice(SPAM if label else HAM, 6).tolist() + rng.choice(FILL + SPAM + HAM, 6).tolist()
        rows.append((" ".join(words) + f" id{i}x", label))
    df = pd.DataFrame(rows, columns=["text", "label"])
    df["source"] = name
    return add_clean_column(df)


def main():
    data = {n: fake_dataset(n, seed=i) for i, n in enumerate(["a", "b", "c"])}
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "results"
        run_explore(data, out)
        run_cv(data, out, folds=3)
        run_cross(data, out)
        run_margin(data, out, "a")
        run_attack(data, out, "a")
        files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    assert any(f.endswith(".png") for f in files) and any(f.endswith(".csv") for f in files)
    print(f"\nSmoke test passed: {len(files)} output files")


if __name__ == "__main__":
    main()
