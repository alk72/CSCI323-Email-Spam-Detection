"""
Collect every results/benchmark/*/metrics.json into one comparison table.

    python -m src.benchmark.summarize    -> prints the table, writes results/benchmark/results_summary.csv
"""
import glob
import json
import os

import pandas as pd

from .common import OUT_DIR


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(OUT_DIR, "*", "metrics.json"))):
        with open(path) as f:
            m = json.load(f)
        for subset, t in m["test"].items():
            if subset != "overall" and m["holdout"] is not None:
                continue  # cross-dataset runs have a single test source
            rows.append({
                "model": m["model"],
                "experiment": "in-domain" if m["holdout"] is None else f"train w/o {m['holdout']}",
                "test_set": subset if m["holdout"] is None else m["holdout"] + " (unseen)",
                **{k: t[k] for k in ("n", "accuracy", "precision", "recall", "f1", "roc_auc", "fp", "fn")},
            })

    if not rows:
        raise SystemExit("no results yet - train a model first")
    df = pd.DataFrame(rows).sort_values(["experiment", "test_set", "model"])
    df.to_csv(os.path.join(OUT_DIR, "results_summary.csv"), index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
