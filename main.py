"""
python main.py download
python main.py all
python main.py cv --datasets enron
python main.py margin --target spamassassin
"""
import argparse
import time

from src.config import DATASETS, RESULTS
from src.download_data import download
from src.experiments import run_attack, run_cross, run_cv, run_explore, run_margin
from src.load_data import load_dataset
from src.preprocess import add_clean_column

STEPS = ["explore", "cv", "cross", "margin", "attack"]


def load_all(names):
    data = {}
    for n in names:
        t = time.time()
        data[n] = add_clean_column(load_dataset(n))
        print(f"loaded {n}: {len(data[n])} messages ({time.time() - t:.1f}s)")
    return data


def main():
    p = argparse.ArgumentParser(description="Email spam detection experiments")
    p.add_argument("step", choices=["download", "all"] + STEPS)
    p.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    p.add_argument("--target", default="spamassassin", choices=DATASETS,
                   help="dataset used by the margin and attack experiments")
    args = p.parse_args()

    if args.step == "download":
        download(args.datasets)
        return

    names = sorted(set(args.datasets) | ({args.target} if args.step in ("margin", "attack", "all") else set()),
                   key=DATASETS.index)
    data = load_all(names)
    steps = STEPS if args.step == "all" else [args.step]
    for step in steps:
        print(f"\n=== {step} ===")
        t = time.time()
        if step == "explore":
            run_explore(data, RESULTS)
        elif step == "cv":
            run_cv({n: data[n] for n in args.datasets}, RESULTS)
        elif step == "cross":
            run_cross({n: data[n] for n in args.datasets}, RESULTS)
        elif step == "margin":
            run_margin(data, RESULTS, args.target)
        elif step == "attack":
            run_attack(data, RESULTS, args.target)
        print(f"({time.time() - t:.1f}s)")
    print(f"\nResults are in {RESULTS}")


if __name__ == "__main__":
    main()
