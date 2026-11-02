"""
Model 1: TF-IDF features + Multi-Layer Perceptron (MLP).

    python -m src.benchmark.train_mlp                       # train on all 3 datasets, test in-domain
    python -m src.benchmark.train_mlp --holdout sms         # cross-dataset: never sees SMS during training
    python -m src.benchmark.train_mlp --tune                # small hyper-parameter search on the validation set

Pipeline
  text -> TF-IDF (word 1-2 grams + character 3-5 grams) -> MLP (ReLU hidden layer(s),
  softmax output) trained with Adam on cross-entropy loss.

Generalisation measures
  * L2 weight penalty (alpha)  - keeps weights small, smoother decision boundary
  * early stopping on a SEPARATE validation set (patience) - stops before overfitting
  * min_df / max_features      - ignores rare tokens that are basically noise
  * sublinear_tf               - log(1+tf), so repeating a word 50x is not 50x stronger
  * character n-grams          - robust to obfuscation like "fr33 m0ney" or "v1agra"
"""
import argparse
import copy
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

from .common import (OUT_DIR, SEED, SOURCES, TfidfFeatures, evaluate_by_source, load_data,
                    make_splits, print_results, run_name, save_json,
                    source_balanced_probs, val_score)


def train_mlp(X_tr, y_tr, X_va, y_va, src_va, hidden, alpha, lr, max_epochs, patience,
              batch_size, sample_p=None):
    """Mini-batch training with early stopping on our own validation set."""
    clf = MLPClassifier(hidden_layer_sizes=hidden, activation="relu", solver="adam",
                        alpha=alpha, learning_rate_init=lr, batch_size=batch_size,
                        random_state=SEED)
    rng = np.random.default_rng(SEED)
    best, best_f1, bad, history = None, -1.0, 0, []
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        n = X_tr.shape[0]
        if sample_p is None:
            idx = rng.permutation(n)  # reshuffle every epoch
        else:
            idx = rng.choice(n, size=n, p=sample_p)  # source-balanced resampling
        clf.partial_fit(X_tr[idx], y_tr[idx], classes=[0, 1])  # = one epoch of mini-batch Adam
        val_prob = clf.predict_proba(X_va)[:, 1]
        val_f1 = val_score(y_va, src_va, val_prob)
        history.append({"epoch": epoch, "train_loss": round(clf.loss_, 5), "val_f1": round(val_f1, 5)})
        print(f"  epoch {epoch:2d}  loss {clf.loss_:.4f}  val F1 {val_f1:.4f}  ({time.time() - t0:.1f}s)")
        if val_f1 > best_f1:
            best, best_f1, bad = copy.deepcopy(clf), val_f1, 0
        else:
            bad += 1
            if bad >= patience:
                print(f"  early stop: no improvement for {patience} epochs")
                break
    return best, best_f1, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", choices=SOURCES, default=None)
    ap.add_argument("--hidden", type=int, nargs="+", default=[128], help="hidden layer sizes")
    ap.add_argument("--alpha", type=float, default=1e-4, help="L2 penalty")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--no-balance", action="store_true",
                    help="do not balance datasets during training")
    ap.add_argument("--tune", action="store_true", help="grid search hidden/alpha on val set")
    args = ap.parse_args()

    df = load_data()
    train, val, test = make_splits(df, args.holdout)
    print(f"train {len(train)}  val {len(val)}  test {len(test)}  "
          f"(holdout = {args.holdout or 'none, in-domain'})")

    feats = TfidfFeatures()
    X_tr = feats.fit_transform(train.text)  # fitted on TRAIN only (no leakage)
    X_va, X_te = feats.transform(val.text), feats.transform(test.text)
    y_tr, y_va, src_va = train.label.values, val.label.values, val.source.values
    sample_p = None if args.no_balance else source_balanced_probs(train.source)
    print(f"feature dimension: {X_tr.shape[1]}")

    grid = [(tuple(args.hidden), args.alpha)]
    if args.tune:
        grid = [((h,), a) for h in (64, 256) for a in (1e-5, 1e-4, 1e-3)] + [((256, 64), 1e-4)]

    best = None
    for hidden, alpha in grid:
        print(f"\nMLP hidden={hidden} alpha={alpha}")
        clf, f1, hist = train_mlp(X_tr, y_tr, X_va, y_va, src_va, hidden, alpha, args.lr,
                                  args.epochs, args.patience, args.batch_size, sample_p)
        print(f"  best val F1 = {f1:.4f}")
        if best is None or f1 > best[1]:
            best = (clf, f1, hist, hidden, alpha)
    clf, val_f1, history, hidden, alpha = best

    res = evaluate_by_source(test, clf.predict_proba(X_te)[:, 1])
    print(f"\nTEST results (hidden={hidden}, alpha={alpha}):")
    print_results(res)

    out = os.path.join(OUT_DIR, run_name("mlp", args.holdout))
    os.makedirs(out, exist_ok=True)
    joblib.dump({"features": feats, "mlp": clf}, os.path.join(out, "model.joblib"))
    pd.DataFrame(history).to_csv(os.path.join(out, "history.csv"), index=False)
    save_json({"model": "mlp", "holdout": args.holdout, "hidden": list(hidden), "alpha": alpha,
               "balanced": not args.no_balance,
               "val_f1": round(val_f1, 4), "test": res}, os.path.join(out, "metrics.json"))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
