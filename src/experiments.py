import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from . import adversarial
from .config import CV_FOLDS, FEATURES, MODELS, RANDOM_STATE, TEST_SIZE
from .evaluate import binned_accuracy, compute_metrics, fpr, heatmap, plot_confusion, save_fig
from .models import build_pipeline, spam_score
from .preprocess import clean_text

COMBOS = [(m, f) for m in MODELS for f in FEATURES]


def _split(df):
    return train_test_split(df, test_size=TEST_SIZE, stratify=df["label"],
                            random_state=RANDOM_STATE)


def _save_table(df, out_dir, name):
    path = out_dir / "tables" / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  saved {path.relative_to(out_dir.parent)}")


def run_explore(data, out_dir):
    stats, top = [], []
    for name, df in data.items():
        words = df["clean"].str.split().str.len()
        stats.append({"dataset": name, "messages": len(df), "spam": int(df.label.sum()),
                      "ham": int((df.label == 0).sum()), "spam_share": round(df.label.mean(), 3),
                      "median_words": int(words.median()), "mean_words": round(words.mean(), 1)})
        for label, cls in [(1, "spam"), (0, "ham")]:
            vec = CountVectorizer(max_features=15)
            counts = vec.fit_transform(df.loc[df.label == label, "clean"]).sum(axis=0).A1
            order = np.argsort(-counts)
            top.append({"dataset": name, "class": cls,
                        "top_words": ", ".join(vec.get_feature_names_out()[order])})
    stats = pd.DataFrame(stats)
    print(stats.to_string(index=False))
    _save_table(stats, out_dir, "dataset_stats")
    _save_table(pd.DataFrame(top), out_dir, "top_words")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    stats.plot.bar(x="dataset", y=["ham", "spam"], ax=axes[0], title="Messages per class")
    for name, df in data.items():
        axes[1].hist(df["clean"].str.split().str.len().clip(upper=1000), bins=50,
                     histtype="step", label=name, density=True)
    axes[1].set(title="Words per message (after cleaning)", xlabel="words", yscale="log")
    axes[1].legend()
    save_fig(fig, out_dir / "figures" / "dataset_overview.png")


def run_cv(data, out_dir, folds=CV_FOLDS):
    scoring = {"accuracy": "accuracy", "precision": "precision", "recall": "recall",
               "f1": "f1", "fpr": make_scorer(fpr)}
    cv = StratifiedKFold(folds, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, df in data.items():
        for model, feats in COMBOS:
            print(f"  {name}: {model} + {feats}")
            res = cross_validate(build_pipeline(model, feats), df["clean"], df["label"],
                                 cv=cv, scoring=scoring, n_jobs=-1)
            row = {"dataset": name, "model": model, "features": feats}
            for k in scoring:
                row[f"{k}_mean"] = res[f"test_{k}"].mean()
                row[f"{k}_std"] = res[f"test_{k}"].std()
            rows.append(row)
    table = pd.DataFrame(rows)
    print(table[["dataset", "model", "features", "accuracy_mean", "f1_mean", "fpr_mean"]]
          .round(4).to_string(index=False))
    _save_table(table, out_dir, "cv_results")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    table["combo"] = table["model"] + "+" + table["features"]
    for ax, metric in zip(axes, ["f1", "fpr"]):
        table.pivot(index="dataset", columns="combo", values=f"{metric}_mean").plot.bar(ax=ax)
        ax.set_title(f"{folds}-fold CV: {metric.upper()} (mean)")
        ax.tick_params(axis="x", rotation=0)
    save_fig(fig, out_dir / "figures" / "cv_results.png")


def run_cross(data, out_dir):
    names = list(data)
    splits = {n: _split(df) for n, df in data.items()}
    rows = []
    for model, feats in COMBOS:
        for train_name in names:
            pipe = build_pipeline(model, feats).fit(splits[train_name][0]["clean"],
                                                    splits[train_name][0]["label"])
            for test_name in names:
                test = splits[test_name][1] if test_name == train_name else data[test_name]
                pred = pipe.predict(test["clean"])
                rows.append({"model": model, "features": feats, "train": train_name,
                             "test": test_name, **compute_metrics(test["label"], pred)})
    table = pd.DataFrame(rows)
    _save_table(table, out_dir, "cross_dataset")

    for metric in ["accuracy", "fpr", "recall"]:
        fig, axes = plt.subplots(len(MODELS), len(FEATURES), figsize=(4.2 * len(FEATURES), 3.6 * len(MODELS)),
                                 squeeze=False)
        for (model, feats), ax in zip(COMBOS, axes.ravel()):
            sub = table[(table.model == model) & (table.features == feats)]
            m = sub.pivot(index="train", columns="test", values=metric).loc[names, names].values
            heatmap(ax, m, [f"train {n}" for n in names], [f"test {n}" for n in names],
                    f"{model} + {feats}")
        fig.suptitle(f"Cross-dataset {metric} (diagonal = same dataset)")
        save_fig(fig, out_dir / "figures" / f"cross_{metric}.png")
    print(table.pivot_table(index=["model", "features", "train"], columns="test",
                            values="accuracy").round(3).to_string())


def run_margin(data, out_dir, dataset, feats="tfidf"):
    train, test = _split(data[dataset])
    summary, bins, borderline, words = [], [], [], []
    fig, axes = plt.subplots(1, len(MODELS), figsize=(5.5 * len(MODELS), 3.8), squeeze=False)
    for model, ax in zip(MODELS, axes.ravel()):
        pipe = build_pipeline(model, feats).fit(train["clean"], train["label"])
        s = spam_score(pipe, test["clean"])
        y, pred = test["label"].values, (s > 0).astype(int)
        plot_confusion(y, pred, f"{model} on {dataset}",
                       out_dir / "figures" / f"confusion_{dataset}_{model}.png")

        near = np.abs(s) < 1
        summary.append({"model": model, "test_emails": len(y), "accuracy": (y == pred).mean(),
                        "share_near_boundary": near.mean(),
                        "error_rate_near": (y[near] != pred[near]).mean() if near.any() else np.nan,
                        "error_rate_far": (y[~near] != pred[~near]).mean()})
        for lo, hi, acc, n in binned_accuracy(s, y, pred):
            bins.append({"model": model, "abs_score": f"{lo:.2f}-{hi:.2f}",
                            "accuracy": acc, "count": n})

        order = np.argsort(np.abs(s))[:15]
        for i in order:
            borderline.append({"model": model, "score": s[i], "true": y[i], "pred": pred[i],
                               "text": test["text"].iloc[i][:300].replace("\n", " ")})

        vocab = pipe.named_steps["vec"].get_feature_names_out()
        clf = pipe.named_steps["clf"]
        w = (clf.feature_log_prob_[1] - clf.feature_log_prob_[0]) if model == "nb" else clf.coef_.ravel()
        words.append({"model": model, "top_spam_words": ", ".join(vocab[np.argsort(-w)[:20]]),
                      "top_ham_words": ", ".join(vocab[np.argsort(w)[:20]])})

        clip = np.percentile(np.abs(s), 98)
        for label, name in [(0, "ham"), (1, "spam")]:
            ax.hist(np.clip(s[y == label], -clip, clip), bins=60, alpha=0.6, label=f"true {name}")
        ax.axvline(0, color="k")
        if model == "svm":
            for v in (-1, 1):
                ax.axvline(v, color="k", ls="--", lw=0.8)
        ax.set(title=f"{model}: spam score on {dataset} test set",
               xlabel="log-odds" if model == "nb" else "w.x + b (dashed = margin)")
        ax.legend()
    save_fig(fig, out_dir / "figures" / f"margin_{dataset}.png")
    summary, bins = pd.DataFrame(summary), pd.DataFrame(bins)
    print(summary.round(4).to_string(index=False))
    print(bins.round(4).to_string(index=False))
    _save_table(summary, out_dir, f"margin_{dataset}")
    _save_table(bins, out_dir, f"margin_bins_{dataset}")
    _save_table(pd.DataFrame(borderline), out_dir, f"borderline_emails_{dataset}")
    _save_table(pd.DataFrame(words), out_dir, f"top_words_by_model_{dataset}")


def run_attack(data, out_dir, dataset,
               rates=(0, 0.25, 0.5, 0.75, 1.0), n_good=(0, 10, 25, 50, 100)):
    rng = np.random.default_rng(RANDOM_STATE)
    train, test = _split(data[dataset])
    spam_raw = test.loc[test.label == 1, "text"].tolist()
    rows = []
    for model, feats in COMBOS:
        pipe = build_pipeline(model, feats).fit(train["clean"], train["label"])
        good = adversarial.top_ham_words(pipe)
        spammy = set(adversarial.top_spam_words(pipe))
        for r in rates:
            attacked = [clean_text(adversarial.obfuscate_text(t, r, rng, spammy)) for t in spam_raw]
            rows.append({"model": model, "features": feats, "attack": "obfuscation",
                         "strength": r, "spam_caught": pipe.predict(attacked).mean()})
        for n in n_good:
            attacked = [clean_text(adversarial.good_word_attack(t, good, n, rng)) for t in spam_raw]
            rows.append({"model": model, "features": feats, "attack": "good_words",
                         "strength": n, "spam_caught": pipe.predict(attacked).mean()})
    table = pd.DataFrame(rows)
    print(table.pivot_table(index=["attack", "strength"], columns=["model", "features"],
                            values="spam_caught").round(3).to_string())
    _save_table(table, out_dir, f"attack_{dataset}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (attack, xlabel) in zip(axes, [("obfuscation", "share of top spam words obfuscated"),
                                          ("good_words", "ham words appended")]):
        for (model, feats), g in table[table.attack == attack].groupby(["model", "features"]):
            ax.plot(g["strength"], g["spam_caught"], marker="o", label=f"{model}+{feats}")
        ax.set(title=f"{attack} attack on {dataset}", xlabel=xlabel,
               ylabel="spam recall (share caught)", ylim=(0, 1.02))
        ax.legend()
    save_fig(fig, out_dir / "figures" / f"attack_{dataset}.png")
