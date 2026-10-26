from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def fpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes False Positive Rate (share of ham incorrectly flagged as spam)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": fpr(y_true, y_pred),
    }


def save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def heatmap(ax: plt.Axes, matrix: np.ndarray, row_labels: List[str], col_labels: List[str],
            title: str, fmt: str = "{:.2f}", vmin: float = 0.0, vmax: float = 1.0):
    im = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap="viridis")
    ax.set_xticks(range(len(col_labels)), col_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.set_title(title)
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            v = matrix[i, j]
            ax.text(j, i, fmt.format(v), ha="center", va="center",
                    color="white" if v < (vmin + vmax) / 2 else "black")
    return im


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, title: str, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    heatmap(ax, cm, ["true ham", "true spam"], ["pred ham", "pred spam"], title,
            fmt="{:d}", vmin=0, vmax=max(1, int(cm.max())))
    save_fig(fig, path)


def binned_accuracy(scores: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 5) -> List[Tuple[float, float, float, int]]:
    dist = np.abs(scores)
    edges = np.quantile(dist, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.searchsorted(edges, dist, side="right") - 1, 0, n_bins - 1)
    correct = (y_true == y_pred)
    return [
        (float(edges[b]), float(edges[b + 1]), 
         float(correct[idx == b].mean()) if (idx == b).any() else float("nan"), 
         int((idx == b).sum())) 
        for b in range(n_bins)
    ]
