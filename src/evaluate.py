"""
Evaluation: 
    -metrics
    -plots (roc, pr, latent umap)
    -feature-importance helpers (gradient importance)
"""

from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

def classification_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "positive_rate": float(y_true.mean()),
        "threshold": float(threshold),
    }

def plot_roc(curves: dict[str, tuple[np.ndarray, np.ndarray]], out: Path | None = None):
    # curves maps model_name -> (y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, (y_true, y_prob) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUROC={roc_auc_score(y_true, y_prob):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=200)
    return fig

def plot_pr(curves: dict[str, tuple[np.ndarray, np.ndarray]], out: Path | None = None):
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, (y_true, y_prob) in curves.items():
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ax.plot(rec, prec, label=f"{name} (AUPRC={average_precision_score(y_true, y_prob):.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=200)
    return fig


def plot_latent_umap(
    Z: np.ndarray,
    obs: pd.DataFrame,
    color_cols: list[str],
    n_neighbors: int = 30,
    min_dist: float = 0.3,
    seed: int = 0,
    out: Path | None = None,
):
    # compute a single UMAP of the latent matrix, color each column
    import umap
    ####
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed)
    emb = reducer.fit_transform(Z)
    n = len(color_cols)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    for ax, col in zip(axes[0], color_cols):
        vals = obs[col].values
        if pd.api.types.is_numeric_dtype(vals):
            sc = ax.scatter(emb[:, 0], emb[:, 1], c=vals, s=2, cmap="viridis", alpha=0.7)
            plt.colorbar(sc, ax=ax)
        else:
            cats = pd.Categorical(vals)
            for i, c in enumerate(cats.categories):
                m = cats == c
                ax.scatter(emb[m, 0], emb[m, 1], s=2, label=str(c), alpha=0.7)
            ax.legend(markerscale=4, fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.set_title(col)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=200, bbox_inches="tight")
    return fig, emb


def top_logreg_genes(
    coefs: np.ndarray, gene_names: list[str], k: int = 25
) -> pd.DataFrame:
    # return the top k positive and top k negative coefficients
    s = pd.Series(coefs.ravel(), index=gene_names).sort_values()
    bottom = s.head(k).rename("coef").reset_index().rename(columns={"index": "gene"})
    bottom["direction"] = "down (singleton)"
    top = s.tail(k).iloc[::-1].rename("coef").reset_index().rename(columns={"index": "gene"})
    top["direction"] = "up (expanded)"
    return pd.concat([top, bottom], ignore_index=True)


def gradient_importance(model, X: np.ndarray, gene_names: list[str], device: str | None = None,
                        batch_size: int = 512) -> pd.DataFrame:
    # mean absolute gradient of the positive-class logit w.r.t. each gene
    import torch
    ##
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    sums = np.zeros(X.shape[1], dtype=np.float64)
    count = 0
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i : i + batch_size].astype(np.float32)).to(device).requires_grad_(True)
        out = model(xb)
        logit = out[1] if isinstance(out, tuple) else out
        logit.sum().backward()
        sums += xb.grad.detach().abs().cpu().numpy().sum(axis=0)
        count += xb.size(0)
    mean_abs_grad = sums / count
    return (
        pd.DataFrame({"gene": gene_names, "mean_abs_grad": mean_abs_grad})
        .sort_values("mean_abs_grad", ascending=False)
        .reset_index(drop=True)
    )
