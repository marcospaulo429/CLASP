"""Save retrieval metric summaries as PNG (headless)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def save_retrieval_plot(
    ranking_metrics: dict[str, float],
    ranks: np.ndarray,
    output_path: Path | str,
    ks: Sequence[int],
    *,
    hist_rank_cap: int | None = 200,
) -> None:
    """
    Two subplots: bar chart of Hits@K (and MRR as text); histogram of 1-based ranks.

    Parameters
    ----------
    hist_rank_cap :
        If set, x-axis for the histogram stops at this rank; counts for ranks above
        are merged into the last bin.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    k_sorted = sorted(set(int(k) for k in ks if k > 0))
    hit_keys = [f"Hits@{k}" for k in k_sorted]
    hit_vals = [ranking_metrics.get(h, 0.0) for h in hit_keys]
    labels = [f"@{k}" for k in k_sorted]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax0 = axes[0]
    x = np.arange(len(hit_vals))
    ax0.bar(x, hit_vals, color="steelblue", edgecolor="navy", alpha=0.85)
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels)
    ax0.set_ylim(0, 1.05)
    ax0.set_ylabel("Rate")
    ax0.set_xlabel("Hits@K")
    ax0.set_title("Hits@K (single relevant per query)")

    mrr = ranking_metrics.get("MRR", float("nan"))
    mean_r = ranking_metrics.get("mean_rank", float("nan"))
    med_r = ranking_metrics.get("median_rank", float("nan"))
    ax0.text(
        0.98,
        0.95,
        f"MRR = {mrr:.4f}\nmean rank = {mean_r:.1f}\nmedian rank = {med_r:.1f}",
        transform=ax0.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax1 = axes[1]
    if ranks.size == 0:
        ax1.text(0.5, 0.5, "no queries", ha="center", va="center")
    else:
        r = ranks.astype(np.int64)
        if hist_rank_cap is not None and r.max() > hist_rank_cap:
            r_plot = np.minimum(r, hist_rank_cap)
            bins = min(40, max(10, hist_rank_cap // 5))
            ax1.hist(r_plot, bins=bins, range=(1, hist_rank_cap), color="seagreen", edgecolor="darkgreen", alpha=0.85)
            ax1.set_xlim(1, hist_rank_cap)
            ax1.set_title(f"Rank of relevant (capped at {hist_rank_cap})")
        else:
            max_r = int(r.max())
            nb = min(50, max(10, int(math.sqrt(len(r)))))
            ax1.hist(r, bins=nb, range=(1, max_r + 1), color="seagreen", edgecolor="darkgreen", alpha=0.85)
            ax1.set_title("Rank of relevant (1-based)")
        ax1.set_xlabel("Rank")
        ax1.set_ylabel("Count")

    fig.suptitle("Retrieval evaluation", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
