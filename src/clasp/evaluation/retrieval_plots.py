"""Save retrieval metric summaries as PNG (headless)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HitsDisplay = Literal["rate", "percent"]


def save_retrieval_plot(
    ranking_metrics: dict[str, float],
    ranks: np.ndarray,
    output_path: Path | str,
    ks: Sequence[int],
    *,
    hist_rank_cap: int | None = 200,
    title: str | None = None,
    subtitle: str | None = None,
    hits_display: HitsDisplay = "rate",
) -> None:
    """
    Two subplots: bar chart of Hits@K (and MRR as text); histogram of 1-based ranks.

    Parameters
    ----------
    hist_rank_cap :
        If set, x-axis for the histogram stops at this rank; counts for ranks above
        are merged into the last bin.
    hits_display :
        ``rate``: y-axis 0–1 (default, backward compatible). ``percent``: 0–100.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    k_sorted = sorted(set(int(k) for k in ks if k > 0))
    hit_keys = [f"Hits@{k}" for k in k_sorted]
    hit_vals = [ranking_metrics.get(h, 0.0) for h in hit_keys]
    labels = [f"@{k}" for k in k_sorted]

    if hits_display == "percent":
        hit_plot = [v * 100.0 for v in hit_vals]
        y_max = 105.0
        y_label = "Hits (%)"
    else:
        hit_plot = hit_vals
        y_max = 1.05
        y_label = "Rate"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax0 = axes[0]
    x = np.arange(len(hit_plot))
    ax0.bar(x, hit_plot, color="steelblue", edgecolor="navy", alpha=0.85)
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels)
    ax0.set_ylim(0, y_max)
    ax0.set_ylabel(y_label)
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
            ax1.set_xlabel("Rank")
            ax1.set_ylabel("Count")
            ax1.set_title(f"Rank of relevant (capped at {hist_rank_cap})")
        else:
            max_r = int(r.max())
            nb = min(50, max(10, int(math.sqrt(len(r)))))
            ax1.hist(r, bins=nb, range=(1, max_r + 1), color="seagreen", edgecolor="darkgreen", alpha=0.85)
            ax1.set_xlabel("Rank")
            ax1.set_ylabel("Count")
            ax1.set_title("Rank of relevant (1-based)")

    st = title or "Retrieval evaluation"
    if subtitle:
        st = f"{st}\n{subtitle}"
    fig.suptitle(st, fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_grouped_hits_plot(
    group_labels: Sequence[str],
    hits_values: Sequence[float],
    sample_counts: Sequence[int],
    output_path: Path | str,
    *,
    ylabel: str = "Hits@1",
    title: str = "Retrieval by group",
    hits_display: HitsDisplay = "percent",
    show_counts_line: bool = True,
) -> None:
    """Bar chart of Hits@K (or Hits@1) per group; optional twin axis for sample counts.

    ``hits_values`` must be rates in ``[0, 1]`` when ``hits_display`` is ``rate`` or ``percent``
    (values are the same; only axis scaling changes).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = list(group_labels)
    hits = np.asarray(hits_values, dtype=np.float64).reshape(-1)
    counts = np.asarray(sample_counts, dtype=np.int64).reshape(-1)
    if hits.size != len(labels) or counts.size != len(labels):
        raise ValueError("group_labels, hits_values, and sample_counts must have the same length")

    if hits_display == "percent":
        y_plot = hits * 100.0
        y_max = 105.0
        y_axis_label = f"{ylabel} (%)"
    else:
        y_plot = hits
        y_max = 1.05
        y_axis_label = ylabel

    fig, ax = plt.subplots(figsize=(max(8.0, len(labels) * 1.2), 5.0))
    x_pos = np.arange(len(labels))

    if show_counts_line:
        ax2 = ax.twinx()
        bars = ax.bar(x_pos, y_plot, color="steelblue", edgecolor="navy", alpha=0.85, width=0.6, label=y_axis_label)
        line = ax2.plot(x_pos, counts, color="firebrick", marker="o", linewidth=2, markersize=8, label="Count")
        for i, c in enumerate(counts):
            ax2.annotate(
                f"n={int(c)}",
                xy=(i, c),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color="firebrick",
            )
        ax.set_ylabel(y_axis_label, color="steelblue")
        ax2.set_ylabel("Sample count", color="firebrick")
        if counts.size and counts.max() > 0:
            ax2.set_ylim(0, float(counts.max()) * 1.25)
        lines1, lab1 = ax.get_legend_handles_labels()
        lines2, lab2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, lab1 + lab2, loc="upper right", fontsize=8)
    else:
        bars = ax.bar(x_pos, y_plot, color="steelblue", edgecolor="navy", alpha=0.85, width=0.6)

    for bar, val in zip(bars, y_plot):
        h = bar.get_height()
        ax.annotate(
            f"{val:.1f}%" if hits_display == "percent" else f"{val:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, y_max)
    ax.set_xlabel("Group")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
