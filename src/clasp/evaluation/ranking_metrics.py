"""Ranking-based retrieval metrics (no similarity threshold).

Assumes one relevant item per query at index i for row i (diagonal ground truth),
with the same tie-breaking as ``evaluate_matrix``: rank = 1 + count of columns j
with score strictly greater than the diagonal score.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from tqdm import tqdm


def similarity_matrix_to_rows(
    sim: np.ndarray | list[list[float]] | Sequence[Sequence[float]],
) -> list[list[float]]:
    """Convert a square similarity matrix to ``list[list[float]]`` for ``compute_ranking_metrics``.

    Accepts ``numpy.ndarray`` of shape ``(n, n)`` or nested lists/tuples.
    """
    if isinstance(sim, np.ndarray):
        if sim.ndim != 2:
            raise ValueError("similarity matrix must be 2-D")
        return sim.astype(np.float64, copy=False).tolist()
    return [[float(x) for x in row] for row in sim]


def _rank_for_row(row: Sequence[float], gold_index: int) -> int:
    gold_score = row[gold_index]
    label_rank = sum(1 for x in row if x > gold_score)
    return label_rank + 1


def compute_ranking_metrics(
    similarity_matrix: list[list[float]],
    ks: Sequence[int] = (1, 5, 10, 50),
) -> tuple[dict[str, float], np.ndarray]:
    """
    Returns
    -------
    metrics : dict
        Hits@K for each K, MRR, MAP (equals MRR for one relevant per query),
        mean_rank, median_rank.
    ranks : ndarray of shape (n_queries,), dtype int64
        1-based rank of the relevant item per query.
    """
    n = len(similarity_matrix)
    if n == 0:
        return {}, np.array([], dtype=np.int64)

    ranks = np.empty(n, dtype=np.int64)
    for i in tqdm(range(n), desc="ranking metrics"):
        ranks[i] = _rank_for_row(similarity_matrix[i], i)

    rr = 1.0 / ranks.astype(np.float64)
    mrr = float(rr.mean())
    map_score = mrr  # one relevant per query: AP_i = 1/rank_i

    metrics: dict[str, float] = {
        "MRR": mrr,
        "MAP": map_score,
        "mean_rank": float(ranks.mean()),
        "median_rank": float(np.median(ranks)),
    }

    k_set = sorted(set(int(k) for k in ks if k > 0))
    for k in k_set:
        metrics[f"Hits@{k}"] = float((ranks <= k).mean())

    return metrics, ranks


def grouped_ranking_summary(
    ranks: np.ndarray,
    groups: Mapping[str, np.ndarray],
    ks: Sequence[int] = (1,),
) -> dict[str, dict[str, Any]]:
    """Summarize Hits@K per query group using precomputed 1-based ranks (no matrix).

    Parameters
    ----------
    ranks :
        Shape ``(n_queries,)``, 1-based rank of the relevant item per query.
    groups :
        Map group name -> 1-D int array of query indices belonging to that group.
    ks :
        Hit cutoffs (e.g. ``(1, 5, 10)``).

    Returns
    -------
    Nested dict: ``group_name -> {"n": int, "Hits@k": float, ...}`` for each ``k``.
    """
    r = np.asarray(ranks, dtype=np.int64).reshape(-1)
    k_set = sorted(set(int(k) for k in ks if k > 0))
    out: dict[str, dict[str, Any]] = {}

    for name, idx in groups.items():
        ind = np.asarray(idx, dtype=np.int64).reshape(-1)
        if ind.size == 0:
            row: dict[str, Any] = {"n": 0}
            for k in k_set:
                row[f"Hits@{k}"] = 0.0
            out[str(name)] = row
            continue
        sub = r[ind]
        row = {"n": int(ind.size)}
        for k in k_set:
            row[f"Hits@{k}"] = float((sub <= k).mean())
        out[str(name)] = row

    return out
