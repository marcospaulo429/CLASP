"""Max-sim (ColBERT-style) similarity for SPIRAL: query vs multi-vector audio, vectorized."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

NEG_INF = -1e4


def max_sim_similarity_matrix(
    text: torch.Tensor,
    audio_padded: torch.Tensor,
    audio_mask: torch.Tensor,
) -> torch.Tensor:
    """
    text: (N, D) on any device; audio_padded: (N, C_max, D); audio_mask: (N, C_max) bool True=valid.
    S[i,j] = max_m cos(t_i, a_{j,m}) over valid chunks of audio j, as dot product of L2-normalized vectors.
    """
    t_n = F.normalize(text, p=2, dim=1, eps=1e-8)
    a_n = F.normalize(audio_padded, p=2, dim=2, eps=1e-8)
    s = torch.einsum("id,jmd->ijm", t_n, a_n)
    # Invalid chunks for audio j apply to all query rows i: mask (1, N, C_max)
    s = s.masked_fill((~audio_mask).unsqueeze(0), NEG_INF)
    return s.max(dim=2).values


def per_query_winning_chunk_on_diagonal(
    text: torch.Tensor,
    audio_padded: torch.Tensor,
    audio_mask: torch.Tensor,
) -> torch.Tensor:
    """
    For each query i, max-sim to audio i's own chunks: argmax over m of (T_i·A[i,m]).
    Returns (N,) long on same device.
    """
    t_n = F.normalize(text, p=2, dim=1, eps=1e-8)
    a_n = F.normalize(audio_padded, p=2, dim=2, eps=1e-8)
    s = torch.einsum("id,imd->im", t_n, a_n)
    s = s.masked_fill(~audio_mask, NEG_INF)
    return s.argmax(dim=1)


def expected_chunk_index_from_time(
    t0_sec: float,
    num_chunks: int,
    chunk_len_sec: float,
) -> int:
    if num_chunks <= 0:
        return 0
    c = int(t0_sec // chunk_len_sec) if chunk_len_sec > 0 else 0
    return int(max(0, min(c, num_chunks - 1)))


def to_numpy_f64(s: torch.Tensor) -> np.ndarray:
    return s.detach().float().cpu().numpy().astype(np.float64, copy=False)
