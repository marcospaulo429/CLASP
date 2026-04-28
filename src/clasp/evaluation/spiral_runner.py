"""SPIRAL JSONL retrieval: embeddings + shared ranking metrics and plots."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoProcessor, HubertModel

from clasp.data.spiral import load_spiral_jsonl, spiral_temporal_bin_indices
from clasp.evaluation.ranking_metrics import (
    compute_ranking_metrics,
    grouped_ranking_summary,
    similarity_matrix_to_rows,
)
from clasp.evaluation.retrieval_plots import save_grouped_hits_plot, save_retrieval_plot
from clasp.inference.audio_preprocess import load_mono_16k_padded
from clasp.inference.embed_audio import hubert_numpy_waveform
from clasp.inference.pipeline import load_model
from clasp.inference.spectrogram_image import (
    efficientnet_embedding_from_waveform,
    load_efficientnet_b7,
)


def _require_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "SPIRAL retrieval requires sentence-transformers. "
            "Install with: pip install sentence-transformers"
        ) from e
    return SentenceTransformer


def encode_texts_labse(
    texts: list[str],
    device: torch.device,
    model_id: str,
    batch_size: int,
) -> torch.Tensor:
    """LaBSE sentence embeddings (same API as pickle builders)."""
    SentenceTransformer = _require_sentence_transformers()
    model = SentenceTransformer(model_id, device=str(device))
    emb = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=len(texts) > 32,
    )
    return emb.detach().cpu().float()


class _SpiralAudioEncoder:
    """HuBERT + EfficientNet on full waveform (chunked)."""

    def __init__(self, device: torch.device, hubert_model_id: str):
        self.device = device
        self.hubert_processor = AutoProcessor.from_pretrained(hubert_model_id)
        self.hubert_model = HubertModel.from_pretrained(hubert_model_id)
        self.hubert_model.to(device)
        self.hubert_model.eval()
        self.vision_model, self.vision_preprocess = load_efficientnet_b7(device)

    def _load_full(self, audio_path: str) -> np.ndarray:
        if not Path(audio_path).exists():
            raise FileNotFoundError(audio_path)
        return load_mono_16k_padded(audio_path)

    @torch.no_grad()
    def embed_paths(
        self,
        audio_paths: list[str],
        chunk_samples: int = 320_000,
        chunk_batch_size: int = 1,
    ) -> torch.Tensor:
        out: list[torch.Tensor] = []
        # Large HuBERT = 1024; base = 768 — must match fusion checkpoint (e.g. Concat encoder).
        hubert_dim = int(self.hubert_model.config.hidden_size)
        spec_dim = 1000  # EfficientNet-B7 ImageNet logits
        for p in tqdm(audio_paths, desc="SPIRAL audio embeddings (full file)"):
            try:
                y = self._load_full(p)
                h = hubert_numpy_waveform(
                    y,
                    self.hubert_processor,
                    self.hubert_model,
                    self.device,
                    chunk_samples=chunk_samples,
                    chunk_batch_size=chunk_batch_size,
                )
                s = efficientnet_embedding_from_waveform(
                    y,
                    self.vision_model,
                    self.vision_preprocess,
                    self.device,
                    chunk_samples=chunk_samples,
                    chunk_batch_size=chunk_batch_size,
                )
                out.append(torch.cat([h.cpu(), s.cpu()], dim=0))
            except Exception as e:
                print(f"\nWarning: audio failed ({p}): {e}", file=sys.stderr)
                out.append(torch.zeros(hubert_dim + spec_dim))
        return torch.stack(out, dim=0)


@torch.no_grad()
def _fuse_clasp(
    model: torch.nn.Module,
    hubert_part: torch.Tensor,
    spec_part: torch.Tensor,
    device: torch.device,
    batch_size: int = 32,
) -> torch.Tensor:
    fused: list[torch.Tensor] = []
    for i in range(0, hubert_part.size(0), batch_size):
        a = hubert_part[i : i + batch_size].to(device)
        b = spec_part[i : i + batch_size].to(device)
        fused.append(model(a, b).cpu())
    return torch.cat(fused, dim=0)


def _cosine_sim_matrix(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    a_n = F.normalize(a, p=2, dim=1)
    b_n = F.normalize(b, p=2, dim=1)
    return torch.mm(a_n, b_n.t()).detach().numpy()


def print_spiral_markdown_tables(
    ranking_metrics: dict[str, float],
    grouped: dict[str, dict[str, Any]],
    bin_order: Sequence[str],
) -> None:
    print("\n| Metric | Value |")
    print("|--------|-------|")
    for k in (1, 5, 10, 50):
        key = f"Hits@{k}"
        if key in ranking_metrics:
            pct = ranking_metrics[key] * 100.0
            print(f"| {key} | {pct:.2f}% |")
    print(f"| MRR | {ranking_metrics.get('MRR', 0.0):.4f} |")
    print(f"| mean_rank | {ranking_metrics.get('mean_rank', 0.0):.2f} |")
    print(f"| median_rank | {ranking_metrics.get('median_rank', 0.0):.2f} |")

    print("\n## By temporal bin (Hits@1)\n")
    print("\n| Group | n | Hits@1 |")
    print("|-------|---|--------|")
    for label in bin_order:
        row = grouped.get(label, {"n": 0, "Hits@1": 0.0})
        n = int(row.get("n", 0))
        h1 = float(row.get("Hits@1", 0.0)) * 100.0
        print(f"| {label} | {n} | {h1:.2f}% |")
    print()


def run_spiral_retrieval_eval(
    data_path: Path,
    model_path: Path,
    output_dir: Path,
    *,
    audio_base_dir: Path | None = None,
    extra_search_roots: Sequence[Path] = (),
    max_samples: int | None = None,
    batch_size_text: int = 32,
    hubert_model_id: str = "facebook/hubert-large-ls960-ft",
    sentence_model_id: str = "sentence-transformers/LaBSE",
    device: torch.device | None = None,
    hits_ks: Sequence[int] = (1, 5, 10, 50),
    chunk_samples: int = 320_000,
    chunk_batch_size_audio: int = 1,
    batch_size_fusion: int = 32,
) -> dict[str, Any]:
    """Load SPIRAL JSONL, compute embeddings, shared metrics, save plots + JSON."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_spiral_jsonl(
        Path(data_path),
        audio_base_dir,
        extra_search_roots=extra_search_roots,
    )
    if max_samples is not None and max_samples < len(samples):
        samples = samples[: int(max_samples)]

    samples = [s for s in samples if Path(s["audio_path"]).exists()]
    if not samples:
        raise SystemExit(
            "No SPIRAL samples with existing audio files. Download wavs from "
            "https://duke.box.com/v/spiral-dataset and set --spiral-audio-base if needed."
        )

    audio_paths = [s["audio_path"] for s in samples]
    key_sentences = [s["key_sentence"] for s in samples]
    timestamps = [s["key_sentence_timestamp"] for s in samples]

    print("Encoding text (LaBSE / SentenceTransformer)...", flush=True)
    text_emb = encode_texts_labse(key_sentences, device, sentence_model_id, batch_size_text)

    print("Encoding audio (HuBERT + EfficientNet)...", flush=True)
    audio_enc = _SpiralAudioEncoder(device, hubert_model_id)
    raw = audio_enc.embed_paths(
        audio_paths,
        chunk_samples=chunk_samples,
        chunk_batch_size=chunk_batch_size_audio,
    )
    hubert_dim = int(audio_enc.hubert_model.config.hidden_size)
    spec_dim = 1000
    if raw.shape[1] != hubert_dim + spec_dim:
        raise ValueError(
            f"SPIRAL audio vector dim {raw.shape[1]} != hubert_dim+spec_dim ({hubert_dim}+{spec_dim}); "
            "check HuBERT variant vs EfficientNet head."
        )
    hubert_part = raw[:, :hubert_dim]
    spec_part = raw[:, hubert_dim : hubert_dim + spec_dim]

    print("Loading CLASP fusion checkpoint...", flush=True)
    clasp = load_model(str(model_path), device)
    fused_audio = _fuse_clasp(
        clasp,
        hubert_part,
        spec_part,
        device,
        batch_size=batch_size_fusion,
    )

    if text_emb.shape[1] != fused_audio.shape[1]:
        raise ValueError(
            f"Text dim {text_emb.shape[1]} != fused audio dim {fused_audio.shape[1]}; "
            "check LaBSE and CLASP output sizes."
        )

    sim = _cosine_sim_matrix(text_emb, fused_audio)
    rows = similarity_matrix_to_rows(sim)
    ranking_metrics, ranks = compute_ranking_metrics(rows, ks=hits_ks)

    temporal = spiral_temporal_bin_indices(timestamps)
    groups_np = {k: np.array(v, dtype=np.int64) for k, v in temporal.items()}
    grouped = grouped_ranking_summary(ranks, groups_np, ks=(1,))
    bin_order = list(temporal.keys())

    ks_list = list(hits_ks)
    save_retrieval_plot(
        ranking_metrics,
        ranks,
        output_dir / "retrieval_summary.png",
        ks_list,
        title="SPIRAL retrieval",
        subtitle=str(data_path.name),
        hits_display="percent",
    )

    g_labels = bin_order
    g_hits = [float(grouped[lb]["Hits@1"]) for lb in bin_order]
    g_n = [int(grouped[lb]["n"]) for lb in bin_order]
    save_grouped_hits_plot(
        g_labels,
        g_hits,
        g_n,
        output_dir / "retrieval_by_temporal_bin.png",
        ylabel="Hits@1",
        title="SPIRAL: Hits@1 by key-sentence start time",
        hits_display="percent",
        show_counts_line=True,
    )

    print("## SPIRAL global ranking metrics\n", flush=True)
    print_spiral_markdown_tables(ranking_metrics, grouped, bin_order)

    payload: dict[str, Any] = {
        "dataset": str(data_path),
        "model_path": str(model_path),
        "n_samples": len(samples),
        "device": str(device),
        "ranking_metrics": ranking_metrics,
        "grouped_temporal_bins": grouped,
        "bin_order": list(bin_order),
    }
    out_json = output_dir / "spiral_evaluation_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_json}", flush=True)

    return payload
