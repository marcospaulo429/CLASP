"""SVQ RAG pipeline: CLASP audio->passage retrieval + LLM answer generation.

Loads an SVQ reasoning config (audio question + gold passage + gold answer span),
retrieves the top-k passages for each spoken query with the CLASP audio encoder
(scored against LaBSE passage embeddings), asks a generator for the answer, and
scores it with EM / token-F1. It also reports retrieval Recall@k (whether the gold
passage was retrieved), so retrieval and generation quality are visible separately.

The generator is pluggable (see ``clasp.rag.generator``); the question text handed
to the LLM is SVQ's gold transcript, so the score isolates CLASP retrieval + LLM
generation without ASR error.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from clasp.evaluation.qa_metrics import score_answer
from clasp.inference.audio_preprocess import MIN_SAMPLES_16K, TARGET_SR
from clasp.inference.embed_audio import hubert_audio_files
from clasp.inference.pipeline import load_model
from clasp.inference.spectrogram_image import (
    efficientnet_embeddings_from_audio_paths,
    load_efficientnet_b7,
)

SVQ_REPO = "google/svq"


# --------------------------------------------------------------------------- data
@dataclass
class RagRow:
    utt_id: str
    question: str
    passage_id: str
    golds: list[str]
    locale: str
    audio_path: str


def _to_mono_16k(value) -> np.ndarray:
    if isinstance(value, dict):
        if value.get("array") is not None:
            arr, sr = np.asarray(value["array"], dtype=np.float32), int(value.get("sampling_rate") or TARGET_SR)
        elif value.get("bytes"):
            arr, sr = sf.read(io.BytesIO(value["bytes"]), dtype="float32", always_2d=False)
        elif value.get("path"):
            arr, sr = sf.read(value["path"], dtype="float32", always_2d=False)
        else:
            raise ValueError("audio dict has no array/bytes/path")
    else:
        arr, sr = np.asarray(value, dtype=np.float32), TARGET_SR
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak > 0:
        arr = arr / peak
    if sr != TARGET_SR:
        arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
    if len(arr) < MIN_SAMPLES_16K:
        arr = np.pad(arr, (0, MIN_SAMPLES_16K - len(arr)), mode="constant")
    return arr


def _golds_from_example(ex: dict) -> list[str]:
    spans = ex.get("spans")
    if isinstance(spans, (list, tuple)) and len(spans) > 0:
        return [str(s) for s in spans]
    span = ex.get("span")
    return [str(span)] if span is not None else []


def load_svq_reasoning_rows(
    config: str,
    split: str,
    locale: str | None,
    max_samples: int | None,
    audio_cache_dir: Path,
    audio_config: str = "audio",
    audio_column: str = "waveform",
) -> list[RagRow]:
    """Load SVQ reasoning rows (question/passage/answer) joined to audio by ``utt_id``.

    The reasoning configs carry ``utt_id`` but not the waveform; the audio lives in a
    separate ``audio*`` config keyed by ``utt_id``. Both datasets are streamed so no
    multi-GB download is needed — the audio stream stops once every needed ``utt_id``
    is found. ``--audio-config audio`` (default) covers every locale; ``audio_en_us_clean``
    is faster but only overlaps the curated clean English subset.
    """
    from datasets import Audio, load_dataset

    audio_cache_dir.mkdir(parents=True, exist_ok=True)

    # 1) Reasoning rows (over-collect so rows dropped for missing audio still hit the cap).
    reasoning_ds = load_dataset(SVQ_REPO, config, split=split, streaming=True)
    over = (max_samples * 4) if max_samples is not None else None
    reasoning: list[dict] = []
    for ex in tqdm(reasoning_ds, desc=f"svq {config}"):
        if locale and str(ex.get("locale", "")).lower() != locale.lower():
            continue
        if not (ex.get("text") or "").strip():
            continue
        reasoning.append(ex)
        if over is not None and len(reasoning) >= over:
            break
    needed = {str(ex["utt_id"]) for ex in reasoning}
    if not needed:
        return []

    # 2) utt_id -> audio cell, streamed from the audio config (stop once all found).
    audio_ds = load_dataset(SVQ_REPO, audio_config, split=split, streaming=True)
    if getattr(audio_ds, "features", None) and audio_column in audio_ds.features:
        try:
            audio_ds = audio_ds.cast_column(audio_column, Audio(decode=False))
        except (TypeError, ValueError, KeyError):
            pass
    utt_to_audio: dict = {}
    for ex in tqdm(audio_ds, desc=f"svq {audio_config} (audio join)"):
        uid = str(ex.get("utt_id", ""))
        if uid in needed:
            utt_to_audio[uid] = ex[audio_column]
            if len(utt_to_audio) == len(needed):
                break

    # 3) Join and materialize WAVs.
    rows: list[RagRow] = []
    for ex in reasoning:
        uid = str(ex["utt_id"])
        if uid not in utt_to_audio:
            continue
        safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", uid)[:150]
        out_path = (audio_cache_dir / f"{len(rows):07d}_{safe}.wav").resolve()
        if not out_path.exists():
            sf.write(str(out_path), _to_mono_16k(utt_to_audio[uid]), TARGET_SR, subtype="PCM_16")
        row = RagRow(
            utt_id=uid,
            question=(ex.get("text") or "").strip(),
            passage_id=str(ex.get("passage_id", f"p_{uid}")),
            golds=_golds_from_example(ex),
            locale=str(ex.get("locale", "")),
            audio_path=str(out_path),
        )
        row.__dict__["passage_text"] = str(ex.get("passage_text", ""))
        rows.append(row)
        if max_samples is not None and len(rows) >= max_samples:
            break
    return rows


def build_corpus(rows: list[RagRow]) -> tuple[list[str], list[str]]:
    """Unique passages across the rows: (passage_ids, passage_texts)."""
    seen: dict[str, str] = {}
    for r in rows:
        pid = r.passage_id
        if pid not in seen:
            seen[pid] = r.__dict__.get("passage_text", "")
    return list(seen.keys()), list(seen.values())


# ----------------------------------------------------------------------- embedder
def _l2(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(2, dim=-1, keepdim=True).clamp_min(1e-12)


class ClaspEmbedder:
    """CLASP audio encoder + LaBSE text encoder in a shared 768-d space."""

    def __init__(self, model_path: str, device: torch.device,
                 hubert_model: str = "facebook/hubert-large-ls960-ft",
                 labse_model: str = "sentence-transformers/LaBSE"):
        from sentence_transformers import SentenceTransformer
        from transformers import AutoProcessor, HubertModel

        self.device = device
        self.clasp = load_model(model_path, device)
        self.hubert_processor = AutoProcessor.from_pretrained(hubert_model)
        self.hubert = HubertModel.from_pretrained(hubert_model).to(device)
        self.hubert.eval()
        self.effnet, self.effnet_preprocess = load_efficientnet_b7(device)
        self.labse = SentenceTransformer(labse_model, device=str(device))

    def embed_audio_paths(self, paths: list[str], vision_batch_size: int = 4) -> torch.Tensor:
        hub = hubert_audio_files(paths, self.hubert_processor, self.hubert, self.device)
        hub = torch.stack([h.squeeze(0) if h.dim() > 1 else h for h in hub]).to(self.device)
        img = efficientnet_embeddings_from_audio_paths(
            paths, self.effnet, self.effnet_preprocess, self.device, batch_size=vision_batch_size
        )
        img = torch.stack([t.squeeze(0) if t.dim() > 1 else t for t in img]).to(self.device)
        with torch.no_grad():
            fused = self.clasp(hub, img)
        return _l2(fused).detach().cpu()

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> torch.Tensor:
        emb = self.labse.encode(texts, batch_size=batch_size, convert_to_tensor=True,
                                show_progress_bar=len(texts) > 64)
        return _l2(emb.detach().cpu().float())


# --------------------------------------------------------------------------- run
@dataclass
class RagResult:
    n: int
    recall_at_k: float
    exact_match: float
    f1: float
    top_k: int
    generator: str
    per_row: list[dict] = field(default_factory=list)


def run_svq_rag(
    embedder: ClaspEmbedder,
    generator,
    rows: list[RagRow],
    top_k: int = 5,
    keep_per_row: bool = True,
) -> RagResult:
    passage_ids, passage_texts = build_corpus(rows)
    pid_to_idx = {pid: i for i, pid in enumerate(passage_ids)}

    q = embedder.embed_audio_paths([r.audio_path for r in rows])   # [N, 768]
    d = embedder.embed_texts(passage_texts)                        # [M, 768]
    sims = q @ d.t()                                               # [N, M]
    k = min(top_k, len(passage_ids))
    top_idx = sims.topk(k=k, dim=1).indices.tolist()

    n_recall = n_em = n_f1 = 0.0
    per_row: list[dict] = []
    for i, r in enumerate(tqdm(rows, desc="rag generate")):
        retrieved_ids = [passage_ids[j] for j in top_idx[i]]
        retrieved_texts = [passage_texts[j] for j in top_idx[i]]
        gold_hit = r.passage_id in retrieved_ids
        pred = generator.generate(r.question, retrieved_texts, language=r.locale)
        em, f1 = score_answer(pred, r.golds)
        n_recall += float(gold_hit)
        n_em += em
        n_f1 += f1
        if keep_per_row:
            per_row.append({
                "utt_id": r.utt_id, "locale": r.locale, "question": r.question,
                "gold_passage_id": r.passage_id, "retrieved_passage_ids": retrieved_ids,
                "gold_in_topk": gold_hit, "golds": r.golds, "prediction": pred,
                "em": em, "f1": f1,
            })

    n = len(rows)
    return RagResult(
        n=n,
        recall_at_k=n_recall / n if n else 0.0,
        exact_match=n_em / n if n else 0.0,
        f1=n_f1 / n if n else 0.0,
        top_k=k,
        generator=getattr(generator, "name", type(generator).__name__),
        per_row=per_row,
    )
