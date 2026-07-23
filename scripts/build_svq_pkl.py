#!/usr/bin/env python3
"""Build a total_dataset pickle from the SVQ dataset (google/svq) for a
**baseline CLASP retrieval eval**, identical in shape to the VoxPopuli /
Spoken-SQuAD PKLs.

Each row pairs the spoken query audio with **its own transcript** (SVQ ``text``
field) — i.e. CLASP self-retrieval (audio -> the text that was spoken), exactly
like ``build_voxpopuli_pkl.py`` pairs audio with ``normalized_text``. This makes
SVQ numbers directly comparable to the other datasets via ``run_retrieval_eval.py``.

Output (per split): ``hubert-emb`` (HuBERT 1024), ``text`` (LaBSE 768),
``image`` (EfficientNet-B7 1000), ``audio_path``.

The default config is ``audio_en_us_clean`` (~1.4k clean English rows) — a small,
clean apples-to-apples baseline. Use ``--config audio`` for the full multilingual
set (171k rows) with ``--locale`` / ``--max-samples`` to subset.

Requires extras with the HF ``datasets`` library, e.g. ``uv sync --extra voxpopuli``.
"""

from __future__ import annotations

import argparse
import copy
import io
import pickle
import re
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import AutoProcessor, HubertModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clasp.config.settings import get_default_device
from clasp.inference.audio_preprocess import MIN_SAMPLES_16K, TARGET_SR
from clasp.inference.embed_audio import hubert_audio_files
from clasp.inference.spectrogram_image import (
    efficientnet_embeddings_from_audio_paths,
    load_efficientnet_b7,
)

try:
    from datasets import Audio, load_dataset
except ImportError as e:
    raise SystemExit(
        "Missing dependency: datasets. Install with: uv sync --extra voxpopuli"
    ) from e

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise SystemExit(
        "Missing dependency: sentence-transformers. Install with: uv sync --extra voxpopuli"
    ) from e


SVQ_REPO = "google/svq"


def _squeeze_hubert_list(embeddings: list[torch.Tensor]) -> list[torch.Tensor]:
    out = []
    for e in embeddings:
        if e.dim() > 1:
            e = e.squeeze(0)
        out.append(e.contiguous().float().cpu())
    return out


def _normalize_resample_pad_mono(arr: np.ndarray, sr: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim > 1:
        arr = np.mean(arr, axis=1)
    arr = arr.astype(np.float32)
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak > 0:
        arr = arr / peak
    if sr != TARGET_SR:
        arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
    if len(arr) < MIN_SAMPLES_16K:
        arr = np.pad(arr, (0, MIN_SAMPLES_16K - len(arr)), mode="constant")
    return arr


def _audio_value_to_mono_16k_padded(value) -> np.ndarray:
    """PCM mono 16 kHz from an HF audio cell (dict with array/bytes/path, or raw array)."""
    if isinstance(value, dict):
        if value.get("array") is not None:
            arr = np.asarray(value["array"], dtype=np.float32)
            sr = int(value.get("sampling_rate") or TARGET_SR)
            return _normalize_resample_pad_mono(arr, sr)
        raw = value.get("bytes")
        if raw:
            arr, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
            return _normalize_resample_pad_mono(np.asarray(arr, dtype=np.float32), int(sr))
        path = value.get("path")
        if path:
            arr, sr = sf.read(path, dtype="float32", always_2d=False)
            return _normalize_resample_pad_mono(np.asarray(arr, dtype=np.float32), int(sr))
        raise ValueError("audio dict has no array, bytes, or path")
    arr = np.asarray(value, dtype=np.float32)
    return _normalize_resample_pad_mono(arr, TARGET_SR)


def _dataset_audio_no_decode(ds, column: str):
    """Avoid torchcodec decode of the audio column in HF datasets 4.x."""
    feats = getattr(ds, "features", None)
    if feats is not None and column in feats:
        try:
            return ds.cast_column(column, Audio(decode=False))
        except (TypeError, ValueError, KeyError):
            pass
    return ds


def _safe_wav_name(audio_id: str, idx: int) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(audio_id))[:150]
    return f"{idx:07d}_{safe}.wav"


def load_svq_rows(
    config: str,
    split: str,
    cache_dir: Path,
    audio_column: str,
    text_column: str,
    id_column: str,
    locale: str | None,
    max_samples: int | None,
) -> list[dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading {SVQ_REPO} config={config} split={split} …")
    ds = load_dataset(SVQ_REPO, config, split=split)
    ds = _dataset_audio_no_decode(ds, audio_column)

    rows: list[dict] = []
    for idx, ex in enumerate(tqdm(ds, desc=f"svq {config}")):
        if locale and str(ex.get("locale", "")).lower() != locale.lower():
            continue
        text = (ex.get(text_column) or "").strip()
        if not text:
            continue
        audio_id = str(ex.get(id_column, f"row_{idx}"))
        out_path = (cache_dir / _safe_wav_name(audio_id, len(rows))).resolve()
        if not out_path.exists():
            arr = _audio_value_to_mono_16k_padded(ex[audio_column])
            sf.write(str(out_path), arr, TARGET_SR, subtype="PCM_16")
        rows.append({"text": text, "_abs_audio": str(out_path)})
        if max_samples is not None and len(rows) >= max_samples:
            break
    print(f"Collected {len(rows)} usable rows.")
    return rows


def build_split_dict(
    rows: list[dict],
    hubert_processor,
    hubert_model,
    sentence_model: SentenceTransformer,
    vision_model,
    vision_preprocess,
    device: torch.device,
    vision_batch_size: int,
    text_batch_size: int,
) -> dict:
    paths = [r["_abs_audio"] for r in rows]
    texts = [r["text"] for r in rows]

    hubert_list = _squeeze_hubert_list(
        hubert_audio_files(paths, hubert_processor, hubert_model, device)
    )

    text_emb = sentence_model.encode(
        texts, batch_size=text_batch_size, convert_to_tensor=True,
        show_progress_bar=len(texts) > 32,
    )
    text_list = [text_emb[j].detach().cpu().float().squeeze() for j in range(text_emb.size(0))]

    image_list = efficientnet_embeddings_from_audio_paths(
        paths, vision_model, vision_preprocess, device, batch_size=vision_batch_size
    )
    image_list = [t.squeeze(0) if t.dim() > 1 else t for t in image_list]

    return {"hubert-emb": hubert_list, "text": text_list, "image": image_list, "audio_path": paths}


def _slice_split(split_dict: dict, start: int, stop: int) -> dict:
    return {key: split_dict[key][start:stop] for key in split_dict}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="audio_en_us_clean", help="SVQ HF config (default: audio_en_us_clean)")
    p.add_argument("--split", default="test", help="HF split (SVQ is eval-only; default: test)")
    p.add_argument("--output", type=Path, default=Path("data/datasets/total_dataset_svq.pkl"))
    p.add_argument("--audio-cache-dir", type=Path, default=Path("data/datasets/svq_wav"))
    p.add_argument("--audio-column", default="waveform")
    p.add_argument("--text-column", default="text")
    p.add_argument("--id-column", default="utt_id")
    p.add_argument("--locale", default=None, help="Filter to a single locale (e.g. en_us); default: all in config")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--hubert-model", default="facebook/hubert-large-ls960-ft")
    p.add_argument("--sentence-transformer", default="sentence-transformers/LaBSE")
    p.add_argument("--vision-batch-size", type=int, default=4)
    p.add_argument("--text-batch-size", type=int, default=32)
    split = p.add_mutually_exclusive_group()
    split.add_argument("--replicate-for-train", action="store_true",
                       help="Set train/validation/test to deep copies of the built split.")
    split.add_argument("--val-fraction", type=float, default=None, metavar="F",
                       help="Hold out fraction F in (0,1) as validation; full set kept as test.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else get_default_device()

    hubert_processor = AutoProcessor.from_pretrained(args.hubert_model)
    hubert_model_obj = HubertModel.from_pretrained(args.hubert_model).to(device)
    hubert_model_obj.eval()
    sentence_model = SentenceTransformer(args.sentence_transformer, device=str(device))
    vision_model, vision_preprocess = load_efficientnet_b7(device)

    rows = load_svq_rows(
        args.config, args.split, args.audio_cache_dir,
        args.audio_column, args.text_column, args.id_column,
        args.locale, args.max_samples,
    )
    if not rows:
        raise SystemExit("No usable SVQ rows (check --config/--locale/--text-column).")

    n = len(rows)
    print(f"Embedding {n} rows on {device} …")
    full_split = build_split_dict(
        rows, hubert_processor, hubert_model_obj, sentence_model,
        vision_model, vision_preprocess, device,
        args.vision_batch_size, args.text_batch_size,
    )

    if args.replicate_for_train:
        total_dataset = {
            "train": copy.deepcopy(full_split),
            "validation": copy.deepcopy(full_split),
            "test": copy.deepcopy(full_split),
        }
        split_msg = "train+validation+test (replicated, deep copy)"
    elif args.val_fraction is not None:
        vf = args.val_fraction
        if not (0.0 < vf < 1.0):
            raise SystemExit("--val-fraction must be strictly between 0 and 1")
        k = int(n * (1.0 - vf))
        if k < 1 or n - k < 1:
            raise SystemExit(f"--val-fraction={vf} with n={n} yields an empty split.")
        total_dataset = {
            "train": _slice_split(full_split, 0, k),
            "validation": _slice_split(full_split, k, n),
            "test": copy.deepcopy(full_split),
        }
        split_msg = f"train={k} val={n - k} test={n}"
    else:
        total_dataset = {"test": full_split}
        split_msg = "test only"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(total_dataset, f)
    print(f"Wrote {args.output} ({split_msg})")


if __name__ == "__main__":
    main()
