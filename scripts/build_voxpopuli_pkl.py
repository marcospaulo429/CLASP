#!/usr/bin/env python3
"""Build total_dataset pickle from Hugging Face facebook/voxpopuli (English, validation split).

Loads only ``en/validation-00000-of-00001.parquet`` (no full ``en/train-*`` download). Optional
``--validation-parquet`` uses a local copy offline.

Each row: transcript = normalized_text, audio written to --audio-cache-dir for HuBERT/EfficientNet.
Output includes test[\"audio_path\"] aligned with embeddings for run_noise_robustness_eval.py.

Use ``--replicate-for-train`` (or ``--val-fraction``) so the pickle also has ``train`` / ``validation``
keys required by ``scripts/train.py``.
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
    from huggingface_hub import hf_hub_download
except ImportError as e:
    raise SystemExit(
        "Missing dependency: huggingface_hub (comes with datasets). "
        "Install with: uv sync --extra voxpopuli"
    ) from e


VOXPOPULI_REPO = "facebook/voxpopuli"
VOXPOPULI_EN_VALIDATION_RELPATH = "en/validation-00000-of-00001.parquet"

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise SystemExit(
        "Missing dependency: sentence-transformers. Install with: uv sync --extra voxpopuli"
    ) from e


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
    peak = float(np.max(np.abs(arr)))
    if peak > 0:
        arr = arr / peak
    if sr != TARGET_SR:
        arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
    if len(arr) < MIN_SAMPLES_16K:
        arr = np.pad(arr, (0, MIN_SAMPLES_16K - len(arr)), mode="constant")
    return arr


def _audio_to_mono_16k_padded(audio_dict_or_array) -> np.ndarray:
    """PCM mono 16 kHz padded from HF row value.

    With ``Audio(decode=False)`` (see ``_dataset_audio_no_decode``), ``audio`` is a dict with
    ``path`` and/or ``bytes`` only — no ``torchcodec`` needed. Older rows may still expose
    ``array`` + ``sampling_rate``.
    """
    if isinstance(audio_dict_or_array, dict):
        if audio_dict_or_array.get("array") is not None:
            arr = np.asarray(audio_dict_or_array["array"], dtype=np.float32)
            sr = int(audio_dict_or_array.get("sampling_rate") or TARGET_SR)
            return _normalize_resample_pad_mono(arr, sr)
        raw = audio_dict_or_array.get("bytes")
        if raw:
            buf = io.BytesIO(raw)
            arr, sr = sf.read(buf, dtype="float32", always_2d=False)
            return _normalize_resample_pad_mono(np.asarray(arr, dtype=np.float32), int(sr))
        path = audio_dict_or_array.get("path")
        if path:
            arr, sr = sf.read(path, dtype="float32", always_2d=False)
            return _normalize_resample_pad_mono(np.asarray(arr, dtype=np.float32), int(sr))
        raise ValueError("audio dict has no array, bytes, or path")
    arr = np.asarray(audio_dict_or_array, dtype=np.float32)
    return _normalize_resample_pad_mono(arr, TARGET_SR)


def _dataset_audio_no_decode(ds):
    """HF ``datasets`` 4.x decodes ``Audio`` with ``torchcodec`` by default; avoid that."""
    feats = getattr(ds, "features", None)
    if feats is not None and "audio" in feats:
        try:
            return ds.cast_column("audio", Audio(decode=False))
        except (TypeError, ValueError, KeyError):
            pass
    return ds


def _safe_wav_name(audio_id: str, idx: int) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(audio_id))[:150]
    return f"{idx:07d}_{safe}.wav"


def _build_split_dict(
    rows: list[dict],
    indices: list[int],
    hubert_processor,
    hubert_model,
    sentence_model: SentenceTransformer,
    vision_model,
    vision_preprocess,
    device: torch.device,
    vision_batch_size: int,
    text_batch_size: int,
) -> dict:
    paths = [rows[i]["_abs_audio"] for i in indices]
    texts = [rows[i]["text"] for i in indices]

    hubert_raw = hubert_audio_files(paths, hubert_processor, hubert_model, device)
    hubert_list = _squeeze_hubert_list(hubert_raw)

    text_emb = sentence_model.encode(
        texts,
        batch_size=text_batch_size,
        convert_to_tensor=True,
        show_progress_bar=len(texts) > 32,
    )
    text_list = []
    for j in range(text_emb.size(0)):
        t = text_emb[j].detach().cpu().float()
        if t.dim() > 1:
            t = t.squeeze(0)
        text_list.append(t)

    image_list = efficientnet_embeddings_from_audio_paths(
        paths,
        vision_model,
        vision_preprocess,
        device,
        batch_size=vision_batch_size,
    )
    image_list = [t.squeeze(0) if t.dim() > 1 else t for t in image_list]

    return {
        "hubert-emb": hubert_list,
        "text": text_list,
        "image": image_list,
        "audio_path": paths,
    }


def _slice_split(split_dict: dict, start: int, stop: int) -> dict:
    return {key: split_dict[key][start:stop] for key in split_dict}


def _resolve_validation_parquet(validation_parquet: Path | None) -> Path:
    if validation_parquet is not None:
        p = validation_parquet.expanduser().resolve()
        if not p.is_file():
            raise SystemExit(f"--validation-parquet not found or not a file: {p}")
        return p
    local = hf_hub_download(
        repo_id=VOXPOPULI_REPO,
        repo_type="dataset",
        filename=VOXPOPULI_EN_VALIDATION_RELPATH,
    )
    return Path(local).resolve()


def load_voxpopuli_rows(
    cache_dir: Path,
    max_samples: int | None,
    require_gold: bool,
    validation_parquet: Path | None,
) -> list[dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = _resolve_validation_parquet(validation_parquet)
    print(f"Loading VoxPopuli EN validation from parquet: {parquet_path}")
    ds = load_dataset(
        "parquet",
        data_files={"validation": str(parquet_path)},
        split="validation",
    )
    ds = _dataset_audio_no_decode(ds)
    rows: list[dict] = []

    for idx, ex in enumerate(tqdm(ds, desc="voxpopuli en/validation")):
        text = (ex.get("normalized_text") or "").strip()
        if not text:
            continue
        if require_gold and not ex.get("is_gold_transcript", True):
            continue
        audio_id = str(ex.get("audio_id", f"row_{idx}"))
        wav_name = _safe_wav_name(audio_id, len(rows))
        out_path = (cache_dir / wav_name).resolve()

        arr = _audio_to_mono_16k_padded(ex["audio"])
        sf.write(str(out_path), arr, TARGET_SR, subtype="PCM_16")

        rows.append({"text": text, "_abs_audio": str(out_path)})
        if max_samples is not None and len(rows) >= max_samples:
            break

    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/datasets/total_dataset_voxpopuli_en_validation.pkl"),
    )
    p.add_argument(
        "--audio-cache-dir",
        type=Path,
        default=Path("data/datasets/voxpopuli_en_validation_wav"),
        help="Where to write per-clip WAV files (reused if already present).",
    )
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument(
        "--require-gold-only",
        action="store_true",
        help="Keep only rows with is_gold_transcript == True.",
    )
    p.add_argument(
        "--validation-parquet",
        type=Path,
        default=None,
        help=(
            "Local path to en/validation-00000-of-00001.parquet (offline). "
            "If omitted, downloads only that file from the Hub via hf_hub_download."
        ),
    )
    p.add_argument("--device", default=None)
    p.add_argument("--hubert-model", default="facebook/hubert-large-ls960-ft")
    p.add_argument("--sentence-transformer", default="sentence-transformers/LaBSE")
    p.add_argument("--vision-batch-size", type=int, default=4)
    p.add_argument("--text-batch-size", type=int, default=32)
    split = p.add_mutually_exclusive_group()
    split.add_argument(
        "--replicate-for-train",
        action="store_true",
        help="Set train and validation to deep copies of the built split (overfit / sanity check for train.py).",
    )
    split.add_argument(
        "--val-fraction",
        type=float,
        default=None,
        metavar="F",
        help="Hold out fraction F in (0,1) for validation; train gets first (1-F) of rows (same order as parquet).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else get_default_device()

    rows = load_voxpopuli_rows(
        args.audio_cache_dir,
        args.max_samples,
        args.require_gold_only,
        args.validation_parquet,
    )
    if not rows:
        raise SystemExit("No usable rows (check text non-empty and filters).")

    n = len(rows)
    print(f"Built {n} rows on {device}")

    hubert_processor = AutoProcessor.from_pretrained(args.hubert_model)
    hubert_model_obj = HubertModel.from_pretrained(args.hubert_model).to(device)
    hubert_model_obj.eval()

    sentence_model = SentenceTransformer(args.sentence_transformer, device=str(device))
    vision_model, vision_preprocess = load_efficientnet_b7(device)

    indices = list(range(n))
    test_split = _build_split_dict(
        rows,
        indices,
        hubert_processor,
        hubert_model_obj,
        sentence_model,
        vision_model,
        vision_preprocess,
        device,
        args.vision_batch_size,
        args.text_batch_size,
    )

    if args.replicate_for_train:
        total_dataset = {
            "train": copy.deepcopy(test_split),
            "validation": copy.deepcopy(test_split),
            "test": copy.deepcopy(test_split),
        }
        split_msg = "train+validation+test (replicated, deep copy)"
    elif args.val_fraction is not None:
        vf = args.val_fraction
        if not (0.0 < vf < 1.0):
            raise SystemExit("--val-fraction must be strictly between 0 and 1")
        k = int(n * (1.0 - vf))
        if k < 1 or n - k < 1:
            raise SystemExit(
                f"--val-fraction={vf} with n={n} yields empty train or validation; "
                "use more rows or a smaller fraction."
            )
        total_dataset = {
            "train": _slice_split(test_split, 0, k),
            "validation": _slice_split(test_split, k, n),
            "test": copy.deepcopy(test_split),
        }
        split_msg = f"train={k} val={n - k} test={n} (full test retained)"
    else:
        total_dataset = {"test": test_split}
        split_msg = "test only"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(total_dataset, f)

    print(f"Wrote {args.output} ({split_msg}, n={n}, audio_path in split)")


if __name__ == "__main__":
    main()
