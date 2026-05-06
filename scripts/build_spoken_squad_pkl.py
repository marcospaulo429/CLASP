#!/usr/bin/env python3
"""Constrói `total_dataset.pkl` a partir de Spoken-SQuAD (JSON SQuAD + `dev_wav`).

Uma linha por parágrafo: LaBSE do `context` escrito; áudio = concatenação temporal dos WAVs
`{article_idx}_{paragraph_idx}_{sentence_idx}.wav` (ordenados pelo terceiro índice; lacunas OK).

Requer extras: `uv sync --extra realdata`

Opcional: `--compute-clasp --model-path` para preencher `test['clasp_emb']` (modo matrix).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path
import numpy as np

import torch
from tqdm import tqdm
from transformers import AutoProcessor, HubertModel
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clasp.config.settings import get_default_device
from clasp.inference.audio_preprocess import load_mono_16k_padded
from clasp.inference.embed_audio import hubert_audio_files
from clasp.inference.spectrogram_image import (
    efficientnet_embeddings_from_audio_paths,
    load_efficientnet_b7,
)
from clasp.audio.noise_augmentation import (
    add_ambient_noise,
    add_reverberation,
    add_white_noise,
    load_esc50_clip,
    scan_esc50_files,
)


def _squeeze_hubert_list(embeddings: list[torch.Tensor]) -> list[torch.Tensor]:
    out = []
    for e in embeddings:
        if e.dim() > 1:
            e = e.squeeze(0)
        out.append(e.contiguous().float().cpu())
    return out


def hubert_audio_files_with_noise(
    paths: list[str],
    processor,
    model,
    device: torch.device,
    noise_prob: float,
    noise_snr: float,
    noise_types: list[str],
    esc50_files: list | None = None,
) -> list[torch.Tensor]:
    embeddings = []
    for path in tqdm(paths, desc="hubert+noise"):
        audio = load_mono_16k_padded(path)
        if noise_prob > 0.0 and np.random.random() < noise_prob:
            noise_type = np.random.choice(noise_types)
            if noise_type == "white":
                audio = add_white_noise(audio, snr_db=noise_snr)
            elif noise_type == "reverb":
                audio = add_reverberation(audio, sr=16000)
            elif noise_type == "ambient" and esc50_files:
                noise_clip = load_esc50_clip(esc50_files, target_sr=16000)
                audio = add_ambient_noise(audio, noise_clip, snr_db=noise_snr)
        t = torch.from_numpy(audio.astype(np.float32))
        inputs = processor(t, sampling_rate=16000, return_tensors="pt").to(device)
        with torch.no_grad():
            hidden = model(**inputs).last_hidden_state
            embeddings.append(torch.mean(hidden, dim=1))
    return embeddings


def _split_indices(
    n: int, train_ratio: float, val_ratio: float, test_ratio: float, seed: int
) -> tuple[list[int], list[int], list[int]]:
    s = train_ratio + val_ratio + test_ratio
    if abs(s - 1.0) > 1e-3:
        raise ValueError(f"train + val + test must sum to 1.0, got {s}")
    indices = list(range(n))
    train_idx, temp_idx = train_test_split(
        indices,
        train_size=train_ratio,
        random_state=seed,
        shuffle=True,
    )
    rel_val = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=rel_val,
        random_state=seed,
        shuffle=True,
    )
    return train_idx, val_idx, test_idx


def load_spoken_squad_rows(json_path: Path, wav_dir: Path, max_samples: int | None) -> list[dict]:
    """Parse Spoken SQuAD JSON and pair each QA entry with its positional WAV file."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    articles = data["data"]
    rows: list[dict] = []

    for a_idx, article in enumerate(tqdm(articles, desc="parse JSON")):
        for p_idx, para in enumerate(article["paragraphs"]):
            for q_idx, qa in enumerate(para["qas"]):
                wav_path = wav_dir / f"{a_idx}_{p_idx}_{q_idx}.wav"
                if not wav_path.is_file():
                    continue
                rows.append({
                    "text": qa["question"],
                    "_abs_audio": str(wav_path),
                })
                if max_samples is not None and len(rows) >= max_samples:
                    return rows

    return rows


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
    noise_prob: float = 0.0,
    noise_snr: float = 20.0,
    noise_types: list[str] | None = None,
    esc50_files: list | None = None,
) -> dict:
    paths = [rows[i]["_abs_audio"] for i in indices]
    texts = [rows[i]["text"] for i in indices]

    if noise_prob > 0.0 and noise_types:
        hubert_raw = hubert_audio_files_with_noise(
            paths, hubert_processor, hubert_model, device,
            noise_prob, noise_snr, noise_types, esc50_files,
        )
    else:
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
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", type=Path, required=True, help="spoken_test-v1.1.json (formato SQuAD)")
    p.add_argument("--wav-dir", type=Path, required=True, help="Pasta com dev_wav (*.wav)")
    p.add_argument("--output", type=Path, default=Path("data/datasets/total_dataset_spoken_squad_test.pkl"))
    p.add_argument("--max-samples", type=int, default=None, help="Limitar número de parágrafos (debug)")
    p.add_argument("--chunk-samples", type=int, default=320_000, help="Janela 16 kHz para HuBERT/EfficientNet (~20s)")
    p.add_argument("--device", default=None, help="cuda, cuda:0, ou cpu")
    p.add_argument("--hubert-model", default="facebook/hubert-large-ls960-ft")
    p.add_argument("--sentence-transformer", default="sentence-transformers/LaBSE")
    p.add_argument("--vision-batch-size", type=int, default=4)
    p.add_argument("--text-batch-size", type=int, default=32)
    # Noise augmentation (applied only to training split)
    p.add_argument("--noise-prob", type=float, default=0.0,
                   help="Probability [0-1] of applying noise to each training sample.")
    p.add_argument("--noise-snr", type=float, default=20.0,
                   help="SNR in dB for noise augmentation (lower = more noise).")
    p.add_argument("--noise-types", nargs="+", default=["white", "reverb"],
                   choices=["white", "reverb", "ambient"],
                   help="Noise types to randomly sample from.")
    p.add_argument("--esc50-dir", type=Path, default=None,
                   help="Path to ESC-50 dataset root (required when 'ambient' is in --noise-types).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.compute_clasp and not args.model_path:
        raise SystemExit("--compute-clasp requer --model-path")

    device = torch.device(args.device) if args.device else get_default_device()

    with open(args.json, encoding="utf-8") as f:
        squad = json.load(f)

    rows = iter_spoken_squad_paragraphs(squad, args.wav_dir)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    n = len(rows)
    if n < 100:
        warnings.warn(
            f"Apenas {n} parágrafos. run_retrieval_eval.py usa --num-candidates 100 por defeito; "
            f"usa --num-candidates <= {max(1, n - 1)}.",
            stacklevel=1,
        )

    print(f"Parágrafos: {n}; device={device}; chunk_samples={args.chunk_samples}")

    hubert_processor = AutoProcessor.from_pretrained(args.hubert_model)
    hubert_model = HubertModel.from_pretrained(args.hubert_model).to(device)
    hubert_model.eval()

    sentence_model = SentenceTransformer(args.sentence_transformer, device=str(device))
    vision_model, vision_preprocess = load_efficientnet_b7(device)

    esc50_files = None
    if "ambient" in args.noise_types:
        if args.esc50_dir is None:
            raise SystemExit("--esc50-dir is required when 'ambient' is in --noise-types.")
        esc50_files = scan_esc50_files(args.esc50_dir)
        print(f"Loaded {len(esc50_files)} ESC-50 clips from {args.esc50_dir}")

    def build(indices: list[int], augment: bool = False) -> dict:
        return _build_split_dict(
            rows,
            indices,
            hubert_processor,
            hubert_model,
            device,
            chunk_samples=args.chunk_samples,
        )
        hubert_list.append(h.detach().cpu().float().squeeze())
        img = efficientnet_embedding_from_waveform(
            y,
            vision_model,
            vision_preprocess,
            device,
            args.vision_batch_size,
            args.text_batch_size,
            noise_prob=args.noise_prob if augment else 0.0,
            noise_snr=args.noise_snr,
            noise_types=args.noise_types,
            esc50_files=esc50_files,
        )
        test_split["clasp_emb"] = clasp_list

    total_dataset = {
        "train": build(train_idx, augment=True),
        "validation": build(val_idx),
        "test": build(test_idx),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(total_dataset, f)

    print(f"Escrito {args.output} (test: {n} amostras)")


if __name__ == "__main__":
    main()
