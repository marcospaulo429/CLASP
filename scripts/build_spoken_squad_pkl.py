#!/usr/bin/env python3
"""Constrói `total_dataset_spoken_squad.pkl` com pareamento correto:

texto = ``paragraphs[*]["context"]`` (transcrição)
áudio = leitura desse contexto, distribuída em chunks ``{a}_{p}_*.wav``.

Modos de pooling:

* ``mean``     — Variante A: 1 amostra por parágrafo. Concatena os WAVs do
                 parágrafo em um único waveform, extrai HuBERT (mean-pool sobre
                 chunks de 20 s) e EfficientNet-B7 (mean-pool sobre janelas).
* ``chunked``  — Variante B (estilo SPIRAL): 1 amostra por chunk individual.
                 Texto do parágrafo é replicado para todos os chunks. A coluna
                 ``paragraph_id`` permite agrupamento no eval (max-sim).

Saída (por split):
    text         : list[Tensor[1024]]    — embedding LaBSE
    hubert-emb   : list[Tensor[1024]]    — HuBERT (mean sobre chunks de 20 s)
    image        : list[Tensor[1000]]    — EfficientNet-B7 logits (mean)
    paragraph_id : list[str]             — "{article_idx}_{paragraph_idx}"

Requer extras: `uv sync --extra realdata`
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
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
from clasp.inference.embed_audio import hubert_numpy_waveform
from clasp.inference.spectrogram_image import (
    efficientnet_embedding_from_waveform,
    load_efficientnet_b7,
)


# --------------------------------------------------------------------------- #
# Parsing                                                                     #
# --------------------------------------------------------------------------- #

def collect_paragraph_chunks(json_path: Path, wav_dir: Path) -> list[dict]:
    """Para cada parágrafo, devolve {paragraph_id, context, wav_paths (ordenados por chunk_idx)}.

    Parágrafos sem nenhum WAV correspondente são descartados.
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    # mapeia (a_idx, p_idx) -> [(chunk_idx, path), ...]
    per_paragraph: dict[tuple[int, int], list[tuple[int, str]]] = defaultdict(list)
    for wav in wav_dir.glob("*.wav"):
        try:
            a, p, c = wav.stem.split("_")
            per_paragraph[(int(a), int(p))].append((int(c), str(wav)))
        except ValueError:
            continue
    for k in per_paragraph:
        per_paragraph[k].sort(key=lambda x: x[0])

    out: list[dict] = []
    articles = data["data"]
    for a_idx, article in enumerate(articles):
        for p_idx, para in enumerate(article["paragraphs"]):
            chunks = per_paragraph.get((a_idx, p_idx))
            if not chunks:
                continue
            out.append({
                "paragraph_id": f"{a_idx}_{p_idx}",
                "context": para["context"],
                "wav_paths": [c[1] for c in chunks],
            })
    return out


# --------------------------------------------------------------------------- #
# Embedding extraction                                                        #
# --------------------------------------------------------------------------- #

def _concat_waveforms(paths: list[str]) -> np.ndarray:
    pieces = [load_mono_16k_padded(p) for p in paths]
    if not pieces:
        return np.zeros(16_000, dtype=np.float32)
    return np.concatenate([np.asarray(x, dtype=np.float32).reshape(-1) for x in pieces])


def build_split_dict_mean(
    paragraphs: list[dict],
    hubert_processor,
    hubert_model,
    sentence_model: SentenceTransformer,
    vision_model,
    vision_preprocess,
    device: torch.device,
    *,
    chunk_samples: int,
    chunk_batch_size: int,
    text_batch_size: int,
) -> dict:
    """Variante A: 1 amostra por parágrafo, mean-pool de chunks."""
    texts: list[str] = []
    hubert_list: list[torch.Tensor] = []
    image_list: list[torch.Tensor] = []
    paragraph_ids: list[str] = []
    audio_paths: list[list[str]] = []

    for para in tqdm(paragraphs, desc="paragraphs (mean)"):
        wav = _concat_waveforms(para["wav_paths"])
        h = hubert_numpy_waveform(
            wav, hubert_processor, hubert_model, device,
            chunk_samples=chunk_samples,
            chunk_batch_size=chunk_batch_size,
            pooling="mean",
        )
        s = efficientnet_embedding_from_waveform(
            wav, vision_model, vision_preprocess, device,
            chunk_samples=chunk_samples,
            chunk_batch_size=chunk_batch_size,
            pooling="mean",
        )
        hubert_list.append(h.detach().cpu().float().contiguous())
        image_list.append(s.detach().cpu().float().contiguous())
        texts.append(para["context"])
        paragraph_ids.append(para["paragraph_id"])
        audio_paths.append(list(para["wav_paths"]))

    text_emb = sentence_model.encode(
        texts, batch_size=text_batch_size, convert_to_tensor=True,
        show_progress_bar=len(texts) > 32,
    )
    text_list = [text_emb[j].detach().cpu().float() for j in range(text_emb.size(0))]

    return {
        "text": text_list,
        "hubert-emb": hubert_list,
        "image": image_list,
        "paragraph_id": paragraph_ids,
        "audio_paths": audio_paths,
    }


def build_split_dict_chunked(
    paragraphs: list[dict],
    hubert_processor,
    hubert_model,
    sentence_model: SentenceTransformer,
    vision_model,
    vision_preprocess,
    device: torch.device,
    *,
    chunk_samples: int,
    chunk_batch_size: int,
    text_batch_size: int,
) -> dict:
    """Variante B (SPIRAL-like): 1 amostra por chunk, texto do parágrafo replicado."""
    flat_texts: list[str] = []
    hubert_list: list[torch.Tensor] = []
    image_list: list[torch.Tensor] = []
    paragraph_ids: list[str] = []
    audio_paths: list[list[str]] = []

    for para in tqdm(paragraphs, desc="paragraphs (chunked)"):
        for wav_path in para["wav_paths"]:
            wav = load_mono_16k_padded(wav_path)
            h = hubert_numpy_waveform(
                wav, hubert_processor, hubert_model, device,
                chunk_samples=chunk_samples,
                chunk_batch_size=chunk_batch_size,
                pooling="mean",
            )
            s = efficientnet_embedding_from_waveform(
                wav, vision_model, vision_preprocess, device,
                chunk_samples=chunk_samples,
                chunk_batch_size=chunk_batch_size,
                pooling="mean",
            )
            hubert_list.append(h.detach().cpu().float().contiguous())
            image_list.append(s.detach().cpu().float().contiguous())
            flat_texts.append(para["context"])
            paragraph_ids.append(para["paragraph_id"])
            audio_paths.append([wav_path])

    # encode com dedup p/ economizar — mas mantém ordem original
    unique_texts: list[str] = []
    text_index: dict[str, int] = {}
    for t in flat_texts:
        if t not in text_index:
            text_index[t] = len(unique_texts)
            unique_texts.append(t)
    encoded = sentence_model.encode(
        unique_texts, batch_size=text_batch_size, convert_to_tensor=True,
        show_progress_bar=len(unique_texts) > 32,
    )
    text_list = [encoded[text_index[t]].detach().cpu().float().clone() for t in flat_texts]

    return {
        "text": text_list,
        "hubert-emb": hubert_list,
        "image": image_list,
        "paragraph_id": paragraph_ids,
        "audio_paths": audio_paths,
    }


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-json", type=Path, required=True)
    p.add_argument("--train-wav-dir", type=Path, required=True)
    p.add_argument("--val-json", type=Path, required=True)
    p.add_argument("--val-wav-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True,
                   help="Caminho do PKL de saída")
    p.add_argument(
        "--pooling-mode",
        choices=["mean", "chunked"],
        required=True,
        help=(
            "mean: 1 amostra/parágrafo (concat+mean-pool). "
            "chunked: 1 amostra/chunk com paragraph_id (eval max-sim)."
        ),
    )
    p.add_argument("--max-train-paragraphs", type=int, default=None,
                   help="Limita # de parágrafos do split de treino (debug)")
    p.add_argument("--max-val-paragraphs", type=int, default=None,
                   help="Limita # de parágrafos do split de validação (debug)")
    p.add_argument("--device", default=None)
    p.add_argument("--hubert-model", default="facebook/hubert-large-ls960-ft")
    p.add_argument("--sentence-transformer", default="sentence-transformers/LaBSE")
    p.add_argument("--chunk-samples", type=int, default=320_000,
                   help="Janela em amostras (16 kHz) para HuBERT/EfficientNet")
    p.add_argument("--chunk-batch-size", type=int, default=1)
    p.add_argument("--text-batch-size", type=int, default=32)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else get_default_device()
    print(f"Device: {device}  |  pooling-mode: {args.pooling_mode}")

    hubert_processor = AutoProcessor.from_pretrained(args.hubert_model)
    hubert_model = HubertModel.from_pretrained(args.hubert_model).to(device)
    hubert_model.eval()
    sentence_model = SentenceTransformer(args.sentence_transformer, device=str(device))
    vision_model, vision_preprocess = load_efficientnet_b7(device)

    builder = (
        build_split_dict_mean if args.pooling_mode == "mean" else build_split_dict_chunked
    )

    print("\n[1/2] Treino …")
    train_paragraphs = collect_paragraph_chunks(args.train_json, args.train_wav_dir)
    if args.max_train_paragraphs is not None:
        train_paragraphs = train_paragraphs[: args.max_train_paragraphs]
    print(f"  {len(train_paragraphs)} parágrafos com áudio")
    train_split = builder(
        train_paragraphs, hubert_processor, hubert_model,
        sentence_model, vision_model, vision_preprocess, device,
        chunk_samples=args.chunk_samples,
        chunk_batch_size=args.chunk_batch_size,
        text_batch_size=args.text_batch_size,
    )
    print(f"  {len(train_split['text'])} amostras de treino geradas")

    print("\n[2/2] Validação …")
    val_paragraphs = collect_paragraph_chunks(args.val_json, args.val_wav_dir)
    if args.max_val_paragraphs is not None:
        val_paragraphs = val_paragraphs[: args.max_val_paragraphs]
    print(f"  {len(val_paragraphs)} parágrafos com áudio")
    val_split = builder(
        val_paragraphs, hubert_processor, hubert_model,
        sentence_model, vision_model, vision_preprocess, device,
        chunk_samples=args.chunk_samples,
        chunk_batch_size=args.chunk_batch_size,
        text_batch_size=args.text_batch_size,
    )
    print(f"  {len(val_split['text'])} amostras de validação geradas")

    total_dataset = {
        "train": train_split,
        "validation": val_split,
        "_meta": {"pooling_mode": args.pooling_mode},
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(total_dataset, f)

    print(f"\nPKL salvo em {args.output}")


if __name__ == "__main__":
    main()
