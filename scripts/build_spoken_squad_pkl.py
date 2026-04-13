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

import torch
from tqdm import tqdm
from transformers import AutoProcessor, HubertModel
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clasp.config.settings import get_default_device
from clasp.data.spoken_squad_wavs import concat_paragraph_wavs, iter_spoken_squad_paragraphs
from clasp.inference.embed_audio import hubert_numpy_waveform
from clasp.inference.pipeline import build_final_embeddings, load_model
from clasp.inference.spectrogram_image import efficientnet_embedding_from_waveform, load_efficientnet_b7


def _empty_split() -> dict:
    return {"hubert-emb": [], "text": [], "image": []}


def _compute_clasp_batched(
    model,
    hubert_list: list[torch.Tensor],
    image_list: list[torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> list[torch.Tensor]:
    n = len(hubert_list)
    out: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, n, batch_size), desc="CLASP forward"):
            end = min(start + batch_size, n)
            ha = torch.stack([hubert_list[i].float() for i in range(start, end)]).to(device)
            im = torch.stack([image_list[i].float() for i in range(start, end)]).to(device)
            fused = build_final_embeddings(model, ha, im)
            for j in range(fused.size(0)):
                out.append(fused[j].detach().cpu().float().squeeze())
    return out


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
    p.add_argument("--compute-clasp", action="store_true", help="Preencher test['clasp_emb']")
    p.add_argument("--model-path", type=Path, default=None, help="Checkpoint CLASP (.pt); obrigatório com --compute-clasp")
    p.add_argument("--clasp-batch-size", type=int, default=16)
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

    hubert_list: list[torch.Tensor] = []
    image_list: list[torch.Tensor] = []
    contexts: list[str] = []

    for row in tqdm(rows, desc="áudio+visão por parágrafo"):
        y = concat_paragraph_wavs(row["wav_paths"])
        h = hubert_numpy_waveform(
            y,
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
            chunk_samples=args.chunk_samples,
        )
        image_list.append(img.squeeze(0) if img.dim() > 1 else img)
        contexts.append(row["context"])

    text_emb = sentence_model.encode(
        contexts,
        batch_size=32,
        convert_to_tensor=True,
        show_progress_bar=len(contexts) > 32,
    )
    text_list: list[torch.Tensor] = []
    for j in range(text_emb.size(0)):
        t = text_emb[j].detach().cpu().float()
        if t.dim() > 1:
            t = t.squeeze(0)
        text_list.append(t)

    test_split = {
        "hubert-emb": hubert_list,
        "text": text_list,
        "image": image_list,
    }

    if args.compute_clasp:
        model = load_model(str(args.model_path), device)
        clasp_list = _compute_clasp_batched(
            model,
            hubert_list,
            image_list,
            device,
            args.clasp_batch_size,
        )
        test_split["clasp_emb"] = clasp_list

    total_dataset = {
        "train": _empty_split(),
        "validation": _empty_split(),
        "test": test_split,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(total_dataset, f)

    print(f"Escrito {args.output} (test: {n} amostras)")


if __name__ == "__main__":
    main()
