#!/usr/bin/env python3
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clasp.config.settings import get_default_device
from clasp.data.datasets import CusDataset
from clasp.models.fusion import HubertLabseConcat
from clasp.train.trainer import train_the_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train CLASP model from extracted modules.")
    parser.add_argument("--dataset-path", required=True, help="Path to total_dataset .pkl file")
    parser.add_argument("--save-path", required=True, help="Output model path (.pt)")
    parser.add_argument("--audio-key", default="hubert-emb", help="Audio embedding key")
    parser.add_argument("--text-key", default="text", help="Text embedding key")
    parser.add_argument("--batch-size-train", type=int, default=32)
    parser.add_argument("--batch-size-val", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--temperature", type=float, default=float(np.log(0.07)))
    parser.add_argument("--in-features-text", type=int, default=1024)
    parser.add_argument("--in-features-image", type=int, default=1000)
    parser.add_argument("--mode", default="joint", choices=["joint", "audio", "image"])
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_default_device()

    with open(args.dataset_path, "rb") as f:
        total_dataset = pickle.load(f)

    train_loader = DataLoader(
        dataset=CusDataset(total_dataset["train"], args.audio_key, args.text_key),
        batch_size=args.batch_size_train,
        shuffle=True,
    )
    val_loader = DataLoader(
        dataset=CusDataset(total_dataset["validation"], args.audio_key, args.text_key),
        batch_size=args.batch_size_val,
        shuffle=False,
    )

    model = HubertLabseConcat(
        in_features_text=args.in_features_text,
        in_features_image=args.in_features_image,
        mode=args.mode,
    ).to(device)

    best_model = train_the_model(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        model_path_save=args.save_path,
        device=device,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
    )
    torch.save(best_model, args.save_path)
    print(f"Saved model to {args.save_path}")


if __name__ == "__main__":
    main()

