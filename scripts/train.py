#!/usr/bin/env python3
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import wandb as _wandb_module
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clasp.config.settings import get_default_device
from clasp.data.datasets import CusDataset
from clasp.inference.pipeline import register_pickled_fusion_classes_for_torch_load
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
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping: stop after this many epochs without validation loss improvement.",
    )
    parser.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="Run all --num-epochs regardless of validation plateau.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional path to a fusion .pt (full HubertLabseConcat module or state_dict). "
            "Weights are loaded into the model built from --in-features-* and --mode; "
            "architecture must match (e.g. official CLASP_Concat_Final_Fusion_Encoder.pt). "
            "Does not restore optimizer or epoch — fusion weights only."
        ),
    )
    # ------------------------------------------------------------------ wandb
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="W&B project name. If omitted, wandb logging is disabled.",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="W&B run name (auto-generated if omitted).",
    )
    parser.add_argument(
        "--wandb-entity",
        default=None,
        help="W&B entity (team/user). Uses default entity if omitted.",
    )
    return parser.parse_args()


def _load_init_weights(model: nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"--init-checkpoint not found: {checkpoint_path}")

    register_pickled_fusion_classes_for_torch_load()
    try:
        obj = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        obj = torch.load(checkpoint_path, map_location=device)

    if isinstance(obj, nn.Module):
        state = obj.state_dict()
    elif isinstance(obj, dict):
        state = obj
    else:
        raise TypeError(
            f"Checkpoint must be an nn.Module or a state_dict dict; got {type(obj).__name__}"
        )

    model.load_state_dict(state, strict=True)
    model.train()
    print(f"Loaded fusion weights from {checkpoint_path}")


def main():
    args = parse_args()
    device = get_default_device()

    # ------------------------------------------------------------------ wandb
    wandb_run = None
    if args.wandb_project:
        if not _WANDB_AVAILABLE:
            raise SystemExit(
                "--wandb-project specified but 'wandb' is not installed. "
                "Install with: pip install wandb"
            )
        wandb_run = _wandb_module.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            config={
                "dataset_path": str(args.dataset_path),
                "mode": args.mode,
                "num_epochs": args.num_epochs,
                "learning_rate": args.learning_rate,
                "temperature": args.temperature,
                "batch_size_train": args.batch_size_train,
                "batch_size_val": args.batch_size_val,
                "patience": args.patience,
                "no_early_stopping": args.no_early_stopping,
                "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
                "in_features_text": args.in_features_text,
                "in_features_image": args.in_features_image,
            },
        )
        print(f"W&B run: {wandb_run.url}")

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

    if args.init_checkpoint is not None:
        _load_init_weights(model, args.init_checkpoint, device)

    best_model = train_the_model(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        model_path_save=args.save_path,
        device=device,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        patience=args.patience,
        no_early_stopping=args.no_early_stopping,
        wandb_run=wandb_run,
    )
    torch.save(best_model, args.save_path)
    print(f"Saved model to {args.save_path}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()

