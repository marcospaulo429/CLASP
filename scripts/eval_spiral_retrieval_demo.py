#!/usr/bin/env python3
"""
CLASP SPIRAL Evaluation - DEMO with Mock Data

Este script demonstra a pipeline completa de avaliação usando dados sintéticos.
Útil para validar o código sem precisar dos áudios reais do SPIRAL.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def generate_mock_spiral_data(num_samples: int = 50, output_dir: Path = None) -> Path:
    """Gera dados SPIRAL mock para teste."""
    if output_dir is None:
        output_dir = ROOT / "results" / "spiral_demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    mock_wavs_dir = output_dir / "mock_wavs"
    mock_wavs_dir.mkdir(exist_ok=True)

    samples = []

    for i in range(num_samples):
        # Gera áudio sintético (ruído/zeros)
        duration = np.random.uniform(30, 120)  # 30-120 segundos
        sample_rate = 16000
        num_samples_audio = int(duration * sample_rate)

        # Cria áudio dummy (silêncio com pequeno ruído)
        audio = np.random.randn(num_samples_audio).astype(np.float32) * 0.01
        audio_path = mock_wavs_dir / f"lecture_{i}.wav"

        import soundfile as sf
        sf.write(audio_path, audio, sample_rate)

        # Timestamp simulado - distribuído pelos bins temporais
        if i < num_samples // 5:
            start_time = np.random.uniform(0, 15)
        elif i < 2 * num_samples // 5:
            start_time = np.random.uniform(20, 35)
        elif i < 3 * num_samples // 5:
            start_time = np.random.uniform(40, 55)
        elif i < 4 * num_samples // 5:
            start_time = np.random.uniform(60, 75)
        else:
            start_time = np.random.uniform(85, 100)

        sample = {
            "audio_path": str(audio_path),
            "key_sentence": f"This is a sample key sentence number {i} for testing the CLASP retrieval system.",
            "key_sentence_timestamp": [start_time, start_time + 5.0],
            "metadata": {"id": f"mock_{i}"},
        }
        samples.append(sample)

    # Salva JSONL
    jsonl_path = output_dir / "mock_data.jsonl"
    with open(jsonl_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"✅ Mock data created: {jsonl_path}")
    print(f"   Samples: {num_samples}")
    print(f"   Audio files: {mock_wavs_dir}")

    return jsonl_path


class MockTextEmbedder:
    """Mock text embedder para demonstração."""

    def __init__(self, device: torch.device, dim: int = 768):
        self.device = device
        self.dim = dim
        print(f"  Mock TextEmbedder (dim={dim})")

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> torch.Tensor:
        """Gera embeddings sintéticos baseados no hash do texto."""
        embeddings = []
        for text in texts:
            # Gera embedding pseudo-aleatório mas determinístico
            np.random.seed(hash(text) % (2**32))
            emb = torch.randn(self.dim)
            embeddings.append(emb)
        return torch.stack(embeddings)


class MockAudioEmbedder:
    """Mock audio embedder para demonstração."""

    def __init__(self, device: torch.device, dim: int = 1768):
        self.device = device
        self.dim = dim  # 768 HuBERT + 1000 EfficientNet
        print(f"  Mock AudioEmbedder (dim={dim})")

    def embed_batch(self, audio_paths: list[str], batch_size: int = 1) -> torch.Tensor:
        """Gera embeddings sintéticos baseados no arquivo."""
        embeddings = []
        for path in tqdm(audio_paths, desc="Mock audio embeddings"):
            # Gera embedding pseudo-aleatório mas determinístico
            np.random.seed(hash(Path(path).name) % (2**32))
            emb = torch.randn(self.dim)
            embeddings.append(emb)
        return torch.stack(embeddings)


class MockCLASPModel(nn.Module):
    """Mock CLASP fusion model."""

    def __init__(self, in_audio: int = 768, in_image: int = 1000, out_dim: int = 768):
        super().__init__()
        self.audio_proj = nn.Linear(in_audio, out_dim)
        self.image_proj = nn.Linear(in_image, out_dim)
        self.fusion = nn.Linear(out_dim * 2, out_dim)

    def forward(self, x_audio, x_image):
        a = self.audio_proj(x_audio)
        i = self.image_proj(x_image)
        fused = torch.cat([a, i], dim=-1)
        return self.fusion(fused)


class MockCLASPEvaluator:
    """Mock CLASP evaluator."""

    def __init__(self, device: torch.device, model_path: Path = None):
        self.device = device
        self.model = MockCLASPModel().to(device)
        self.model.eval()
        print("  Mock CLASPEvaluator loaded")

    def fuse_embeddings(self, audio_emb: torch.Tensor, image_emb: torch.Tensor, batch_size: int = 32) -> torch.Tensor:
        with torch.no_grad():
            return self.model(audio_emb.to(self.device), image_emb.to(self.device)).cpu()


def compute_similarity_matrix(text_embeddings: torch.Tensor, audio_embeddings: torch.Tensor) -> np.ndarray:
    """Computa matriz de similaridade cosseno."""
    text_norm = F.normalize(text_embeddings, p=2, dim=1)
    audio_norm = F.normalize(audio_embeddings, p=2, dim=1)
    return torch.mm(text_norm, audio_norm.t()).detach().numpy()


def compute_recall_at_k(similarity_matrix: np.ndarray, ks: list[int] = [1, 5, 10]) -> tuple:
    """Computa métricas de recall."""
    n = similarity_matrix.shape[0]
    ranks = []

    for i in range(n):
        sorted_indices = np.argsort(-similarity_matrix[i])
        rank = np.where(sorted_indices == i)[0][0] + 1
        ranks.append(rank)

    ranks = np.array(ranks)
    metrics = {}

    for k in ks:
        metrics[f"Recall@{k}"] = float(np.mean(ranks <= k)) * 100

    metrics["MRR"] = float(np.mean(1.0 / ranks))
    metrics["mean_rank"] = float(np.mean(ranks))
    metrics["median_rank"] = float(np.median(ranks))

    return metrics, ranks


def compute_temporal_bins(similarity_matrix: np.ndarray, timestamps: list, bins: list = None) -> list:
    """Análise por bins temporais."""
    if bins is None:
        bins = [(0, 20), (20, 40), (40, 60), (60, 80), (80, None)]

    results = []
    for bin_start, bin_end in bins:
        indices = []
        for i, ts in enumerate(timestamps):
            start = ts[0] if isinstance(ts, (list, tuple)) and len(ts) > 0 else 0.0
            if bin_end is None:
                if start >= bin_start:
                    indices.append(i)
            else:
                if bin_start <= start < bin_end:
                    indices.append(i)

        if len(indices) == 0:
            results.append({
                "bin": f"{bin_start}-{bin_end if bin_end else '+'}s",
                "total_samples": 0,
                "success_count": 0,
                "recall_at_1": 0.0,
            })
            continue

        success = sum(1 for i in indices if np.argsort(-similarity_matrix[i])[0] == i)
        results.append({
            "bin": f"{bin_start}-{bin_end if bin_end else '+'}s",
            "total_samples": len(indices),
            "success_count": success,
            "recall_at_1": (success / len(indices)) * 100,
        })

    return results


def plot_metrics(metrics: dict, temporal_bins: list, output_dir: Path):
    """Gera gráficos."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    # Global metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    recall_keys = [k for k in metrics.keys() if k.startswith("Recall@")]
    recall_vals = [metrics[k] for k in recall_keys]

    colors = ["#2ecc71", "#3498db", "#9b59b6"]
    bars = ax.bar(recall_keys, recall_vals, color=colors[:len(recall_keys)],
                  edgecolor="black", linewidth=1.5)

    for bar, val in zip(bars, recall_vals):
        ax.annotate(f"{val:.1f}%", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points", ha="center", va="bottom",
                    fontsize=14, fontweight="bold")

    ax.set_ylabel("Recall (%)", fontsize=14, fontweight="bold")
    ax.set_title("CLASP SPIRAL (DEMO) - Global Metrics", fontsize=16, fontweight="bold")
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(output_dir / "demo_global_metrics.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Temporal bins
    fig, ax = plt.subplots(figsize=(12, 6))

    valid_bins = [b for b in temporal_bins if b["total_samples"] > 0]
    if valid_bins:
        bins = [b["bin"] for b in valid_bins]
        recalls = [b["recall_at_1"] for b in valid_bins]
        counts = [b["total_samples"] for b in valid_bins]

        ax2 = ax.twinx()

        x_pos = np.arange(len(bins))
        bars1 = ax.bar(x_pos, recalls, color="#3498db", alpha=0.8,
                       edgecolor="black", linewidth=1.5, width=0.6)

        ax2.plot(x_pos, counts, color="#e74c3c", marker="o", markersize=10, linewidth=2.5)

        for bar, val in zip(bars1, recalls):
            ax.annotate(f"{val:.1f}%", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 5), textcoords="offset points", ha="center", va="bottom",
                        fontsize=11, fontweight="bold")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(bins, fontsize=11)
        ax.set_xlabel("Temporal Position (seconds)", fontsize=14, fontweight="bold")
        ax.set_ylabel("Recall@1 (%)", fontsize=14, fontweight="bold", color="#3498db")
        ax2.set_ylabel("Sample Count", fontsize=14, fontweight="bold", color="#e74c3c")
        ax.set_title("CLASP SPIRAL (DEMO) - Accuracy by Temporal Position",
                     fontsize=16, fontweight="bold")
        ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(output_dir / "demo_temporal_bins.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Plots saved to: {output_dir}")


def print_results(metrics: dict, bins: list):
    """Imprime tabelas."""
    print("\n" + "=" * 70)
    print("CLASP SPIRAL (DEMO) - Global Metrics")
    print("=" * 70)
    print("\n| Metric | Value |")
    print("|--------|-------|")
    for k in [1, 5, 10]:
        key = f"Recall@{k}"
        if key in metrics:
            print(f"| {key} | {metrics[key]:.2f}% |")
    print(f"| MRR | {metrics.get('MRR', 0):.4f} |")

    print("\n" + "=" * 70)
    print("CLASP SPIRAL (DEMO) - Accuracy by Temporal Position")
    print("=" * 70)
    print("\n| Temporal Bin | Samples | Success | Recall@1 |")
    print("|--------------|---------|---------|----------|")
    for b in bins:
        print(f"| {b['bin']} | {b['total_samples']} | {b['success_count']} | {b['recall_at_1']:.2f}% |")
    print()


def run_demo(num_samples: int = 50, output_dir: Path = None):
    """Executa demo completa."""
    if output_dir is None:
        output_dir = ROOT / "results" / "spiral_demo"

    print("\n" + "=" * 70)
    print("CLASP SPIRAL Evaluation - DEMO with Mock Data")
    print("=" * 70)

    # 1. Gera dados mock
    print("\n📂 Gerando dados mock...")
    data_path = generate_mock_spiral_data(num_samples, output_dir)

    # 2. Carrega dados
    samples = []
    with open(data_path, "r") as f:
        for line in f:
            samples.append(json.loads(line.strip()))

    print(f"   Loaded {len(samples)} samples")

    # 3. Inicializa embedders mock
    print("\n🔧 Inicializando modelos mock...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")

    text_embedder = MockTextEmbedder(device)
    audio_embedder = MockAudioEmbedder(device)
    clasp_evaluator = MockCLASPEvaluator(device)

    # 4. Gera embeddings
    print("\n📝 Gerando embeddings de texto...")
    texts = [s["key_sentence"] for s in samples]
    text_emb = text_embedder.embed_batch(texts)

    print("\n🔊 Gerando embeddings de áudio...")
    audio_paths = [s["audio_path"] for s in samples]
    raw_audio_emb = audio_embedder.embed_batch(audio_paths)

    # 5. Fusão CLASP
    print("\n🔗 Aplicando fusão CLASP...")
    audio_hubert = raw_audio_emb[:, :768]
    audio_spec = raw_audio_emb[:, 768:1768]
    fused_audio = clasp_evaluator.fuse_embeddings(audio_hubert, audio_spec)

    # Para demo, faz texto = áudio fundido (simulando retrieval perfeito)
    # Na prática, texto seria projetado para espaço conjunto
    text_emb = nn.Linear(768, 768)(text_emb)

    # 6. Similaridade
    print("\n📊 Computando similaridade...")
    sim_matrix = compute_similarity_matrix(text_emb, fused_audio)

    # 7. Métricas
    print("\n📈 Calculando métricas...")
    # Adiciona correlação artificial para simular retrieval
    for i in range(len(samples)):
        sim_matrix[i, i] += 0.5  # Aumenta diagonal

    global_metrics, _ = compute_recall_at_k(sim_matrix)
    timestamps = [s["key_sentence_timestamp"] for s in samples]
    temporal_bins = compute_temporal_bins(sim_matrix, timestamps)

    # 8. Resultados
    print_results(global_metrics, temporal_bins)

    # 9. Gráficos
    print("\n📊 Gerando gráficos...")
    plot_metrics(global_metrics, temporal_bins, output_dir)

    # 10. Salva JSON
    results = {
        "global_metrics": global_metrics,
        "temporal_bins": temporal_bins,
        "total_samples": len(samples),
        "note": "DEMO with mock data - not real SPIRAL evaluation",
    }
    results_path = output_dir / "demo_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Results saved: {results_path}")
    print("\n" + "=" * 70)
    print("Demo completed!")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="CLASP SPIRAL Demo with Mock Data")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of mock samples")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "spiral_demo",
                        help="Output directory")
    args = parser.parse_args()

    run_demo(args.num_samples, args.output)


if __name__ == "__main__":
    main()
