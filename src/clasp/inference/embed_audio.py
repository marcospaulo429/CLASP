import numpy as np
import torch
from tqdm import tqdm

from clasp.inference.audio_preprocess import MIN_SAMPLES_16K, load_mono_16k_padded


def hubert_numpy_waveform(
    waveform: np.ndarray,
    hubert_processor,
    hubert_model,
    device: torch.device,
    chunk_samples: int = 320_000,
) -> torch.Tensor:
    """HuBERT embedding for long mono 16 kHz audio: janelas fixas, média dos vetores por janela."""
    y = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if y.size == 0:
        y = np.zeros(MIN_SAMPLES_16K, dtype=np.float32)
    chunk_vecs: list[torch.Tensor] = []
    start = 0
    while start < y.size:
        end = min(start + chunk_samples, y.size)
        piece = y[start:end].copy()
        if piece.size < MIN_SAMPLES_16K:
            piece = np.pad(piece, (0, MIN_SAMPLES_16K - piece.size), mode="constant")
        audio = torch.from_numpy(piece)
        inputs = hubert_processor(audio, sampling_rate=16000, return_tensors="pt").to(device)
        with torch.no_grad():
            hidden = hubert_model(**inputs).last_hidden_state
            chunk_vecs.append(torch.mean(hidden, dim=1).squeeze(0))
        start = end
    stacked = torch.stack(chunk_vecs, dim=0)
    return torch.mean(stacked, dim=0)


def hubert_audio_files(audio_file_list, hubert_processor, hubert_model, device):
    embeddings = []
    for file_path in tqdm(audio_file_list):
        data = load_mono_16k_padded(file_path)
        audio = torch.from_numpy(data.astype(np.float32))
        inputs = hubert_processor(audio, sampling_rate=16000, return_tensors="pt").to(device)
        with torch.no_grad():
            hidden_states = hubert_model(**inputs).last_hidden_state
            avg_embedding = torch.mean(hidden_states, dim=1)
            embeddings.append(avg_embedding)
    return embeddings
