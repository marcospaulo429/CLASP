import numpy as np
import torch
from tqdm import tqdm

from clasp.inference.audio_preprocess import load_mono_16k_padded


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
