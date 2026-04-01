import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm


def hubert_audio_files(audio_file_list, hubert_processor, hubert_model, device):
    embeddings = []
    for file_path in tqdm(audio_file_list):
        data, samplerate = sf.read(file_path)
        data = data / np.max(np.abs(data))
        data = librosa.resample(data, orig_sr=samplerate, target_sr=16000)
        audio = torch.from_numpy(data)
        inputs = hubert_processor(audio, sampling_rate=16000, return_tensors="pt").to(device)
        with torch.no_grad():
            hidden_states = hubert_model(**inputs).last_hidden_state
            avg_embedding = torch.mean(hidden_states, dim=1)
            embeddings.append(avg_embedding)
    return embeddings

