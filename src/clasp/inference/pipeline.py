import torch
import torch.nn as nn


def load_model(model_path, device):
    model_to_test = torch.load(model_path, map_location=device)
    model_to_test = model_to_test.to(device)
    model_to_test.eval()
    return model_to_test


def build_final_embeddings(model, audio_embeddings, image_embeddings):
    with torch.no_grad():
        return model(audio_embeddings, image_embeddings)


def retrieve_top1(text_embeddings, final_embeddings, sample_index, device):
    sample_embedding = text_embeddings[sample_index].unsqueeze(0).to(device)
    cos_sim = nn.CosineSimilarity(dim=1)
    similarities = cos_sim(sample_embedding, final_embeddings)
    most_similar_idx = torch.argmax(similarities).item()
    return most_similar_idx, similarities[most_similar_idx].item()

