import torch
import torch.nn.functional as F
from tqdm import tqdm


def cosine_similarity(embedding1, embedding2):
    dim = 1
    embedding1 = F.normalize(embedding1, p=2, dim=dim)
    embedding2 = F.normalize(embedding2, p=2, dim=dim)
    dot_product = torch.sum(embedding1 * embedding2, dim=dim)
    magnitude1 = torch.norm(embedding1, p=2, dim=dim)
    magnitude2 = torch.norm(embedding2, p=2, dim=dim)
    return dot_product / (magnitude1 * magnitude2)


def build_similarity_matrix(query_embeddings, candidate_embeddings):
    similarity_matrix = []
    for i in tqdm(range(len(query_embeddings))):
        similarity_matrix.append([])
        for j in range(len(candidate_embeddings)):
            similarity_matrix[i].append(
                cosine_similarity(
                    query_embeddings[i].unsqueeze(0),
                    candidate_embeddings[j].unsqueeze(0),
                ).item()
            )
    return similarity_matrix


def retrieve_topk(query_embedding, candidate_embeddings, k=10):
    sims = []
    for i, candidate_embedding in enumerate(candidate_embeddings):
        sim = cosine_similarity(query_embedding.unsqueeze(0), candidate_embedding.unsqueeze(0)).item()
        sims.append((i, sim))
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:k]

