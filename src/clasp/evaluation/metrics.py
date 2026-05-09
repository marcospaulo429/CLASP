from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from clasp.evaluation.ranking_metrics import compute_ranking_metrics, similarity_matrix_to_rows


def _cosine_similarity(embedding1, embedding2):
    dim = 1
    embedding1 = F.normalize(embedding1, p=2, dim=dim)
    embedding2 = F.normalize(embedding2, p=2, dim=dim)
    dot_product = torch.sum(embedding1 * embedding2, dim=dim)
    magnitude1 = torch.norm(embedding1, p=2, dim=dim)
    magnitude2 = torch.norm(embedding2, p=2, dim=dim)
    return dot_product / (magnitude1 * magnitude2)


def evaluate_model_on_candidates(model, dataloader, device, threshold=0.5):
    model.eval()
    total_hits_1 = 0
    total_mrr = 0
    total_instances = 0
    total_labels = []
    total_predictions = []
    number_of_golden_predictions = 0

    with torch.no_grad():
        for text_embedding, audio_candidates, image_candidates, label in tqdm(dataloader):
            label = label[0]
            text_embedding = text_embedding[0].to(device)
            label = label.to(device)
            audio_candidates = audio_candidates[0].to(device)
            image_candidates = image_candidates[0].to(device)
            final_embs = model(audio_candidates, image_candidates)

            similarities = [
                _cosine_similarity(text_embedding.unsqueeze(0), item.unsqueeze(0)).item()
                for item in final_embs
            ]
            predicted_idx = np.argmax(similarities)
            label_similarity = similarities[label.item()]

            if predicted_idx == label.item():
                total_hits_1 += 1

            label_rank = sum(1 for x in similarities if x > similarities[label.item()])
            reciprocal_rank = 1 / (label_rank + 1)
            total_mrr += reciprocal_rank

            predictions = [0 if sim < threshold else 1 for sim in similarities]
            total_labels.extend([0 if i != label.item() else 1 for i in range(len(similarities))])
            total_predictions.extend(predictions)
            if label_similarity >= threshold:
                number_of_golden_predictions += 1

            total_instances += 1

    avg_hits_1 = total_hits_1 / total_instances
    avg_mrr = total_mrr / total_instances
    precision = precision_score(total_labels, total_predictions, average="macro")
    recall = recall_score(total_labels, total_predictions, average="macro")
    f1 = f1_score(total_labels, total_predictions, average="macro")
    precision_micro = precision_score(total_labels, total_predictions, average="micro")
    recall_micro = recall_score(total_labels, total_predictions, average="micro")
    f1_micro = f1_score(total_labels, total_predictions, average="micro")
    accuracy = accuracy_score(total_labels, total_predictions)
    golden_prediction_accuracy = number_of_golden_predictions / total_instances

    return {
        "Hits@1": avg_hits_1,
        "MRR": avg_mrr,
        "Macro Precision": precision,
        "Macro Recall": recall,
        "Macro F1": f1,
        "Micro Precision": precision_micro,
        "Micro Recall": recall_micro,
        "Micro F1": f1_micro,
        "Accuracy": accuracy,
        "Golden Accuracy": golden_prediction_accuracy,
    }


def evaluate_matrix(similarity_matrix, threshold=0.5):
    total_hits_1 = 0
    total_mrr = 0
    total_instances = 0
    total_labels = []
    total_predictions = []
    number_of_golden_predictions = 0

    for i in tqdm(range(len(similarity_matrix))):
        predicted_idx = np.argmax(similarity_matrix[i])
        label_similarity = similarity_matrix[i][i]

        if predicted_idx == i:
            total_hits_1 += 1

        label_rank = sum(1 for x in similarity_matrix[i] if x > similarity_matrix[i][i])
        reciprocal_rank = 1 / (label_rank + 1)
        total_mrr += reciprocal_rank

        predictions = [0 if sim < threshold else 1 for sim in similarity_matrix[i]]
        total_labels.extend([0 if k != i else 1 for k in range(len(similarity_matrix[i]))])
        total_predictions.extend(predictions)
        if label_similarity >= threshold:
            number_of_golden_predictions += 1

        total_instances += 1

    avg_hits_1 = total_hits_1 / total_instances
    avg_mrr = total_mrr / total_instances
    precision = precision_score(total_labels, total_predictions, average="macro")
    recall = recall_score(total_labels, total_predictions, average="macro")
    f1 = f1_score(total_labels, total_predictions, average="macro")
    precision_micro = precision_score(total_labels, total_predictions, average="micro")
    recall_micro = recall_score(total_labels, total_predictions, average="micro")
    f1_micro = f1_score(total_labels, total_predictions, average="micro")
    accuracy = accuracy_score(total_labels, total_predictions)
    golden_prediction_accuracy = number_of_golden_predictions / total_instances

    return {
        "Hits@1": avg_hits_1,
        "MRR": avg_mrr,
        "Macro Precision": precision,
        "Macro Recall": recall,
        "Macro F1": f1,
        "Micro Precision": precision_micro,
        "Micro Recall": recall_micro,
        "Micro F1": f1_micro,
        "Accuracy": accuracy,
        "Golden Accuracy": golden_prediction_accuracy,
    }


def evaluate_matrix_by_source(similarity_matrix, sources, threshold=0.5):
    def _evaluate(indices):
        total_hits_1 = 0
        total_mrr = 0
        total_instances = 0
        total_labels = []
        total_predictions = []
        number_of_golden_predictions = 0
        total_rank = 0

        for i in tqdm(indices):
            predicted_idx = np.argmax(similarity_matrix[i])
            label_similarity = similarity_matrix[i][i]

            if predicted_idx != i and similarity_matrix[i][predicted_idx] == label_similarity:
                continue

            if predicted_idx == i:
                total_hits_1 += 1

            total_instances += 1
            label_rank = sum(1 for x in similarity_matrix[i] if x > similarity_matrix[i][i])
            reciprocal_rank = 1 / (label_rank + 1)
            total_mrr += reciprocal_rank
            total_rank += label_rank + 1

            predictions = [0 if sim < threshold else 1 for sim in similarity_matrix[i]]
            total_labels.extend([0 if k != i else 1 for k in range(len(similarity_matrix[i]))])
            total_predictions.extend(predictions)
            if label_similarity >= threshold:
                number_of_golden_predictions += 1

        avg_hits_1 = total_hits_1 / total_instances
        avg_mrr = total_mrr / total_instances
        avg_rank = total_rank / total_instances
        f1 = f1_score(total_labels, total_predictions, average="macro")
        golden_prediction_accuracy = number_of_golden_predictions / total_instances

        return {
            "Hits@1": avg_hits_1,
            "MRR": avg_mrr,
            "meanR": avg_rank,
            "Macro F1": f1,
            "Golden Accuracy": golden_prediction_accuracy,
        }

    source_metrics = defaultdict(dict)
    for source in set(sources):
        indices = [i for i, s in enumerate(sources) if s == source]
        source_metrics[source] = _evaluate(indices)
    return source_metrics



def evaluate_model_on_paragraph_groups(
    model,
    test_data: dict,
    device: torch.device,
    *,
    audio_key: str = "hubert-emb",
    text_key: str = "text",
    image_key: str = "image",
    paragraph_key: str = "paragraph_id",
    ks=(1, 5, 10, 50),
    batch_size: int = 64,
):
    """Paragraph-level retrieval (max-sim across chunks) for chunked PKLs.

    Each row of ``test_data`` is one *chunk*; rows sharing ``paragraph_id`` form
    a single paragraph candidate. Query = LaBSE text embedding (one per
    paragraph). Candidate score = max cosine over fused chunk embeddings of
    that paragraph (ColBERT/SPIRAL-style).

    Returns a dict with Hits@K, MRR, MAP, mean_rank, median_rank.
    """
    model.eval()
    audio_all = [x.contiguous().float() for x in test_data[audio_key]]
    image_all = [x.contiguous().float() for x in test_data[image_key]]
    text_all = [x.contiguous().float() for x in test_data[text_key]]
    paragraph_ids = list(test_data[paragraph_key])
    n = len(audio_all)
    if n == 0:
        return {}

    # 1. Run model over every chunk -> fused embeddings (n, D)
    fused_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in tqdm(range(0, n, batch_size), desc="fuse chunks"):
            end = min(start + batch_size, n)
            a = torch.stack(audio_all[start:end]).to(device)
            im = torch.stack(image_all[start:end]).to(device)
            f = model(a, im).detach().cpu().float()
            for j in range(f.size(0)):
                fused_chunks.append(f[j])

    # 2. Group chunks by paragraph_id (preserving first-seen order)
    pid_order: list[str] = []
    pid_to_rows: dict[str, list[int]] = defaultdict(list)
    for idx, pid in enumerate(paragraph_ids):
        if pid not in pid_to_rows:
            pid_order.append(pid)
        pid_to_rows[pid].append(idx)

    n_para = len(pid_order)
    c_max = max(len(rows) for rows in pid_to_rows.values())
    d = fused_chunks[0].shape[-1]

    audio_padded = torch.zeros(n_para, c_max, d, dtype=torch.float32)
    audio_mask = torch.zeros(n_para, c_max, dtype=torch.bool)
    text_per_paragraph = torch.zeros(n_para, text_all[0].shape[-1], dtype=torch.float32)
    for j, pid in enumerate(pid_order):
        rows = pid_to_rows[pid]
        for k, idx in enumerate(rows):
            audio_padded[j, k] = fused_chunks[idx]
            audio_mask[j, k] = True
        text_per_paragraph[j] = text_all[rows[0]]  # all chunks share the same text

    # 3. max-sim similarity (n_para, n_para)
    NEG_INF = -1e4
    t_n = F.normalize(text_per_paragraph, p=2, dim=1, eps=1e-8)
    a_n = F.normalize(audio_padded, p=2, dim=2, eps=1e-8)
    s = torch.einsum("id,jmd->ijm", t_n, a_n)
    s = s.masked_fill((~audio_mask).unsqueeze(0), NEG_INF)
    sim_matrix = s.max(dim=2).values.numpy()

    # 4. Ranking metrics on diagonal
    ks_clean = sorted({int(k) for k in ks if k > 0})
    rows_list = similarity_matrix_to_rows(sim_matrix)
    metrics, _ranks = compute_ranking_metrics(rows_list, ks=ks_clean)
    metrics["n_paragraphs"] = float(n_para)
    metrics["n_chunks"] = float(n)
    metrics["max_chunks_per_paragraph"] = float(c_max)
    return metrics
