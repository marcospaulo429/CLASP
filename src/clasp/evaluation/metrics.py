from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm


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

