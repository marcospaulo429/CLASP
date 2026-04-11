import random

import torch
from torch.utils.data import Dataset


class CusDataset(Dataset):
    def __init__(self, dataset, audio_name, text_name):
        self.dataset = dataset
        self.text_name = text_name
        self.audio_name = audio_name

    def __len__(self):
        return len(self.dataset[self.audio_name])

    def __getitem__(self, i):
        return (
            self.dataset[self.text_name][i],
            self.dataset[self.audio_name][i],
            self.dataset["image"][i],
        )


def build_test_metadata(test_len_data, number_of_candidates_per_sample=100):
    test_metadata = []
    for index in range(test_len_data):
        candidate_indexes = random.sample(
            [i for i in range(test_len_data) if i != index],
            number_of_candidates_per_sample - 1,
        )
        candidate_indexes += [index]
        test_metadata.append(candidate_indexes)
    return test_metadata


class TestDataset(Dataset):
    def __init__(self, test_dataset, metadata, audio_name, text_name):
        self.data = test_dataset
        self.metadata = metadata
        self.audio_name = audio_name
        self.text_name = text_name

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        candidate_indexes = self.metadata[index]
        text_embedding = self.data[self.text_name][index]
        audio_embeddings = [self.data[self.audio_name][i] for i in candidate_indexes]
        image_embeddings = [self.data["image"][i] for i in candidate_indexes]
        label_index = len(candidate_indexes) - 1
        audio_embeddings = torch.stack(audio_embeddings)
        image_embeddings = torch.stack(image_embeddings)

        return text_embedding, audio_embeddings, image_embeddings, label_index

