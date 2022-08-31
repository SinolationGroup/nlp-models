from typing import List

import torch
from PIL import Image
from torch.utils.data import Dataset


class OHLDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        n_classes: int = 5,
        max_length: int = 512,
        augmentation=None,
        tokenizer=None,
    ):

        self.texts = texts
        self.labels = labels
        self.augmentation = augmentation
        self.tokenizer = tokenizer
        self.n_classes = n_classes
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):

        text = self.texts[idx]
        label = self.labels[idx]

        label_ohe = torch.zeros(self.n_classes, dtype=torch.int)
        label_ohe[label] = 1

        if self.augmentation:
            text = self.augmentation(text)

        if self.tokenizer:
            text = self.tokenizer(text, padding="max_length", truncation=True)
            data = {}
            data["input_ids"] = torch.IntTensor(text["input_ids"])
            data["attention_mask"] = torch.FloatTensor(text["attention_mask"])
            data["labels"] = torch.IntTensor(label_ohe).squeeze(0)

        # return text, torch.FloatTensor(label_ohe)
        return data
