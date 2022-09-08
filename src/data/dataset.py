from typing import List

import torch
from torch.utils.data import Dataset


class OHLDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        n_classes: int = 296,
        max_length: int = 512,
        augmentation=None,
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

        if self.augmentation:
            text = self.augmentation(text)

        text = self.tokenizer(
            text, padding="max_length", truncation=True, max_length=self.max_length
        )
        data = {}
        data["input_ids"] = torch.IntTensor(text["input_ids"])
        data["attention_mask"] = torch.FloatTensor(text["attention_mask"])
        data["labels"] = torch.LongTensor([label])

        return data
