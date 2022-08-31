import os
from typing import List, Optional

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.flow as naf
import numpy as np
import pandas as pd
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from src.config import SEED
from src.data.augmentation import ComposeAug, NLPTransform, OneOfTransfroms
from src.data.dataset import OHLDataset


class OHLDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        max_length: int,
        train_bs: int,
        valid_bs: int,
        test_size: float,
        model_name=None,
        aug=False,
    ):

        super().__init__()

        self.data_path = data_path
        self.max_length = max_length
        self.train_aug = self.get_training_augmentation() if aug else None
        self.valid_aug = self.get_validation_augmentation() if aug else None
        self.model_name = model_name

        if self.train_aug:
            self.hparams["train_aug_params"] = self.train_aug.to_dict()
        if self.valid_aug:
            self.hparams["valid_aug_params"] = self.valid_aug.to_dict()
        self.save_hyperparameters(
            "train_bs", "valid_bs", "max_length", "test_size", "aug", "model_name"
        )

    def setup(self, stage: Optional[str] = None):

        dataset_df = pd.read_csv(self.data_path)
        texts: List[str] = dataset_df["question"].values

        l_encoder = LabelEncoder()
        l_encoder = l_encoder.fit(dataset_df["label"].values)
        labels: List[int] = l_encoder.transform(dataset_df["label"].values)
        self.num_classes = np.unique(labels).shape[0]

        print("Loading data ...")

        # train test split
        train_ids, valid_ids = train_test_split(
            np.arange(len(labels)),
            test_size=self.hparams.test_size,
            random_state=SEED,
            stratify=labels,
        )

        print(train_ids.shape[0], "train questions")
        print(valid_ids.shape[0], "validation questions")

        tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name)

        self.train_ds = OHLDataset(
            [texts[idx] for idx in train_ids],
            [labels[idx] for idx in train_ids],
            self.num_classes,
            self.hparams.max_length,
            augmentation=self.train_aug,
            tokenizer=tokenizer,
        )

        self.valid_ds = OHLDataset(
            [texts[idx] for idx in valid_ids],
            [labels[idx] for idx in valid_ids],
            self.num_classes,
            self.hparams.max_length,
            augmentation=self.valid_aug,
            tokenizer=tokenizer,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.train_bs,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_ds,
            batch_size=self.hparams.valid_bs,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
            num_workers=4,
        )

    def test_dataloader(self):
        return None

    def get_training_augmentation(self):
        transforms = [
            OneOfTransfroms(
                [
                    NLPTransform(nac.KeyboardAug(aug_char_max=1, aug_word_max=3), p=1),
                    NLPTransform(naw.SpellingAug(aug_max=3), p=1),
                ],
                p=0.3,
            ),
        ]
        return ComposeAug(transforms)

    def get_validation_augmentation(self):
        transforms = []
        return ComposeAug(transforms)

    def get_test_augmentation(self):
        transforms = []
        return ComposeAug(transforms)
