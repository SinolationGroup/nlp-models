from typing import Dict, List, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.optim import AdamW
from torchmetrics.classification import (
    Accuracy,
    ConfusionMatrix,
    F1Score,
    Precision,
    Recall,
)
from transformers import AutoModelForSequenceClassification


class OHLModel(LightningModule):
    def __init__(
        self,
        n_classes,
        class_names,
        model_name="distilbert-base-uncased",
        lr=0.001,
        class_weights=None,
    ):
        super().__init__()

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=n_classes
        )

        self.save_hyperparameters()

        self.softmax = nn.Softmax(dim=1)
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(num_classes=self.hparams.n_classes)
        self.accuracy_top5 = Accuracy(num_classes=self.hparams.n_classes, top_k=5)
        self.f1 = F1Score(num_classes=self.hparams.n_classes, average="macro")
        self.f1_top5 = F1Score(
            num_classes=self.hparams.n_classes, average="macro", top_k=5
        )
        self.metrics = [
            ("accuracy", self.accuracy),
            ("accuracy_top5", self.accuracy_top5),
            ("f1", self.f1),
            ("f1_top5", self.f1_top5),
        ]

        self.val_precision = Precision(num_classes=self.hparams.n_classes, average=None)
        self.val_recall = Recall(num_classes=self.hparams.n_classes, average=None)
        self.val_f1 = F1Score(num_classes=self.hparams.n_classes, average="macro")
        self.val_f1_top5 = F1Score(
            num_classes=self.hparams.n_classes, average="macro", top_k=5
        )
        self.val_accuracy = Accuracy(num_classes=self.hparams.n_classes)
        self.val_accuracy_top5 = Accuracy(num_classes=self.hparams.n_classes, top_k=5)
        # self.val_cm = ConfusionMatrix(num_classes=self.hparams.n_classes)
        self.val_metrics = [
            # ("precision", self.val_precision),
            # ("recall", self.val_recall),
            ("f1", self.val_f1),
            ("f1_top5", self.val_f1_top5),
            ("accuracy", self.val_accuracy),
            ("accuracy_top5", self.val_accuracy_top5),
        ]

    def forward(self, x):
        x = self.model(x)
        # x = self.softmax(x.logits)
        return x

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)

        # loss = self.loss_fn(preds, y)
        loss = outputs.loss

        log_values = {}
        y = batch["labels"].squeeze(-1)
        preds = outputs.logits
        for metric_name, metric_fn in self.metrics:
            log_values["train_{}".format(metric_name)] = metric_fn(preds, y)

        log_values["train_loss"] = loss
        self.log_dict(
            log_values, prog_bar=True, logger=True, on_epoch=True, on_step=True
        )

        return loss

    def validation_step(self, batch, batch_idx):

        outputs = self.model(**batch)

        # loss = self.loss_fn(preds, y)
        loss = outputs.loss

        log_values = {}
        y = batch["labels"].squeeze(-1)
        preds = outputs.logits
        for metric_name, metric_fn in self.val_metrics:
            log_values["val_{}".format(metric_name)] = metric_fn(preds, y)
        log_values["val_loss"] = loss
        self.log_dict(
            log_values, prog_bar=True, logger=True, on_epoch=True, on_step=False
        )

        return {"loss": loss, "preds": preds, "target": y}

    # def train_epoch_end(self, outputs):
    #     print("train", self.current_epoch)

    def validation_epoch_end(self, outputs):
        pass
        # print("val", self.current_epoch)
        # preds = torch.cat([tmp["preds"] for tmp in outputs])
        # targets = torch.cat([tmp["target"] for tmp in outputs])

        # log classification report
        # report = self.get_classification_report(preds, targets)
        # self.logger.log_text(
        #     key="Metrics", dataframe=report, step=self.trainer.global_step
        # )

        # log confusion matrix
        # cm_img = self.get_confusion_matrix(preds, targets)
        # self.logger.log_image(
        #     key="Conf matrix", images=[cm_img], step=self.trainer.global_step
        # )

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer, T_0=10, T_mult=2, eta_min=0.0001, last_epoch=-1
        # )

        lambda1 = lambda epoch: 0.9**epoch
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

        return {
            "optimizer": optimizer,
            # "lr_scheduler": {
            #     "scheduler": lr_scheduler,
            #     "frequency": 1,
            #     "interval": "epoch",
            # },
        }

    def get_progress_bar_dict(self):
        # don't show the version number
        metrics = super().get_progress_bar_dict()
        metrics.pop("v_num", None)
        return metrics

    def get_classification_report(self, preds, targets):
        metric_values = {
            metric_name: f(preds, targets).detach().cpu().numpy()
            for metric_name, f in self.val_metrics
        }
        report = pd.DataFrame(
            metric_values,
            index=self.hparams.class_names,
        )

        return report

    def get_confusion_matrix(self, preds, targets):
        cm_df = pd.DataFrame(
            self.val_cm(preds, targets).detach().cpu().numpy(),
            columns=self.hparams.class_names,
            index=self.hparams.class_names,
        )

        sns.set_theme()
        plt.figure(figsize=(10, 9), tight_layout=True)
        ax = sns.heatmap(
            cm_df,
            annot=True,
            cmap="Pastel1",
            linewidths=0.5,
            mask=~cm_df.values.astype(bool),
            linecolor="white",
        )
        plt.xlabel("Predicted", fontsize=15)
        plt.ylabel("Actual", fontsize=15)
        plt.xticks(rotation=60, fontsize=15)
        plt.yticks(rotation="horizontal", fontsize=15)

        fig_ = ax.figure

        fig_.canvas.draw()
        fig_.tight_layout()
        # Now we can save it to a numpy array.
        data = np.frombuffer(fig_.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig_.canvas.get_width_height()[::-1] + (3,))

        return data

    def warmup(self):
        img = torch.rand(1, *self.input_size).to(self.device)
        self(img)
