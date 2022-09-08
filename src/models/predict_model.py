from typing import List, Text

import torch

# from service_streamer import ThreadedStreamer
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class OHLProductionModel(torch.nn.Module):
    def __init__(
        self,
        n_classes,
        model_name="distilbert-base-uncased",
        device="cpu",
        # batch_size=1,
        # max_latency=0.1,
    ):
        super().__init__()

        self.n_classes = n_classes
        self.model_name = model_name
        self.device = device
        # self.batch_size = batch_size
        # self.max_latency = max_latency

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.n_classes
        ).to(self.device)
        self.softmax = torch.nn.Softmax(dim=1)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # self.streamer = ThreadedStreamer(
        #     self.predict_batch, self.batch_size, self.max_latency
        # )

    def forward(self, x):
        x = self.model(**x)
        x = self.softmax(x.logits)
        return x

    def preprocess_input(self, text_list: List[Text]):
        data = self.tokenizer(
            text_list, padding="longest", truncation=True, return_tensors="pt"
        )
        data = {k: v.to(self.device) for k, v in data.items()}
        return data

    # def predict_batch(self, text_batch):
    #     """
    #     Predict function for service streamer
    #     """
    #     text_batch = self.preprocess_input(text_batch)
    #     with torch.no_grad():
    #         preds = self(text_batch)
    #     preds = preds.cpu().detach().numpy()
    #     return preds

    def warmup(self):
        input = self.preprocess_input("warmup text")
        with torch.no_grad():
            self(input)
