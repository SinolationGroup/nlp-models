import torch
from service_streamer import ThreadedStreamer
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class OHLProductionModel(torch.nn.Module):
    def __init__(
        self,
        n_classes,
        model_name="distilbert-base-uncased",
        device="cpu",
        batch_size=1,
        max_latency=0.1,
    ):
        super().__init__()

        self.n_classes = n_classes
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_latency = max_latency

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.n_classes
        ).to(self.device)
        self.softmax = torch.nn.Softmax()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.streamer = ThreadedStreamer(
            self.predict_batch, self.batch_size, self.max_latency
        )

    def forward(self, x):
        x = self.model(**x)
        x = self.softmax(x.logits)
        return x

    def preprocess_input(self, text: str):
        text = self.tokenizer(text, padding="max_length", truncation=True)
        data = {}
        data["input_ids"] = torch.IntTensor(text["input_ids"])
        data["attention_mask"] = torch.FloatTensor(text["attention_mask"])
        return data

    def predict_batch(self, text_batch):
        """
        Predict function for service streamer
        """
        for idx, image in enumerate(text_batch):
            text_batch[idx] = self.preprocessing_fn(image)
        with torch.no_grad():
            preds = self(torch.stack(text_batch).to(self.device))
        preds = preds.cpu().detach().numpy()
        return preds

    def warmup(self):
        input = self.preprocess_input("warmup text")
        input = {k: v.unsqueeze(0).to(self.device) for k, v in input.items()}
        with torch.no_grad():
            self(input)
