"""Load trusted local artifacts only; joblib is not safe for untrusted files."""

from pathlib import Path

from .common import clean_text, read_json


class Predictor:
    def __init__(self, directory, backend="distilbert"):
        self.directory = Path(directory)
        self.labels = read_json(self.directory / "labels.json")
        self.config = read_json(self.directory / "config.json")
        self.backend = backend
        if backend == "baseline":
            import joblib

            self.model = joblib.load(self.directory / "baseline.joblib")
        elif backend == "distilbert":
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.directory / "distilbert", local_files_only=True
            )
            self.model = (
                AutoModelForSequenceClassification.from_pretrained(
                    self.directory / "distilbert", local_files_only=True
                )
                .to(self.device)
                .eval()
            )
            if [
                self.model.config.id2label[i] for i in range(len(self.labels))
            ] != self.labels:
                raise ValueError("Label mapping mismatch")
        else:
            raise ValueError("Unsupported backend")

    def __call__(self, text):
        text = clean_text(text)
        truncated = False
        if self.backend == "baseline":
            from .baseline import probabilities

            scores = probabilities(self.model, [text], self.labels)[0]
        else:
            import torch

            tokens = self.tokenizer(text, truncation=False, add_special_tokens=True)[
                "input_ids"
            ]
            truncated = len(tokens) > self.config["max_length"]
            inputs = self.tokenizer(
                text,
                truncation=True,
                max_length=self.config["max_length"],
                return_tensors="pt",
            )
            with torch.inference_mode():
                scores = (
                    self.model(**{k: v.to(self.device) for k, v in inputs.items()})
                    .logits.softmax(-1)[0]
                    .cpu()
                    .numpy()
                )
        index = int(scores.argmax())
        return {
            "label": self.labels[index],
            "score": float(scores[index]),
            "model": self.backend,
            "mode": self.config["mode"],
            "truncated": truncated,
        }
