"""Explicit PyTorch loop: easy to inspect, no Trainer/accelerate required."""

import logging
import math
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .common import clean_text, save_json, seed_all
from .evaluation import evaluate

MODEL = "distilbert/distilbert-base-uncased"


def loader(rows, tokenizer, labels, config, batch_size, shuffle=False):
    ids = {label: i for i, label in enumerate(labels)}

    def collate(batch):
        encoded = tokenizer(
            [clean_text(r["text"]) for r in batch],
            padding=True,
            truncation=True,
            max_length=config["max_length"],
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor([ids[r["label"]] for r in batch])
        return encoded

    return DataLoader(
        rows, batch_size=batch_size, shuffle=shuffle, collate_fn=collate, num_workers=0
    )


def predict(model, tokenizer, rows, labels, config, device, batch_size):
    model.eval()
    results = []
    with torch.inference_mode():
        for batch in loader(rows, tokenizer, labels, config, batch_size):
            batch.pop("labels")
            logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
            results.append(logits.softmax(-1).cpu().numpy())
    return np.concatenate(results)


def train(partitions, labels, config, artifacts, reports):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    batch_size = config[
        "batch_size_cuda" if device.type == "cuda" else "batch_size_cpu"
    ]
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    # Record immutable HF revision resolved by tokenizer/model, including source checkpoint.
    revision = tokenizer.init_kwargs.get("_commit_hash")
    best_score, runs = -1, []
    artifacts.mkdir(parents=True, exist_ok=True)
    for lr in config["learning_rates"]:
        seed_all(config["seed"])
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL,
            revision=revision,
            num_labels=len(labels),
            id2label=dict(enumerate(labels)),
            label2id={l: i for i, l in enumerate(labels)},
        )
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        batches = loader(
            partitions["train"], tokenizer, labels, config, batch_size, shuffle=True
        )
        accumulation = config["gradient_accumulation"]
        max_steps = config.get(
            "max_steps", math.ceil(len(batches) / accumulation) * config["epochs"]
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step, total=max_steps: min(
                (step + 1) / max(1, int(0.1 * total)), max(0.0, 1 - step / total)
            ),
        )
        steps = 0
        started = time.monotonic()
        for epoch in range(config["epochs"]):
            model.train()
            optimizer.zero_grad()
            losses = []
            for index, batch in enumerate(batches):
                loss = model(**{k: v.to(device) for k, v in batch.items()}).loss
                losses.append(float(loss.detach()))
                # Normalize a final partial accumulation window correctly.
                window = min(
                    accumulation, len(batches) - (index // accumulation) * accumulation
                )
                (loss / window).backward()
                if (index + 1) % accumulation == 0 or index + 1 == len(batches):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    steps += 1
                    if steps >= max_steps:
                        break
            metrics = evaluate(
                partitions["validation"],
                predict(
                    model,
                    tokenizer,
                    partitions["validation"],
                    labels,
                    config,
                    device,
                    batch_size,
                ),
                labels,
                reports / f"distilbert_lr{lr}_epoch{epoch + 1}" / "validation",
            )
            run = {
                "model": MODEL,
                "revision": getattr(model.config, "_commit_hash", revision),
                "seed": config["seed"],
                "learning_rate": lr,
                "epoch": epoch + 1,
                "optimizer_steps": steps,
                "mean_train_loss": float(np.mean(losses)),
                "elapsed_seconds": time.monotonic() - started,
                "device": str(device),
                "batch_size": batch_size,
                "split": "validation",
                "mode": config["mode"],
                **metrics,
            }
            runs.append(run)
            save_json(
                reports / "transformer_experiments.json",
                {"config": config, "runs": runs},
            )
            logging.getLogger(__name__).info("DistilBERT validation: %s", run)
            if metrics["macro_f1"] > best_score:
                best_score = metrics["macro_f1"]
                model.save_pretrained(artifacts / "distilbert")
                tokenizer.save_pretrained(artifacts / "distilbert")
            if steps >= max_steps:
                break
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return runs
