from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from .common import save_json


def evaluate(rows, probabilities, labels, destination):
    destination = Path(destination)
    y = [r["label"] for r in rows]
    predicted = [labels[i] for i in np.argmax(probabilities, axis=1)]
    metrics = {
        "accuracy": accuracy_score(y, predicted),
        "macro_f1": f1_score(
            y, predicted, labels=labels, average="macro", zero_division=0
        ),
        "n": len(rows),
        "per_class": classification_report(
            y, predicted, labels=labels, output_dict=True, zero_division=0
        ),
    }
    save_json(destination / "metrics.json", metrics)
    cm = confusion_matrix(y, predicted, labels=labels)
    save_json(
        destination / "confusion_matrix.json", {"labels": labels, "matrix": cm.tolist()}
    )
    fig, ax = plt.subplots(figsize=(9, 8) if len(labels) < 15 else (15, 13))
    image = ax.imshow(cm, cmap="Blues")
    ax.set(
        xlabel="Predicted class ID (labels.json)",
        ylabel="True class ID (labels.json)",
        title=f"{destination}: confusion matrix",
    )
    ticks = (
        list(range(len(labels))) if len(labels) < 15 else list(range(0, len(labels), 5))
    )
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    if len(labels) < 15:
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(destination / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    errors = [
        {
            "id": r["id"],
            "text": r["text"],
            "true": truth,
            "predicted": pred,
            "score": float(max(probs)),
        }
        for r, truth, pred, probs in zip(rows, y, predicted, probabilities)
        if truth != pred
    ]
    save_json(destination / "errors.json", errors)
    pairs = Counter((e["true"], e["predicted"]) for e in errors)
    save_json(
        destination / "error_summary.json",
        {
            "error_count": len(errors),
            "top_confusions": [
                {"true": a, "predicted": b, "count": n}
                for (a, b), n in pairs.most_common(15)
            ],
            "review_note": "Inspect errors.json manually before claiming linguistic failure patterns.",
        },
    )
    return {k: metrics[k] for k in ("accuracy", "macro_f1", "n")}
