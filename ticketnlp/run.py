import argparse
import logging
import platform
import sys
from pathlib import Path

import torch

from . import baseline
from .common import read_json, save_json, seed_all
from .data import assert_no_leakage, prepare
from .evaluation import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=["prepare", "baseline", "transformer", "final"]
    )
    parser.add_argument("--config", default="configs/smoke.json")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    config = read_json(args.config)
    seed_all(config["seed"])
    mode = config["mode"]
    reports, artifacts = Path("reports") / mode, Path("artifacts") / mode
    reports.mkdir(parents=True, exist_ok=True)
    logging.getLogger().addHandler(
        logging.FileHandler(reports / "run.log", encoding="utf-8")
    )
    if args.stage == "prepare":
        prepare(config)
        save_json(reports / "config.json", config)
        save_json(
            reports / "hardware.json",
            {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda": torch.cuda.is_available(),
                "gpu": torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None,
            },
        )
        return
    if read_json(reports / "config.json") != config:
        raise ValueError(
            "Config changed since prepare. Prepare again in a fresh mode directory."
        )
    partitions = {
        s: read_json(Path("data") / mode / f"{s}.json")
        for s in ["train", "validation", "test"]
    }
    labels = read_json(Path("data") / mode / "labels.json")
    assert_no_leakage(partitions)
    artifacts.mkdir(parents=True, exist_ok=True)
    save_json(artifacts / "labels.json", labels)
    save_json(artifacts / "config.json", config)
    if args.stage in ("baseline", "transformer"):
        if args.stage == "baseline":
            _, runs = baseline.train(partitions, labels, config, artifacts, reports)
        else:
            from .transformer import train

            runs = train(partitions, labels, config, artifacts, reports)
        save_json(
            reports / f"{args.stage}_experiments.json", {"config": config, "runs": runs}
        )
        return
    # Final stage is deliberately separate; refuses accidental test reruns.
    if (reports / "final_comparison.json").exists():
        raise FileExistsError(
            "Test already evaluated. Preserve the original result; do not tune on test."
        )
    import joblib
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from .transformer import predict

    model = joblib.load(artifacts / "baseline.joblib")
    transformer = AutoModelForSequenceClassification.from_pretrained(
        artifacts / "distilbert"
    )
    tokenizer = AutoTokenizer.from_pretrained(artifacts / "distilbert")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer.to(device)
    test = partitions["test"]
    base_metrics = evaluate(
        test,
        baseline.probabilities(model, [r["text"] for r in test], labels),
        labels,
        reports / "baseline_final_test",
    )
    bert_metrics = evaluate(
        test,
        predict(
            transformer,
            tokenizer,
            test,
            labels,
            config,
            device,
            8 if device.type == "cuda" else 4,
        ),
        labels,
        reports / "distilbert_final_test",
    )
    save_json(
        reports / "final_comparison.json",
        {
            "mode": mode,
            "split": "final_test",
            "baseline": base_metrics,
            "distilbert": bert_metrics,
            "distilbert_macro_f1_delta": bert_metrics["macro_f1"]
            - base_metrics["macro_f1"],
        },
    )


if __name__ == "__main__":
    main()
