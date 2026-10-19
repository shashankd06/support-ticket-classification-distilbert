"""Offline integration check using invented fixtures, never benchmark results."""

import platform
import subprocess
import sys
from pathlib import Path

import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ticketnlp.baseline import probabilities
from ticketnlp.common import save_json
from ticketnlp.evaluation import evaluate


def main():
    reports = Path("reports/local_checks")
    reports.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        capture_output=True,
        text=True,
        check=False,
    )
    (reports / "pytest.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
    # Exercise report writing on invented data without implying dataset training.
    texts = ["card missing", "card lost", "transfer late", "transfer pending"]
    y = ["card_issue", "card_issue", "transfer_issue", "transfer_issue"]
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("classifier", LogisticRegression(random_state=42)),
        ]
    )
    model.fit(texts, y)
    labels = ["card_issue", "transfer_issue"]
    rows = [
        {"id": "invented-1", "text": "card missing", "label": "card_issue"},
        {"id": "invented-2", "text": "pending transfer", "label": "transfer_issue"},
    ]
    evaluate(
        rows,
        probabilities(model, [r["text"] for r in rows], labels),
        labels,
        reports / "invented_fixture_ONLY",
    )
    save_json(
        reports / "status.json",
        {
            "scope": "Offline code checks on invented fixtures only; NOT BANKING77 results",
            "pytest_exit_code": result.returncode,
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda_available": torch.cuda.is_available(),
            "real_dataset_downloaded": False,
            "baseline_dataset_training": "not run",
            "distilbert_finetuning": "not run",
            "test_evaluation": "not run",
        },
    )
    print(result.stdout)
    print("Offline report generation completed; no benchmark metrics measured.")
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
