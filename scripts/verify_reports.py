"""Independently check saved metric arithmetic; never rerun model evaluation."""

import math
from pathlib import Path

from ticketnlp.common import read_json, save_json


def close(actual, expected):
    assert math.isclose(actual, expected, abs_tol=1e-10), (actual, expected)


def main():
    root = Path("reports/full")
    checked = []
    for path in sorted(root.rglob("metrics.json")):
        metrics = read_json(path)
        confusion = read_json(path.parent / "confusion_matrix.json")
        labels, matrix = confusion["labels"], confusion["matrix"]
        assert len(labels) == len(set(labels)) == len(matrix) == 77
        assert all(len(row) == 77 for row in matrix)
        assert all(isinstance(x, int) and x >= 0 for row in matrix for x in row)
        n = sum(map(sum, matrix))
        assert n == metrics["n"]
        close(metrics["accuracy"], sum(matrix[i][i] for i in range(77)) / n)
        f1_values = []
        for i, label in enumerate(labels):
            support = sum(matrix[i])
            predicted = sum(row[i] for row in matrix)
            precision = matrix[i][i] / predicted if predicted else 0
            recall = matrix[i][i] / support if support else 0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0
            )
            saved = metrics["per_class"][label]
            close(saved["precision"], precision)
            close(saved["recall"], recall)
            close(saved["f1-score"], f1)
            close(saved["support"], support)
            f1_values.append(f1)
        close(metrics["macro_f1"], sum(f1_values) / 77)
        checked.append(path.as_posix())
    assert (
        len(checked) == 10
    )  # 2 baseline validations, 6 Transformer validations, 2 tests.
    final = read_json(root / "final_comparison.json")
    for name in ("baseline", "distilbert"):
        saved = read_json(root / f"{name}_final_test/metrics.json")
        for key in ("accuracy", "macro_f1", "n"):
            close(final[name][key], saved[key])
    close(
        final["distilbert_macro_f1_delta"],
        final["distilbert"]["macro_f1"] - final["baseline"]["macro_f1"],
    )
    save_json(
        "reports/verification/report_integrity.json",
        {
            "passed": True,
            "scope": "Saved metric arithmetic recomputed from confusion matrices; no training or test inference rerun",
            "metric_reports_checked": len(checked),
            "classes_per_report": 77,
            "checks": [
                "accuracy",
                "per-class precision/recall/F1/support",
                "macro-F1",
                "final comparison",
            ],
            "files": checked,
        },
    )
    print(
        f"Verified {len(checked)} metric reports, 77 classes each, and final comparison"
    )


if __name__ == "__main__":
    main()
