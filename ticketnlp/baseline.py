import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .common import clean_text
from .evaluation import evaluate


def probabilities(model, texts, labels):
    scores = model.predict_proba([clean_text(t) for t in texts])
    order = [list(model.classes_).index(label) for label in labels]
    return scores[:, order]


def train(partitions, labels, config, artifacts, reports):
    train_rows = partitions["train"]
    validation = partitions["validation"]
    best_score, best_model, runs = -1, None, []
    for c in config["baseline_C"]:
        # C tests regularization strength; same partitions and feature settings.
        model = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=1,
                        sublinear_tf=True,
                        max_features=50000,
                        strip_accents="unicode",
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(C=c, max_iter=1000, random_state=config["seed"]),
                ),
            ]
        )
        model.fit(
            [clean_text(r["text"]) for r in train_rows],
            [r["label"] for r in train_rows],
        )
        metrics = evaluate(
            validation,
            probabilities(model, [r["text"] for r in validation], labels),
            labels,
            reports / f"baseline_C{c}" / "validation",
        )
        runs.append(
            {
                "model": "tfidf_logistic",
                "C": c,
                "seed": config["seed"],
                "split": "validation",
                "mode": config["mode"],
                **metrics,
            }
        )
        if metrics["macro_f1"] > best_score:
            best_score, best_model = metrics["macro_f1"], model
    artifacts.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, artifacts / "baseline.joblib")
    return best_model, runs
