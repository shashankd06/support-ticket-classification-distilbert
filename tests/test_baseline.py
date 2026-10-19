import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ticketnlp.baseline import probabilities
from ticketnlp.common import clean_text


def test_train_only_vocabulary_preprocessing_and_label_order():
    model = Pipeline(
        [("tfidf", TfidfVectorizer()), ("classifier", LogisticRegression())]
    )
    model.fit(
        [
            clean_text(t)
            for t in ["card lost", "card missing", "transfer slow", "transfer pending"]
        ],
        ["lost", "lost", "pending", "pending"],
    )
    labels = ["pending", "lost"]  # Deliberately reverse sklearn class ordering.
    p = probabilities(model, ["  card\n lost  ", "unseen_validation_secret"], labels)
    assert "unseen_validation_secret" not in model.named_steps["tfidf"].vocabulary_
    assert np.allclose(p.sum(axis=1), 1)
    assert labels[int(p[0].argmax())] == "lost"
    assert np.allclose(p[0], probabilities(model, ["card lost"], labels)[0])
