import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from ticketnlp.api import create_app
from ticketnlp.common import read_json


def test_validation_and_one_load_for_multiple_requests():
    loads = []

    def factory():
        loads.append(1)
        return lambda text: {
            "label": "card_arrival",
            "score": 0.7,
            "model": "test_double",
            "mode": "test",
            "truncated": False,
        }

    with TestClient(create_app(factory)) as client:
        for payload in [
            {"text": ""},
            {"text": "   "},
            {"text": "x" * 4001},
            {},
            {"text": 123},
        ]:
            assert client.post("/predict", json=payload).status_code == 422
        for _ in range(2):
            response = client.post("/predict", json={"text": "where is my card"})
            assert response.status_code == 200
            assert response.json()["label"] == "card_arrival"
        assert client.get("/health").status_code == 200
    assert len(loads) == 1


def test_frontend_assets_and_evaluation_use_committed_source():
    from ticketnlp.api import EVALUATION

    with TestClient(create_app(lambda: lambda text: None)) as client:
        page = client.get("/")
        assert page.status_code == 200 and "IntentDesk" in page.text
        assert "text/html" in page.headers["content-type"]
        assert client.get("/assets/app.js").status_code == 200
        assert client.get("/assets/style.css").status_code == 200
        assert client.get("/evaluation").json() == read_json(EVALUATION)
        assert client.get("/docs").status_code == 200
        assert client.get("/assets/../common.py").status_code == 404


def test_missing_evaluation_does_not_fabricate_results(monkeypatch, tmp_path):
    monkeypatch.setattr("ticketnlp.api.EVALUATION", tmp_path / "missing.json")
    with TestClient(create_app(lambda: lambda text: None)) as client:
        assert client.get("/evaluation").status_code == 503
