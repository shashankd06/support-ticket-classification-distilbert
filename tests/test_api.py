import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from ticketnlp.api import create_app


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
