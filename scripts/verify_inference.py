"""Verify real local checkpoints through a temporary loopback HTTP server.

Run from the project root: python -m scripts.verify_inference
No weights are downloaded and no model-quality evaluation is performed.
"""

import hashlib
import importlib.metadata
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from ticketnlp.common import read_json, save_json


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_backend(backend, directory, destination):
    labels = read_json(directory / "labels.json")
    checkpoint = directory / (
        "distilbert/model.safetensors" if backend == "distilbert" else "baseline.joblib"
    )
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    environment = os.environ.copy()
    environment.update(
        MODEL_DIR=str(directory.resolve()),
        MODEL_BACKEND=backend,
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    base = f"http://127.0.0.1:{port}"
    result = {
        "backend": backend,
        "scope": "Local HTTP inference using trained full-mode artifacts; not deployment or a quality benchmark",
        "checkpoint_sha256": file_hash(checkpoint),
        "checks": [],
        "passed": False,
    }

    def request(path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(
            base + path,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with opener.open(req, timeout=15) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as error:
            return error.code, json.load(error)

    with (destination / f"{backend}_server.log").open("w", encoding="utf-8") as log:
        server = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "ticketnlp.api:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--no-access-log",
            ],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            started = time.monotonic()
            while time.monotonic() - started < 90:
                if server.poll() is not None:
                    raise RuntimeError(
                        "Inference server exited during startup; inspect log"
                    )
                try:
                    status, body = request("/health")
                    if status == 200 and body == {"status": "ready"}:
                        break
                except (OSError, urllib.error.URLError):
                    pass
                time.sleep(0.3)
            else:
                raise TimeoutError(
                    "Inference server did not become ready within 90 seconds"
                )
            result["startup_seconds"] = time.monotonic() - started
            result["checks"].append("health_after_real_checkpoint_startup")
            status, spec = request("/openapi.json")
            assert status == 200 and "/predict" in spec["paths"]
            result["checks"].append("openapi_route_available")
            example = "My card has not arrived yet."
            status, prediction = request("/predict", {"text": example})
            assert status == 200
            assert prediction["label"] in labels
            assert 0 <= prediction["score"] <= 1
            assert prediction["model"] == backend and prediction["mode"] == "full"
            assert prediction["truncated"] is False
            result["example"] = {"text": example, "response": prediction}
            result["checks"].append("real_prediction_schema_label_score_and_mode")
            status, repeat = request("/predict", {"text": example})
            assert status == 200 and repeat["label"] == prediction["label"]
            assert abs(repeat["score"] - prediction["score"]) < 1e-5
            result["checks"].append("repeat_request_stable")
            status, spaced = request(
                "/predict", {"text": "  My card\n has not arrived yet.  "}
            )
            assert status == 200 and spaced["label"] == prediction["label"]
            assert abs(spaced["score"] - prediction["score"]) < 1e-5
            result["checks"].append("whitespace_preprocessing_consistent")
            for name, payload in [
                ("empty", {"text": ""}),
                ("whitespace_only", {"text": " \n "}),
                ("missing", {}),
                ("wrong_type", {"text": 123}),
                ("over_character_limit", {"text": "x" * 4001}),
            ]:
                status, _ = request("/predict", payload)
                assert status == 422, (name, status)
                result["checks"].append(f"reject_{name}_422")
            status, long_prediction = request("/predict", {"text": "card " * 800})
            assert status == 200
            assert long_prediction["truncated"] is (backend == "distilbert")
            result["checks"].append("accepted_4000_chars_and_expected_token_truncation")
            result["long_input_response"] = long_prediction
            result["passed"] = True
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)
            save_json(destination / f"{backend}.json", result)
    return result


def main():
    directory = Path("artifacts/full")
    destination = Path("reports/inference_verification")
    destination.mkdir(parents=True, exist_ok=True)
    results = []
    for backend in ("baseline", "distilbert"):
        print(f"Verifying {backend} with local trained artifacts", flush=True)
        result = verify_backend(backend, directory, destination)
        results.append(result)
        print(f"{backend}: {len(result['checks'])} checks passed", flush=True)
    save_json(
        destination / "summary.json",
        {
            "all_passed": all(r["passed"] for r in results),
            "python": sys.version.split()[0],
            "packages": {
                name: importlib.metadata.version(name)
                for name in (
                    "torch",
                    "transformers",
                    "scikit-learn",
                    "fastapi",
                    "starlette",
                    "httpx",
                    "uvicorn",
                )
            },
            "backends": [
                {"name": r["backend"], "checks_passed": len(r["checks"])}
                for r in results
            ],
            "deployment": "Local loopback only; no public or production deployment performed",
        },
    )


if __name__ == "__main__":
    main()
