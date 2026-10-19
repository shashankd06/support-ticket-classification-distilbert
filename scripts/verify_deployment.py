"""Check the public demo without rerunning model-quality evaluation."""

import argparse
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    args = parser.parse_args()
    checks = []

    def request(path, payload=None, expected=200):
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(args.url.rstrip("/") + path, data=data,
                                     headers={"Content-Type": "application/json"})
        started = time.monotonic()
        try:
            response = urllib.request.urlopen(req, timeout=120)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            body = response.read().decode()
            status = response.code
        checks.append({"path": path, "method": "GET" if data is None else "POST",
                       "status": status, "elapsed_seconds": round(time.monotonic() - started, 3),
                       "passed": status == expected})
        assert status == expected, (path, status, body[:200])
        return body

    assert "IntentDesk" in request("/")
    assert json.loads(request("/health"))["status"] == "ready"
    assert "macro_f1" in request("/evaluation")
    assert "swagger-ui" in request("/docs")
    predictions = []
    for text, label in [("My card has not arrived yet.", "card_arrival"),
                        ("I need to cancel a bank transfer.", "cancel_transfer")]:
        result = json.loads(request("/predict", {"text": text}))
        assert result["label"] == label and result["model"] == "distilbert"
        assert result["mode"] == "full" and 0 <= result["score"] <= 1
        predictions.append({"text": text, "response": result})
    request("/predict", {"text": "   "}, 422)
    request("/predict", {"text": "a" * 4001}, 422)
    long_result = json.loads(request("/predict", {"text": "Where is my card? " * 100}))
    assert long_result["truncated"] is True
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "url": args.url, "scope": "Sequential public demo checks; not load testing or model-quality evaluation",
              "checks": checks, "predictions": predictions, "long_input_response": long_result,
              "passed": True}
    destination = Path("reports/deployment/verification.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Passed {len(checks)} public HTTP checks; saved {destination}")


if __name__ == "__main__":
    main()
