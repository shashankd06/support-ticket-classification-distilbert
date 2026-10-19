# Deploy IntentDesk

The page, `/predict`, `/health`, `/evaluation` and `/docs` are served by one FastAPI application. There is no separate frontend framework, build pipeline or development CORS origin. The original PyTorch training workflow, weights and held-out evaluation are preserved.

## Local preview

From the repository root, activate the training environment and use the existing full artifacts:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn ticketnlp.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/` for IntentDesk and `/docs` for the API schema. `MODEL_BACKEND=baseline` selects the locally trained sparse baseline; the default is DistilBERT. Smoke artifacts are identified explicitly in results if `MODEL_DIR=artifacts/smoke` is selected.

## Container and model storage

The Docker image installs CPU-only PyTorch and minimal serving dependencies, loads one full DistilBERT instance at startup and serves both UI and API. A pinned, checksum-verified archive is downloaded **during image build**, never per request. The archive contains unchanged original FP32 weights, tokenizer/configuration files and attribution/license notices, with no training queries or baseline pickle. Checkpoints are stored as a GitHub Release asset rather than committed into source history. `deployment/model-release.json` records the archive and original checkpoint SHA-256 values.

To package a new release from already trained files, preserve the upstream license in `deployment/LICENSE-BASE-MODEL.txt`, then run `python -m scripts.package_model`. This packages existing weights without retraining, quantization or final-test inference. Do not replace the release asset without changing its version/URL and pinned checksum.

When Docker is running, a local container check is:

```powershell
docker build -t intentdesk .
docker run --rm -p 8000:10000 --memory=512m intentdesk
```

The Docker daemon was not running in the build environment, so container build execution and a 512 MB memory-limit test have not been verified. Native Windows inference/browser checks are evidence for local functionality, not proof that a Linux container fits a hosting plan.

## Render

The repository includes a `render.yaml` for **one free Docker web service**, no databases/disks and no paid fallback. It binds to Render's supplied `PORT`; `/health` becomes available after model loading. Auto-deploy is disabled, so future pushes do not silently restart the demonstration.

1. Connect Render and authorize the repository. Select the intended workspace before creating resources; confirm no service with the intended name already exists.
2. Validate `render.yaml` with `render blueprints validate render.yaml` when the authenticated Render CLI is available. Local JSON Schema validation checks structure only, not account permissions or capacity.
3. Ensure the source changes and pinned model release exist on GitHub.
4. Create a Blueprint from the repository, review that the compute plan remains **free**, and apply it. The [Blueprint deployment link](https://dashboard.render.com/blueprint/new?repo=https://github.com/shashankd06/support-ticket-classification-distilbert) opens this configuration.
5. Wait for an actual live deployment; inspect build/runtime logs, open `/`, `/health`, `/evaluation` and `/docs`, and submit at least two real `/predict` requests. Record the returned URL and checks before claiming deployment success.

Render's [free plan](https://render.com/docs/free) has cold starts and resource limits; the [current compute reference](https://render.com/docs/compute-plans) lists 512 MB RAM for free web services. This PyTorch model's fit on that plan is unverified. If it runs out of memory, stop and report the constraint; do not upgrade or change/quantize the weights automatically. A live free deployment may sleep when idle, so demonstrate the API-unavailable/retry state honestly.

No public deployment is verified at the time these files were prepared. Installing a plugin alone does not establish an authenticated deployment or a live URL. Native Sites hosting cannot execute this PyTorch/FastAPI service; a disconnected static page would not satisfy the real-model demo requirement.

## Verification

```powershell
python -m pytest -q -o faulthandler_timeout=30
node --test tests/frontend.test.cjs
python -m scripts.verify_inference
python -m scripts.verify_reports
```

Node is needed only for the frontend state tests; the application itself has no Node runtime/build dependency. Frontend fixtures mock API responses for loading/error/race-condition checks and are separate from real browser predictions. Browser QA screenshots and the verification record are under `reports/frontend/`.

Allow loopback sockets for local tests/preview. Do not retrain or rerun final test evaluation merely to verify the UI. [Demo script](DEMO_SCRIPT.md).
