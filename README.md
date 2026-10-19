# Support Ticket Classification with DistilBERT

A manageable Python project for comparing a TF-IDF + logistic regression classifier with a fine-tuned DistilBERT model on banking support intents. The purpose is to demonstrate the whole experiment lifecycle, including honest negative results.

**Current status:** code implemented; offline verification results are in `reports/local_checks`. No BANKING77 training or pretrained-model download completed in this environment. No benchmark metrics or model winner are claimed. FastAPI and Transformer checks require installing the dependencies. See `reports/EXPERIMENT_REPORT.md` for the exact boundary between completed and pending work.

## Dataset and attribution

Use [BANKING77 from PolyAI](https://github.com/PolyAI-LDN/task-specific-datasets), introduced by Casanueva, Temcinas, Gerz, Henderson, and Vulić, *Efficient Intent Detection with Dual Sentence Encoders*, 2020 ([paper](https://arxiv.org/abs/2003.04807)). The upstream repository lists 10,003 train and 3,080 test customer queries, across 77 fine-grained English online-banking intents. This is customer-intent routing, a proxy for ticket classification, not a collection of complete support threads.

The actual upstream CSV URLs were reachable through web inspection on 2026-10-19: [train](https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv) and [test](https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv). The header is `text,category`; text fields are quoted where necessary. The [Hugging Face metadata](https://huggingface.co/datasets/PolyAI/banking77/blob/main/dataset_infos.json) independently describes the 77 labels and official split sizes. These sizes have not yet been independently counted by the local preparation run; the loader checks them before accepting data.

The upstream [LICENSE](https://github.com/PolyAI-LDN/task-specific-datasets/blob/master/LICENSE) is **Creative Commons Attribution 4.0 International**. Preserve attribution, link the license, indicate modifications (this project filters overlap/conflicts and creates validation partitions), and do not imply endorsement. See [license terms](https://creativecommons.org/licenses/by/4.0/). No paid service is required. Dataset license applies to the data; it does not automatically license pretrained weights or your code. The selected [DistilBERT model card](https://huggingface.co/distilbert/distilbert-base-uncased) declares Apache-2.0. Keep the model notices with redistributed artifacts.

Labels are categorical intents such as `card_arrival` (a card has not arrived or delivery tracking), `cancel_transfer` (cancel a transfer), `card_not_working` (a card fails to work), and `pending_transfer` (a transfer remains pending). The machine-readable glossary for **all 77 labels** is generated from the source categories in `data/<mode>/labels.json`; their human-readable meanings are the underscore-separated category names. Read the dataset card for the complete source list. These are routing targets, not sentiment, urgency, ticket priority, or banking advice. See `docs/DATASET.md` for finer distinctions and limits.

## Layout

```text
configs/                 full and smoke experiment settings
ticketnlp/data.py        immutable source download, schema checks, leakage audit, splits
ticketnlp/baseline.py    training-only TF-IDF and logistic regression
ticketnlp/transformer.py PyTorch fine-tuning and batched prediction
ticketnlp/evaluation.py  metrics, confusion plots, misclassifications
ticketnlp/inference.py   consistent artifact loading and prediction
ticketnlp/api.py         small FastAPI service
ticketnlp/run.py         explicit experiment stages
tests/                  leakage, mapping, preprocessing and API checks
reports/                logs and measured results (no checkpoints)
data/ artifacts/        generated locally, ignored by Git
```

## Setup (Windows PowerShell)

Python 3.10–3.12 is recommended for broad wheel availability. An isolated environment avoids changing your existing installation. The inspected host has 12 logical CPUs, approximately 15.7 GiB RAM and an NVIDIA RTX 3050 laptop GPU with 6 GiB VRAM; its current global PyTorch is CPU-only. An attempted CUDA environment install was blocked by the download approval service, so GPU execution is **unverified**.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# CUDA wheels for this NVIDIA machine; use the current official selector if needed:
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[test]'
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m pytest -q
```

For CPU, install PyTorch using the [official installation selector](https://pytorch.org/get-started/locally/) before the editable install. Training automatically uses CUDA when available. The full default batch is 8 on CUDA / 4 on CPU with two accumulation steps, maximum 96 tokens, FP32, no DataLoader workers. If GPU memory runs out, reduce `batch_size_cuda` to 4 or 2 before preparation and increase accumulation to preserve the effective batch. This is a practical starting point, not a verified VRAM guarantee. Do not change settings after seeing test results.

Dependency ranges are in `pyproject.toml`. `reports/local_checks/environment.txt` records the actual environment used for completed offline checks; it is **not** a validated training lockfile. After a successful install, save `python -m pip freeze > requirements-lock.txt` for your training environment.

## Smoke mode: functionality, not model quality

```powershell
python -m ticketnlp.run prepare --config configs/smoke.json
python -m ticketnlp.run baseline --config configs/smoke.json
python -m ticketnlp.run transformer --config configs/smoke.json
python -m ticketnlp.run final --config configs/smoke.json
```

Smoke uses all 77 labels but only up to 3 train and 2 validation/test examples per class, one learning rate, one epoch, and at most **6 optimizer steps** for DistilBERT. It still downloads the real pretrained weights and audits the entire dataset first. Expect poor, noisy predictions: this verifies wiring rather than learning quality. Smoke and full data/artifacts/reports have distinct directories. Smoke models identify their mode in API responses.

## Full experiment

```powershell
python -m ticketnlp.run prepare --config configs/full.json
python -m ticketnlp.run baseline --config configs/full.json
python -m ticketnlp.run transformer --config configs/full.json
# Review validation reports before proceeding. Settings are now frozen.
python -m ticketnlp.run final --config configs/full.json
```

Or run `powershell -ExecutionPolicy Bypass -File scripts/train_gpu.ps1` from the project folder after setup. The commands use free local compute only. CPU full training may take substantially longer; no duration estimate has been measured. For a free GPU notebook, see `docs/GPU_RUN.md`.

Preparation resolves the upstream Git commit once, downloads CSV/license files at that revision, checks schema/size/labels, and records SHA-256 hashes. Keep `data/raw/revision.txt` and `reports/<mode>/data_audit.json` to reproduce the exact data revision. Never commit raw datasets or weights.

### Leakage control and official split changes

NFKC normalization, case folding, punctuation removal, and whitespace normalization identify exact duplicates. A **stateless** hashed character 3–5 gram cosine threshold of 0.90 identifies close lexical variants. All detected duplicate pairs form connected components (including transitive links). This audit sees held-out text for contamination checks only; it learns no IDF or classifier features from held-out data. It does not guarantee detection of semantic paraphrases; hashing collisions and the threshold can over/under-group sentences.

Preserve the official test partition except for conflicting-label components, which are quarantined from all partitions. Remove official training examples whose component touches test. Split remaining training data with seeded five-fold stratified group splitting; one fold (~20%) becomes validation. Whole groups stay together, even across different intent labels. Fail if a class disappears from any partition. Exact source, removal counts, components, and per-class counts are saved in the audit. Therefore results are **not directly comparable** with published unfiltered BANKING77 benchmarks. Test-internal repetitions can remain and overweight repeated queries; document their audit counts when reporting results.

Both models consume the same saved row IDs, labels, and partitions. Only training fits the TF-IDF vocabulary/IDF and classifier. DistilBERT uses its pretrained tokenizer and only training examples update weights. Neither evaluation split is used to fit preprocessing. Fixed seed 42 controls split generation, initial classification head, and shuffling; GPU/library differences can still change numerical results.

### Selection and final results

Baseline experiments compare `C=1` and `C=4`: a small regularization sweep for sparse features. DistilBERT experiments compare learning rates `2e-5` and `5e-5` over three epochs, with AdamW, weight decay 0.01, gradient clipping 1.0, and a short warmup/decay schedule. Select each family's checkpoint on **validation macro-F1**, with the first candidate winning ties. No seed sweep or statistical significance claim is implied.

The `final` stage loads both selected models and evaluates the common held-out test set. It refuses to overwrite a completed final comparison. Do not delete this guard to tune against test. A new experiment after inspecting test requires a genuinely fresh holdout for an unbiased estimate.

Reports contain accuracy, macro-F1, per-class precision/recall/F1/support, raw confusion matrices and PNG plots. Experiment JSON files include configurations, seeds, validation metrics, and (for DistilBERT) training loss, epoch, device, source revision and elapsed time. Errors retain local ticket text and scores for manual review; avoid publishing potentially sensitive real production tickets. `final_comparison.json` states the observed macro-F1 difference. A positive delta means DistilBERT beats the baseline on that run's macro-F1; it does not establish statistical significance.

## API

After training, launch a single process (each worker would load its own model):

```powershell
$env:MODEL_DIR='artifacts/full'
$env:MODEL_BACKEND='distilbert' # or baseline
python -m uvicorn ticketnlp.api:app --host 127.0.0.1 --port 8000
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType 'application/json' -Body '{"text":"My card has not arrived yet"}'
```

The response schema is `{label, score, model, mode, truncated}`. The score is the predicted class's softmax probability (DistilBERT) or logistic regression probability, **not calibrated confidence**. No sample numeric prediction is claimed here. `/docs` provides interactive API documentation; `/health` reports ready only after startup loads artifacts. Missing artifacts fail startup, and there is no silent model fallback. Load only trusted local joblib files.

Requests must contain a string with 1–4,000 characters and at least one non-whitespace character; invalid input returns HTTP 422. Whitespace is collapsed in training and serving. DistilBERT retains only the first `max_length` WordPiece tokens including special tokens (96 full / 64 smoke), and reports `truncated=true` when text exceeds that limit. Baseline uses all accepted input. Information near the end of a long ticket may be lost. The development API has no authentication, rate limits or production monitoring; keep it local.

## Limits and next steps

English banking domain only, closed-set single-label output, no unknown/abstention class, no calibrated thresholds, and no conversation context. Related intents and short ambiguous messages can be difficult. No actual failure pattern is asserted until error files are inspected. Full benchmark metrics and an API runtime check remain pending in the current delivery. Read `docs/INTERVIEW_GUIDE.md` and `reports/RESUME_BULLETS.md` before describing the work in interviews.

Implementation references: [Hugging Face sequence classification](https://huggingface.co/docs/transformers/tasks/sequence_classification), [DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert), [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html), [scikit-learn group splitting](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html), and [FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/).
