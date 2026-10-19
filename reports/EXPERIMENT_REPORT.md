# Experiment and error-analysis report

## Status on 2026-10-19

Implementation is complete enough for an initial run; real-data execution is **incomplete**. The dataset CSV endpoints/schema, upstream split sizes, 77-label metadata and CC BY 4.0 license were inspected via official web sources. Source data was not downloaded into the local workflow. Direct networking failed, and both dependency-install attempts were rejected because automatic approval review could not complete (review model at capacity), including a retry explicitly approved by the user. This was not a safety determination. No compute was purchased.

Hardware inspection found an RTX 3050 laptop GPU (6 GiB VRAM), 12 logical CPU cores, and approximately 15.7 GiB RAM. Global PyTorch is 2.14.1+cpu, reporting CUDA unavailable; the physical GPU is present but is not usable from that installation. Python 3.10 exists but its ML/API dependencies are not installed. Creating the requested GPU environment never executed.

## Completed checks

- Offline leakage/grouping, label-order and training-only vocabulary/preprocessing checks: see `local_checks/pytest.txt` for the actual run count and result.
- Report-writing integration exercise with **invented fixtures**, including macro-F1/accuracy/per-class metrics and confusion-matrix PNG. These outputs live in `local_checks/invented_fixture_ONLY` and are not BANKING77 or smoke-mode results. They demonstrate file generation only; they say nothing about generalization.
- Static source compilation check; see `local_checks/compile.txt`.

The FastAPI test module was skipped because FastAPI is absent. Transformer loading, tokenizer truncation on a real model, CUDA training, real-data split audits, smoke training, full training, final test evaluation, and live API requests remain **not run**. Source review does not establish that these unexecuted integrations will succeed.

A later repeat of the offline integration check stalled and was interrupted; its cause was not established. The saved fixture reports and pytest output are from the earlier completed run, before final formatting and plot-label refinements. Final source compilation and lint checks passed, but the later repeat is not counted as a successful integration run.

## Planned experiments (no measured outcomes yet)

| Family | Configuration | Selection split | Status |
|---|---|---|---|
| TF-IDF + logistic regression | unigram/bigram, C=1, seed=42 | validation macro-F1 | not run |
| TF-IDF + logistic regression | unigram/bigram, C=4, seed=42 | validation macro-F1 | not run |
| DistilBERT | LR 2e-5, 3 epochs, seed=42 | validation macro-F1 / epoch | not run |
| DistilBERT | LR 5e-5, 3 epochs, seed=42 | validation macro-F1 / epoch | not run |

Full configuration: length 96, AdamW, weight decay 0.01, FP32, GPU batch 8 or CPU batch 4, accumulation 2. Rationale: a small regularization comparison for the baseline and a small learning-rate comparison for fine-tuning sensitivity. No elaborate sweep is necessary for a manageable project. Smoke is a separate six-step plumbing check, not an experiment on model quality.

No macro-F1, accuracy, per-class scores, class distribution after filtering, training duration or winner can be reported for BANKING77. The pipeline records these only when the corresponding stage actually runs. Do not copy fixture scores into this table.

## Error analysis

There are no measured real-data misclassifications to inspect. Candidate distinctions for future review include card arrival versus delivery estimate, pending versus failed transfer, and requesting versus following up on a refund. These are review hypotheses from label meanings, **not observed failure patterns**.

After training, inspect the top confusion pairs from each model’s `error_summary.json`, then at least 20 actual error rows from `errors.json`. Record row IDs, true/predicted labels, the decisive wording, whether the baseline and Transformer disagree, and whether the issue appears to involve ambiguity, missing context, lexical shortcut, truncation or label uncertainty. Separate validation exploration from final test findings. Include counterexamples; do not infer a failure cause solely from a confusion count.

## Completion checklist

Install dependencies; run tests without skipped API tests; prepare and inspect the real leakage audit; execute smoke; execute full baseline and Transformer sweeps; freeze selection; run final test once; inspect concrete errors; update this report and résumé drafts with measured full-run results. Exact commands are in the README and GPU run guide.
