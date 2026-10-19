# Published evidence

## Model training and evaluation

- [Final comparison](full/final_comparison.json): full-training held-out test metrics for both selected models.
- [Baseline experiments](full/baseline_experiments.json) and [DistilBERT experiments](full/transformer_experiments.json): configurations, seed and measured validation results (including every Transformer epoch).
- [Training log](full/run.log), [configuration](full/config.json), [hardware](full/hardware.json).
- [Full data audit](full/data_audit.json): source revision/hashes and per-class split counts; [compact summary](full/data_audit_summary.json).
- [Baseline final metrics](full/baseline_final_test/metrics.json), [JSON matrix](full/baseline_final_test/confusion_matrix.json), [plot](full/baseline_final_test/confusion_matrix.png), [error counts](full/baseline_final_test/error_summary.json).
- [DistilBERT final metrics](full/distilbert_final_test/metrics.json), [JSON matrix](full/distilbert_final_test/confusion_matrix.json), [plot](full/distilbert_final_test/confusion_matrix.png), [error counts](full/distilbert_final_test/error_summary.json).

Each validation experiment folder also contains per-class metrics, a labeled JSON confusion matrix, PNG and aggregate error summary. Validation results are exploratory selection evidence; folders ending in `final_test` contain the final held-out results. [Experiment report](EXPERIMENT_REPORT.md) explains selection and concrete errors.

## Verification

- [Metric integrity](verification/report_integrity.json): independent arithmetic checks for ten saved metric reports, 77 classes each.
- [Passing tests](verification/pytest.txt) and [JUnit XML](verification/pytest.xml): six tests, no skips, one dependency deprecation warning.
- [Real inference summary](inference_verification/summary.json), [baseline checks](inference_verification/baseline.json), [DistilBERT checks](inference_verification/distilbert.json), and corresponding server logs: 11 HTTP checks per trained backend, served only on localhost.
- [Verification environment](verification/requirements-environment.txt): post-training environment snapshot, not a reconstructed exact training lockfile. CUDA wheels need the PyTorch index.

The earlier `local_checks/` folder is retained as historical evidence from the initial CPU-only build. Its invented-fixture scores are not BANKING77 results. Its skipped API test and incomplete-training status are superseded by the evidence above. Smoke training was a functionality check; its scores are not the full experiment's results.

## Publication scope

Datasets, full ticket/error text dumps, secrets and model checkpoints are excluded. Confusion matrices and audit files contain aggregate counts and canonical label names. Data-derived reports attribute BANKING77 to PolyAI and Casanueva et al. (2020), [source](https://github.com/PolyAI-LDN/task-specific-datasets), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); modifications include overlap/conflict filtering and validation splitting. Summarized source examples in the experiment report retain row IDs for provenance.

No training or final test inference was rerun for publication. Local HTTP checks use invented demonstration inputs and verify serving behavior, not model quality. No public deployment, statistical significance, calibration, production performance or all-class improvement is claimed.
