# Resume bullets based on verified work

- Fine-tuned DistilBERT for 77 banking support intents using PyTorch and Hugging Face, achieving 89.60% accuracy and 0.8907 macro-F1 on 3,020 filtered held-out queries, improving macro-F1 by 0.0243 over a TF-IDF/logistic regression baseline.
- Implemented duplicate-aware shared data partitions, validation-based experiment selection and per-class error analysis; deployed a responsive FastAPI/DistilBERT demo on Render Free, verified with nine public HTTP checks, 22 local HTTP checks, eight Python/API tests and six frontend state tests.

These bullets describe verified local training/evaluation and public portfolio-demo deployment, not production readiness. Results are one seeded run on a filtered BANKING77 population. Checkpoints, raw datasets and text-bearing error files are excluded from Git. Evidence: [experiment report](EXPERIMENT_REPORT.md), [final results](full/final_comparison.json), [API checks](inference_verification/summary.json), [tests](frontend/pytest.txt).
