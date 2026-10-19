# Interview guide

## A truthful opening

"I built a banking support-intent classifier using BANKING77. I compared a TF-IDF/logistic regression baseline with actual DistilBERT fine-tuning on a local GPU. I selected settings on validation macro-F1 and evaluated both models on the same filtered held-out set. DistilBERT reached 89.60% accuracy and 0.8907 macro-F1, versus 0.8664 macro-F1 for the baseline. I also verified the trained models through a local FastAPI service and analyzed where the Transformer still fails."

Explain your actual role and any AI assistance honestly. Be ready to trace the code, reproduce a run, and change a setting yourself. Do not claim a publicly deployed or production-ready application.

## Why this problem and dataset?

Customer-query routing is a practical multiclass NLP task. BANKING77 has 13,083 queries and 77 fine-grained intents; 77 is the number of classes, not examples. The source license is CC BY 4.0, with attribution and modification notices preserved. These are short single-turn English banking queries, not full support threads or a general-purpose ticket dataset.

## How did you build it?

Verified the source, schema, license and official split; implemented lexical overlap auditing and shared partitions; trained the sparse baseline; fine-tuned DistilBERT using a small learning-rate comparison; selected checkpoints on validation; evaluated final test once; inspected per-class failures; and verified real-model local HTTP inference. Modules separate preparation, baseline, Transformer training, evaluation and inference. JSON configs and a fixed seed keep the project understandable without extra infrastructure.

## Why use a baseline?

TF-IDF turns words and bigrams into sparse features, and logistic regression learns class probabilities. It is cheap and often strong for keyword-heavy intents. Vocabulary and IDF are fitted on training only. DistilBERT must justify extra training and inference cost with measured benefit. It improved overall accuracy by 2.81 percentage points and made 85 fewer errors, but the baseline was better on some classes.

## What was actually fine-tuned?

The pretrained DistilBERT encoder and a randomly initialized 77-way classification head both receive cross-entropy gradients. This is not frozen embedding extraction or only calling a hosted pretrained API. Each learning rate ran three epochs and 1,368 optimizer steps, using batch 8, accumulation 2, maximum length 96, AdamW and FP32 on an RTX 3050 6GB GPU. The model checkpoint and tokenizer are saved locally and excluded from Git.

## How did you avoid leakage?

Normalize exact matches, then group character-ngram cosine matches at threshold 0.90 into connected components. A stateless hashing vectorizer audits overlap without fitting IDF on held-out text. Remove official training rows whose component touches test, quarantine conflicting-label components, and split remaining training groups into train and validation. The resulting counts are 7,296 / 1,810 / 3,020. Saved group IDs are disjoint. The audit cannot guarantee removal of every semantic paraphrase, can over-group, and changes the official population, so published benchmark comparisons are inappropriate.

## How did you choose settings?

Compare logistic regression C=1 versus C=4, and DistilBERT learning rates 2e-5 versus 5e-5. Select each family's checkpoint by validation macro-F1, including epoch selection. C=4 reached validation macro-F1 0.8551; LR 5e-5 at epoch 3 reached 0.8872. The selected models achieved test macro-F1 0.8664 and 0.8907 respectively. One seed does not establish statistical significance or capture run-to-run variability.

## Why macro-F1 as well as accuracy?

Accuracy is the fraction of correct predictions. Macro-F1 averages each intent's F1 equally, making weak classes more visible. Per-class precision measures whether predictions for a label are correct; recall measures how many true examples are found. Support counts and confusion matrices reveal what an aggregate score hides. Ten saved metric reports were independently checked against their confusion matrices, including all 77 classes' precision, recall, F1 and support.

## Which failures matter most?

`virtual_card_not_working` has 23 training and 7 validation examples after filtering. DistilBERT recognizes only 2/39 test queries (F1 0.0976), versus the baseline's 14/39 (F1 0.5283). It confuses broken virtual cards with requests to obtain a virtual card: 18 go to `getting_virtual_card`, 16 to `get_disposable_virtual_card`. Sparse training coverage is a hypothesis for the weakness, not a proven cause.

Identity verification also shows related-intent confusion: 10 why-verification queries become how-to-verify predictions, and 7 go the other way. Card arrival versus delivery estimate is ambiguous for short queries such as `test-5`. Some errors are partly about label granularity or missing context; do not describe every wrong label as proof of a single model flaw.

## What does the score mean?

The selected class's softmax or logistic regression probability, not calibrated confidence. Wrong predictions with scores above 0.9 occur in the saved error review. Calibration requires a separate quantitative study. There is no unknown-intent class or abstention threshold, so out-of-domain input still receives a banking label.

## Is the application working?

Yes, local serving has been verified with the actual saved checkpoints. Temporary Uvicorn servers loaded each model once at startup and received real HTTP requests. Both backends passed 11 checks covering readiness, OpenAPI, predictions, repeat requests, whitespace consistency, input errors and truncation; eight Python/API tests passed without skips, alongside six fixture-based frontend state tests. The single-load behavior is also checked by a factory-counting unit test. This is not a public deployment, concurrency benchmark or production-readiness claim.

An earlier test stall was traced to Windows asyncio creating a loopback socket in the restricted execution environment. The minimal client stalled before application startup; the same test suite passed when loopback sockets were allowed. Explain the environmental cause and the evidence, rather than claiming a model or application bug was fixed.

## What are the engineering tradeoffs?

Explicit modules, a short PyTorch loop and JSON configs are enough for this project. Loading weights once per server process avoids per-request downloads/loading. The API accepts 1-4,000 characters and rejects blank inputs; DistilBERT retains only the first 96 WordPiece tokens in full mode and reports truncation. The baseline uses all accepted text. The service is local-only and lacks authentication, rate limiting, monitoring and load testing.

## What would you improve next?

Assess calibration, unknown-intent handling, seed variability and inference performance. A class-weighted loss could be tested using validation, but no improvement is promised. The existing test set has now been inspected, so any further test-driven iteration is exploratory; a fresh untouched holdout is needed for an unbiased new improvement claim. The current measured result remains preserved.

Evidence: [experiment report](../reports/EXPERIMENT_REPORT.md), [final comparison](../reports/full/final_comparison.json), [inference verification](../reports/inference_verification/summary.json), [test results](../reports/frontend/pytest.txt).
