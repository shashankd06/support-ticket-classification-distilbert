# Interview guide

## A truthful opening

“I implemented a support-intent classification experiment using BANKING77, a TF-IDF/logistic regression baseline, a DistilBERT fine-tuning loop, shared leakage-controlled splits, and a small inference API. Offline tests passed for the portions available locally. Dataset training and API integration are pending because dependencies and model downloads were blocked in the build environment.”

Replace the final sentence only once you have run and inspected the real results. Do not present invented-fixture metrics as benchmark performance.

## Why this problem and dataset?

Routing a customer query to a support category is a concrete multiclass NLP task. Fine-grained banking categories expose ambiguity beyond broad topic classification. Dataset availability, official splits and licensing were checked before implementation. However, these are short customer intents, not complete support-ticket threads; the title describes an application use case rather than an exact production-data source.

## Why start with TF-IDF?

TF-IDF emphasizes informative terms and bigrams; logistic regression is inexpensive, interpretable and often strong on constrained intents. A Transformer must earn its extra latency, memory and training cost. Sparse lexical features struggle with paraphrases or word order; their success on keywords makes the baseline a useful diagnostic. Fit vocabulary and IDF on training only.

## What does fine-tuning change?

The pretrained DistilBERT encoder represents tokens using bidirectional self-attention. A randomly initialized 77-way classification head predicts routing intent. Cross-entropy backpropagates through the head and encoder; this is actual fine-tuning rather than frozen embeddings. The tokenizer is pretrained, and dynamic padding reduces wasted computation. Smaller batches plus accumulation manage GPU memory. DistilBERT trades some capacity for a smaller model than BERT; no performance advantage over this baseline is presumed.

## How was leakage addressed?

An identical query appearing in training and test can inflate reported performance. Normalize and group exact and close lexical matches before splitting. Any group touching official test is removed from official train; conflicting-label groups are quarantined; the remaining training groups form train and validation. Connected components enforce transitivity. Stateless hashing supports auditing without learning IDF from test. The audit is lexical, so explain that semantic paraphrases may remain and threshold choices can over-group. Test results on this filtered population are not directly comparable with published official-split results.

## How are settings selected?

Two regularization values test baseline overfitting, and two learning rates test fine-tuning sensitivity. The best validation macro-F1 selects each family’s checkpoint, including the epoch. Test is reserved for the final comparison. One fixed seed is manageable but does not quantify run-to-run variability. If another experiment is motivated by looking at test errors, a fresh holdout is required to make an unbiased new claim.

## Why macro-F1 and accuracy?

Macro-F1 gives each intent equal importance, averaging per-class F1 rather than pooling decisions across all examples. Accuracy gives the overall fraction correct. Per-class precision measures how often predictions for a class are correct; recall measures how many real examples of that class are recovered. Zero-division handling prevents undefined numbers for unpredicted classes. Support counts and a confusion matrix expose poor minority-class performance that one scalar can hide.

## Which errors actually happened?

No BANKING77 model errors have been measured yet. Plausible review targets include arrival versus delivery estimate, pending versus failed transfer, and refund request versus missing refund. These are hypotheses, not findings. After a run, read the highest-count confusion pairs and at least 20 concrete examples, compare both models, and categorize lexical ambiguity, missing context, apparent labeling ambiguity, truncation, and true domain misunderstanding. Quote actual row IDs and counts in the report. A few examples do not establish a population-wide cause.

## Is the API score confidence?

It is a model probability, not a guarantee of correctness. Neural softmax probabilities can be overconfident, especially for out-of-domain text. Calibration needs independent validation and measurement (e.g. reliability plots or expected calibration error); it is intentionally not claimed here. The service has no abstention policy and always selects one known intent. Adding an unknown class or calibrated rejection threshold is a useful future experiment.

## What are the engineering tradeoffs?

Explicit modules and a short PyTorch loop are easy to inspect. Files and JSON configs are sufficient for a small project; there is no experiment tracking server or distributed infrastructure. Startup loads weights once per process; synchronous request handling avoids blocking the async event loop with model work, but concurrency is not load-tested. Inputs have a 4,000-character limit, Transformers truncate by token length, and outputs disclose truncation and smoke/full mode. Production readiness would require security, rate limits, out-of-domain handling, drift checks and operational evaluation.

## What would you improve next?

First finish smoke and full runs and inspect errors. Then test seed variability, assess calibration, measure latency/VRAM, inspect near-duplicate threshold sensitivity and consider an out-of-domain holdout. Choose changes based on evidence rather than making the system larger by default.
