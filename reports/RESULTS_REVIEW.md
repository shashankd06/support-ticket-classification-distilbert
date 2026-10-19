# Review of completed training results

Reviewed local saved outputs on 2026-10-19. Both smoke and full workflows have final comparison files, selected baseline artifacts and DistilBERT safetensors/tokenizers. Full Transformer logs record CUDA execution on an NVIDIA RTX 3050 6GB Laptop GPU using PyTorch 2.11.0+cu128 and Python 3.10.11.

## Full final test results

| Model | Accuracy | Macro-F1 | Correct / total | Errors |
|---|---:|---:|---:|---:|
| TF-IDF + logistic regression | 86.7881% | 0.866395 | 2,621 / 3,020 | 399 |
| DistilBERT | 89.6026% | 0.890706 | 2,706 / 3,020 | 314 |

DistilBERT improves macro-F1 by 0.024311 (2.43 percentage points on a percentage scale), and accuracy by 2.81 percentage points. It makes 85 fewer mistakes. This is one seed/run on a filtered test population, not a statistical significance result or a published official-split benchmark comparison.

Selected using validation: baseline C=4 (macro-F1 0.855068); DistilBERT learning rate 5e-5, epoch 3 (macro-F1 0.887182). The alternative DistilBERT rate 2e-5 reached validation macro-F1 0.730368 at epoch 3. Both learning-rate runs completed 1,368 optimizer steps each. No training or final test evaluation was rerun during this review.

## Integrity checks completed during review

- Saved duplicate-component IDs are disjoint across train/validation/test.
- Split sizes are 7,296 / 1,810 / 3,020 and labels match the selected model configuration.
- Both final confusion matrices sum to 3,020; their diagonal counts reproduce saved accuracy.
- Saved error-row counts equal total minus correct predictions.
- Source train/test SHA-256 hashes match the values previously reported by official dataset metadata.

The data audit reports 60 normalized duplicate rows, 1,962 close-similarity pairs, 839 official training rows removed for test overlap, and 50 conflicting-label components. The original source has 10,003 train and 3,080 test rows. Filtering changes class coverage and the test population. This review checked saved group assignments; it did not recompute the full pairwise duplicate audit.

## Concrete failure patterns

The largest weakness is `virtual_card_not_working`: only 23 training and 7 validation examples remain, compared with 39 test examples. DistilBERT correctly recognizes 2/39 (recall 5.13%, F1 0.097561), whereas the baseline recognizes 14/39 (recall 35.90%, F1 0.528302). DistilBERT sends 18 of these queries to `getting_virtual_card` and 16 to `get_disposable_virtual_card`. Low training coverage is a plausible contributor, not an established causal finding.

For example, `test-2522` describes a broken disposable virtual card; both models classify it as requesting a disposable card. DistilBERT similarly misses `test-2521`, a virtual card that does not work. These errors suggest that product-type words can outweigh the distinction between obtaining a card and troubleshooting it.

Identity verification is another related-intent confusion: DistilBERT predicts `verify_my_identity` for 10 `why_verify_identity` queries and makes the reverse error 7 times. Card arrival versus delivery estimate contributes 4 errors; `test-5` asks when the card will arrive and receives the estimate label despite the source arrival label. Some short questions are inherently ambiguous without context.

Incorrect high scores also occur: `test-156` is assigned general `exchange_rate` rather than `card_payment_wrong_exchange_rate` with score 0.918. This is evidence that a high individual model score is not a correctness guarantee; it is not a quantitative calibration study.

## Scope and remaining checks

Smoke completed but is only a functionality check (six optimizer steps); its low Transformer score must not be presented as full-training quality. The initial follow-up pytest run stalled and was interrupted. Subsequent diagnostics located the stall in Windows asyncio creating a loopback socket before application startup in the restricted execution environment. With local socket access enabled, all six tests passed without skips. Both trained models then passed 11 actual HTTP checks each through temporary local Uvicorn servers. See [current verification](README.md) for reports and reproducible scripts. No public deployment is claimed.

The README, experiment report, resume bullets and interview guide now reflect the completed training and local serving verification. Supporting aggregate results are published without datasets, text-bearing error dumps or checkpoints. Preserve the original final test results; any further model selection must use validation, and an unbiased new post-test claim needs a fresh holdout.
