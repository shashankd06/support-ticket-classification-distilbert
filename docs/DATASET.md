# Dataset decision and label interpretation

BANKING77 was chosen because its fine-grained banking intent labels resemble routing categories in a support queue, and its size permits a meaningful baseline and small Transformer experiments. It is more challenging than classifying a few broad topics. Public source, CSV endpoints, metadata and the CC BY 4.0 license were verified through the browser; local row counts, duplicates and distributions remain unmeasured until preparation succeeds.

## What labels mean

Use source category names as canonical labels, with sorted alphabetical IDs persisted in `labels.json`. Do not rely on the Hugging Face dataset's numeric IDs, which use a different order. Human-readable descriptions can be obtained by replacing underscores with spaces; retain original punctuation/case in canonical names (including `Refund_not_showing_up` and `reverted_card_payment?`). The model, reports and API all use the same saved mapping.

Useful distinctions when manually reviewing examples:

| Labels | Intended distinction from category names |
|---|---|
| `card_arrival` / `card_delivery_estimate` | Actual delivery or missing-card status versus expected delivery timing |
| `pending_transfer` / `failed_transfer` / `declined_transfer` | A transfer in progress, failed, or declined |
| `card_payment_not_recognised` / `card_payment_fee_charged` | An unfamiliar card payment versus a fee associated with it |
| `cash_withdrawal_not_recognised` / `wrong_amount_of_cash_received` | Unrecognized withdrawal versus ATM dispensed the wrong amount |
| `request_refund` / `Refund_not_showing_up` | Requesting a refund versus following up on an expected refund |
| `verify_my_identity` / `why_verify_identity` / `unable_to_verify_identity` | How to verify, reason for verification, or verification failing |
| `lost_or_stolen_card` / `compromised_card` | Missing physical card versus suspected compromised card details |

These are interpretations of the published names, not an additional authoritative annotation manual. Some queries lack enough context to disambiguate them. Inspect source examples and avoid claiming that every borderline prediction is a clear model error.

The full source label inventory is linked in the README. Preparation validates 77 distinct nonempty categories, writes their exact canonical names, and checks each class remains represented in every final partition.

## Limits

Short English single-turn customer queries, one banking domain, one label each. There is no representative production ticket arrival rate, no severity target, no multilingual coverage, no explicit unknown class, and no measured label reliability in this project. Results do not establish suitability for a live banking system. Similar wording can occur under different labels. Quarantining conflicting lexical components sacrifices examples to avoid silently choosing labels.

The duplication audit uses text only, prior to any model fitting. It keeps detected connected components out of multiple splits, but does not identify every semantic paraphrase. Cosine threshold 0.90 is a predeclared engineering choice, not a tuned optimal value. Inspect components and document sensitivity before making stronger claims. Changing the official train/test population means reporting your exact retained counts is essential.
