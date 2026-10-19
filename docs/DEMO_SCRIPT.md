# IntentDesk: 60–90-second demo

**0–15 seconds — problem.** "IntentDesk routes short English banking queries into 77 support intents. This is a portfolio demonstration of an actual fine-tuned classifier, rather than a general-purpose chatbot."

**15–35 seconds — first prediction.** Choose **Card delivery**, then **Classify query**. "This goes to my saved model through FastAPI. The result shows a readable intent, its canonical label, the model and the exact submitted query. The score is an uncalibrated model probability, not guaranteed confidence." Read the response actually displayed; do not promise a fixed score.

**35–50 seconds — second prediction.** Choose **Bank transfer** and classify. "This is a different topic using the same endpoint. If I edit the input, the previous result is explicitly marked outdated, so a result cannot silently appear to describe a different query."

**50–70 seconds — evidence.** Open **View evaluation**. "I compared DistilBERT with TF-IDF plus logistic regression on the same 3,020 filtered held-out queries. DistilBERT reached 89.60% accuracy and 0.8907 macro-F1, versus 86.79% and 0.8664 for the baseline. I selected settings on validation, and the experiment logs and per-class reports are linked here. These are one-seed filtered results, not a directly comparable official benchmark."

**70–90 seconds — limitation.** "Overall performance hides a real failure: DistilBERT recognized only two of 39 virtual-card troubleshooting queries, often confusing broken cards with requests for new cards. The system has no unknown-intent rejection and cannot replace human support judgment."

If the service is unavailable or waking up, say so and show the published evaluation; never substitute a fabricated prediction. A local demonstration is not a public deployment. Describe hosting as live only after the deployed URL and real `/predict` requests have been verified.
