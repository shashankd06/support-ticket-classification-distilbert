# IntentDesk frontend verification

## Implemented

Buildless HTML/CSS/JavaScript integrated into the existing FastAPI app. `/` serves the page, `/assets/` serves packaged assets, `/evaluation` reads the committed full final comparison, and `/predict`, `/health` and `/docs` retain their contracts. Evaluation metrics and model weights were not changed. No final-test inference or retraining was performed.

The interface labels the input, counts/limits characters, offers three examples, shows actual model/label/score/mode/truncation fields, and binds results to the submitted text. It uses text rendering rather than HTML interpolation for inputs and outputs. A changed input is visibly distinguished from a previous result. The page includes a baseline comparison, source links, dataset/model attribution, scope limitations and the observed virtual-card failure.

## Completed checks

- **8 Python/API tests passed**, including static routes, preservation of `/docs`, evaluation equality with the committed report, missing-evaluation 503 behavior, input validation, single startup loading, leakage grouping and label/preprocessing checks. One upstream Starlette/httpx deprecation warning remains. See [output](pytest.txt) and [JUnit XML](pytest.xml).
- **6 frontend state tests passed** using explicitly mocked DOM/API fixtures. Covered empty/overlong input, loading and duplicate submission prevention, input editing during an outstanding request, unavailable API/retry, smoke/truncation notices, safe text rendering, 422/malformed responses, and report-sourced metrics. See [output](frontend-tests.txt) and [test source](../../tests/frontend.test.cjs). These tests are not real-model predictions.
- **Real browser requests against the local trained DistilBERT** returned `card_arrival` (score displayed 84.4%) for the card-delivery example, `cancel_transfer` (93.3%) for the transfer example, and `change_pin` (94.1%) for the PIN example. These are invented demonstration inputs, not additional quality measurements. Empty input displayed validation, and editing an input after a completed request visibly marked the result outdated.
- Desktop layout was inspected in the in-app browser and captured in [desktop.png](desktop.png). Mobile layout was inspected using the actual app inside a narrow QA iframe because the browser viewport override was unreliable. The frame's measured document client width and scroll width were both **360px**, confirming no horizontal overflow at that inspection. [mobile.png](mobile.png) records that layout. Cross-frame click/keyboard automation was unreliable, so no separate successful mobile-interaction run is claimed; real inference was verified in the desktop page.
- Reduced-motion rules, visible focus states, labeled inputs, live status/error regions and semantic table/navigation structure are implemented. No automated accessibility certification or exhaustive browser compatibility claim is made.
- The prepared Render YAML passed the current official public JSON Schema. This is structural validation only, not authenticated Render CLI/API validation or a deployment.
- The serving archive's weight SHA-256 matches the previously verified full model exactly. No quantization or weight modification occurred. CPU-only Docker build/run and its 512 MB limit test remain unverified because Docker's daemon was not running.

## Deployment boundary

No public hosting URL has been verified. The hosting integration was installed during this active turn, but publishing tools have not been exposed to this session and the available browser was not authenticated to Render. No paid service, substitute model, fabricated prediction fallback, or disconnected static-only deployment was created. See [deployment guide](../../docs/DEPLOYMENT.md).

The initial six-test API verification remains in `reports/verification/`; this folder adds the current frontend/API checks. The two trained backends' earlier real HTTP verification is in `reports/inference_verification/`. Source datasets, full error text dumps and model checkpoints remain excluded from Git history.
