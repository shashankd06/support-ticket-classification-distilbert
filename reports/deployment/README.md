# Public deployment verification

Verified on 2026-10-19: [IntentDesk](https://intentdesk-banking-distilbert.onrender.com/) and [API documentation](https://intentdesk-banking-distilbert.onrender.com/docs).

- Render workspace: My Workspace. Service: `intentdesk-banking-distilbert`, Free Docker plan, one worker. No paid resources selected.
- Source commit: `07060e5207af935fcceff6733610a4b0ef79370c`.
- Service ID: `srv-dbb321navr4c738icig0`; deployment ID: `dep-dbb321vavr4c738icj40`.
- Render reported **Deploy succeeded | Live**, duration 2m25s. Image build fetched and verified the released checkpoint; application startup completed.
- Nine public HTTP checks passed: page, health, evaluation, API docs, two real predictions, blank-input rejection, 4,001-character rejection, and token-truncation reporting. See [actual responses](verification.json).
- Browser predictions independently showed `card_arrival` with score 0.8444 and `cancel_transfer` with score 0.9330. Scores are uncalibrated model probabilities.

Evidence: [app screenshot](live-app.png), [Render screenshot](render-dashboard.png), [visible build/startup logs](render-log.txt). Reproduce the sequential checks from the repository root:

```powershell
python -m scripts.verify_deployment --url https://intentdesk-banking-distilbert.onrender.com
```

This verifies a working portfolio demo at the recorded time. It is not a load test, uptime guarantee, calibration study, or new model-quality evaluation. The original weights and held-out metrics are unchanged. Free instances sleep when idle and may need 50 seconds or more to wake. Concurrent-load behavior, peak memory, and sustained stability are unmeasured. Auto-deploy is disabled; subsequent documentation commits do not alter this recorded deployed revision.
