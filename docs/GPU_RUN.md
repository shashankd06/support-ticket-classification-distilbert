# Run training on a GPU

The included `scripts/train_gpu.ps1` is the local GPU script. Use the README setup commands first, activate the environment, check that `torch.cuda.is_available()` is true, then run the script from the project root. Smoke and full GPU training have now completed on an RTX 3050 6GB using Python 3.10.11 and PyTorch 2.11.0+cu128; see the [published experiment report](../reports/EXPERIMENT_REPORT.md). Existing final results are preserved, and the `final` command refuses to overwrite them. Use these instructions for a fresh reproduction, not to repeatedly inspect test scores while tuning.

## Optional free notebook runtime

If local setup remains unavailable, use a notebook provider's free GPU tier if offered. Availability and session limits vary; do not enable billing or upgrade to paid compute for this project. This procedure is prepared but has not been executed.

1. Create a notebook, select a GPU runtime, and upload the project ZIP. Extract it. Ensure the current directory contains `pyproject.toml`.
2. Run these cells (adapt the extraction path if needed):

```python
import zipfile, os
with zipfile.ZipFile('/content/support-ticket-classification.zip') as z:
    z.extractall('/content/project')
os.chdir('/content/project/support-ticket-classification')
```

```python
%pip install -e '.[test]'
import torch
print(torch.__version__, torch.cuda.is_available())
assert torch.cuda.is_available(), 'Select a free GPU runtime before full training'
```

```python
!python -m pytest -q
!python -m ticketnlp.run prepare --config configs/smoke.json
!python -m ticketnlp.run baseline --config configs/smoke.json
!python -m ticketnlp.run transformer --config configs/smoke.json
!python -m ticketnlp.run final --config configs/smoke.json
```

Inspect each command's exit status/output; stop if a stage fails. Then run full mode:

```python
import subprocess, sys
for stage in ['prepare', 'baseline', 'transformer']:
    subprocess.run([sys.executable, '-m', 'ticketnlp.run', stage,
                    '--config', 'configs/full.json'], check=True)
```

Review validation metrics, freeze settings, and evaluate test **once**:

```python
subprocess.run([sys.executable, '-m', 'ticketnlp.run', 'final',
                '--config', 'configs/full.json'], check=True)
```

3. Save `reports/`, `artifacts/full/`, `data/raw/revision.txt`, and your environment lock to your own storage before the runtime expires. Checkpoints can be hundreds of MB. Keep them out of Git. The code saves the best checkpoint after validation, but does not resume optimizer state after interruption; rerun the affected training stage from the fixed seed if needed.
4. Read the saved error examples and compare your reproduction with the published experiment report. Update any new claims only from the results you actually measured. The existing resume bullets describe the completed local run, not the unexecuted notebook procedure.

## Troubleshooting

- No CUDA: verify the NVIDIA driver, CUDA-enabled PyTorch wheel and correct activated environment. System CUDA toolkit installation is not normally required for a bundled PyTorch wheel.
- Out of memory: reduce batch size before starting the experiment; keep maximum length reasonable. Set accumulation to preserve effective batch if comparing settings.
- Failed downloads: confirm network access to GitHub, Hugging Face and the PyTorch package index. The dataset requires no account; never paste access tokens into code.
- Slow duplicate audit: it uses brute-force sparse cosine search over ~13k queries, chunked for memory. It is quadratic in total pair work; it is intended for this small corpus, not millions of tickets.
- Expired free runtime: checkpoints and reports must be saved before shutdown. No automatic paid fallback exists.
