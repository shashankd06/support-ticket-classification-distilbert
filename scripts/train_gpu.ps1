$ErrorActionPreference = 'Stop'
python -m ticketnlp.run prepare --config configs/full.json
if ($LASTEXITCODE -ne 0) { throw 'Preparation failed' }
python -m ticketnlp.run baseline --config configs/full.json
if ($LASTEXITCODE -ne 0) { throw 'Baseline failed' }
python -m ticketnlp.run transformer --config configs/full.json
if ($LASTEXITCODE -ne 0) { throw 'Transformer failed' }
python -m ticketnlp.run final --config configs/full.json
if ($LASTEXITCODE -ne 0) { throw 'Final evaluation failed' }
