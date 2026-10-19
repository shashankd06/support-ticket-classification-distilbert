FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/artifacts/full MODEL_BACKEND=distilbert \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
    MALLOC_ARENA_MAX=2
WORKDIR /app
COPY deployment/requirements-serving.txt /app/deployment/requirements-serving.txt
RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r deployment/requirements-serving.txt
COPY ticketnlp /app/ticketnlp
COPY reports/full/final_comparison.json /app/reports/full/final_comparison.json
COPY deployment/fetch_model.py deployment/model-release.json /app/deployment/
RUN python deployment/fetch_model.py
EXPOSE 10000
CMD ["sh", "-c", "exec python -m uvicorn ticketnlp.api:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1 --limit-concurrency 4 --no-access-log"]
