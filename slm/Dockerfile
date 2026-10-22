# syntax=docker/dockerfile:1
# The small language model service (optional compose profile "slm"). NOT built or run in the release
# that added it: it is untested. It uses the hf backend on CPU unless a GPU runtime is added by you.
# The adapter (GHOSTBHAIYA/climate-slm-1.5b-sft) and the base model are downloaded from Hugging Face when
# the service starts; no weights are baked into the image.
FROM python:3.11.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    PHASE5_BACKEND=hf PHASE5_MODEL=sft_1.5b PHASE5_HOST=0.0.0.0 PHASE5_PORT=8001 \
    HF_HOME=/home/app/.cache/huggingface

WORKDIR /app
RUN useradd --system --create-home --uid 10001 app && chown app:app /app

COPY requirements-serve.txt requirements.txt ./
# mlflow is only for logging results on a laptop; the service does not need it.
RUN grep -v "^mlflow" requirements-serve.txt > /tmp/req.txt \
    && pip install --upgrade pip \
    && pip install -r requirements.txt -r /tmp/req.txt \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install transformers==5.19.0 peft==0.21.2 huggingface_hub accelerate

COPY --chown=app:app climate_slm ./climate_slm
COPY --chown=app:app serving ./serving
COPY --chown=app:app configs ./configs
USER app

EXPOSE 8001
CMD ["python", "-m", "serving.app"]
