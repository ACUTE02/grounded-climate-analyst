# syntax=docker/dockerfile:1
# The analyst's backend. Demo mode by default: it answers only the saved examples and needs
# no key and no network. No .env, no data, no generated models and no secrets are baked in.
FROM python:3.11.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CLIMATE_SUITE_MODE=demo

WORKDIR /app
RUN useradd --system --create-home --uid 10001 app
# the app creates its data folders under /app at start-up, so it must own the folder
RUN chown app:app /app

# requirements.txt says what is needed; requirements.lock.txt pins the versions it was
# tested with (used as constraints, so only what requirements.txt needs is installed).
COPY requirements.txt requirements.lock.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt -c requirements.lock.txt

COPY --chown=app:app . .
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
