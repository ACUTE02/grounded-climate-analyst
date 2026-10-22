"""FastAPI service: POST /answer and GET /health.

    PHASE5_BACKEND=hf PHASE5_MODEL=sft_1.5b python -m serving.app          # needs torch, a GPU and the adapter download
    uvicorn serving.app:create_app --factory --host 127.0.0.1 --port 8000

The prompt is built EXACTLY like the evaluator's (serving/prompting.build_messages, tested against sft_val.jsonl), the decoding
is the evaluator's greedy mode (repetition_penalty 1.0, at most 256 new tokens), and the refusal flag is the evaluator's STRICT
detector (climate_slm.evaluation.metrics.is_refusal_strict) applied to the answer text.

Behaviour that is tested on the laptop with a fake backend:
  * empty / missing context or question, an unknown field, max_new_tokens outside 1..256, an unknown model name: HTTP 422
  * a prompt longer than service.max_prompt_tokens: HTTP 413 with the numbers (the context is never cut silently)
  * a backend error or a backend that is not loaded: HTTP 503 with a short message, no stack trace
  * logs carry request id, model, latency and token counts, never the context, the question or the answer text
  * no CORS middleware, no secret or token anywhere; every response has an X-Request-ID header."""
from __future__ import annotations

import logging
import os
import sys
import time
import uuid
from importlib import metadata

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from climate_slm.config import load_config, load_eval_config, load_phase5_config
from climate_slm.evaluation import metrics as M
from serving.backends import Backend, make_backend
from serving.prompting import build_messages
from serving.schemas import MAX_NEW_TOKENS, AnswerRequest, AnswerResponse, HealthResponse

log = logging.getLogger("serving")
VERSION_PACKAGES = ("fastapi", "pydantic", "uvicorn", "torch", "transformers", "peft", "vllm")


def package_versions() -> dict:
    """Versions of the installed packages (read from the package metadata: nothing is imported)."""
    out = {"python": sys.version.split()[0]}
    for p in VERSION_PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def check_service_config(cfg5: dict) -> None:
    s = cfg5["service"]
    if s["max_new_tokens"] != MAX_NEW_TOKENS or s["max_prompt_tokens"] != s["max_model_len"] - s["max_new_tokens"]:
        raise ValueError("service.max_new_tokens must be 256 and max_prompt_tokens = max_model_len - max_new_tokens")
    if s.get("cors"):
        raise ValueError("CORS must stay off by default (service.cors: false)")


def choose_backend_name(cfg5: dict, env: dict | None = None) -> str:
    env = os.environ if env is None else env
    return env.get(cfg5["backend"]["env_backend"]) or cfg5["backend"]["default"]


def choose_model_key(cfg5: dict, env: dict | None = None) -> str:
    env = os.environ if env is None else env
    return env.get(cfg5["backend"]["env_model"]) or cfg5["default_model"]


def create_app(backend: Backend | None = None, cfg5: dict | None = None, cfg1: dict | None = None) -> FastAPI:
    """Build the app. With no `backend` the one named by the environment / config is created and loaded on start-up."""
    cfg5 = cfg5 or load_phase5_config()
    check_service_config(cfg5)
    cfg1 = cfg1 or load_config()
    refusal_text, svc = cfg1["refusal_text"], cfg5["service"]
    logging.getLogger("serving").setLevel(getattr(logging, svc["log_level"], logging.INFO))
    own_backend = backend is None
    if own_backend:
        backend = make_backend(choose_backend_name(cfg5), choose_model_key(cfg5), cfg5, load_eval_config())

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if own_backend:
            t0 = time.perf_counter()
            if not backend.loaded:
                backend.load()                               # real backends load here, once
            app.state.startup_seconds = time.perf_counter() - t0
            log.info("backend %s loaded model %s in %.1f s", backend.name, backend.model_name, app.state.startup_seconds)
        yield
        backend.close()

    app = FastAPI(title="Climate SLM grounded QA", version="1", lifespan=lifespan)
    app.state.backend, app.state.cfg5, app.state.startup_seconds = backend, cfg5, None

    @app.middleware("http")
    async def request_id_header(request: Request, call_next):
        request.state.request_id = uuid.uuid4().hex
        try:
            response = await call_next(request)
        except Exception:                                    # unhandled: a short JSON message, no stack trace, the type is logged
            log.exception("request_id=%s unhandled error", request.state.request_id)
            response = JSONResponse({"detail": "internal error"}, status_code=500)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.get("/health", response_model=HealthResponse)
    def health():
        return HealthResponse(status="ok" if backend.loaded else "loading", model_loaded=bool(backend.loaded), backend=backend.name,
                              model=backend.model_name, models=sorted(cfg5["models"]), versions=package_versions(),
                              startup_seconds=app.state.startup_seconds,
                              batching=backend.batch_stats() if hasattr(backend, "batch_stats") else None)

    @app.post("/answer", response_model=AnswerResponse)
    def answer(req: AnswerRequest, request: Request):
        rid = request.state.request_id
        if req.model is not None and req.model != backend.model_name:
            raise HTTPException(422, f"this service serves model {backend.model_name!r}, not {req.model!r}")
        if not backend.loaded:
            raise HTTPException(503, "model not loaded yet")
        messages = build_messages(req.context, req.question, cfg1)
        try:
            n_prompt = backend.count_prompt_tokens(messages)
        except Exception as err:
            log.error("request_id=%s backend error while counting tokens: %s", rid, type(err).__name__)
            raise HTTPException(503, "backend error") from None
        if n_prompt > svc["max_prompt_tokens"]:
            raise HTTPException(413, f"prompt has {n_prompt} tokens, the limit is {svc['max_prompt_tokens']}; "
                                     "shorten the context (it is never cut silently)")
        max_new = req.max_new_tokens or svc["max_new_tokens"]
        t0 = time.perf_counter()
        try:
            g = backend.generate(messages, max_new)
        except Exception as err:                            # any backend failure: 503, short message, only the type is logged
            log.error("request_id=%s backend error: %s", rid, type(err).__name__)
            raise HTTPException(503, "backend error") from None
        latency_ms = (time.perf_counter() - t0) * 1000.0
        refused = M.is_refusal_strict(g.text, refusal_text)
        log.info("request_id=%s model=%s latency_ms=%.1f n_prompt_tokens=%d n_new_tokens=%d refused=%s", rid, backend.model_name,
                 latency_ms, g.n_prompt_tokens, g.n_new_tokens, refused)
        return AnswerResponse(answer=g.text, refused=refused, model=backend.model_name, n_prompt_tokens=g.n_prompt_tokens,
                              n_new_tokens=g.n_new_tokens, latency_ms=latency_ms, request_id=rid)

    return app


def main() -> int:
    import uvicorn
    cfg5 = load_phase5_config()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    s = cfg5["service"]
    port = int(os.environ.get(cfg5["backend"]["env_port"]) or s["port"])
    host = os.environ.get("PHASE5_HOST") or s["host"]        # 127.0.0.1 unless asked; a container sets 0.0.0.0 (never public by default)
    uvicorn.run(create_app(cfg5=cfg5), host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
