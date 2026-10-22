"""Optional passage check by the small language model (SLM). Default OFF.

After retrieval, ask the SLM service, for each of the first retrieved passages, whether that one passage answers the
user's question (the SLM's trained task: answer from the given passage, or refuse with one exact sentence). The result
is shown next to the report as a separate item. It is NOT given to the synthesis step and NOT added to the sources the
number checker accepts, so it cannot change the report, its numbers or its grounding result.

    CLIMATE_SUITE_SLM=on        # default off; read from the environment or backend/.env
    SLM_URL=http://127.0.0.1:8001

A missing URL, an unreachable service or an error never stops a report: the step records why it was skipped.
The step makes no Gemini call. Nothing here claims that the SLM improves or worsens the analyst: that is not measured.
"""

from __future__ import annotations

import os
import time

import httpx

from orchestrator import config

NOTE = ("Answers of a small separate model, from one retrieved passage each. Not this project's "
        "measurement and not used to write the report.")


def _env(env: dict | None) -> dict:
    if env is not None:
        return env
    from retrieval.embed import _load_dotenv_if_present

    _load_dotenv_if_present()
    return os.environ


def enabled(env: dict | None = None) -> bool:
    return (_env(env).get(config.SLM_FLAG_ENV) or "").strip().lower() in ("on", "1", "true")


def service_url(env: dict | None = None) -> str | None:
    value = (_env(env).get(config.SLM_URL_ENV) or "").strip()
    return value.rstrip("/") or None


def passage_answers(question: str, chunks: list[dict], env: dict | None = None,
                    client: httpx.Client | None = None, clock=time.monotonic) -> dict:
    """Ask the SLM about the first retrieved passages. Always returns a dict; never raises."""
    result: dict = {"enabled": True, "model": None, "answers": [], "skipped_reason": None, "note": NOTE}
    url = service_url(env)
    if url is None:
        result["skipped_reason"] = f"{config.SLM_URL_ENV} is not set"
        return result
    passages = [c for c in chunks if (c.get("text") or "").strip()][: config.SLM_MAX_PASSAGES]
    if not passages:
        result["skipped_reason"] = "no retrieved passage to ask about"
        return result

    own_client = client is None
    client = client or httpx.Client(timeout=config.SLM_CALL_TIMEOUT_S)
    started = clock()
    try:
        for chunk in passages:
            if clock() - started > config.SLM_TOTAL_DEADLINE_S:
                result["skipped_reason"] = "the total time limit was reached"
                break
            entry = {"source": chunk.get("source"), "citation": chunk.get("citation")}
            try:
                response = client.post(f"{url}/answer", timeout=config.SLM_CALL_TIMEOUT_S,
                                       json={"context": chunk["text"], "question": question,
                                             "max_new_tokens": config.SLM_MAX_NEW_TOKENS})
            except httpx.HTTPError as exc:
                result["skipped_reason"] = f"the SLM service did not answer ({type(exc).__name__})"
                break
            if response.status_code == 413:
                result["answers"].append({**entry, "answer": None, "refused": None,
                                          "error": "this passage is too long for the SLM"})
                continue
            if response.status_code != 200:
                result["skipped_reason"] = f"the SLM service answered HTTP {response.status_code}"
                break
            body = response.json()
            result["model"] = body.get("model")
            result["answers"].append({**entry, "answer": body.get("answer"), "refused": body.get("refused"),
                                      "latency_ms": body.get("latency_ms"),
                                      "n_prompt_tokens": body.get("n_prompt_tokens")})
    finally:
        if own_client:
            client.close()
    return result
