"""A thin proxy to the optional small language model (SLM) service, for the website's "Try it" card.

    POST /slm/answer   {context, question}  ->  the SLM service's /answer, reduced to answer, refused, model
    GET  /slm/status   is SLM_URL set, and does the service answer /health

It works in demo mode too: the SLM service is separate from Gemini, so no key and no Gemini call is involved.
Nothing is stored and the text of the passage, the question and the answer is never logged.
No claim is made here about how fast or how accurate the model is.
"""

from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from orchestrator import config, slm

MAX_CONTEXT_CHARS = 4000
MAX_QUESTION_CHARS = 500
CALL_TIMEOUT_S = 60
STATUS_TIMEOUT_S = 3
UNSET_MESSAGE = "SLM service is not running; start it and set SLM_URL"

router = APIRouter(prefix="/slm", tags=["slm"])


class SlmAnswerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    context: str = Field(..., min_length=1, max_length=MAX_CONTEXT_CHARS,
                         description=f"The passage to answer from (at most {MAX_CONTEXT_CHARS} characters).")
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_CHARS,
                          description=f"The question (at most {MAX_QUESTION_CHARS} characters).")


class SlmAnswerResponse(BaseModel):
    answer: str
    refused: bool
    model: str | None = None


def make_client(timeout: float) -> httpx.Client:
    """One place to build the HTTP client, so tests can put a fake service behind it."""
    return httpx.Client(timeout=timeout)


def _url() -> str:
    url = slm.service_url()
    if url is None:
        raise HTTPException(status_code=503, detail=UNSET_MESSAGE)
    return url


@router.post("/answer", response_model=SlmAnswerResponse)
def answer(body: SlmAnswerRequest) -> dict:
    url = _url()
    client = make_client(CALL_TIMEOUT_S)
    try:
        response = client.post(f"{url}/answer", timeout=CALL_TIMEOUT_S,
                               json={"context": body.context, "question": body.question,
                                     "max_new_tokens": config.SLM_MAX_NEW_TOKENS})
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="The SLM service did not answer in time.") from None
    except httpx.HTTPError:
        raise HTTPException(status_code=503, detail=UNSET_MESSAGE) from None
    finally:
        client.close()

    if response.status_code == 413:
        raise HTTPException(status_code=413, detail="The passage and question are too long for the SLM; shorten the passage.")
    if response.status_code != 200:
        raise HTTPException(status_code=503, detail=f"The SLM service answered HTTP {response.status_code}.")
    try:
        data = response.json()
        return {"answer": str(data["answer"]), "refused": bool(data["refused"]),
                "model": data.get("model")}
    except (ValueError, KeyError, TypeError):
        raise HTTPException(status_code=502, detail="The SLM service sent an answer in an unexpected shape.") from None


@router.get("/status")
def status() -> dict:
    """Whether the Try-it card can be used. Reads the SLM service's /health only."""
    url = slm.service_url()
    if url is None:
        return {"configured": False, "available": False, "model": None, "message": UNSET_MESSAGE}
    client = make_client(STATUS_TIMEOUT_S)
    try:
        response = client.get(f"{url}/health", timeout=STATUS_TIMEOUT_S)
        data = response.json() if response.status_code == 200 else {}
    except (httpx.HTTPError, ValueError):
        return {"configured": True, "available": False, "model": None,
                "message": "SLM_URL is set but the service did not answer."}
    finally:
        client.close()
    loaded = bool(data.get("model_loaded"))
    return {"configured": True, "available": loaded, "model": data.get("model"),
            "message": "" if loaded else "The SLM service is up but its model is not loaded yet."}
