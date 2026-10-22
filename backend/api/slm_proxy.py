"""A thin proxy to the optional small language model (SLM) service, for the website's "Try it" card.

    POST /slm/answer   {context, question}  ->  the SLM service's /answer, reduced to answer, refused, model
    POST /slm/ask      {question, report_id | report_text}  ->  the same, answered from a report plus the best-matching
                       source passages (api/slm_context.py builds the context); also returns the sources used
    GET  /slm/status   is SLM_URL set, and does the service answer /health

It works in demo mode too: the SLM service is separate from Gemini, so no key and no Gemini call is involved.
Nothing is stored and the text of the passage, the question and the answer is never logged.
No claim is made here about how fast or how accurate the model is.
"""

from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from api import slm_context
from orchestrator import config, slm

MAX_CONTEXT_CHARS = 4000
MAX_QUESTION_CHARS = 500
CALL_TIMEOUT_S = 60
STATUS_TIMEOUT_S = 3
UNSET_MESSAGE = "SLM service is not running; start it and set SLM_URL"
NO_MATERIAL_MESSAGE = ("There is no report or source corpus to answer from: pick a report, or build the corpus with "
                       "`python -m retrieval.build`.")
MAX_REPORT_TEXT = 20000

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


class SlmAskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_CHARS)
    report_id: str | None = Field(None, max_length=80, description="A saved example to include (its report and forecast labels).")
    report_text: str | None = Field(None, max_length=MAX_REPORT_TEXT, description="Or the written report text the page already holds.")


class SlmSource(BaseModel):
    kind: str
    name: str


class SlmAskResponse(SlmAnswerResponse):
    sources: list[SlmSource]
    truncated: bool


def make_client(timeout: float) -> httpx.Client:
    """One place to build the HTTP client, so tests can put a fake service behind it."""
    return httpx.Client(timeout=timeout)


def _url() -> str:
    url = slm.service_url()
    if url is None:
        raise HTTPException(status_code=503, detail=UNSET_MESSAGE)
    return url


def _ask_service(context: str, question: str) -> dict:
    """Forward one context and question to the SLM service; map every failure to a short HTTP error."""
    url = _url()
    client = make_client(CALL_TIMEOUT_S)
    try:
        response = client.post(f"{url}/answer", timeout=CALL_TIMEOUT_S,
                               json={"context": context, "question": question,
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


@router.post("/answer", response_model=SlmAnswerResponse)
def answer(body: SlmAnswerRequest) -> dict:
    return _ask_service(body.context, body.question)


@router.post("/ask", response_model=SlmAskResponse)
def ask(body: SlmAskRequest) -> dict:
    """Answer a question from the project's own material: a report (saved example or the page's text) plus the best-matching
    source passages. The context is built here, within the same size limit as /answer."""
    from api import demo

    example = None
    if body.report_id:
        example = demo.load_example(body.report_id)
        if example is None:
            raise HTTPException(status_code=404, detail=f"No saved example {body.report_id!r}.")
    built = slm_context.build_context(body.question, MAX_CONTEXT_CHARS, example=example, report_text=body.report_text)
    if not built["context"].strip():
        raise HTTPException(status_code=404, detail=NO_MATERIAL_MESSAGE)
    result = _ask_service(built["context"], body.question)
    return {**result, "sources": built["sources"], "truncated": built["truncated"]}


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
