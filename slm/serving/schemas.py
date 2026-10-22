"""Request and response schemas of the service. Validation lives here: empty or missing context / question is rejected (422),
max_new_tokens must be 1..256 (422)."""
from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_NEW_TOKENS = 256            # the upper bound; configs/phase5.yaml (service.max_new_tokens) is checked against it


class AnswerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())
    context: str
    question: str
    model: str | None = None
    max_new_tokens: int | None = Field(default=None, ge=1, le=MAX_NEW_TOKENS)

    @field_validator("context", "question")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty")
        return v                 # returned unchanged: the prompt must be built from exactly what was sent


class AnswerResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    answer: str
    refused: bool
    model: str
    n_prompt_tokens: int
    n_new_tokens: int
    latency_ms: float
    request_id: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: str
    model_loaded: bool
    backend: str
    model: str | None
    models: list[str]
    versions: dict
    startup_seconds: float | None = None       # time of backend.load() in this process (None for a backend given ready)
    batching: dict | None = None               # batch statistics of the real backends
