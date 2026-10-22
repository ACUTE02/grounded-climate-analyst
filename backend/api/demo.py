"""Demo mode: saved examples served instead of a live run.

CLIMATE_SUITE_MODE selects the mode: "demo" (default) or "live". In demo mode the API
answers only the examples saved in backend/demo/. It never builds the graph, never
creates a Gemini client and never opens a network connection, so a fresh clone works
with no key and no network. Free-text questions are refused with a plain message.

Layout of backend/demo/:
    index.json            the examples: id, label, shows, body (a ReportRequest), kind, file
    reports/<id>.json     one saved ReportResponse per example, with its `meta` object
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone

from forecasting import config as fconfig

MODE_ENV = "CLIMATE_SUITE_MODE"
MODES = ("demo", "live")
DEFAULT_MODE = "demo"

DEMO_DIR = fconfig.REPO_ROOT / "demo"
INDEX_PATH = DEMO_DIR / "index.json"
REPORTS_DIR = DEMO_DIR / "reports"

# Shown whenever the horizon manifest in use is the generated one (decision 16, 21): its
# numbers can differ a little from the frozen, published evidence.
EVIDENCE_NOTE = ("The numbers here may differ slightly from the published evidence; "
                 "see the Evidence page.")
# The note is shown only when the data behind the models was fetched again (data_source
# "refetched"); with the checked-in snapshot the numbers equal the published evidence.

FREE_TEXT_CODE = "demo_free_text_disabled"
FREE_TEXT_MESSAGE = (
    "Demo mode answers only the saved examples. To ask your own question, run in live "
    "mode with your own Gemini key (set CLIMATE_SUITE_MODE=live in backend/.env; see "
    "the README).")


def _load_env_file() -> None:
    from retrieval.embed import _load_dotenv_if_present

    _load_dotenv_if_present()


def requested_mode() -> str:
    """The raw value of CLIMATE_SUITE_MODE (after backend/.env is read), or the default."""
    _load_env_file()
    return os.environ.get(MODE_ENV, "").strip().lower() or DEFAULT_MODE


def mode() -> str:
    """"demo" or "live". An unknown value falls back to demo (no cost, no network)."""
    value = requested_mode()
    return value if value in MODES else DEFAULT_MODE


def mode_problem() -> str | None:
    value = requested_mode()
    if value in MODES:
        return None
    return (f"{MODE_ENV}={value!r} is not one of {', '.join(MODES)}; running in "
            f"{DEFAULT_MODE} mode.")


def is_demo() -> bool:
    return mode() == "demo"


# --------------------------------------------------------------------------- #
# Saved examples
# --------------------------------------------------------------------------- #
def load_index() -> list[dict]:
    if not INDEX_PATH.exists():
        return []
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))["examples"]


def load_example(example_id: str) -> dict | None:
    for item in load_index():
        if item["id"] == example_id:
            path = REPORTS_DIR / item["file"]
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
    return None


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


def find_match(body) -> dict | None:
    """The saved example whose request matches this body, else None.

    Same question (case and spacing ignored) and the same optional fields. A request
    that sets a field the example does not set, or a different value, is not a match.
    """
    for item in load_index():
        saved = item["body"]
        if _normalise(saved["request"]) != _normalise(body.request):
            continue
        if all(getattr(body, f) == saved.get(f)
               for f in ("region", "risk_types", "month", "crop")):
            return load_example(item["id"])
    return None


def free_text_detail() -> dict:
    return {"code": FREE_TEXT_CODE, "message": FREE_TEXT_MESSAGE,
            "examples": [item["id"] for item in load_index()]}


def examples_payload() -> dict:
    items = []
    for item in load_index():
        items.append({"id": item["id"], "label": item["label"], "body": item["body"],
                      "shows": item["shows"], "saved": True, "kind": item["kind"],
                      "has_report": item["has_report"]})
    return items


# --------------------------------------------------------------------------- #
# meta
# --------------------------------------------------------------------------- #
def _chunks_hash() -> str | None:
    from retrieval import config as rconfig

    if not rconfig.CHUNKS_PATH.exists():
        return None
    return hashlib.sha256(rconfig.CHUNKS_PATH.read_bytes()).hexdigest()


def corpus_info() -> dict | None:
    """Manifest time, chunk count and chunk hash of the retrieval corpus, if built."""
    from retrieval import config as rconfig

    if not rconfig.MANIFEST_PATH.exists():
        return None
    manifest = json.loads(rconfig.MANIFEST_PATH.read_text(encoding="utf-8"))
    return {"manifest_generated_at": manifest.get("generated_at"),
            "total_chunks": manifest.get("total_chunks"),
            "embedding_model": manifest.get("embedding_model"),
            "sha256_chunks": _chunks_hash()}


def _current_through(tool_outputs: dict) -> dict | None:
    out = {}
    drought = (tool_outputs or {}).get("forecast_drought_risk") or {}
    currency = drought.get("data_currency") or {}
    if drought.get("region") and currency.get("data_current_through"):
        out[drought["region"]] = str(currency["data_current_through"])
    return out or None


def make_meta(*, how_made: str, tool_outputs: dict, report: str, mode_: str,
              saved_example: bool = False, example_id: str | None = None,
              banner: str | None = None, used_models: bool = False,
              use_corpus: bool = False, details: dict | None = None,
              uses_manifest: bool = True) -> dict:
    """Build the `meta` object from the run, never by hand."""
    from orchestrator import config as oconfig
    from retrieval import config as rconfig

    from forecasting import snapshot

    source = fconfig.manifest_source() if uses_manifest else None
    data_source = snapshot.data_source() if uses_manifest else None
    if data_source == "refetched":
        banner = f"{banner} {EVIDENCE_NOTE}" if banner else EVIDENCE_NOTE

    return {
        "mode": mode_,
        "saved_example": saved_example,
        "example_id": example_id,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "how_made": how_made,
        "report_generated": bool((report or "").strip()),
        "chat_model": oconfig.CHAT_MODEL if used_models else None,
        "embedding_model": rconfig.EMBED_MODEL if (used_models and use_corpus) else None,
        "corpus": corpus_info() if use_corpus else None,
        "data_current_through": _current_through(tool_outputs),
        "horizon_manifest_source": source,
        "data_source": data_source,
        "banner": banner,
        "details": details,
    }


def live_start_problems() -> list[str]:
    """What live mode lacks, as plain sentences, for a start-up warning."""
    from retrieval.embed import MissingAPIKey, get_api_key

    problems = []
    try:
        get_api_key()
    except MissingAPIKey:
        problems.append("no Gemini key found (GEMINI_API_KEY in backend/.env): "
                        "reports cannot be written")
    missing = []
    for region in fconfig.REGIONS:
        for path in (fconfig.scaler_path(region), fconfig.spi_params_path(region),
                     *(fconfig.horizon_model_path(region, h) for h in (1, 2, 3))):
            if not path.exists():
                missing.append(path.name)
    if missing:
        problems.append(f"{len(missing)} model file(s) missing: run "
                        "`python -m scripts.setup`")
    from retrieval import config as rconfig

    if not rconfig.CHUNKS_PATH.exists():
        problems.append("the retrieval corpus is not built: run `python -m "
                        "retrieval.build`")
    return problems
