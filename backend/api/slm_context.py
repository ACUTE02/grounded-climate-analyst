"""Build the context for "ask the SLM about the project's own material".

The SLM answers from ONE text it is given, and its service accepts a bounded one (see slm_proxy.MAX_CONTEXT_CHARS). So the
backend assembles that text from material the project already holds and sends it with the question:

  1. a report (a saved example by id, or the written report text the page already has) with its forecast labels and what the
     system does not have, and
  2. the passages of the source corpus that best match the question (BM25, no key and no network).

What is NOT given to the model: the raw weather tables and the model files. Only the forecast results that appear in a report reach it.
Nothing is stored or logged. That the small model answers well from such a mixed text has not been measured.
"""

from __future__ import annotations

REPORT_BUDGET = 1800          # characters of the context reserved for the report part
PASSAGES = 4                  # corpus passages looked up for a question
TRUNCATION_MARK = " [cut]"


def report_part(example: dict | None, report_text: str | None) -> tuple[str, str | None]:
    """The report section of the context and a name for it, or ("", None)."""
    lines: list[str] = []
    name = None
    if example:
        name = f"report: {example.get('request', 'saved example')}"
        for h in example.get("horizon_confidence") or []:
            lines.append(f"Forecast {h.get('horizon')}: skill score {h.get('skill_score')}, label: {h.get('label')}.")
        for m in example.get("missing_data") or []:
            lines.append(f"Not available: {m.get('what')} ({m.get('reason')}).")
        report_text = report_text or example.get("report") or ""
    if report_text and report_text.strip():
        name = name or "report text"
        lines.append(report_text.strip())
    return "\n".join(lines), name


def corpus_passages(question: str, n: int = PASSAGES) -> list[dict]:
    """The best-matching source passages, or [] when the corpus is not built (no error).

    bm25_search returns hits shaped {id, bm25_score, rank, chunk}; the passage itself is in "chunk".
    """
    try:
        from retrieval.lexical import bm25_search

        hits = bm25_search(question, n=n)
    except Exception:
        return []
    chunks = [hit.get("chunk") or {} for hit in hits]
    return [c for c in chunks if (c.get("text") or "").strip()]


def build_context(question: str, limit: int, example: dict | None = None, report_text: str | None = None,
                  passages: list[dict] | None = None) -> dict:
    """{context, sources, truncated}. Every part is labelled with where it came from and the total is at most `limit`."""
    parts: list[tuple[str, str, str]] = []                    # (kind, name, text)
    truncated = False

    text, name = report_part(example, report_text)
    if text:
        if len(text) > REPORT_BUDGET:
            text, truncated = text[: REPORT_BUDGET - len(TRUNCATION_MARK)] + TRUNCATION_MARK, True
        parts.append(("report", name or "report", text))

    for chunk in (corpus_passages(question) if passages is None else passages):
        parts.append(("corpus", str(chunk.get("source") or chunk.get("id") or "source passage"), chunk["text"].strip()))

    out: list[str] = []
    sources: list[dict] = []
    used = 0
    for kind, name, text in parts:
        header = f"[{kind}: {name}]\n"
        room = limit - used - len(header) - (2 if out else 0)
        if room < 200:                                         # not worth a fragment
            truncated = truncated or kind == "corpus"
            continue
        if len(text) > room:
            text, truncated = text[: room - len(TRUNCATION_MARK)] + TRUNCATION_MARK, True
        block = header + text
        out.append(block)
        used += len(block) + (2 if len(out) > 1 else 0)
        sources.append({"kind": kind, "name": name})
    return {"context": "\n\n".join(out), "sources": sources, "truncated": truncated}
