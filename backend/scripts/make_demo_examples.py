"""Build the saved demo examples that need no Gemini call.

    python -m scripts.make_demo_examples              # tools only: no key, no network
    python -m scripts.make_demo_examples --live       # written reports: key, index, quota

Writes backend/demo/index.json and backend/demo/reports/<id>.json. The default needs the
model files from `python -m scripts.setup`, no key, no network. A file that already holds a
live-generated report is kept, never replaced by the tools-only version.

`--live` runs the real pipeline (Gemini) for the examples that need a written report, one
at a time, inside the free-tier budget (20 calls a day, 5 a minute): it stops before an
example whose worst case (4 calls) would not fit, waits a minute between examples, and
saves each result as soon as it exists, so a later run continues where this one stopped.
It never prints the key.

What each file holds in this step:
- drought_one_month, drought_no_skill_horizon, heat_null_result, crop_impact_wheat:
  the real output of the deterministic tools ("tools_only"). The written report, the
  retrieval and the live outside sources are NOT in these files yet: they need a Gemini
  key and the retrieval index and are added in a later step. `grounding.status` says
  "not_generated" and `meta.report_generated` is false, which is the truth.
- grounding_catch: one of the checker's own labelled test cases, run through the real
  checker ("constructed_fixture"), with the recorded real case beside it.
- out_of_scope: the refusals the code itself produces for an unsupported region, crop
  and horizon ("validators_only").

Every value is read from a tool, a file or the checker. Nothing is typed in by hand
except the questions and the labels.
"""

from __future__ import annotations

import json

import re
import time

from api import app as api_app
from api import demo
from api import quota
from crop_impact import config as cconfig
from crop_impact.tool import assess_crop_impact_core
from evaluation.checker_eval import CASES
from forecasting import config as fconfig
from forecasting.tool import forecast_drought_risk
from heat.tool import forecast_heat_stress_risk
from orchestrator import config as oconfig
from orchestrator.grounding import check_grounding, warning_banner

TOOLS_ONLY_BANNER = (
    "Saved example. These are the real outputs of the forecasting tools; the written "
    "report is not part of this saved file yet.")
TOOLS_ONLY_WARNING = (
    "tools_only: no written report, no retrieved sources and no live outside sources "
    "in this saved example")

OUT_OF_SCOPE_REQUEST = ("Give me the six-month drought forecast for Jaisalmer district "
                        "and the expected percentage yield loss for mustard there.")


def _shape(state: dict, request: str) -> dict:
    """The same fields build_response gives, without the live quota and meta."""
    return {
        "request": request,
        "report": state.get("report", ""),
        "grounding": api_app._grounding_status(state),
        "horizon_confidence": api_app._horizon_confidence(state),
        "missing_data": api_app._missing_data(state),
        "tools_called": [c.get("name") for c in state.get("tool_calls", [])],
        "tool_outputs": state.get("tool_outputs", {}),
        "retrieved_sources": [],
        "external_sources": [],
        "warnings": state.get("warnings", []),
        "quota": {},
    }


def _no_report_state(calls: list[tuple[str, dict]], outputs: dict) -> dict:
    return {
        "tool_calls": [{"name": name, "args": args} for name, args in calls],
        "tool_outputs": outputs,
        "grounding": {"report_missing": True},
        "warnings": [TOOLS_ONLY_WARNING],
    }


def _tools_only(request: str, calls, outputs, example_id: str) -> dict:
    response = _shape(_no_report_state(calls, outputs), request)
    response["meta"] = demo.make_meta(
        how_made="tools_only", tool_outputs=outputs, report="", mode_="demo",
        saved_example=True, example_id=example_id, banner=TOOLS_ONLY_BANNER)
    return response


def build_drought(example_id: str, request: str, region: str) -> dict:
    out = forecast_drought_risk.invoke({"region": region})
    return _tools_only(request, [("forecast_drought_risk", {"region": region})],
                       {"forecast_drought_risk": out}, example_id)


def build_heat(example_id: str, request: str, region: str, month: str) -> dict:
    args = {"region": region, "month": month}
    out = forecast_heat_stress_risk.invoke(args)
    return _tools_only(request, [("forecast_heat_stress_risk", args)],
                       {"forecast_heat_stress_risk": out}, example_id)


def build_crop(example_id: str, request: str, region: str, crop: str,
               month: str) -> dict:
    out = assess_crop_impact_core(region, crop, month, with_narrative=False)
    args = {"region": region, "crop": crop, "month": month}
    return _tools_only(request, [("assess_crop_impact", args)],
                       {"assess_crop_impact": out}, example_id)


def build_grounding_catch(example_id: str, request: str) -> dict:
    case = next(c for c in CASES if c.id == "fabricated_skill_score")
    result = check_grounding(case.report, case.tool_outputs, case.chunks or [])
    report = warning_banner(result) + "\n" + case.report
    outputs = {"forecast_drought_risk": case.tool_outputs}

    real = json.loads(oconfig.CAUGHT_SAMPLE_PATH.read_text(encoding="utf-8"))
    details = {
        "fixture_case": case.id,
        "fixture_why": case.why,
        "invented_numbers": list(case.fabricated),
        "fixture_note": (
            "The report text and the tool output in this example are a labelled test "
            "case from evaluation/checker_eval.py, not a real forecast: the report "
            "was written wrong on purpose. The warning above it, the flagged "
            "number and the counts are the real checker's output for that text."),
        "related_real_case": {
            "source_file": "orchestrator/grounding_caught_sample.json",
            "what_happened": real["what_happened"],
            "why_it_matters": real["why_it_matters"],
            "how_the_pipeline_behaved": real["how_the_pipeline_behaved"],
            "fix_applied": real["fix_applied"],
            "request": real["caught_failures"][0]["request"],
            "attempts": [{"attempt": f["attempt"], "logged_at": f["logged_at"],
                          "unverified_numbers": f["unverified_numbers"],
                          "total_checked": f["total_checked"]}
                         for f in real["caught_failures"]],
        },
    }
    state = {"report": report, "grounding": result, "tool_outputs": outputs,
             "tool_calls": [], "warnings": []}
    response = _shape(state, request)
    response["meta"] = demo.make_meta(
        how_made="constructed_fixture", tool_outputs=outputs, report=report,
        mode_="demo", saved_example=True, example_id=example_id,
        banner=("Saved example. The report below was made wrong on purpose to show "
                "the number checker; a real recorded case is described in details."),
        details=details, uses_manifest=False)
    return response


def build_out_of_scope(example_id: str, request: str) -> dict:
    def refusal(call, *args) -> str:
        try:
            call(*args)
        except ValueError as exc:
            return str(exc)
        raise RuntimeError(f"{call.__name__}{args} did not refuse")

    manifest = json.loads(fconfig.manifest_path().read_text(encoding="utf-8"))
    horizons = [h["horizon"] for h in manifest["regions"][fconfig.DEFAULT_REGION]]
    flags = [
        {"what": "district-level drought forecast for Jaisalmer", "available": False,
         "reason": refusal(fconfig.check_region, "jaisalmer")},
        {"what": "six-month drought forecast", "available": False,
         "reason": (f"this system's drought forecast covers {', '.join(horizons)} "
                    "only (the horizons in the horizon manifest)")},
        {"what": "yield-loss estimate for mustard", "available": False,
         "reason": refusal(cconfig.check_crop, "mustard")},
    ]
    state = {"grounding": {"report_missing": True}, "tool_outputs": {},
             "tool_calls": [],
             "warnings": ["validators_only: these refusals come from the region, crop "
                          "and horizon checks, not from a written report"]}
    response = _shape(state, request)
    response["missing_data"] = flags
    response["meta"] = demo.make_meta(
        how_made="validators_only", tool_outputs={}, report="", mode_="demo",
        saved_example=True, example_id=example_id,
        banner=("Saved example, not written by the live model. These refusals come from "
                "the system's own region, crop and horizon checks; there is no written "
                "report for this example."),
        details={"live_written": False,
                 "note": "A live written report was tried and could not be made (the daily "
                         "Gemini quota was used up), so this example stays on the refusals "
                         "from the system's own checks."})
    return response


def examples() -> list[dict]:
    return [
        {"id": "drought_one_month",
         "label": "Drought next month, Rajasthan",
         "shows": "The one forecast horizon this project validated, with its measured skill.",
         "body": {"request": "What is the drought risk for Rajasthan next month, and how "
                             "much should I trust it?",
                  "region": "rajasthan", "risk_types": ["drought"]},
         "build": lambda i, b: build_drought(i, b["request"], "rajasthan")},
        {"id": "drought_no_skill_horizon",
         "label": "Drought for Barmer, three months, with reliability",
         "shows": "Per-horizon skill labels, including a horizon with no skill that "
                  "must not be relied on.",
         "body": {"request": "What is the drought risk for Barmer over the next three "
                             "months, and how much should I trust each month's number?",
                  "region": "barmer", "risk_types": ["drought"]},
         "build": lambda i, b: build_drought(i, b["request"], "barmer")},
        {"id": "heat_null_result",
         "label": "Heat stress for Barmer: an honest null result",
         "shows": "Observed heat only. Forecasting heat stress was tested and has no "
                  "usable skill, so no forecast is offered.",
         "body": {"request": "How much heat stress did Barmer see in May 2024, and can "
                             "you forecast it?",
                  "region": "barmer", "risk_types": ["heat_stress"], "month": "2024-05"},
         "build": lambda i, b: build_heat(i, b["request"], "barmer", b["month"])},
        {"id": "crop_impact_wheat",
         "label": "Heat impact on wheat, Feb 2006",
         "shows": "Which risk binds yield, and the one sourced yield coefficient with "
                  "its caveat, or an explicit 'no sourced estimate'.",
         "body": {"request": "How did heat affect wheat in Rajasthan in February 2006, "
                             "and what does it mean for yield?",
                  "crop": "wheat", "month": "2006-02"},
         "build": lambda i, b: build_crop(i, b["request"], "rajasthan", "wheat", "2006-02")},
        {"id": "grounding_catch",
         "label": "The number checker catching an invented figure",
         "shows": "A figure not found in any source is flagged and named in a warning "
                  "above the report.",
         "body": {"request": "What is the one-month drought forecast for Rajasthan?",
                  "region": "rajasthan", "risk_types": ["drought"]},
         "build": lambda i, b: build_grounding_catch(i, b["request"])},
        {"id": "out_of_scope",
         "label": "Something the system cannot know",
         "shows": "Three unsupported asks, each declined rather than invented.",
         "body": {"request": OUT_OF_SCOPE_REQUEST},
         "build": lambda i, b: build_out_of_scope(i, b["request"])},
    ]


LIVE_IDS = ("drought_one_month", "drought_no_skill_horizon", "heat_null_result",
            "crop_impact_wheat", "out_of_scope")
LIVE_BANNER = ("Saved example. Written by the live pipeline on the date in generated_at "
               "and saved as it was; asking again can give different text.")
KEY_LIKE = re.compile(r"AIza[0-9A-Za-z_-]{20,}|sk-[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}")


def _safe(text: str, limit: int = 160) -> str:
    return KEY_LIKE.sub("[redacted]", str(text))[:limit]


def _existing(example_id: str) -> dict | None:
    path = demo.REPORTS_DIR / f"{example_id}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _out_of_scope_flags() -> list[dict]:
    return json.loads((demo.REPORTS_DIR / "out_of_scope.json").read_text(
        encoding="utf-8"))["missing_data"]


def build_live(example: dict) -> tuple[dict | None, str]:
    """Run the real pipeline for one example. Returns (response or None, one-line note)."""
    from orchestrator.graph import analyse

    body = example["body"]
    state = analyse(request=body["request"], region=body.get("region"),
                    risk_types=body.get("risk_types"), month=body.get("month"),
                    crop=body.get("crop"))
    if (state.get("grounding") or {}).get("report_missing"):
        warnings = state.get("warnings", [])
        return None, "no report: " + _safe("; ".join(warnings), 700)
    response = api_app.build_response(state, body["request"])
    response["quota"] = {}
    if example["id"] == "out_of_scope":
        known = {m["what"] for m in response["missing_data"]}
        response["missing_data"] = [m for m in _out_of_scope_flags()
                                    if m["what"] not in known] + response["missing_data"]
    response["meta"] = demo.make_meta(
        how_made="live_generation", tool_outputs=response["tool_outputs"],
        report=response["report"], mode_="demo", saved_example=True,
        example_id=example["id"], banner=LIVE_BANNER, used_models=True, use_corpus=True,
        details=({"added_missing_data": "the first three missing_data entries come from "
                  "the region, crop and horizon checks, not from the model"}
                 if example["id"] == "out_of_scope" else None))
    return response, f"grounding {response['grounding']['status']}"


def live_main(only: list[str] | None) -> int:
    ids = [i for i in LIVE_IDS if only is None or i in only]
    by_id = {e["id"]: e for e in examples()}
    saved, skipped = [], []
    for position, example_id in enumerate(ids):
        existing = _existing(example_id)
        if existing and existing["meta"]["how_made"] == "live_generation":
            print(f"{example_id}: already live-generated, kept")
            continue
        used = quota.calls_used_today()
        if used + quota.WORST_CASE_CALLS_PER_REPORT > quota.DAILY_CALL_BUDGET:
            skipped.append(example_id)
            print(f"{example_id}: NOT run, {used} of {quota.DAILY_CALL_BUDGET} calls used "
                  f"in this run and one report can cost {quota.WORST_CASE_CALLS_PER_REPORT}")
            continue
        try:
            response, note = build_live(by_id[example_id])
        except Exception as exc:                  # keep going with the next example
            response, note = None, f"error {type(exc).__name__}: {_safe(exc)}"
        calls = quota.calls_used_today() - used
        if response is None:
            skipped.append(example_id)
            print(f"{example_id}: NOT saved ({calls} calls) {note}")
        else:
            (demo.REPORTS_DIR / f"{example_id}.json").write_text(
                json.dumps(response, indent=2, ensure_ascii=False, default=str) + "\n",
                encoding="utf-8")
            saved.append(example_id)
            print(f"{example_id}: saved ({calls} calls) {note}")
        _reindex()
        if position < len(ids) - 1:
            time.sleep(60)                        # 5 requests a minute
    print(f"saved {len(saved)}, not saved {len(skipped)} {skipped}; calls used in this "
          f"run: {quota.calls_used_today()} of {quota.DAILY_CALL_BUDGET}")
    return 0 if not skipped else 2


def _reindex() -> None:
    index = []
    for ex in examples():
        data = _existing(ex["id"])
        index.append({"id": ex["id"], "label": ex["label"], "shows": ex["shows"],
                      "body": ex["body"], "kind": data["meta"]["how_made"],
                      "has_report": data["meta"]["report_generated"],
                      "file": f"{ex['id']}.json"})
    demo.INDEX_PATH.write_text(
        json.dumps({"examples": index}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build the saved demo examples.")
    parser.add_argument("--live", action="store_true",
                        help="generate the written reports with the live pipeline")
    parser.add_argument("--only", default=None, help="comma-separated example ids (--live)")
    args = parser.parse_args()
    if args.live:
        return live_main(args.only.split(",") if args.only else None)

    demo.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    index = []
    for ex in examples():
        kept = _existing(ex["id"])
        if kept and kept["meta"]["how_made"] == "live_generation":
            response = kept
            print(f"kept demo/reports/{ex['id']}.json (live_generation)")
            index.append({"id": ex["id"], "label": ex["label"], "shows": ex["shows"],
                          "body": ex["body"], "kind": "live_generation",
                          "has_report": kept["meta"]["report_generated"],
                          "file": f"{ex['id']}.json"})
            continue
        response = ex["build"](ex["id"], ex["body"])
        file_name = f"{ex['id']}.json"
        (demo.REPORTS_DIR / file_name).write_text(
            json.dumps(response, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8")
        index.append({"id": ex["id"], "label": ex["label"], "shows": ex["shows"],
                      "body": ex["body"], "kind": response["meta"]["how_made"],
                      "has_report": response["meta"]["report_generated"],
                      "file": file_name})
        print(f"wrote demo/reports/{file_name} ({response['meta']['how_made']})")
    demo.INDEX_PATH.write_text(
        json.dumps({"examples": index}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(f"wrote demo/index.json ({len(index)} examples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
