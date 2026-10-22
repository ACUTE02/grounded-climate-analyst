"""Phase 5 — the evaluation suite.

Entirely offline and free. The live faithfulness run is a script
(``python -m evaluation.faithfulness``), not a test, because it costs 3-5 Gemini
calls per item against a 20/day budget; its *results* are asserted here from the
checked-in JSON instead, so the suite verifies the measurement without repeating
its cost.

The centrepiece is `test_the_evaluation_summary_numbers_match_the_saved_results`:
the numbers the API reports are held to the result file they come from.
"""

from __future__ import annotations

import json

import pytest

from evaluation import checker_eval, config

# --------------------------------------------------------------------------- #
# The API's evaluation summary: every number must come from a saved result file
# --------------------------------------------------------------------------- #
def _summary() -> dict:
    from api.app import evaluation

    return evaluation()["summary"]


def test_the_evaluation_summary_numbers_match_the_saved_results():
    """The numbers the API reports are held to the result file they come from."""
    stored = json.loads(config.CHECKER_EVAL_PATH.read_text(encoding="utf-8"))
    reported = _summary()["grounding_checker"]

    for key in ("precision", "recall", "f1", "cases"):
        assert reported[key] == stored[key], key
    assert "known_defect" in reported


@pytest.mark.skipif(not config.FAITHFULNESS_PATH.exists(),
                    reason="evaluation/faithfulness_results.json is not in this checkout")
def test_the_faithfulness_summary_matches_the_saved_results():
    stored = json.loads(config.FAITHFULNESS_PATH.read_text(encoding="utf-8"))
    reported = _summary()["faithfulness"]

    assert reported["set_size"] == stored["set_size"]
    for key, value in stored["summary"].items():
        assert reported[key] == value, key

# --------------------------------------------------------------------------- #
# Grounding-checker evaluation (offline, deterministic)
# --------------------------------------------------------------------------- #
def test_checker_eval_labels_are_exhaustive():
    """Every extracted number in every case must be labelled.

    score_case() raises otherwise; this asserts the whole set passes that gate,
    so the measurement cannot drift behind the fixtures.
    """
    for case in checker_eval.CASES:
        checker_eval.score_case(case)      # raises on an unlabelled token


def test_checker_eval_is_reproducible_and_matches_the_checked_in_result():
    live = checker_eval.evaluate()
    stored = json.loads(config.CHECKER_EVAL_PATH.read_text(encoding="utf-8"))

    for key in ("precision", "recall", "f1", "true_positives", "false_positives",
                "false_negatives", "true_negatives", "cases"):
        assert live[key] == stored[key], (
            f"{key} drifted: recomputed {live[key]} vs checked-in {stored[key]} "
            "— re-run `python -m evaluation.checker_eval`")


def test_checker_eval_covers_the_failure_modes_that_actually_happened():
    ids = {c.id for c in checker_eval.CASES}
    for required in ("mckee_band_recitation",     # the real Phase-3 fabrication
                     "content_block_leakage",     # the Phase-3.1 bug's signature
                     "number_fragment_attack",    # boundary matching
                     "invented_yield_percentage"):  # the Phase-4 failure mode
        assert required in ids, f"the labelled set must cover {required}"


def test_checker_has_no_false_positives_on_this_set():
    """Precision 1.0 is load-bearing: a checker that cries wolf gets ignored."""
    result = checker_eval.evaluate()
    assert result["false_positives"] == 0
    assert result["precision"] == 1.0


def test_the_known_false_negative_is_recorded_not_hidden():
    """The percent/fraction defect must stay documented and stay failing.

    If someone fixes the checker, this test fails loudly — which is correct: the
    fix must come with a re-measurement and an update of the saved results, not silently.
    """
    assert "PERCENT_FRACTION_DEFECT" in dir(checker_eval)
    assert "12%" in checker_eval.PERCENT_FRACTION_DEFECT

    result = checker_eval.evaluate()
    failing = [c for c in result["per_case"] if not c["passed"]]
    assert len(failing) == 1
    assert failing[0]["id"] == "invented_yield_percentage"
    assert failing[0]["missed"] == ["12%"]


def test_the_summary_reports_the_defect_rather_than_only_the_headline():
    defect = _summary()["grounding_checker"]["known_defect"]
    assert "percentage" in defect
    assert "not yet fixed" in defect.lower()


# --------------------------------------------------------------------------- #
# Faithfulness set and results
# --------------------------------------------------------------------------- #
def test_held_out_set_was_written_before_results_and_states_its_size_rationale():
    spec = checker_eval.json.loads(
        config.REQUESTS_PATH.read_text(encoding="utf-8"))

    assert spec["written_before_results"] is True
    assert spec["size_rationale"].strip()
    assert spec["known_coverage_gap"].strip()
    assert len(spec["requests"]) == config.HELD_OUT_SET_SIZE
    for item in spec["requests"]:
        assert item["targets"], "every item must name the tools it should route to"
        assert item["what_a_faithful_report_must_do"].strip()


def test_held_out_set_includes_an_adversarial_item():
    spec = json.loads(config.REQUESTS_PATH.read_text(encoding="utf-8"))
    assert any("impossible" in i["id"] or "unsupported" in i["why_this_item"].lower()
               for i in spec["requests"])


def test_quota_budget_was_stated_before_the_set_was_sized():
    """The set size must actually fit the budget it claims to fit."""
    worst = config.HELD_OUT_SET_SIZE * config.CALLS_PER_ITEM_WORST
    assert worst <= config.DAILY_REQUEST_BUDGET, (
        f"{config.HELD_OUT_SET_SIZE} items x {config.CALLS_PER_ITEM_WORST} calls "
        f"= {worst} exceeds the {config.DAILY_REQUEST_BUDGET}/day budget")


@pytest.mark.skipif(not config.FAITHFULNESS_PATH.exists(),
                    reason="run `python -m evaluation.faithfulness` first")
def test_faithfulness_results_are_complete_and_labelled_lower_trust():
    result = json.loads(config.FAITHFULNESS_PATH.read_text(encoding="utf-8"))

    assert result["set_size"] == config.HELD_OUT_SET_SIZE
    assert "lower-trust" in result["trust_note"]
    assert "mechanical checker wins" in result["trust_note"]

    summary = result["summary"]
    assert summary["items_scored"] == summary["items_run"], \
        "an item whose judge output could not be parsed must not be silently dropped"
    assert 0.0 <= summary["mean_faithfulness"] <= 1.0
    assert 0.0 <= summary["mean_relevance"] <= 1.0


@pytest.mark.skipif(not config.FAITHFULNESS_PATH.exists(),
                    reason="run `python -m evaluation.faithfulness` first")
def test_mechanical_grounding_was_clean_on_every_judged_report():
    """The high-trust check must pass even where the soft one has quibbles."""
    result = json.loads(config.FAITHFULNESS_PATH.read_text(encoding="utf-8"))
    for item in result["items"]:
        grounding = item["mechanical_grounding"]
        assert grounding.get("grounded") is True, (
            f"{item['id']} was not mechanically grounded: "
            f"{grounding.get('unverified_numbers')}")
        assert grounding.get("total_checked", 0) > 0, \
            f"{item['id']} passed over zero numbers — a vacuous pass"


# --------------------------------------------------------------------------- #
# The judge is the documented exception, and it stays documented
# --------------------------------------------------------------------------- #
def test_the_llm_judge_exception_is_explicit_in_code():
    import inspect

    from evaluation import faithfulness
    source = inspect.getsource(faithfulness)
    assert "one place in the entire project where an LLM judges an LLM" in source
    assert "lower trust" in source or "lower-trust" in source


@pytest.mark.skipif(not config.FAITHFULNESS_PATH.exists(),
                    reason="evaluation/faithfulness_results.json is not in this checkout")
def test_the_llm_judge_exception_is_explicit_in_the_summary():
    assert "mechanical checker wins" in _summary()["faithfulness"]["trust_note"]


def test_judge_prompt_is_checked_in_and_forbids_authoring():
    text = config.JUDGE_PROMPT_PATH.read_text(encoding="utf-8")
    assert "You are a grader, not an author" in text
    assert "unsupported" in text
