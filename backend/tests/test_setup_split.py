"""scripts.setup must never write the tracked evidence, and readers must prefer
what it generated.

Frozen = the tracked evidence files in models/ (metrics_t1_ridge.json,
horizon_comparison.json, horizon_manifest.json). Generated = models/generated/,
written by t1_model.run and recursive.run (the steps scripts.setup calls) and
gitignored.
"""

import inspect

from fastapi.testclient import TestClient

from forecasting import config, iod_check, recursive, rolling_check, t1_model
from heat import model as heat_model


def test_manifest_path_prefers_generated_then_falls_back_to_frozen(monkeypatch, tmp_path):
    frozen = tmp_path / "frozen_manifest.json"
    generated = tmp_path / "generated_manifest.json"
    monkeypatch.setattr(config, "HORIZON_MANIFEST_PATH", frozen)
    monkeypatch.setattr(config, "HORIZON_MANIFEST_GENERATED_PATH", generated)

    assert config.manifest_path() == frozen
    assert config.manifest_source() == "frozen"

    generated.write_text("{}", encoding="utf-8")
    assert config.manifest_path() == generated
    assert config.manifest_source() == "generated"


def test_t1_metrics_path_prefers_generated_then_falls_back_to_frozen(monkeypatch, tmp_path):
    frozen = tmp_path / "frozen_t1.json"
    generated = tmp_path / "generated_t1.json"
    monkeypatch.setattr(config, "T1_METRICS_PATH", frozen)
    monkeypatch.setattr(config, "T1_METRICS_GENERATED_PATH", generated)

    assert config.t1_metrics_path() == frozen
    generated.write_text("{}", encoding="utf-8")
    assert config.t1_metrics_path() == generated


def test_generated_paths_live_under_models_generated_and_frozen_ones_do_not():
    for path in (config.T1_METRICS_GENERATED_PATH,
                 config.HORIZON_COMPARISON_GENERATED_PATH,
                 config.HORIZON_MANIFEST_GENERATED_PATH):
        assert path.parent == config.GENERATED_DIR
    for path in (config.T1_METRICS_PATH, config.HORIZON_COMPARISON_PATH,
                 config.HORIZON_MANIFEST_PATH):
        assert path.parent == config.MODELS_DIR


def test_the_setup_steps_never_write_the_frozen_evidence():
    """t1_model.run and recursive.run are what scripts.setup calls."""
    for module, frozen_names in ((t1_model, ("T1_METRICS_PATH",)),
                                 (recursive, ("HORIZON_COMPARISON_PATH",
                                              "HORIZON_MANIFEST_PATH"))):
        source = inspect.getsource(module)
        for name in frozen_names:
            assert f"config.{name}.write_text" not in source, (
                f"{module.__name__} writes the frozen {name}")


def test_health_says_which_manifest_is_used(monkeypatch, tmp_path):
    from api.app import app

    frozen = tmp_path / "horizon_manifest.json"
    frozen.write_text("{}", encoding="utf-8")
    generated_dir = tmp_path / "generated"
    generated = generated_dir / "horizon_manifest.json"
    monkeypatch.setattr(config, "HORIZON_MANIFEST_PATH", frozen)
    monkeypatch.setattr(config, "HORIZON_MANIFEST_GENERATED_PATH", generated)
    client = TestClient(app)

    assert client.get("/health").json()["horizon_manifest_source"] == "frozen"

    generated_dir.mkdir()
    generated.write_text("{}", encoding="utf-8")
    assert client.get("/health").json()["horizon_manifest_source"] == "generated"


def test_manual_checks_write_only_under_models_generated():
    """rolling_check, iod_check and heat.model never write the tracked evidence."""
    frozen_writers = {
        rolling_check: ("ROLLING_CHECK_PATH",),
        iod_check: ("IOD_COMPARISON_PATH",),
        heat_model: ("HEAT_COMPARISON_PATH", "HEAT_MANIFEST_PATH", "heat_metrics_path"),
    }
    for module, names in frozen_writers.items():
        source = inspect.getsource(module)
        for name in names:
            assert f"config.{name}.write_text" not in source, (
                f"{module.__name__} writes the frozen {name}")
            assert f"config.{name}(region).write_text" not in source, (
                f"{module.__name__} writes the frozen {name}")
    for path in (config.ROLLING_CHECK_GENERATED_PATH,
                 config.IOD_COMPARISON_GENERATED_PATH,
                 config.HEAT_COMPARISON_GENERATED_PATH,
                 config.HEAT_MANIFEST_GENERATED_PATH,
                 config.heat_metrics_generated_path("rajasthan")):
        assert path.parent == config.GENERATED_DIR


def test_rolling_and_iod_readers_prefer_generated_then_fall_back_to_frozen(
        monkeypatch, tmp_path):
    for frozen_name, generated_name, getter in (
            ("ROLLING_CHECK_PATH", "ROLLING_CHECK_GENERATED_PATH", config.rolling_check_path),
            ("IOD_COMPARISON_PATH", "IOD_COMPARISON_GENERATED_PATH", config.iod_comparison_path)):
        frozen = tmp_path / f"frozen_{frozen_name}.json"
        generated = tmp_path / f"generated_{frozen_name}.json"
        monkeypatch.setattr(config, frozen_name, frozen)
        monkeypatch.setattr(config, generated_name, generated)
        assert getter() == frozen
        generated.write_text("{}", encoding="utf-8")
        assert getter() == generated
