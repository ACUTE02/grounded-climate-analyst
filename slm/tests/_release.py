"""Skip rules for files that are not part of this release.

The three SLM training data files (and their .sha256 files) and the SLM README are left out of the public repository, so the tests
that read them are skipped when the file is missing and run as before when it is present.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TRAINING_DATA = [ROOT / "data" / "instruction" / n for n in (
    "sft_train.jsonl", "sft_train.sha256", "dpo_train.jsonl", "dpo_train.sha256", "dpo2_mix.jsonl", "dpo2_mix.sha256")]

needs_training_data = pytest.mark.skipif(not all(p.exists() for p in TRAINING_DATA), reason="training data not included in this release")
needs_readme = pytest.mark.skipif(not (ROOT / "README.md").exists(), reason="the SLM README is not included in this release")
