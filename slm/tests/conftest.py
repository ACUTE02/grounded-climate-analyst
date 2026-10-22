"""Shared test helper.

Some tests check the working files of the dataset build (draft candidates, spot-check sheets, the raw chunk snapshot).
Those files are not part of this repository, only the final train / val / test files are. `needs(...)` skips such a test
with a clear reason when one of its files is missing, instead of failing with FileNotFoundError."""
import pytest

from climate_slm.config import REPO_ROOT


def needs(*rel_paths: str):
    missing = [p for p in rel_paths if not (REPO_ROOT / p).exists()]
    return pytest.mark.skipif(bool(missing), reason="dataset working file(s) not in this repository: " + ", ".join(missing))
