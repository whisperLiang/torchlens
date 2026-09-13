"""Shared test helpers must not depend on owning the generic ``tests`` package."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("filename", "helper_module", "helper_name"),
    [
        ("test_distributed_census_skeleton.py", "support.census_harness", "run_census_criterion_1"),
        ("test_distributed_census_topologies.py", "support.census_harness", "run_census_row"),
        ("test_merged_save_durability.py", "test_merged_load_identity", "_rank_trace"),
    ],
)
def test_helper_imports_ignore_an_installed_tests_package(
    monkeypatch: pytest.MonkeyPatch, filename: str, helper_module: str, helper_name: str
) -> None:
    """Load test suites while an unrelated installed ``tests`` owns that name.

    Parameters
    ----------
    monkeypatch:
        Restores the import cache after simulating the foreign package.
    filename:
        Test suite whose module-level imports must remain collectable.
    helper_module:
        Repository module that exports the shared helper.
    helper_name:
        Imported helper that must resolve to this repository's support package.
    """

    foreign_tests = ModuleType("tests")
    foreign_tests.__path__ = []
    for name in tuple(sys.modules):
        if name == "tests" or name.startswith("tests."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "tests", foreign_tests)

    spec = importlib.util.spec_from_file_location(
        "_suite_import_regression", Path(__file__).with_name(filename)
    )
    assert spec is not None and spec.loader is not None
    suite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(suite)

    assert getattr(suite, helper_name) is getattr(
        importlib.import_module(helper_module), helper_name
    )
    assert sys.modules["tests"] is foreign_tests
