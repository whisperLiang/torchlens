"""Consumer-ledger generation and closure gates (P0).

Regenerates the ledger from the tree and the scenario battery every run,
asserts the closure properties, and DIFFS the regenerated inventory against the
git-tracked artifacts next to this test, so P2 keeps a reviewable migration
checklist and producer/consumer drift is a red test rather than a silent
rewrite.

Finding B1-14: the gate used to ``write_text`` all four inventories and then
assert only set/closure properties, so a run was green *while overwriting the
tracked files*. The checked-in content had rotted accordingly -- ``runtime_reads
.json`` recorded absolute caller paths from a long-deleted worktree, and every
location was stale by thousands of lines -- and CI could not distinguish a
reviewed ledger refresh from unreviewed drift.

Two halves fix it:

1. **Normalized artifacts.** The tracked payloads are machine-independent and
   stable under unrelated edits: locations are repo-relative, line numbers are
   replaced by a per-``(file, ...)`` SITE COUNT (so a new site in an
   already-listed file is still drift), and callers outside the repo collapse to
   ``<external>`` (a venv path and its torch-version line numbers are not a
   producer fact). Exact lines stay greppable from the file+field pair.
2. **Compare, never write.** A normal run only compares. Refreshing is an
   explicit, reviewed act behind :data:`_REFRESH_ENV`.
"""

from __future__ import annotations

import difflib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from ._ledger import (
    OPEVENT_FIELDS,
    mutator_inventory,
    runtime_read_recorder,
    static_scan,
    step0_trace_read_recorder,
)
from ._models import SCENARIOS
from ._snapshot import run_scenario

pytestmark = pytest.mark.heavy

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_ROOT = _REPO_ROOT / "torchlens"
_LEDGER_DIR = Path(__file__).resolve().parent / "ledger"

# The exact post-commit mutation channel set AFTER the P4 migration (DoR
# 4.1/4.9): the typed amendment lane is the ONE channel. Emit sites are the
# seven torch families' files plus the five preview backends; the journal's
# own concat transport re-appends merged amendments. The legacy channels
# (replace_op_event callers, in-place op_events[i] writes) must stay at ZERO.
# ANY change to this set is a reviewed diff here.
EXPECTED_APPEND_AMENDMENT_CALLER_FILES = {
    "torchlens/backends/torch/_ops_retention.py",  # lookback_retention
    "torchlens/user_funcs.py",  # graph_edge_insertion (register_tensor_connection)
    "torchlens/backends/torch/model_prep.py",  # raw hook / module exit / boundary retention
    "torchlens/backends/torch/backend.py",  # output_parent_promotion
    "torchlens/postprocess/graph_traversal.py",  # late_buffer_output_parent (pre-0)
    "torchlens/backends/tf/backend.py",  # preview_output_parent_mark x2
    "torchlens/backends/tf/interventions.py",  # module_exit_intervention (site fire)
    "torchlens/backends/mlx/backend.py",  # preview_output_parent_mark
    "torchlens/backends/paddle/backend.py",  # preview_output_parent_mark
    "torchlens/backends/jax/backend.py",  # preview_output_parent_rebind
    "torchlens/backends/tinygrad/backend.py",  # preview_output_parent_rebind
    "torchlens/ir/capture_events.py",  # concat lane transport (re-append)
}


#: Opt-in for the one sanctioned artifact refresh. A normal run NEVER writes.
_REFRESH_ENV = "TORCHLENS_REFRESH_PRODUCER_LEDGER"

#: Command the failure message hands the developer.
_REFRESH_COMMAND = (
    f"{_REFRESH_ENV}=1 pytest tests/producer_parity/test_ledger.py::test_generate_and_close_ledger"
)

#: Location stand-in for a caller outside the repository (venv / stdlib). Their
#: absolute paths and line numbers are environment facts, not producer facts.
_EXTERNAL = "<external>"


def normalize_repo_location(location: str, repo_root: Path) -> str:
    """Return a machine-independent, line-free form of one recorded location.

    Parameters
    ----------
    location:
        Recorded ``path`` or ``path:line`` string, absolute or repo-relative.
    repo_root:
        Repository root every in-tree path is expressed against.

    Returns
    -------
    str
        Repo-relative path for an in-tree location, else :data:`_EXTERNAL`.
    """

    path_text = location.rsplit(":", 1)[0] if ":" in location else location
    candidate = Path(path_text)
    if not candidate.is_absolute():
        return candidate.as_posix()
    try:
        return candidate.relative_to(repo_root).as_posix()
    except ValueError:
        return _EXTERNAL


def normalized_static_scan(sites: list[Any]) -> list[dict[str, Any]]:
    """Return the static inventory as sorted per-file/field site COUNTS.

    Parameters
    ----------
    sites:
        ``ReadSite`` records from :func:`static_scan`.

    Returns
    -------
    list[dict[str, Any]]
        One row per distinct ``(file, field, via_getattr, has_default,
        ambiguous)`` tuple with the number of sites, sorted. Dropping the line
        number keeps the artifact stable under unrelated edits; keeping the count
        keeps a NEW site in an already-listed file visible as drift.
    """

    counts = Counter(
        (site.file, site.field, site.via_getattr, site.has_default, site.ambiguous)
        for site in sites
    )
    return [
        {
            "file": file,
            "field": field,
            "via_getattr": via_getattr,
            "has_default": has_default,
            "ambiguous": ambiguous,
            "sites": total,
        }
        for (file, field, via_getattr, has_default, ambiguous), total in sorted(counts.items())
    ]


def normalized_runtime_reads(
    reads: dict[str, set[str]],
    repo_root: Path,
) -> dict[str, dict[str, int]]:
    """Return recorded runtime reads as field -> normalized caller -> count.

    Parameters
    ----------
    reads:
        Field -> ``{caller file:line}`` map from the runtime recorder.
    repo_root:
        Repository root for normalization.

    Returns
    -------
    dict[str, dict[str, int]]
        Sorted, machine-independent view. External callers collapse to
        :data:`_EXTERNAL` so a venv path or a torch-version line number cannot
        make the tracked artifact undiffable.
    """

    normalized: dict[str, dict[str, int]] = {}
    for field, callers in sorted(reads.items()):
        counts = Counter(normalize_repo_location(caller, repo_root) for caller in callers)
        normalized[field] = {caller: counts[caller] for caller in sorted(counts)}
    return normalized


def normalized_mutators(mutators: dict[str, list[str]]) -> dict[str, dict[str, int]]:
    """Return the mutator inventory as channel -> file -> site count.

    Parameters
    ----------
    mutators:
        Channel -> ``["file:line", ...]`` inventory.

    Returns
    -------
    dict[str, dict[str, int]]
        Sorted, line-free view of each mutation channel.
    """

    normalized: dict[str, dict[str, int]] = {}
    for channel, sites in sorted(mutators.items()):
        counts = Counter(site.rsplit(":", 1)[0] for site in sites)
        normalized[channel] = {file: counts[file] for file in sorted(counts)}
    return normalized


def artifact_payload(content: Any) -> str:
    """Return the canonical serialization every tracked artifact is stored in.

    Parameters
    ----------
    content:
        JSON-serializable normalized inventory.

    Returns
    -------
    str
        Deterministic, newline-terminated JSON text.
    """

    return json.dumps(content, indent=1, sort_keys=True) + "\n"


def assert_artifact_current(path: Path, payload: str) -> None:
    """Diff one tracked ledger artifact against freshly generated content.

    Writes ONLY under :data:`_REFRESH_ENV`; otherwise a mismatch (or a missing
    artifact) is a failure naming the refresh command, so an unreviewed
    producer/consumer change can no longer overwrite its own evidence.

    Parameters
    ----------
    path:
        Tracked artifact path.
    payload:
        Freshly generated canonical content.
    """

    if os.environ.get(_REFRESH_ENV):
        path.write_text(payload, encoding="utf-8")
        return
    checked_in = path.read_text(encoding="utf-8") if path.exists() else ""
    if checked_in == payload:
        return
    diff = list(
        difflib.unified_diff(
            checked_in.splitlines(),
            payload.splitlines(),
            fromfile=f"{path.name} (checked in)",
            tofile=f"{path.name} (regenerated)",
            lineterm="",
            n=1,
        )
    )
    shown = "\n".join(diff[:60])
    truncated = "" if len(diff) <= 60 else f"\n... {len(diff) - 60} more diff lines"
    try:
        display = path.relative_to(_REPO_ROOT).as_posix()
    except ValueError:  # a self-test artifact under tmp_path
        display = path.name
    raise AssertionError(
        f"{display} does not match the regenerated ledger. "
        "Either a producer/consumer site moved without review, or the artifact "
        f"needs a reviewed refresh: {_REFRESH_COMMAND}\n{shown}{truncated}"
    )


def test_generate_and_close_ledger(tmp_path: Path) -> None:
    """Regenerate every ledger artifact, diff it, and assert the closure properties."""

    _LEDGER_DIR.mkdir(exist_ok=True)

    # ---- source 1: static scan (with getattr default-reliers) -------------
    sites = static_scan(_PACKAGE_ROOT)
    static_fields = {site.field for site in sites}
    assert_artifact_current(
        _LEDGER_DIR / "static_scan.json",
        artifact_payload(normalized_static_scan(sites)),
    )
    # the pinned getattr default-relier examples (hashing.py) must be present
    hashing_defaults = [
        site
        for site in sites
        if site.file.endswith("utils/hashing.py") and site.via_getattr and site.has_default
    ]
    assert hashing_defaults, "pinned getattr default-reliers in utils/hashing.py not found"

    # ---- source 2: runtime instrumentation over the battery ----------------
    with runtime_read_recorder() as reads:
        for scenario in SCENARIOS:
            with tempfile.TemporaryDirectory() as tmp:
                run_scenario(scenario, Path(tmp), with_artifact=False)
    assert_artifact_current(
        _LEDGER_DIR / "runtime_reads.json",
        artifact_payload(normalized_runtime_reads(reads, _REPO_ROOT)),
    )
    # Closure: every field read at runtime FROM INSIDE torchlens/ appears in
    # the static inventory. Harness-internal and stdlib-dataclasses machinery
    # reads are not consumer sites; a torchlens-internal read with no static
    # site would be string dispatch the migration must not miss.
    internal_marker = str(_PACKAGE_ROOT)
    internal_fields = {
        field
        for field, callers in reads.items()
        if any(caller.startswith(internal_marker) for caller in callers)
    }
    unexplained = internal_fields - static_fields
    assert not unexplained, (
        f"torchlens-internal runtime reads with no static site (string dispatch "
        f"closure hole): {unexplained}"
    )

    # ---- step-0 trace-read recorder (feeds the IngestInputs v1 freeze) -----
    from ._models import scenario_by_name

    with step0_trace_read_recorder() as observed, tempfile.TemporaryDirectory() as tmp:
        run_scenario(
            scenario_by_name("cnn_exhaustive"),
            Path(tmp),
            with_artifact=False,
        )
    assert observed, "step-0 recorder observed nothing (hook broken)"
    assert_artifact_current(
        _LEDGER_DIR / "step0_trace_reads.json",
        artifact_payload(sorted(observed)),
    )

    # ---- source 4: mutator inventory (exact) --------------------------------
    mutators = mutator_inventory(_PACKAGE_ROOT)
    assert_artifact_current(
        _LEDGER_DIR / "mutators.json",
        artifact_payload(normalized_mutators(mutators)),
    )

    amendment_files = {site.rsplit(":", 1)[0] for site in mutators["append_amendment_callers"]}
    assert amendment_files == EXPECTED_APPEND_AMENDMENT_CALLER_FILES, (
        "append_amendment caller set drifted — a new post-commit mutation "
        "channel needs a registry family: "
        f"{amendment_files ^ EXPECTED_APPEND_AMENDMENT_CALLER_FILES}"
    )
    # Legacy channels stay dead: no replace_op_event callers outside its
    # definition module, no in-place op_events[i] list writes anywhere.
    legacy_callers = {
        site for site in mutators["replace_op_event_callers"] if "ir/capture_events.py" not in site
    }
    assert not legacy_callers, (
        f"replace_op_event callers resurfaced after the P4 migration: {legacy_callers}"
    )
    inplace_writes = {
        site
        for site in mutators["op_events_inplace_writes"]
        if "ir/capture_events.py" not in site  # replace_op_event's own body, deleted in P4
    }
    assert not inplace_writes, (
        f"in-place op_events[i] writes resurfaced after the P4 migration: {inplace_writes}"
    )


def test_all_54_fields_classified_for_migration() -> None:
    """Sanity: the OpEvent field universe is exactly the documented 54."""

    assert len(OPEVENT_FIELDS) == 54


class TestLedgerDiffGateIsRedCapable:
    """The B1-14 mechanism itself: prove the comparison can fail (and refuse to write)."""

    @pytest.fixture(autouse=True)
    def _no_refresh(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Run these checks in normal (compare-only) mode."""

        monkeypatch.delenv(_REFRESH_ENV, raising=False)

    def test_matching_artifact_passes_and_is_not_rewritten(self, tmp_path: Path) -> None:
        """An up-to-date artifact is left byte-identical on disk."""

        artifact = tmp_path / "ledger.json"
        payload = artifact_payload({"a": 1})
        artifact.write_text(payload, encoding="utf-8")
        stat_before = artifact.stat().st_mtime_ns
        assert_artifact_current(artifact, payload)
        assert artifact.read_text(encoding="utf-8") == payload
        assert artifact.stat().st_mtime_ns == stat_before

    def test_drifted_artifact_fails_instead_of_being_overwritten(self, tmp_path: Path) -> None:
        """The pre-fix behavior (silent rewrite + green) is now impossible."""

        artifact = tmp_path / "ledger.json"
        stale = artifact_payload({"torchlens/a.py": 1})
        artifact.write_text(stale, encoding="utf-8")
        with pytest.raises(AssertionError, match=_REFRESH_ENV):
            assert_artifact_current(artifact, artifact_payload({"torchlens/a.py": 2}))
        assert artifact.read_text(encoding="utf-8") == stale

    def test_missing_artifact_fails(self, tmp_path: Path) -> None:
        """A deleted inventory cannot pass by regenerating itself."""

        with pytest.raises(AssertionError):
            assert_artifact_current(tmp_path / "absent.json", artifact_payload([]))

    def test_refresh_env_writes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The sanctioned refresh path still updates the artifact."""

        monkeypatch.setenv(_REFRESH_ENV, "1")
        artifact = tmp_path / "ledger.json"
        payload = artifact_payload({"b": 2})
        assert_artifact_current(artifact, payload)
        assert artifact.read_text(encoding="utf-8") == payload

    def test_an_added_site_in_a_listed_file_is_drift(self) -> None:
        """Line numbers are dropped, but site COUNTS keep a new site visible."""

        class _Site:
            def __init__(self, line: int) -> None:
                self.file = "torchlens/a.py"
                self.line = line
                self.field = "func_name"
                self.via_getattr = False
                self.has_default = False
                self.ambiguous = False

        one = normalized_static_scan([_Site(10)])
        moved = normalized_static_scan([_Site(99)])
        added = normalized_static_scan([_Site(10), _Site(11)])
        assert one == moved, "a pure line shift must not be reported as drift"
        assert one != added, "a NEW read site in a listed file must be drift"

    def test_locations_are_normalized_machine_independently(self) -> None:
        """Repo paths become relative; foreign paths collapse to one token."""

        repo_root = Path("/repo")
        assert normalize_repo_location("/repo/torchlens/a.py:12", repo_root) == "torchlens/a.py"
        assert normalize_repo_location("torchlens/a.py:12", repo_root) == "torchlens/a.py"
        assert (
            normalize_repo_location("/somewhere/else/site-packages/torch/_tensor.py:623", repo_root)
            == _EXTERNAL
        )
