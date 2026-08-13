"""Consumer-ledger generation and closure gates (P0).

Regenerates the ledger from the tree and the scenario battery every run and
asserts the closure properties; the generated inventory is written next to
this test (git-tracked) so P2 has a reviewable migration checklist and any
drift shows up as a diff.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

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
    "torchlens/backends/torch/ops.py",  # lookback_retention
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


def test_generate_and_close_ledger(tmp_path: Path) -> None:
    """Generate all ledger artifacts; assert the closure properties."""

    _LEDGER_DIR.mkdir(exist_ok=True)

    # ---- source 1: static scan (with getattr default-reliers) -------------
    sites = static_scan(_PACKAGE_ROOT)
    static_fields = {site.field for site in sites}
    (_LEDGER_DIR / "static_scan.json").write_text(
        json.dumps(
            [site.__dict__ for site in sites],
            indent=0,
            sort_keys=True,
        )
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
    (_LEDGER_DIR / "runtime_reads.json").write_text(
        json.dumps(
            {field: sorted(callers) for field, callers in sorted(reads.items())},
            indent=0,
        )
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

    with step0_trace_read_recorder() as observed:
        with tempfile.TemporaryDirectory() as tmp:
            run_scenario(
                scenario_by_name("cnn_exhaustive"),
                Path(tmp),
                with_artifact=False,
            )
    assert observed, "step-0 recorder observed nothing (hook broken)"
    (_LEDGER_DIR / "step0_trace_reads.json").write_text(
        json.dumps(sorted(observed), indent=0)
    )

    # ---- source 4: mutator inventory (exact) --------------------------------
    mutators = mutator_inventory(_PACKAGE_ROOT)
    (_LEDGER_DIR / "mutators.json").write_text(json.dumps(mutators, indent=1, sort_keys=True))

    amendment_files = {
        site.rsplit(":", 1)[0] for site in mutators["append_amendment_callers"]
    }
    assert amendment_files == EXPECTED_APPEND_AMENDMENT_CALLER_FILES, (
        "append_amendment caller set drifted — a new post-commit mutation "
        "channel needs a registry family: "
        f"{amendment_files ^ EXPECTED_APPEND_AMENDMENT_CALLER_FILES}"
    )
    # Legacy channels stay dead: no replace_op_event callers outside its
    # definition module, no in-place op_events[i] list writes anywhere.
    legacy_callers = {
        site
        for site in mutators["replace_op_event_callers"]
        if "ir/capture_events.py" not in site
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
        "in-place op_events[i] writes resurfaced after the P4 migration: "
        f"{inplace_writes}"
    )


def test_all_54_fields_classified_for_migration() -> None:
    """Sanity: the OpEvent field universe is exactly the documented 54."""

    assert len(OPEVENT_FIELDS) == 54
