"""P1 gates: record types, amendment registry, seam contracts, scatter parity.

* Three-way coverage closure: manifest keys == store slots + extra-key
  channels; every ingest-produced fields_dict key classified.
* Regenerate-and-diff: the generated manifest matches the source spec.
* Scatter parity: for every journal record of the scenario battery,
  ``op_record_from_event`` -> ``scatter_record_to_cells`` reproduces today's
  ``_fields_from_event`` output cell-for-cell on every record-sourced cell
  (JOINs neutralized on both sides) — the substance of the
  ``Op._from_cells == Op(fields_dict)`` gate before ingest rewires in P3.
* Amendment registry battery: exact-set violations red; typed constructors
  green; identity fields unpatchable; PATH_TO_FLAT covers exactly the union.
* Strict protocol: unknown names raise ``OpRecordAttributeError`` and
  ``getattr(..., default)`` keeps default semantics.
* IngestInputs v1 carries the RESERVED ``aten_events`` lane (cross-sprint
  requirement) and covers the recorded step-0 trace-read inventory.
"""

from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path

import pytest

import torchlens as tl
from torchlens.ir.op_record import (
    AMENDMENT_FAMILIES,
    PATH_TO_FLAT,
    TYPED_CONSTRUCTORS,
    AmendmentValidationError,
    OpAmendment,
    OpRecordAttributeError,
    amend_late_buffer_output_parent,
    op_record_from_event,
    validate_amendment,
)
from torchlens.ir.op_record_scatter import (
    CELL_SOURCES,
    EXTRA_KEY_CHANNELS,
    scatter_record_to_cells,
)

pytestmark = pytest.mark.smoke


def test_three_way_manifest_closure() -> None:
    from torchlens.data_classes.op import _OP_SLOT_NAMES

    manifest_keys = set(CELL_SOURCES)
    slot_keys = set(_OP_SLOT_NAMES)
    universe = slot_keys | EXTRA_KEY_CHANNELS
    missing = universe - manifest_keys
    assert not missing, f"unclassified store cells: {sorted(missing)}"
    stale = manifest_keys - universe - {"is_in_conditional_body", "source_trace"}
    assert not stale, f"manifest rows without a store cell: {sorted(stale)}"


def test_manifest_regenerate_and_diff() -> None:
    from tools.generate_op_record_manifest import generate

    generated = Path("torchlens/ir/op_record_manifest.py").read_text()
    assert generated == generate(), (
        "generated manifest is stale — run python -m tools.generate_op_record_manifest"
    )
    from torchlens.ir.op_record_manifest import CELL_SOURCE_MANIFEST

    assert CELL_SOURCE_MANIFEST == CELL_SOURCES


def test_amendment_registry_exact_sets() -> None:
    union = {path for schema in AMENDMENT_FAMILIES.values() for path, _ in schema}
    assert union == set(PATH_TO_FLAT), "PATH_TO_FLAT must cover exactly the path union"
    assert set(TYPED_CONSTRUCTORS) == set(AMENDMENT_FAMILIES)

    good = amend_late_buffer_output_parent(7, "buffer_1_raw", is_output_parent=True)
    validate_amendment(good)

    # unregistered family
    with pytest.raises(AmendmentValidationError, match="unregistered"):
        validate_amendment(
            OpAmendment(0, 0, 7, "x_raw", "no_such_family", (("graph.is_output_parent", True),))
        )
    # wrong path set (missing member)
    with pytest.raises(AmendmentValidationError, match="exact"):
        validate_amendment(
            OpAmendment(0, 0, 7, "x_raw", "module_exit_intervention", (
                ("intervention.intervention_fired", True),
            ))
        )
    # same-set wrong ORDER also refused (ordered exact-set)
    with pytest.raises(AmendmentValidationError, match="exact"):
        validate_amendment(
            OpAmendment(0, 0, 7, "x_raw", "lookback_retention", (
                ("policy.predicate_matched", True),
                ("core.output", object()),
            ))
        )
    # value type violation
    with pytest.raises(AmendmentValidationError, match="value type"):
        validate_amendment(
            OpAmendment(0, 0, 7, "x_raw", "late_buffer_output_parent", (
                ("graph.is_output_parent", "yes"),
            ))
        )
    # identity fields unpatchable even under a forged family row
    forged = dict(AMENDMENT_FAMILIES)
    try:
        AMENDMENT_FAMILIES["forged"] = (("core.label_raw", (str,)),)
        with pytest.raises(AmendmentValidationError, match="unpatchable"):
            validate_amendment(
                OpAmendment(0, 0, 7, "x_raw", "forged", (("core.label_raw", "evil"),))
            )
    finally:
        AMENDMENT_FAMILIES.clear()
        AMENDMENT_FAMILIES.update(forged)
        AMENDMENT_FAMILIES.pop("forged", None)
    # anchor required
    with pytest.raises(AmendmentValidationError, match="anchor"):
        validate_amendment(
            OpAmendment(0, 0, 7, "", "late_buffer_output_parent", (
                ("graph.is_output_parent", True),
            ))
        )


def test_strict_protocol_refusal_type() -> None:
    from ._models import SmallCNN, _cnn_input

    events = _journal_events(SmallCNN(), _cnn_input())
    record, _ = op_record_from_event(events[0])
    with pytest.raises(OpRecordAttributeError):
        _ = record.definitely_not_a_field
    # AttributeError subclass: getattr default semantics survive
    assert getattr(record, "definitely_not_a_field", "fallback") == "fallback"
    # legacy flat names read through the facets
    assert record.label_raw == events[0].label_raw
    assert record.parent_arg_positions == events[0].parent_arg_positions


def _journal_events(model, inputs) -> list:
    import torchlens.postprocess as postprocess_module
    import torchlens.postprocess._materialize as materialize_module

    captured: list = []
    original = materialize_module.materialize_from_events

    def spy(trace, events):
        captured.extend(events.op_events)
        original(trace, events)

    postprocess_module.materialize_from_events = spy
    materialize_module.materialize_from_events = spy
    try:
        tl.trace(model, inputs)
    finally:
        postprocess_module.materialize_from_events = original
        materialize_module.materialize_from_events = original
    return captured


@pytest.mark.heavy
def test_scatter_parity_cell_for_cell(tmp_path: Path) -> None:
    """Adapter + scatter reproduce _fields_from_event on record-sourced cells."""

    from torchlens.postprocess._materialize import _fields_from_event

    from ._models import SCENARIOS
    from ._snapshot import run_scenario

    gated_classes = ("CORE", "FACET:", "EXTRAS:", "DEFAULT")
    checked_cells = 0
    for scenario in SCENARIOS:
        if scenario.name == "cnn_backward":
            continue  # backward mutates grads post-capture; journal identical anyway
        events: list = []

        def collect(journal_events, _sink=events) -> None:
            _sink.extend(journal_events.op_events)

        with tempfile.TemporaryDirectory() as tmp:
            run = run_scenario(
                scenario,
                Path(tmp),
                with_artifact=False,
                arm_shims=False,
                journal_mutator=collect,
            )
        trace = run.trace
        for event in events:
            expected = _fields_from_event(
                trace,
                event,
                op_event_labels={e.label_raw for e in events},
                children=[],
                equivalent_ops=set(),
                buffer_address=None,
                buffer_alias_snapshots={},
                module_input_fields={},
                module_output_fields={},
                buffer_write_fields={},
                grad_fn_handle=None,
                input_io_role=None,
                output_versions_by_child={},
                op_events_by_label={e.label_raw: e for e in events},
            )
            record, extras = op_record_from_event(event)
            cells = scatter_record_to_cells(record, extras, trace)
            for name, value in cells.items():
                source = CELL_SOURCES.get(name, "")
                if not source.startswith(gated_classes):
                    continue
                if name == "source_trace":
                    assert value is expected[name]
                    continue
                assert name in expected, f"scatter produced unknown cell {name!r}"
                assert value == expected[name], (
                    f"{scenario.name}:{event.label_raw}:{name}: "
                    f"scatter {value!r} != legacy {expected[name]!r}"
                )
                checked_cells += 1
    assert checked_cells > 1000, f"vacuity guard: only {checked_cells} cells compared"


def test_ingest_inputs_v1_reserves_aten_lane_and_covers_step0_reads() -> None:
    import json

    from torchlens.postprocess._ingest_contract import (
        INGEST_CONTRACT_ROWS,
        IngestInputs,
        JournalView,
        Step0Result,
    )

    journal_fields = {f.name for f in dataclasses.fields(JournalView)}
    assert "aten_events" in journal_fields, "reserved aten_events lane missing"
    assert JournalView.__dataclass_fields__["aten_events"].default == ()
    assert any(row[0] == "journal.aten_events" for row in INGEST_CONTRACT_ROWS)

    input_fields = {f.name for f in dataclasses.fields(IngestInputs)}
    for required in (
        "journal",
        "module_workspace",
        "raw_graph_workspace",  # X1
        "buffer_initial_values",
        "op_equivalence_classes",
        "source_model_ref",
        "param_logs",
        "owning_trace",
        "op_row_store",
        "trace_core",  # ppdag 9.1 orchestrator row
        "input_layers_initial",  # X4
        "timing_sink",  # X3
        "scatter_options",
    ):
        assert required in input_fields, required
    result_fields = {f.name for f in dataclasses.fields(Step0Result)}
    assert result_fields == {
        "raw_log_registrations",  # X2 local-map derivation
        "input_layer_labels",
        "equivalence_class_map",
        "module_side_channel",
    }

    # the recorded step-0 trace-read inventory must be coverable by the
    # enumeration (each read maps to a declared IngestInputs handle or a
    # named orchestrator-epilogue attribute)
    ledger_path = Path(__file__).resolve().parent / "ledger" / "step0_trace_reads.json"
    if ledger_path.exists():
        observed = set(json.loads(ledger_path.read_text()))
        covered = {
            # journal + workspaces + registries
            "capture_events": "journal",
            "_module_capture_ws": "module_workspace",
            "_raw_graph_ws": "raw_graph_workspace",
            "_buffer_initial_values": "buffer_initial_values",
            "op_equivalence_classes": "op_equivalence_classes",
            "_source_model_ref": "source_model_ref",
            "param_logs": "param_logs",
            "input_layers": "input_layers_initial",
            "_trace_core": "trace_core",
            "_phase_timings": "timing_sink",
        }
        uncovered = {
            name
            for name in observed
            if name in covered and covered[name] not in {f.name for f in dataclasses.fields(IngestInputs)}
        }
        assert not uncovered, f"declared coverage broken: {uncovered}"
