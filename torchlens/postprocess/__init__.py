"""Postprocessing pipeline for cleaning up the model log after the forward pass.

After the forward pass captures raw tensor metadata into a Trace, this pipeline
transforms the raw graph into its user-facing form. The full pipeline runs
stable contract steps 0-20, split into thematic submodules:

- graph_traversal (Steps 1-4): Add output nodes, trace ancestry, remove orphans,
  compute input/output distances.
- control_flow (Steps 5-6): Mark conditional branches and deduplicate/merge buffer layers.
- loop_detection (Step 7): Identify repeated operations (loops/recurrence), assign
  same-layer groupings via BFS isomorphic subgraph expansion.
- labeling (Steps 8-11): Generate final human-readable labels, rename all internal
  references, trim/reorder fields, build lookup keys, and finalize retained layer lists.
- finalization (Steps 12-20): Undecorate saved tensors, log timing, finalize
  ParamLogs, build Layer/Module aggregates, mark pass as finished, then
  finalize any streamed bundle, optionally evict in-memory outs, and release
  live parameter references.

Step ordering invariants:
- Steps 1-3 MUST precede Step 5 (conditional branch detection needs orphan-free graph).
- Module suffixes must be present on equivalence_class before Step 7 loop detection.
- Step 7 MUST precede Step 8 (label generation needs recurrent_ops).
- Step 8 MUST precede Step 9 (final info logging uses finalized labels).
- Step 9 MUST precede Step 11 (lookup key generation needs module hierarchy data
  populated in Step 9).
- Step 10 (rename) MUST precede Step 11 (lookup keys use renamed labels).
- Step 15.5 (_build_layer_logs) MUST precede Step 16 (_build_module_logs) because
  Module.layers references Layer keys.

"""

from dataclasses import dataclass
import os
from typing import TYPE_CHECKING, List

import time
import torch
import warnings

from ..capture.session import capture_session_for_events
from ..ir.capture_events import _clone_op_event_for_replay
from ..backends.torch.ops import _compact_ancestor_sets
from ..data_classes._compaction import compact_op_metadata as _compact_op_metadata
from .._trace_core.relation_views import freeze_trace_relation_views as _freeze_relation_views
from ..utils.tensor_utils import _is_cuda_available
from ..utils.hashing import (
    compute_graph_shape_hash,
    compute_raw_event_shape_hash,
    populate_normalized_layer_addresses,
)

from .control_flow import (
    _fix_buffer_layers,
    _mark_conditional_branches,
)
from .finalization import (
    _build_layer_logs,
    _build_module_logs,
    _evict_streamed_outs,
    _finalize_streamed_bundle,
    _finalize_param_logs,
    _log_time_elapsed,
    _set_tracing_finished,
    _undecorate_all_saved_tensors,
)
from .graph_traversal import (
    _add_output_layers,
    _find_output_ancestors,
    _mark_layer_depths,
    _remove_orphan_nodes,
    _resolve_output_parent_labels,
)
from .labeling import (
    _log_final_info_for_layers,
    _map_raw_labels_to_final_labels,
    _build_lookup_keys_and_finalize_retained_layers,
    _rename_model_history_layer_names,
)
from .loop_detection import _detect_and_label_loops, _group_by_shared_params
from .loop_grouping_adapter import (
    RecurrenceAssignment,
    RecurrenceGroupingGraph,
    RecurrenceNode,
    group_recurrent_nodes,
)
from .saved_summary import refresh_saved_module_call_count
from ._materialize import materialize_from_events
from .ast_branches import resolve_var_names

if TYPE_CHECKING:
    from ..data_classes.trace import Trace

from ..quantities import Bytes

__all__ = [
    "RecurrenceAssignment",
    "RecurrenceGroupingGraph",
    "RecurrenceNode",
    "group_recurrent_nodes",
    "postprocess",
]
from ..utils.display import _vprint, _vtimed


_POSTPROCESS_ASSERT_ENV = "TORCHLENS_POSTPROCESS_ASSERTIONS"


@dataclass(frozen=True)
class PostprocessStepContract:
    """Declared contract for one postprocess pipeline step.

    Parameters
    ----------
    step:
        Stable step identifier used by the pipeline.
    name:
        Human-readable step name.
    contract:
        Short consumes/produces/mutation contract.
    writes:
        Declared op-store COLUMN write set for this step (M10): the exact
        cell columns the step may write or delete, enforced under
        ``TORCHLENS_POSTPROCESS_ASSERTIONS`` by the zero-cost-when-off
        write audit (``op_store.begin_cell_write_audit``). ``None`` means
        undeclared (wildcard) — steps that legitimately touch the whole
        row (materialize, undecorate) or run before the store exists.
        A step writing an undeclared column fails the tripwire; widening
        a set is a REVIEWED schema-contract diff, never a silent drift.
    removes_rows:
        Whether the step is sanctioned to remove whole op rows (removal
        husking releases every cell of the row). Whole-row release is the
        row-lifecycle twin of row creation — audited separately from
        column writes, so an unsanctioned removal fails with a precise
        message instead of a wall of column names, and a sanctioned one
        (orphan removal) stops false-positively tripping the column
        tripwire on removal-heavy paths such as the fastlog cook.
    """

    step: str
    name: str
    contract: str
    writes: frozenset[str] | None = None
    removes_rows: bool = False


POSTPROCESS_STEP_CONTRACTS: dict[str, PostprocessStepContract] = {
    "0": PostprocessStepContract(
        "0",
        "Materialize capture events",
        "Consumes capture events; rebuilds raw Op state; mutates Trace in place.",
        writes=frozenset(),
    ),
    "1": PostprocessStepContract(
        "1",
        "Add output layers",
        "Consumes model outputs and parent labels; produces output Ops in raw state.",
        writes=frozenset(
            (
                "_arg_expressions_cache",
                "_edge_uses",
                "_label_raw",
                "_layer_label_raw",
                "_param_barcodes",
                "_param_logs",
                "activation_memory",
                "arg_names",
                "atomic_module_call",
                "autograd_memory",
                "bytes_delta_at_call",
                "bytes_peak_at_call",
                "children",
                "code_context",
                "container_path",
                "container_spec",
                "dropped_edge_tensor_args",
                "dtype",
                "equivalence_class",
                "equivalent_ops",
                "func",
                "func_config",
                "func_duration",
                "func_name",
                "func_non_tensor_args",
                "func_rng_states",
                "grad_fn_class_name",
                "has_children",
                "has_out_variations",
                "has_output_descendant",
                "input_to_module_calls",
                "intervention_replaced",
                "interventions",
                "io_role",
                "is_atomic_module",
                "is_buffer",
                "is_final_output",
                "is_input",
                "is_internal_source",
                "is_module_output",
                "is_output",
                "is_transform",
                "module",
                "module_call_stack",
                "modules",
                "non_tensor_kwargs",
                "non_tensor_pos_args",
                "num_args_total",
                "num_autograd_tensors",
                "num_kwargs",
                "num_params",
                "num_params_frozen",
                "num_params_trainable",
                "num_passes",
                "num_pos_args",
                "out",
                "out_versions_by_child",
                "output_descendants",
                "output_of_module_calls",
                "output_of_modules",
                "param_memory",
                "param_shapes",
                "parent_arg_positions",
                "parent_param_ops",
                "parent_params",
                "parents",
                "pass_index",
                "raw_index",
                "recurrent_ops",
                "saved_args",
                "saved_kwargs",
                "shape",
                "transform_chain",
                "transform_config",
                "transform_fn_name",
                "transform_fn_qualname",
                "transform_fn_source",
                "transform_kind",
                "transformed_activation_memory",
                "transformed_out",
                "transformed_out_dtype",
                "transformed_out_shape",
                "type",
                "unattributed_tensor_args",
                "var_names",
            )
        ),
    ),
    "2": PostprocessStepContract(
        "2",
        "Trace output ancestors",
        "Consumes raw graph links; mutates output-descendant ancestry flags in place.",
        # Reviewed widening (sol finding 6 in-place audit): the traversal
        # mutates each op's output_descendants staging SET in place.
        writes=frozenset(
            (
                "has_output_descendant",
                "output_descendants",
            )
        ),
    ),
    "3": PostprocessStepContract(
        "3",
        "Remove orphan nodes",
        "Consumes ancestry flags; removes or records orphan raw Ops in place.",
        # Design-ppdag-v3 defect 4, evidence-narrowed: step 3's undeclared
        # writes are two config-gated sets. (4a) keep_orphans=True on an
        # orphan-bearing model writes is_orphan on every retained orphan and
        # returns before the batch removal. (4b) default keep_orphans=False
        # runs the removal scrub (_batch_remove_log_entries with
        # remove_references=True), which rebinds SURVIVING rows'
        # equivalent_ops when an orphan shared an equivalence class
        # (verified live on the OrphanTensors fixture). The design's wider
        # parents/children/recurrent_ops scrub columns CANNOT fire at step 3:
        # the orphan flood is undirected, so orphans are whole disconnected
        # components — no survivor holds a dataflow edge to one — and
        # recurrence groups are not built until step 7. Declaring them here
        # would be phantom declarations (the Opus-4 anti-laundering guard).
        writes=frozenset(
            (
                "_edge_uses",
                "args_template",
                "conditional_arm_children",
                "conditional_elif_children",
                "conditional_else_children",
                "conditional_entry_children",
                "conditional_then_children",
                "equivalent_ops",
                "interventions",
                "is_internal_sink",
                "is_orphan",
                "is_terminal_bool",
                "kwargs_template",
            )
        ),
        removes_rows=True,
    ),
    "4": PostprocessStepContract(
        "4",
        "Input/output distances",
        "Consumes orphan-free graph; mutates distance fields in place.",
        writes=frozenset(
            (
                "has_input_ancestor",
                "has_output_descendant",
                # Reviewed widening (sol finding 6 in-place audit): the
                # distance traversal mutates input_ancestors sets in place.
                "input_ancestors",
                "max_distance_from_input",
                "max_distance_to_output",
                "min_distance_from_input",
                "min_distance_to_output",
            )
        ),
    ),
    "5": PostprocessStepContract(
        "5",
        "Mark conditional branches",
        "Consumes orphan-free graph; mutates conditional metadata in place.",
        writes=frozenset(
            (
                "_is_in_conditional_body",
                "conditional_arm_children",
                "conditional_branch_depth",
                "conditional_branch_stack",
                "conditional_context_kind",
                "conditional_elif_children",
                "conditional_else_children",
                "conditional_entry_children",
                "conditional_then_children",
                "conditional_wrapper_kind",
                # Reviewed widening (sol finding 6 in-place audit): terminal
                # scalar-bool classification writes is_terminal_bool.
                "is_terminal_bool",
                "is_terminal_conditional_bool",
                "terminal_conditional_id",
            )
        ),
    ),
    "6": PostprocessStepContract(
        "6",
        "Fix buffer layers",
        "Consumes buffer events and graph links; mutates buffer metadata in place.",
        writes=frozenset(
            (
                "buffer_pass",
                "buffer_replay_validated",
                "func",
                "func_name",
                "has_children",
                "has_input_ancestor",
                # Reviewed widening (sol finding 6 in-place audit): buffer
                # rewiring mutates root_ancestors closure sets in place.
                "root_ancestors",
            )
        ),
    ),
    "7": PostprocessStepContract(
        "7",
        "Loop detection",
        "Consumes final raw graph structure; mutates recurrence/equivalence metadata.",
        writes=frozenset(
            (
                "_layer_label_raw",
                "equivalence_class",
                "num_passes",
                "pass_index",
                "recurrent_ops",
            )
        ),
    ),
    "8": PostprocessStepContract(
        "8",
        "Map labels",
        "Consumes raw labels and recurrence metadata; produces raw-to-final maps.",
        writes=frozenset(
            (
                "label",
                "label_short",
                "layer_label",
                "layer_label_short",
                "step_index",
                "type_index",
            )
        ),
    ),
    "9": PostprocessStepContract(
        "9",
        "Log final info",
        "Consumes mapped labels; mutates final Op metadata and module build data.",
        writes=frozenset(
            (
                "_edge_uses",
                # Reviewed widening (closure review, enforcement leg over the
                # intervention/observer suites): the final-label rename in
                # _replace_layer_names_for_layer_entry rewrites raw parent
                # refs inside replay templates and intervention records —
                # cells that exist only on intervention-ready captures, an
                # axis absent from the six surface-oracle recording models.
                "args_template",
                "atomic_module_call",
                "children",
                "conditional_arm_children",
                "conditional_elif_children",
                "conditional_else_children",
                "conditional_entry_children",
                "conditional_then_children",
                "equivalent_ops",
                "fx_call_index",
                "fx_qualpath",
                "input_ancestors",
                "internal_source_ancestors",
                # Same reviewed widening as args_template above.
                "interventions",
                "is_buffer",
                "is_input",
                "is_output",
                # Same reviewed widening as args_template above.
                "kwargs_template",
                "output_descendants",
                # Reviewed widening (sol finding 6 in-place audit): final-info
                # logging mutates parent_arg_positions dicts in place.
                "parent_arg_positions",
                "parents",
                "recurrent_ops",
                "root_ancestors",
                "step_index",
            )
        ),
    ),
    "10": PostprocessStepContract(
        "10",
        "Rename labels",
        "Consumes raw-to-final maps; mutates graph references to final labels.",
        writes=frozenset(),
    ),
    "11": PostprocessStepContract(
        "11",
        "Build lookup keys",
        "Consumes final labels; rebuilds final lookup containers in place.",
        writes=frozenset(
            (
                "input_to_module_calls",
                "lookup_keys",
                "module",
                "modules",
                "ordinal_index",
                "output_of_module_calls",
            )
        ),
    ),
    # Step 11.5 previously declared an EMPTY write set, silently wrong under
    # save_code_context=True where it assigns op.var_names on every op
    # (design-ppdag-v3 defect 3). The audit never tripped because no recorded
    # enforcement axis enabled save_code_context. Note: the design's expected
    # _arg_expressions_cache companion write does NOT fire here —
    # resolve_var_names never reads op.arg_expressions (verified), so
    # declaring it would be a phantom declaration.
    "11.5": PostprocessStepContract(
        "11.5",
        "Populate source var names",
        "Consumes code context; mutates Op var_names in place.",
        writes=frozenset(("var_names",)),
    ),
    # Step 11.75 previously had NO contract boundary, so its writes were
    # misattributed to step 12's window and only surfaced on the selective/
    # fastlog axis (deferred retention runs only with a capture session) —
    # an axis absent from the recorded contract runs (closure review,
    # enforcement leg). The declared set is retention's payload family:
    # saving a deferred out writes the payload cells and their derived
    # shape/dtype/memory metadata.
    "11.75": PostprocessStepContract(
        "11.75",
        "Resolve deferred retention",
        "Consumes deferred retention decisions; saves selected payloads.",
        writes=frozenset(
            (
                "activation_memory",
                "annotations",
                "dtype",
                "has_saved_activation",
                "out",
                "saved_args",
                "saved_kwargs",
                "shape",
                "transformed_activation_memory",
                "transformed_out",
                "transformed_out_dtype",
                "transformed_out_shape",
            )
        ),
    ),
    "12": PostprocessStepContract(
        "12",
        "Undecorate tensors",
        "Consumes saved tensors; mutates payload wrappers in place.",
        writes=frozenset(),
    ),
    "13": PostprocessStepContract(
        "13",
        "Clear CUDA cache",
        "Runs optional CUDA allocator cleanup; leaves Trace metadata unchanged.",
        writes=frozenset(),
    ),
    "14": PostprocessStepContract(
        "14",
        "Log timing",
        "Consumes capture timestamps; mutates duration fields in place.",
        writes=frozenset(),
    ),
    "15": PostprocessStepContract(
        "15",
        "Finalize params",
        "Consumes Op param references; mutates Param reverse mappings.",
        # Reviewed widening (sol finding 6 in-place audit): param
        # finalization mutates the _param_logs containers in place.
        writes=frozenset(
            (
                "_param_logs",
                "parent_params",
            )
        ),
    ),
    "15.5": PostprocessStepContract(
        "15.5",
        "Build layer logs",
        "Consumes final Op list; rebuilds aggregate Layer logs and pass index.",
        writes=frozenset(
            (
                "in_conditionals",
                "terminal_bool_for",
            )
        ),
    ),
    "16": PostprocessStepContract(
        "16",
        "Build module logs",
        "Consumes module build data and layer logs; rebuilds Module/ModuleCall logs.",
        # Reviewed widening (sol finding 6 in-place audit): module-log
        # building mutates the _param_logs containers in place.
        writes=frozenset(("_param_logs",)),
    ),
    "16.5": PostprocessStepContract(
        "16.5",
        "Graph shape hash",
        "Consumes final graph; mutates normalized addresses and graph hash.",
        writes=frozenset(
            (
                "_address_normalized",
            )
        ),
    ),
    "17": PostprocessStepContract(
        "17",
        "Mark pass finished",
        "Consumes finalized containers; mutates Trace to user-facing finished state.",
        writes=frozenset(
            (
                "_tracing_finished",
            )
        ),
    ),
    # Steps 18/19 previously declared writes=None (wildcard), which the audit
    # skipped entirely — the streaming axis ran unaudited (design-ppdag-v3
    # defect 1). The sets below are HAND-DERIVED from finalization.py and
    # verified by a streaming recording run: step 18's only live-op writes are
    # the LazyActivationRef attachments in _attach_streamed_tensor_refs
    # (out_ref always; grad_ref only when grads streamed), and step 19's are
    # the _internal_set evictions (out always; transformed_out only when a
    # transformed payload was streamed). The grad/transform halves are
    # config-gated, not phantom.
    "18": PostprocessStepContract(
        "18",
        "Finalize streamed bundle",
        "Consumes stream writer state; finalizes bundle metadata in place.",
        writes=frozenset(
            (
                "grad_ref",
                "out_ref",
            )
        ),
    ),
    "19": PostprocessStepContract(
        "19",
        "Evict streamed outs",
        "Consumes finalized stream state; drops in-memory output payloads.",
        writes=frozenset(
            (
                "out",
                "transformed_out",
            )
        ),
    ),
    "20": PostprocessStepContract(
        "20",
        "Release param refs",
        "Consumes finalized Param logs; drops live parameter references in place.",
        writes=frozenset(),
    ),
}


def _postprocess_assertions_enabled() -> bool:
    """Return whether postprocess boundary assertions are enabled.

    Returns
    -------
    bool
        ``True`` when ``TORCHLENS_POSTPROCESS_ASSERTIONS`` is set to a truthy value.
    """

    return os.environ.get(_POSTPROCESS_ASSERT_ENV, "").lower() in {"1", "true", "yes", "on"}


_WRITE_AUDIT_RECORD_ENV = "TORCHLENS_POSTPROCESS_WRITE_AUDIT"

#: Recording-mode sink: step id -> union of observed written column names
#: across every audited postprocess run in this process. Read by the
#: declaration-generation tooling; never consulted in enforcement mode.
RECORDED_STEP_WRITES: dict[str, set[str]] = {}


def _write_audit_record_mode() -> bool:
    """Return whether the write audit RECORDS instead of enforcing."""

    return os.environ.get(_WRITE_AUDIT_RECORD_ENV, "").lower() == "record"


def _open_step_write_audit(self: "Trace") -> None:
    """Start the op-store column write audit for the next step window."""

    core = self.__dict__.get("_trace_core")
    if core is None or core.ops is None:
        return
    from .._trace_core.op_store import begin_cell_write_audit

    begin_cell_write_audit(core.ops)


def _close_step_write_audit(self: "Trace") -> tuple[set[str], int] | None:
    """Stop the audit; return (written column names, released-row count).

    ``None`` when unarmed (no core-backed store yet).
    """

    core = self.__dict__.get("_trace_core")
    if core is None or core.ops is None:
        return None
    from .._trace_core.op_store import end_cell_write_audit

    return end_cell_write_audit(core.ops)


def _assert_postprocess_contract(self: "Trace", step: str) -> None:
    """Assert cheap postconditions for one completed postprocess step.

    Parameters
    ----------
    self:
        Trace being postprocessed.
    step:
        Step identifier from ``POSTPROCESS_STEP_CONTRACTS``.
    """

    if not _postprocess_assertions_enabled():
        return
    contract = POSTPROCESS_STEP_CONTRACTS.get(step)
    assert contract is not None, f"Unknown postprocess step contract: {step!r}"
    audit_result = _close_step_write_audit(self)
    if audit_result is not None:
        observed_writes, released_rows = audit_result
        assert not released_rows or contract.removes_rows, (
            f"Step {step} ({contract.name}) released {released_rows} whole op "
            "row(s) without a removes_rows sanction in "
            "POSTPROCESS_STEP_CONTRACTS; declaring row removal is a reviewed "
            "contract diff, never a silent drift."
        )
        if _write_audit_record_mode():
            RECORDED_STEP_WRITES.setdefault(step, set()).update(observed_writes)
        elif contract.writes is not None:
            undeclared_writes = observed_writes - contract.writes
            assert not undeclared_writes, (
                f"Step {step} ({contract.name}) wrote undeclared op-store "
                f"columns {sorted(undeclared_writes)}; widen the declared "
                "write set in POSTPROCESS_STEP_CONTRACTS as a reviewed "
                "schema-contract diff if the writes are intended."
            )
    _open_step_write_audit(self)
    step_name = f"Step {contract.step} ({contract.name})"
    if step == "1":
        assert self.output_layers, f"{step_name} must register output layers"
    elif step == "8":
        assert self._raw_to_final_layer_labels, "Step 8 must build raw-to-final layer labels"
        assert self._raw_to_final_op_labels, "Step 8 must build raw-to-final op labels"
    elif step == "11":
        for op in self.layer_list:
            assert op.label, f"Step 11 left {op!r} without a final op label"
            assert op.layer_label, f"Step 11 left {op!r} without a final layer label"
            assert op.lookup_keys, f"Step 11 left {op.label} without lookup keys"
            assert self.layer_dict_all_keys[op.label] is op
            # The bare layer label resolves to ONE pass of that layer (the
            # public lookup contract keeps the LAST pass for multi-pass
            # layers), never to a foreign layer's op. The former exact
            # `is op` form was wrong by construction for every recurrent
            # model and unreachable outside the debug env flag.
            resolved = self.layer_dict_all_keys[op.layer_label]
            assert resolved.layer_label == op.layer_label, (
                f"Step 11 mapped layer label {op.layer_label!r} to a foreign "
                f"op {resolved.label!r}"
            )
    elif step == "15.5":
        assert self.layer_logs, "Step 15.5 must build aggregate layer logs"
        assert len(self.layer_logs) == len(self.layer_labels)
        assert isinstance(self.by_pass, dict)
    elif step == "16":
        assert getattr(self, "_module_logs", None) is not None, "Step 16 must build module logs"
    elif step == "16.5":
        assert self.graph_shape_hash is not None, "Step 16.5 must compute graph_shape_hash"
    elif step == "17":
        assert self._tracing_finished is True, "Step 17 must mark tracing finished"


def _warn_unattributed_tensor_args(self: "Trace") -> None:
    """Warn once for tensor arguments without graph/source provenance.

    Parameters
    ----------
    self:
        Trace being postprocessed.

    Returns
    -------
    None
        Emits at most one aggregate warning.
    """

    offenders: list[str] = []
    for op in getattr(self, "layer_list", ()):
        if getattr(op, "type", None) == "output":
            continue
        positions = tuple(getattr(op, "unattributed_tensor_args", ()) or ())
        if not positions:
            continue
        label = getattr(op, "label", None) or getattr(op, "layer_label", None) or op._label_raw
        offenders.append(f"{label} ({', '.join(positions)})")
    if not offenders:
        return
    warnings.warn(
        "TorchLens found tensor arguments with no graph/source provenance. "
        "These are usually tensors captured from outside the traced model; "
        "module tensor attributes, inputs, parameters, and buffers are known sources. "
        "Offending ops/arg positions: " + "; ".join(offenders),
        UserWarning,
        stacklevel=2,
    )


def _populate_var_names(self: "Trace") -> None:
    """Populate source assignment names for captured operation call sites.

    Parameters
    ----------
    self:
        Trace being postprocessed.
    """

    if not getattr(self, "save_code_context", False):
        return
    for op in getattr(self, "layer_list", ()):
        if getattr(op, "type", None) == "output":
            op.var_names = []
            continue
        op.var_names = resolve_var_names(
            getattr(op, "code_context", []) or [],
            getattr(op, "func_name", None),
        )


def _drop_transient_capture_state(self: "Trace") -> None:
    """Remove capture/session scratch that must not survive on final traces.

    Args:
        self: Trace whose postprocess-local state should be discarded.

    Returns:
        None. Mutates ``self.__dict__``.
    """

    keep_deferred_streaming = bool(
        self.__dict__.get("_defer_streaming_bundle_finalization", False)
        and self.__dict__.get("_out_writer") is not None
    )
    keep_selective_sink = self.__dict__.get("_out_sink") is not None
    wrapper_ws = self.__dict__.get("_wrapper_runtime_ws")
    if wrapper_ws is not None:
        registry = getattr(wrapper_ws, "container_registry", None)
        if registry is not None:
            registry.clear_live_state()
    field_names = [
        "_raw_graph_ws",
        "_module_capture_ws",
        "_wrapper_runtime_ws",
        "capture_events",
        "_output_container_specs_by_raw_label",
    ]
    if not keep_deferred_streaming and not keep_selective_sink:
        field_names.extend(
            [
                "_out_writer",
                "_out_sink",
                "_keep_outs_in_memory",
                "_keep_grads_in_memory",
                "_grad_stream_retain_in_memory",
                "_defer_streaming_bundle_finalization",
            ]
        )
    elif not keep_deferred_streaming:
        field_names.extend(
            [
                "_out_writer",
                "_keep_outs_in_memory",
                "_keep_grads_in_memory",
                "_grad_stream_retain_in_memory",
                "_defer_streaming_bundle_finalization",
            ]
        )
    for field_name in field_names:
        self.__dict__.pop(field_name, None)


def _refresh_fast_saved_summary(self: "Trace") -> None:
    """Refresh saved-output counters after retained layers are finalized.

    Args:
        self: Trace whose final retained layer entries were updated.

    Returns:
        None. Mutates aggregate saved-output fields on ``self``.
    """

    saved_layers = [
        layer_entry
        for layer_entry in self.layer_list
        if getattr(layer_entry, "has_saved_activation", False)
        and not getattr(layer_entry, "is_orphan", False)
    ]
    self.num_saved_ops = len(saved_layers)
    self.saved_activation_memory = Bytes(
        sum(int(getattr(layer_entry, "activation_memory", 0) or 0) for layer_entry in saved_layers)
    )
    self.num_saved_layers = len({layer_entry.layer_label for layer_entry in saved_layers})
    refresh_saved_module_call_count(self, {layer_entry.label for layer_entry in saved_layers})


def postprocess(
    self: "Trace", output_tensors: List[torch.Tensor], output_tensor_addresses: List[str]
) -> None:
    """Run the full postprocessing pipeline in exhaustive mode.

    Transforms the raw Trace captured during the forward pass into its
    final user-facing form.

    Parameters
    ----------
    output_tensors:
        Actual output tensors returned by the model's forward call.
    output_tensor_addresses:
        Hierarchical address strings for each output, for example ``"0.1"``
        for nested tuple outputs.
    """
    capture_events = getattr(self, "capture_events", None)
    capture_session = None
    # Resolve each output tensor's graph parent BEFORE materializing events:
    # a registered buffer returned directly from forward() without ever being
    # used by a traced op has no graph node yet, and is logged here as a late
    # buffer source event so it materializes with everything else.
    output_parent_labels = _resolve_output_parent_labels(self, output_tensors)
    if capture_events is not None:
        self._raw_event_shape_hash = compute_raw_event_shape_hash(capture_events)
        capture_session = capture_session_for_events(capture_events)
        sealed_op_events = (
            [_clone_op_event_for_replay(event) for event in capture_session.seal().events]
            if capture_session is not None
            else list(capture_events.op_events)
        )
        working_events = capture_events.copy_for_replay()
        working_events.op_events = sealed_op_events
        working_events.op_event_by_label_raw = {
            event.label_raw: event for event in sealed_op_events
        }
        self._capture_events = working_events
        with _vtimed(self, "  Step 0: Materialize capture events"):
            materialize_from_events(self, working_events)
        working_events.release_working_projection()
        _assert_postprocess_contract(self, "0")
        delattr(self, "capture_events")

    # Guard: if the model produced no logged layers, skip postprocessing (#153)
    if len(self._raw_graph_ws.raw_layer_labels_list) == 0:
        import warnings

        warnings.warn("No layers were logged during the forward pass; skipping postprocessing.")
        _set_tracing_finished(self)
        _drop_transient_capture_state(self)
        if capture_events is not None:
            capture_events.release_runtime_sidecars()
            # The trace is the sole strong owner of its (sidecar-released)
            # event stream; the former _EVENT_STREAMS weak registry is gone.
            self.__dict__["_capture_events"] = capture_events
        return

    _vprint(
        self,
        f"Postprocessing {len(self._raw_graph_ws.raw_layer_labels_list):,} layers "
        f"({len(self.buffer_layers):,} buffers)...",
    )
    _post_t0 = time.time() if getattr(self, "verbose", False) else 0

    # Step 1: Add dedicated output nodes
    with _vtimed(self, "  Step 1: Add output layers"):
        _add_output_layers(self, output_tensors, output_tensor_addresses, output_parent_labels)
    _assert_postprocess_contract(self, "1")

    # Step 2: Trace which nodes are ancestors of output nodes
    with _vtimed(self, "  Step 2: Trace output ancestors"):
        _find_output_ancestors(self)
    _assert_postprocess_contract(self, "2")

    # Step 3: Remove orphan nodes, find nodes that don't terminate in output node
    with _vtimed(self, "  Step 3: Remove orphan nodes"):
        _remove_orphan_nodes(self)
    _assert_postprocess_contract(self, "3")

    # Step 4: Find min/max distance from input and output nodes.
    # Conditional: only runs when the user requested distance metadata.
    if self.mark_layer_depths:
        with _vtimed(self, "  Step 4: Input/output distances"):
            _mark_layer_depths(self)
        _assert_postprocess_contract(self, "4")

    # Step 5: Starting from terminal single boolean tensors, mark the conditional branches.
    with _vtimed(self, "  Step 5: Mark conditional branches"):
        _mark_conditional_branches(self)
    _assert_postprocess_contract(self, "5")

    # Step 6: Fix the buffer ops and parent information.
    with _vtimed(self, "  Step 6: Fix buffer layers"):
        _fix_buffer_layers(self)
    _assert_postprocess_contract(self, "6")

    # Step 7: Identify all loops, mark repeated layers.
    loop_desc = (
        "  Step 7: Loop detection (full)"
        if self.recurrence_detection
        else "  Step 7: Loop detection (params only)"
    )
    with _vtimed(self, loop_desc):
        if self.recurrence_detection:
            _detect_and_label_loops(self)
        else:
            _group_by_shared_params(self)
    _assert_postprocess_contract(self, "7")

    # Step 8: Go down tensor list, get the mapping from raw tensor names to final tensor names.
    with _vtimed(self, "  Step 8: Map labels"):
        _map_raw_labels_to_final_labels(self)
    _assert_postprocess_contract(self, "8")

    # Step 9: Log final info for all layers
    with _vtimed(self, "  Step 9: Log final info"):
        _log_final_info_for_layers(self)
    _assert_postprocess_contract(self, "9")

    # Step 10: Rename all raw labels to final labels
    with _vtimed(self, "  Step 10: Rename labels"):
        _rename_model_history_layer_names(self)
    _assert_postprocess_contract(self, "10")

    # Step 11: Build lookup keys and finalize retained layer lists
    with _vtimed(self, "  Step 11: Build lookup keys"):
        _build_lookup_keys_and_finalize_retained_layers(self)
        _refresh_fast_saved_summary(self)
        _warn_unattributed_tensor_args(self)
    _assert_postprocess_contract(self, "11")

    # Step 11.5: Populate source assignment names from full-file AST context.
    with _vtimed(self, "  Step 11.5: Populate source var names"):
        _populate_var_names(self)
    _assert_postprocess_contract(self, "11.5")

    if capture_session is not None:
        with _vtimed(self, "  Step 11.75: Resolve deferred retention"):
            capture_session.resolve_deferred_retention(self, list(output_tensors))
    _assert_postprocess_contract(self, "11.75")

    # Step 12: Undecorate all saved tensors and remove saved grad_fns.
    with _vtimed(self, "  Step 12: Undecorate tensors"):
        _undecorate_all_saved_tensors(self)
    _assert_postprocess_contract(self, "12")

    # Step 13: Clear the cache after any tensor deletions for garbage collection purposes.
    # Gated behind cached cuda.is_available() so CPU-only runs don't pay the
    # CUDA driver / NVML probe cost (per profiling audit 2026-04-27 finding #4).
    if _is_cuda_available():
        torch.cuda.empty_cache()
    _assert_postprocess_contract(self, "13")

    # Step 14: Log time elapsed.
    with _vtimed(self, "  Step 14: Log timing"):
        _log_time_elapsed(self)
    _assert_postprocess_contract(self, "14")

    # Step 15: Populate Param reverse mappings, linked params, num_calls, and grad metadata.
    with _vtimed(self, "  Step 15: Finalize params"):
        _finalize_param_logs(self)
    _assert_postprocess_contract(self, "15")

    # Step 15.5: Build aggregate Layer objects from per-pass Op entries.
    with _vtimed(self, "  Step 15.5: Build layer logs"):
        _build_layer_logs(self)
        self.by_pass = {}
        for index, op in enumerate(self.layer_list):
            pass_index = getattr(op, "pass_index", None)
            if pass_index is not None:
                self.by_pass.setdefault(pass_index, []).append(index)
    _assert_postprocess_contract(self, "15.5")

    # Step 16: Build structured Module objects from raw module_* dicts.
    with _vtimed(self, "  Step 16: Build module logs"):
        _build_module_logs(self)
        refresh_saved_module_call_count(self)
    _assert_postprocess_contract(self, "16")

    # Step 16.5: Compute graph shape hash before _set_tracing_finished changes access behavior.
    with _vtimed(self, "  Step 16.5: Graph shape hash"):
        populate_normalized_layer_addresses(self)
        self.graph_shape_hash = compute_graph_shape_hash(self)
    _assert_postprocess_contract(self, "16.5")

    # Step 17: log the pass as finished, changing the Trace behavior to its user-facing version.
    with _vtimed(self, "  Step 17: Mark pass finished"):
        _set_tracing_finished(self)
    _assert_postprocess_contract(self, "17")

    wrapper_ws = self.__dict__.get("_wrapper_runtime_ws")
    if wrapper_ws is not None:
        registry = getattr(wrapper_ws, "container_registry", None)
        if registry is not None:
            if registry.records:
                self.__dict__["_containers"] = dict(registry.records)
            registry.clear_live_state()

    for field_name in (
        "_raw_graph_ws",
        "_module_capture_ws",
        "_wrapper_runtime_ws",
        "capture_events",
        "_output_container_specs_by_raw_label",
    ):
        self.__dict__.pop(field_name, None)

    should_finalize_streaming = getattr(self, "_out_writer", None) is not None and not getattr(
        self, "_defer_streaming_bundle_finalization", False
    )
    if should_finalize_streaming:
        with _vtimed(self, "  Step 18: Finalize streamed bundle"):
            _finalize_streamed_bundle(self)
        _assert_postprocess_contract(self, "18")

    if should_finalize_streaming and not self._keep_outs_in_memory:
        with _vtimed(self, "  Step 19: Evict streamed outs"):
            _evict_streamed_outs(self)
        _assert_postprocess_contract(self, "19")

    with _vtimed(self, "  Step 20: Release param refs"):
        self.release_param_refs(allow_iter_rehydrate=True)
    _assert_postprocess_contract(self, "20")
    # Discard the trailing audit window opened by the step-20 assertion: the
    # freeze conversion below legitimately rewrites relation cells wholesale.
    _close_step_write_audit(self)

    # The compaction passes belong to the freeze (M11 fold): ancestor
    # closures intern into shared bitmaps and repeated immutable Op metadata
    # pools onto shared instances, right before the physical seal.
    _compact_ancestor_sets(self)
    _compact_op_metadata(self)

    # The core freeze point (trace_core_design.md section 3.3): forward
    # topology froze logically at step 17, the payload plane settled through
    # step 20, and the ancestor closures were just interned. The M6 relation
    # conversion runs here — parents/children project into the core's
    # canonical dataflow edge table (differentially verified before the
    # staging cells die) and the remaining relation families become interned
    # immutable views — the M7 group and shared-fact conversions follow
    # (equivalence/recurrence GroupRefs, FunctionCall/ParamAlias fact
    # blocks) — then the Op row store seals (columnar transpose on large
    # traces). Later public writes land in the store's sparse overlay;
    # facade behavior is otherwise unchanged.
    _core = self.__dict__.get("_trace_core")
    if _core is not None and _core.ops is not None:
        _freeze_relation_views(self)
        _core.ops.freeze()
        # Adopt the trace-scoped FuncCallLocation records (cached per call
        # site in _code_context_cache) into their kind table before sealing.
        # Cache entries mix FuncCallLocation records with plain metadata, so
        # filter by type.
        from ..data_classes.func_call_location import FuncCallLocation as _FCL

        _fcl_seen: dict[int, object] = {}
        for _fcl_group in (self.__dict__.get("_code_context_cache") or {}).values():
            for _fcl in _fcl_group or ():
                if isinstance(_fcl, _FCL):
                    _fcl_seen.setdefault(id(_fcl), _fcl)
        if _fcl_seen:
            from .._trace_core.record_rows import adopt_records

            adopt_records(_core, "func_call_location", _fcl_seen.values())
        # The M8 non-Op kind tables (param/module/module_call/buffer/
        # func_call_location) seal with the same lifecycle: appends stop,
        # later writes keep landing in row cells via the sealed-store path.
        for _kind_store in _core.kind_rows.values():
            _kind_store.freeze()

    if getattr(self, "verbose", False):
        print(f"[torchlens] Postprocessing complete ({time.time() - _post_t0:.2f}s)")
    _drop_transient_capture_state(self)
    if capture_events is not None:
        capture_events.release_runtime_sidecars()
        # The trace is the sole strong owner of its (sidecar-released) event
        # stream; the former _EVENT_STREAMS weak registry is gone. Sidecar
        # release already stripped payloads, native handles, and the
        # source_trace backrefs, so this strong edge closes no new cycle.
        self.__dict__["_capture_events"] = capture_events
