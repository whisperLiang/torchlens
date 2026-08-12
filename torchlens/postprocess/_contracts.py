"""Postprocess step contracts, frozen rank, and the pinned-pair corpus.

The three frozen artifacts of the postprocess dependency derivation
(design-ppdag-v3):

- ``POSTPROCESS_STEP_CONTRACTS`` — each step's declared writes/reads/
  probes/row_effects/trace_state (the derivation's inputs).
- ``LEGACY_STEP_RANK`` — key 1 of the two-key direction authority: the
  historically-established order as a frozen semantic constant. Every edge
  orients by rank; registry position is never an input to derivation.
- ``PINNED_ORDER_PAIRS`` — key 2: the named producer->consumer pair corpus.
  Reordering steps mechanically requires editing a reason-bearing corpus
  entry (import check K1), and every newly derived RAW/WW edge must be
  pinned before CI passes (test-side check K2).

The read-triggers-write class (design-ppdag-v3 §5.6): exactly one member is
live inside steps 1-20 — reading ``op.arg_expressions`` writes
``_arg_expressions_cache`` (a lazy cache behind a property). The ``_CSR``/
``_FACT`` fact-block hydration writes can only fire after the freeze
installs their sentinels, which happens in the epilogue AFTER step 20 —
out-of-window by construction. PRECOMMIT: a write-audit trip on a
``*_cache`` column from a read site is ROOT-CAUSED (is the read intended?
is the cache column declared for that step?), never resolved by
reflexively widening a declared set.

Named model limits (unchanged by the derivation): warnings and
first-exception order are pinned solely by day-1 order identity — any
future reorder adds a warnings/exception-order review gate; deferred
gradient streaming re-runs step-18/19-equivalent code after backward,
outside the pipeline and its windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

#: Closed trace-state token vocabulary (design-ppdag-v3 §2.2). Each token
#: names one non-column state surface a step may consume or produce; the
#: derivation orients token conflicts exactly like column conflicts. The
#: three workspace tokens are capture-produced (legal to read with no
#: earlier pipeline writer — the token analogue of the capture baseline).
_TRACE_STATE_TOKENS: frozenset[str] = frozenset(
    (
        "raw_graph_ws",
        "module_capture_ws",
        "wrapper_runtime_ws",
        "label_maps",
        "lookup_containers",
        "conditional_records",
        "layer_logs",
        "module_logs",
        "module_build",
        "param_logs_kind",
        "stream_writer",
        "payload_tensors",
        "timing",
        "warnings",
        "cuda_cache",
        "finished_flag",
        "saved_summary",
        "graph_hash",
        "containers",
        "stream_lifecycle",
    )
)

#: Tokens produced by capture/step 0 itself: a declared read with no earlier
#: pipeline writer is legal for exactly these (import check 7.1-5). Beyond
#: the three per-phase workspaces: param_logs_kind because the raw ParamLog
#: kind records are built during capture (step 15 finalizes pre-existing
#: state — its declared rw: read has no earlier pipeline writer), and
#: stream_writer because disk streaming creates the writer and streams
#: payloads DURING the forward (step 18's rw: read consumes capture-created
#: writer state).
CAPTURE_BASELINE_TOKENS: frozenset[str] = frozenset(
    (
        "raw_graph_ws",
        "module_capture_ws",
        "wrapper_runtime_ws",
        "param_logs_kind",
        "stream_writer",
    )
)


def tokens(*declarations: str) -> frozenset[str]:
    """Normalize trace-state token declarations to the stored vocabulary.

    The stored vocabulary is ``r:<token>`` / ``w:<token>`` ONLY. ``rw:<token>``
    is construction-time shorthand expanded to both entries; any other prefix
    or unknown token raises at import time (design-ppdag-v3 §2.2).
    """

    normalized: set[str] = set()
    for declaration in declarations:
        prefix, _, token = declaration.partition(":")
        if token not in _TRACE_STATE_TOKENS:
            raise ValueError(
                f"Unknown trace-state token {token!r} in {declaration!r}; the "
                "closed vocabulary lives in _TRACE_STATE_TOKENS and growing it "
                "is a reviewed contract diff."
            )
        if prefix == "rw":
            normalized.add(f"r:{token}")
            normalized.add(f"w:{token}")
        elif prefix in ("r", "w"):
            normalized.add(declaration)
        else:
            raise ValueError(
                f"Invalid trace-state prefix {prefix!r} in {declaration!r}; "
                "only r:/w: are stored (rw: is construction-time shorthand)."
            )
    return frozenset(normalized)


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
        write audit (``op_store.begin_cell_write_audit``). The former
        ``None`` wildcard is DELETED (design-ppdag-v3): every step declares
        an exact set. A step writing an undeclared column fails the
        tripwire; widening a set is a REVIEWED schema-contract diff, never
        a silent drift.
    reads:
        Declared op-store COLUMN read set. Authority for the dependency
        derivation: seeded from the recording matrix, hand-reviewed against
        the step source, shipped as a reviewed diff (recordings are
        evidence, never auto-regenerated declarations).
    placeholder_probes:
        Reviewed reads that legally observe the step-0 schema placeholder
        ("has this been set yet"). Exempt from read-before-write findings,
        NEVER from WAR edges — a probe's correctness depends on staying
        pinned before the column's writer.
    row_effects:
        Whole-row lifecycle sanctions: ``"creates"`` (the step may build op
        rows — also the legality condition for row-clone reads) and/or
        ``"deletes"`` (removal husking releases every cell of the row).
        Row lifecycle is audited separately from column writes, so an
        unsanctioned removal fails with a precise message instead of a
        wall of column names. A step with either effect is a two-sided
        barrier against every op-column-touching step (edge rule 4).
    trace_state:
        Non-column state tokens, ``r:<token>``/``w:<token>`` over the
        closed ``_TRACE_STATE_TOKENS`` vocabulary. Construct with
        ``tokens()`` so ``rw:`` shorthand normalizes and typos refuse at
        import.
    barrier:
        Full ordering barrier (step 17 only: ``_tracing_finished`` flips
        global facade behavior).
    """

    step: str
    name: str
    contract: str
    writes: frozenset[str]
    reads: frozenset[str]
    placeholder_probes: frozenset[str] = frozenset()
    row_effects: frozenset[str] = frozenset()
    trace_state: frozenset[str] = frozenset()
    barrier: bool = False

    def __post_init__(self) -> None:
        """Refuse malformed contracts at construction (plain raise, not assert)."""

        if self.writes is None or self.reads is None:  # type: ignore[unreachable]
            raise ValueError(
                f"Step {self.step}: writes/reads must be exact frozensets; the "
                "None wildcard is deleted (design-ppdag-v3 defect 1)."
            )
        unknown_effects = self.row_effects - {"creates", "deletes"}
        if unknown_effects:
            raise ValueError(
                f"Step {self.step}: unknown row_effects {sorted(unknown_effects)}; "
                "the vocabulary is {'creates', 'deletes'}."
            )
        for entry in self.trace_state:
            prefix, _, token = entry.partition(":")
            if prefix not in ("r", "w") or token not in _TRACE_STATE_TOKENS:
                raise ValueError(
                    f"Step {self.step}: invalid trace_state entry {entry!r}; "
                    "construct with tokens() (stored vocabulary is r:/w: over "
                    "_TRACE_STATE_TOKENS)."
                )


#: FROZEN semantic constant (design-ppdag-v3 §2.1, key 1 of the two-key
#: direction authority): the historically-established producer/consumer
#: order as ground truth. Every RAW/WW/WAR/row/token edge orients by this
#: rank; registry position is NOT an input to derivation. Editing it is a
#: reviewed semantic diff under the same governance as widening a write
#: set. It is NOT derived from the step registry and is NOT regenerated by
#: any tool. Step "0" is deliberately absent (fenced prologue, producer
#: lane); step "17.5" is the contracted container-adoption seam.
LEGACY_STEP_RANK: "Mapping[str, int]" = MappingProxyType(
    {
        "1": 10,
        "2": 20,
        "3": 30,
        "4": 40,
        "5": 50,
        "6": 60,
        "7": 70,
        "8": 80,
        "9": 90,
        "10": 100,
        "11": 110,
        "11.5": 115,
        "11.75": 118,
        "12": 120,
        "13": 130,
        "14": 140,
        "15": 150,
        "15.5": 155,
        "16": 160,
        "16.5": 165,
        "17": 170,
        "17.5": 175,
        "18": 180,
        "19": 190,
        "20": 200,
    }
)


POSTPROCESS_STEP_CONTRACTS: dict[str, PostprocessStepContract] = {
    # Step 0 runs before any audit window can arm (the store is born inside
    # it), so its writes line is vacuously green; it exists for the
    # producer-lane fence and is JOINT-SIGNOFF with that lane.
    "0": PostprocessStepContract(
        "0",
        "Materialize capture events",
        "Consumes capture events; rebuilds raw Op state; mutates Trace in place.",
        writes=frozenset(),
        reads=frozenset(),
        row_effects=frozenset(("creates",)),
        trace_state=tokens("w:raw_graph_ws"),
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
        reads=frozenset(),
        # Row creation carries the row-clone read legality (design-ppdag-v3
        # §2.4d): step 1 clones the output node via Op.copy(), whose
        # whole-schema getattr loop is a mechanical row_clone access kind,
        # not a per-column dependency.
        row_effects=frozenset(("creates",)),
        trace_state=tokens("rw:raw_graph_ws", "w:lookup_containers"),
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
        reads=frozenset(),
        trace_state=tokens("r:raw_graph_ws"),
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
        reads=frozenset(),
        row_effects=frozenset(("deletes",)),
        trace_state=tokens("rw:raw_graph_ws"),
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
        reads=frozenset(),
        trace_state=tokens("r:raw_graph_ws"),
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
        reads=frozenset(),
        trace_state=tokens("r:raw_graph_ws", "w:conditional_records"),
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
        reads=frozenset(),
        # Buffer dedup removes merged duplicate rows through the same husking
        # path as orphan removal (_remove_log_entry at control_flow.py:951);
        # previously unsanctioned — a latent released-row trip on any
        # buffer-merging axis (design-ppdag-v3 inventory row 6).
        row_effects=frozenset(("deletes",)),
        trace_state=tokens("rw:raw_graph_ws"),
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
        reads=frozenset(),
        trace_state=tokens("r:raw_graph_ws"),
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
        reads=frozenset(),
        trace_state=tokens("r:raw_graph_ws", "w:label_maps"),
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
        reads=frozenset(),
        trace_state=tokens("r:label_maps", "rw:raw_graph_ws", "w:module_build"),
    ),
    "10": PostprocessStepContract(
        "10",
        "Rename labels",
        "Consumes raw-to-final maps; mutates graph references to final labels.",
        writes=frozenset(),
        reads=frozenset(),
        trace_state=tokens("r:label_maps", "r:raw_graph_ws", "w:lookup_containers"),
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
        reads=frozenset(),
        # r:module_build makes the 9 -> 11 edge derivable: step 11 reads
        # module_build_data["module_num_calls"] (labeling.py:847).
        trace_state=tokens(
            "r:label_maps",
            "r:module_build",
            "rw:lookup_containers",
            "w:saved_summary",
            "w:warnings",
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
        reads=frozenset(),
        trace_state=tokens("r:raw_graph_ws"),
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
        reads=frozenset(),
        trace_state=tokens("w:payload_tensors"),
    ),
    "12": PostprocessStepContract(
        "12",
        "Undecorate tensors",
        "Consumes saved tensors; mutates payload wrappers in place.",
        writes=frozenset(),
        reads=frozenset(),
        trace_state=tokens("w:payload_tensors"),
    ),
    "13": PostprocessStepContract(
        "13",
        "Clear CUDA cache",
        "Runs optional CUDA allocator cleanup; leaves Trace metadata unchanged.",
        writes=frozenset(),
        reads=frozenset(),
        trace_state=tokens("r:payload_tensors", "w:cuda_cache"),
    ),
    "14": PostprocessStepContract(
        "14",
        "Log timing",
        "Consumes capture timestamps; mutates duration fields in place.",
        writes=frozenset(),
        reads=frozenset(),
        trace_state=tokens("w:timing"),
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
        reads=frozenset(),
        trace_state=tokens("rw:param_logs_kind"),
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
        reads=frozenset(),
        trace_state=tokens("r:conditional_records", "w:layer_logs"),
    ),
    "16": PostprocessStepContract(
        "16",
        "Build module logs",
        "Consumes module build data and layer logs; rebuilds Module/ModuleCall logs.",
        # Reviewed widening (sol finding 6 in-place audit): module-log
        # building mutates the _param_logs containers in place.
        writes=frozenset(("_param_logs",)),
        reads=frozenset(),
        trace_state=tokens(
            "r:layer_logs",
            "r:module_build",
            "r:module_capture_ws",
            "rw:param_logs_kind",
            "rw:saved_summary",
            "w:module_logs",
        ),
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
        reads=frozenset(),
        trace_state=tokens("r:lookup_containers", "w:graph_hash"),
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
        reads=frozenset(),
        trace_state=tokens("w:finished_flag"),
        barrier=True,
    ),
    # Step 17.5: the container-adoption + workspace-drop seam, contracted at
    # its exact historical position between 17 and the streaming snapshot
    # (design-ppdag-v3 §5.3). Trace-side only: adopts the wrapper runtime
    # registry's container records into trace._containers and drops all
    # three per-phase workspaces (terminal consumes). Step 18 declares
    # r:containers, which is the edge that makes this seam's position
    # derivable. Unwrapped by _vtimed today — stays unwrapped.
    "17.5": PostprocessStepContract(
        "17.5",
        "Adopt containers, drop workspaces",
        "Consumes wrapper runtime registry; adopts container records; drops workspaces.",
        writes=frozenset(),
        reads=frozenset(),
        trace_state=tokens(
            "r:wrapper_runtime_ws",
            "w:containers",
            "w:raw_graph_ws",
            "w:module_capture_ws",
            "w:wrapper_runtime_ws",
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
        reads=frozenset(),
        trace_state=tokens(
            "r:containers",
            "r:payload_tensors",
            "rw:stream_writer",
            "w:stream_lifecycle",
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
        reads=frozenset(),
        trace_state=tokens("r:stream_writer", "rw:stream_lifecycle"),
    ),
    "20": PostprocessStepContract(
        "20",
        "Release param refs",
        "Consumes finalized Param logs; drops live parameter references in place.",
        writes=frozenset(),
        reads=frozenset(),
        # r:stream_lifecycle is the explicit hand-declared token that makes
        # the 18/19 -> 20 ordering derivable (release after optional stream
        # finalization, AGENTS.md).
        trace_state=tokens("rw:param_logs_kind", "r:stream_lifecycle"),
    ),
}


@dataclass(frozen=True)
class PinnedPair:
    """One reviewed semantic producer->consumer fact (corpus key 2).

    Parameters
    ----------
    carrier:
        What carries the dependency: ``"columns"`` (op-store columns),
        ``"tokens"`` (trace-state tokens), or ``"structure"`` (row-effects
        barriers / the step-17 barrier — prose invariants absorbed from the
        docstring-invariant map, whose ``carriers`` set may then be empty of
        columns; review note N16 keeps K2's subset check well-typed).
    carriers:
        The column or token names carrying the dependency (may be empty for
        ``"structure"`` entries).
    reason:
        One-line reviewed semantic justification. Deleting or editing an
        entry is THE reviewed act that blesses a reorder — no tooling
        regenerates this corpus.
    """

    carrier: str
    carriers: frozenset[str]
    reason: str

    def __post_init__(self) -> None:
        """Refuse malformed corpus entries at construction."""

        if self.carrier not in ("columns", "tokens", "structure"):
            raise ValueError(
                f"PinnedPair carrier must be columns/tokens/structure, got "
                f"{self.carrier!r}."
            )
        if self.carrier != "structure" and not self.carriers:
            raise ValueError(
                "A columns/tokens PinnedPair must name its carriers."
            )


#: Key 2 of the two-key direction authority: the semantic producer->consumer
#: corpus. Seeded ONCE from the day-1 derived RAW and WW edge sets, then
#: hand-reviewed; NEVER regenerated by tooling. Import check K1 refuses any
#: entry contradicting LEGACY_STEP_RANK by name; test-side check K2 refuses
#: any derived RAW/WW edge not pinned here. Content lands with the
#: declaration freeze (implementation plan step 5); the empty corpus is the
#: pre-seed state, not a steady state.
PINNED_ORDER_PAIRS: Mapping[tuple[str, str], PinnedPair] = MappingProxyType({})


def _validate_contract_artifacts() -> None:
    """Import-time structural binding of contracts and the frozen rank.

    Check 7.1-0 (design-ppdag-v3): the rank's key set equals the contract
    key set minus the fenced step "0", refused by name — a step
    insertion/removal diff hits this first, never a bare ``KeyError``
    inside derivation. Plain ``raise`` (``python -O`` strips asserts).
    """

    contract_steps = set(POSTPROCESS_STEP_CONTRACTS) - {"0"}
    rank_steps = set(LEGACY_STEP_RANK)
    if contract_steps != rank_steps:
        missing_rank = sorted(contract_steps - rank_steps)
        missing_contract = sorted(rank_steps - contract_steps)
        raise ValueError(
            "POSTPROCESS_STEP_CONTRACTS and LEGACY_STEP_RANK disagree: "
            f"steps missing a rank: {missing_rank}; ranks missing a "
            f"contract: {missing_contract}. Adding or removing a pipeline "
            "step edits both artifacts (and the pinned-pair corpus) in one "
            "reviewed diff."
        )
    ranks = [LEGACY_STEP_RANK[step] for step in LEGACY_STEP_RANK]
    if len(set(ranks)) != len(ranks):
        raise ValueError("LEGACY_STEP_RANK ranks must be unique integers.")
    # K1: every pinned pair is rank-consistent, refused BY NAME with its
    # reason — a coordinated rank+registry reversal re-orients every derived
    # edge and passes R1/R2, but cannot pass this without editing the named
    # reason-bearing corpus entry (design-ppdag-v3 §2.1).
    for (producer, consumer), pair in PINNED_ORDER_PAIRS.items():
        if producer not in LEGACY_STEP_RANK or consumer not in LEGACY_STEP_RANK:
            raise ValueError(
                f"PINNED_ORDER_PAIRS[({producer!r}, {consumer!r})] names a "
                "step absent from LEGACY_STEP_RANK."
            )
        if LEGACY_STEP_RANK[producer] >= LEGACY_STEP_RANK[consumer]:
            raise ValueError(
                f"Reordering {consumer} before {producer} contradicts "
                f"PINNED_ORDER_PAIRS[({producer!r}, {consumer!r})]: "
                f"{pair.reason}"
            )


_validate_contract_artifacts()
