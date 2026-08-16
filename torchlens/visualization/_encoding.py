"""Encoding-channel core (S5/L5): declarative value -> visual channels for draw().

v1 ships the COLOR channel (``color_by``): a value source (Layer/Op field
name, scalar node-overlay builtin, or callable ``node -> value``) compiled
down to the existing NodeSpec chain as a fillcolor transform in the C3
precedence slot (after the node-style preset and intervention styling, before
the user ``node_spec_fn``), plus an auto-legend that disclosures the
transform. Size (``size_by``) and rank (``stack_by``) channels are wave-1.

Resolution is TWO-PHASE (design memo 2.3):

* PHASE A -- a presentation PREPASS over the already-chosen visible-node
  universe (:func:`populate_encoding_state`, called from
  ``build_render_ir``): collect each eligible node's raw value exactly once
  (user callables are invoked once per node HERE and never again), apply the
  closed value/type rules and the rolled-aggregate source allowlist, and
  min-max normalize over the finite values. This is a data prepass over
  records; it renders nothing.
* PHASE B -- during per-node spec resolution, apply the precomputed color as
  a spec transform (:meth:`EncodingState.fillcolor_for`); the user
  ``node_spec_fn`` still sees and may override the channel's output.

NAMING: ``color_by``, the ``encoding_*`` error codes, and the ``show_legend``
tri-state are DOCUMENTED-UNSTABLE spellings (no deprecation shim owed) until
the naming session ratifies them (METAPLAN naming protocol).

HONESTY TRIPWIRE (never weaken): an encoding must never imply uniformity it
cannot prove. On a rolled multi-pass Layer node, a FIELD source resolves
through the NAME-KEYED rolled-aggregate allowlist below; a source in no
declared row REFUSES rather than silently projecting pass-1 or an aggregate
bound.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .._errors import InvalidArgumentError

if TYPE_CHECKING:  # pragma: no cover - typing only
    import graphviz

    from ..data_classes.trace import Trace
    from .themes import VisualizationTheme

# ---------------------------------------------------------------------------
# Rolled-aggregate source allowlist rows (design memo 2.3b, r4 shape).
#
# A FIELD or BUILTIN source resolving on a rolled MULTI-PASS Layer node
# (len(layer.ops) > 1) is classified by the NAME-KEYED table below. A name in
# no row refuses ``encoding_source_invalid`` at the prepass (allowlist
# default): a new record field cannot silently become encodable.
# ---------------------------------------------------------------------------

#: All passes agree unless annotations["varying_across_passes"] names the
#: field (the finalization reconciler's any-variation marker). Unmarked ->
#: exact -> encode; marked -> the stored aggregate is a per-pass MAXIMUM
#: (upper bound) -> color UNENCODES + legend note (size_by will refuse).
ROW_RECONCILED = "reconciled_aggregate"
#: Exact cross-pass aggregate (sum or distinct-union). Uniformly defined on
#: every node ("this node's total over the whole trace"), encode WITH a
#: mandatory aggregation legend line.
ROW_SUMMED = "summed_aggregate"
#: Mirrored per-call NUMERICS: stored names resolving through
#: _LAYER_MIRROR_SPEC to the representative FIRST pass with NO variation
#: marker and NO reconciliation -- a first-pass projection that cannot be
#: certified single-valued. Color unencodes + legend note; size refuses.
ROW_MIRRORED_PER_CALL = "mirrored_per_call_numeric"
#: Value determined by the grouping identity (function, parameters, output
#: slot, module address, non-tensor args) or by group construction
#: (num_passes). Encodes normally; op-backed members additionally get a
#: DEFENSIVE per-pass equality check at resolution (mismatch degrades to
#: unencoded + note rather than painting a dishonest uniform value).
ROW_STRUCTURAL = "structurally_uniform"
#: Properties computed from other classified fields; they inherit the MOST
#: CONSERVATIVE verdict among their inputs (see _DERIVED_COMPOSITE_INPUTS).
ROW_DERIVED = "derived_composite"
#: Per-pass attributes with no aggregate meaning: the multipass-safe read
#: degrades to None on a rolled aggregate -> unencoded + "n/a" legend note.
ROW_PER_PASS = "per_pass"
#: Shape-valued sources: never a scalar color source (size_by consumes them
#: via the wave-1 "dims" typed shape path only).
ROW_SHAPE = "shape"
#: Everything else readable off the record: strings, containers, callables,
#: bools (a bool is a truth value, not an encodable magnitude), weakrefs.
#: Resolving one raises the 2.2 wrong-type rule (encoding_value_invalid).
ROW_NON_NUMERIC = "non_numeric"

#: THE TRUE MIRRORED PER-CALL NUMERIC ENUMERATION (sol r4 MAJOR-1 residual
#: fix). The design memo claimed this set was exactly
#: {transformed_gradient_memory}; the live capture/postprocess code disproves
#: that five ways:
#:   raw_index               -- fresh incremented counter per emitted op
#:   step_index              -- labeling step 9 reassigns SEQUENTIALLY PER OP
#:                              (labeling.py: layer_entry.step_index = step_index)
#:   ordinal_index           -- zero-based position of EACH op in the final
#:                              ordered layer list (labeling.py step 11)
#:   grad_fn_object_id       -- per-call autograd object identity
#:   buffer_pass             -- sequential per-address buffer version number
#:                              (control_flow.py step 6)
#:   transformed_gradient_memory -- per-call Bytes, no marker (the memo's one)
#: Plus the first-pass-projection numeric PROPERTY conditional_depth
#: (len over the mirror-copied in_conditionals) classified in the same row.
MIRRORED_PER_CALL_NUMERIC_FIELDS = frozenset(
    {
        "raw_index",
        "step_index",
        "ordinal_index",
        "grad_fn_object_id",
        "buffer_pass",
        "transformed_gradient_memory",
        "conditional_depth",
    }
)

_RECONCILED_FIELDS = frozenset(
    {"activation_memory", "transformed_activation_memory", "flops_forward", "flops_backward"}
)

_SUMMED_FIELDS = frozenset(
    {
        # Stored fields _build_layer_logs overwrites with cross-pass sums.
        "autograd_memory",
        "total_autograd_memory",
        "num_autograd_tensors",
        # total_* sum properties over Layer.ops.
        "total_activation_memory",
        "total_gradient_memory",
        "total_flops_forward",
        "total_flops_backward",
        "total_flops_total",
        "total_macs_forward",
        "total_macs_backward",
        "total_macs_total",
        "total_func_duration",
        # Aggregate distinct-union cardinalities (exact on the rolled node).
        "num_children",
        "num_parents",
    }
)

#: Legend wording per summed-family source (default: "total across passes").
_AGGREGATE_LEGEND_WORDING = {
    "num_children": "distinct across all passes",
    "num_parents": "distinct across all passes",
}

_STRUCTURAL_FIELDS = frozenset(
    {
        "type_index",  # inherited from pass 1 for every pass (labeling step 8)
        "num_passes",  # group-level, assigned to every member
        "num_ops",  # == number of passes
        "num_args_total",
        "num_pos_args",
        "num_kwargs",
        "multi_output_index",  # output slot, part of the grouping identity
        "num_params",
        "num_params_trainable",
        "num_params_frozen",
        "total_param_memory",
        "num_param_tensors",
        "num_param_tensors_trainable",
        "num_param_tensors_frozen",
    }
)

#: Derived-composite properties -> their input fields. Verdict = the most
#: conservative verdict among the inputs' rows (aliasing rule: verdicts key
#: on the DECLARED row, never on string-prefix accident).
_DERIVED_COMPOSITE_INPUTS: dict[str, tuple[str, ...]] = {
    "flops_total": ("flops_forward", "flops_backward"),
    "macs_forward": ("flops_forward",),
    "macs_backward": ("flops_backward",),
    "macs_total": ("flops_forward", "flops_backward"),
    "buffer_overwrite_index": ("buffer_pass",),
    # len(modules): the reconciler writes a "modules" variation marker.
    "module_call_depth": ("modules",),
}

_PER_PASS_FIELDS = frozenset({"func_duration", "fx_call_index", "pass_index"})

_SHAPE_FIELDS = frozenset({"shape", "transformed_out_shape"})

#: Sources classified per-name for rolled multi-pass resolution. Built below
#: from the row sets plus the NON_NUMERIC remainder of the readable Layer
#: namespace; the classification completeness pin
#: (tests/test_encoding_channels.py) asserts this dict covers EXACTLY
#: _LAYER_STATE_ORDER + _LAYER_MIRROR_SPEC keys + the public Layer
#: properties, each name in exactly one row.
_NON_NUMERIC_FIELDS = frozenset(
    {
        # Identity / naming strings.
        "layer_label",
        "layer_label_short",
        "layer_type",
        "label",
        "label_short",
        "fx_label",
        "fx_qualpath",
        "lookup_keys",
        "op_labels",
        "call_labels",
        # Function / autograd objects and strings.
        "func",
        "func_name",
        "func_qualname",
        "func_config",
        "func_rng_states",
        "grad_fn",
        "grad_fn_handle",
        "grad_fn_class_name",
        "grad_fn_class_qualname",
        "code_context",
        "arg_names",
        "saved_args",
        "saved_kwargs",
        # Bools (a truth value is not an encodable magnitude; the runtime
        # wrong-type rule rejects bool explicitly).
        "is_inplace",
        "in_multi_output",
        "is_input",
        "input_was_parameter",
        "is_output",
        "is_final_output",
        "is_buffer",
        "is_buffer_source",
        "is_compute_layer",
        "is_internal_source",
        "is_internal_sink",
        "is_terminal_bool",
        "is_scalar_bool",
        "bool_value",
        "buffer_value_changed",
        "buffer_replay_validated",
        "has_input_ancestor",
        "is_atomic_module",
        "intervention_replaced",
        "detach_saved_activations",
        "save_grads",
        "edges_vary_across_ops",
        "has_children",
        "has_co_parents",
        "has_frozen_params",
        "has_grad",
        "has_parents",
        "has_saved_activation",
        "has_siblings",
        "has_trainable_params",
        "in_submodule",
        "is_in_conditional",
        "is_in_conditional_body",
        "is_in_conditional_evaluation",
        "is_module_input",
        "is_orphan",
        "uses_params",
        # Dtypes / devices / addresses / roles (marker-only family included).
        "dtype",
        "dtype_ref",
        "transformed_out_dtype",
        "transformed_grad_dtype",
        "device_ref",
        "output_device",
        "backend_address",
        "resolver_status",
        "address",
        "io_role",
        "equivalence_class",
        "multi_output_name",
        "buffer_source",
        "buffer_write_kind",
        "buffer_source_func_name",
        "visualizer_path",
        "activation_transform",
        # Containers / graph views / payloads.
        "transformed_grad_shape",
        "param_shapes",
        "params",
        "param_names",
        "param_dtypes",
        "modules",
        "module",
        "output_of_modules",
        "output_of_module_calls",
        "in_conditionals",
        "terminal_bool_for",
        "conditional_entry_children",
        "conditional_then_children",
        "conditional_elif_children",
        "conditional_else_children",
        "conditional_arm_children",
        "conditional_role_stacks",
        "conditional_branch_stack_ops",
        "annotations",
        "equivalent_ops",
        "ops",
        "children",
        "parents",
        "children_per_pass",
        "parents_per_pass",
        "child_ops_per_layer",
        "parent_ops_per_layer",
        "parent_arg_positions",
        "co_parents",
        "siblings",
        "leaf_module_ops",
        "out",
        "grad",
        "tensor",
        "transformed_out",
        "transformed_grad",
        "receptive_field",
        "projective_field",
        "source_trace",
        "trace",
        # Private stored state.
        "_source_trace_ref",
        "_is_in_conditional_body",
        "_param_barcodes",
        "_param_logs",
    }
)


def _build_source_rows() -> dict[str, str]:
    """Build the name -> row classification table."""

    rows: dict[str, str] = {}
    for names, row in (
        (_RECONCILED_FIELDS, ROW_RECONCILED),
        (_SUMMED_FIELDS, ROW_SUMMED),
        (MIRRORED_PER_CALL_NUMERIC_FIELDS, ROW_MIRRORED_PER_CALL),
        (_STRUCTURAL_FIELDS, ROW_STRUCTURAL),
        (frozenset(_DERIVED_COMPOSITE_INPUTS), ROW_DERIVED),
        (_PER_PASS_FIELDS, ROW_PER_PASS),
        (_SHAPE_FIELDS, ROW_SHAPE),
        (_NON_NUMERIC_FIELDS, ROW_NON_NUMERIC),
    ):
        for name in names:
            if name in rows:  # pragma: no cover - guarded by the completeness pin
                raise AssertionError(f"source {name!r} classified into two rows")
            rows[name] = row
    return rows


LAYER_SOURCE_ROWS: dict[str, str] = _build_source_rows()

# ---------------------------------------------------------------------------
# Builtin scalar sources: the node_overlay builtin names where scalar, so the
# two surfaces converge (memo 2.2). "nan" is boolean-valued and NOT a scalar
# color source. Field-backed builtins resolve through the FIELD path so the
# rolled-aggregate allowlist governs them; payload builtins (magnitude,
# grad_norm) resolve through the multipass-safe overlay helper.
# ---------------------------------------------------------------------------
_BUILTIN_FIELD_SOURCES = {
    "flops": "flops_forward",
    "bytes": "activation_memory",
    "time": "func_duration",
}
_BUILTIN_PAYLOAD_SOURCES = frozenset({"magnitude", "grad_norm"})
SCALAR_BUILTIN_SOURCES = frozenset(_BUILTIN_FIELD_SOURCES) | _BUILTIN_PAYLOAD_SOURCES

#: Legend notes (fixed wording pinned by tests).
NOTE_NA_UNENCODED = "n/a = unencoded"
NOTE_VARIES = "varies across passes -- unencoded"
NOTE_FIRST_PASS_ONLY = "first-pass-only field -- unencoded on rolled nodes"
NOTE_CONSTANT = "constant value -- midpoint encoding"
NOTE_CALLABLE = "value from user callable"

#: Sequential colormap anchors (Okabe-Ito adjacent, colorblind-safe).
LIGHT_RAMP = ("#FFFFFF", "#0072B2")
DARK_RAMP = ("#1F2937", "#56B4E9")


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    """Parse ``#RRGGBB`` into an RGB tuple."""

    stripped = color.lstrip("#")
    return (int(stripped[0:2], 16), int(stripped[2:4], 16), int(stripped[4:6], 16))


def interpolate_hex(start: str, end: str, fraction: float) -> str:
    """Linearly interpolate between two ``#RRGGBB`` colors.

    The ONE colormap interpolation home (generalized from
    ``bundle_diff._interpolate``, which now consumes this helper).

    Parameters
    ----------
    start:
        Start color.
    end:
        End color.
    fraction:
        Interpolation fraction in ``[0, 1]``.

    Returns
    -------
    str
        Interpolated hex color.
    """

    fraction = max(0.0, min(1.0, fraction))
    start_rgb = _hex_to_rgb(start)
    end_rgb = _hex_to_rgb(end)
    values = [
        round(start_value + (end_value - start_value) * fraction)
        for start_value, end_value in zip(start_rgb, end_rgb, strict=True)
    ]
    return "#{:02X}{:02X}{:02X}".format(*values)


@dataclass(frozen=True)
class EncodingChannelSpec:
    """One resolved channel request (option-validation product).

    Parameters
    ----------
    channel:
        Channel kind; v1 supports ``"color"`` only.
    source_kind:
        ``"builtin"``, ``"field"``, or ``"callable"``.
    source:
        The validated user source (token, field name, or callable).
    display_name:
        Human-readable source name for legend disclosure.
    """

    channel: str
    source_kind: str
    source: Any
    display_name: str


@dataclass
class EncodingState:
    """Mutable per-draw channel state: prepass products + legend disclosures.

    Created at request resolution, populated exactly once by the Phase-A
    prepass in ``build_render_ir``, and read by Phase B and the legend.
    """

    spec: EncodingChannelSpec
    dark_theme: bool = False
    populated: bool = False
    colors: dict[str, str] = field(default_factory=dict)
    values: dict[str, float] = field(default_factory=dict)
    domain: tuple[float, float] | None = None
    notes: list[str] = field(default_factory=list)
    aggregation_lines: list[str] = field(default_factory=list)
    eligible_count: int = 0

    def note(self, text: str) -> None:
        """Record a legend note once."""

        if text not in self.notes:
            self.notes.append(text)

    def fillcolor_for(self, node: Any) -> str | None:
        """Phase B: return the precomputed fill for ``node`` (None = unencoded)."""

        return self.colors.get(_node_key(node))

    @property
    def ramp(self) -> tuple[str, str]:
        """Return the theme-aware sequential ramp anchors."""

        return DARK_RAMP if self.dark_theme else LIGHT_RAMP


def _encoding_error(
    problem: str, *, code: str, remedy: str, **context: Any
) -> InvalidArgumentError:
    """Build a typed encoding refusal."""

    return InvalidArgumentError(problem, code=code, remedy=remedy, **context)


def resolve_color_by(color_by: Any) -> EncodingChannelSpec | None:
    """Validate ``color_by`` at option validation, before any render work.

    Parameters
    ----------
    color_by:
        ``None``, a field-name string, a scalar builtin token, or a callable
        ``node -> value``.

    Returns
    -------
    EncodingChannelSpec | None
        Resolved channel spec, or ``None`` when the channel is inactive.

    Raises
    ------
    InvalidArgumentError
        ``encoding_source_invalid`` for an unknown field name / builtin
        token or a non-string non-callable source.
    """

    if color_by is None:
        return None
    if callable(color_by) and not isinstance(color_by, str):
        name = getattr(color_by, "__name__", type(color_by).__name__)
        return EncodingChannelSpec(
            channel="color",
            source_kind="callable",
            source=color_by,
            display_name=f"callable {name}",
        )
    if isinstance(color_by, str):
        normalized = color_by.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in SCALAR_BUILTIN_SOURCES:
            return EncodingChannelSpec(
                channel="color",
                source_kind="builtin",
                source=normalized,
                display_name=normalized,
            )
        from ..constants import LAYER_PASS_LOG_FIELD_ORDER

        if color_by in LAYER_SOURCE_ROWS or color_by in LAYER_PASS_LOG_FIELD_ORDER:
            return EncodingChannelSpec(
                channel="color",
                source_kind="field",
                source=color_by,
                display_name=color_by,
            )
        raise _encoding_error(
            f"color_by source {color_by!r} is not a known record field, scalar "
            "builtin, or callable",
            code="encoding_source_invalid",
            remedy=(
                "pass a Layer/Op field name, one of the scalar builtins "
                f"({', '.join(sorted(SCALAR_BUILTIN_SOURCES))}), or a callable node -> value"
            ),
            argument="color_by",
        )
    raise _encoding_error(
        f"color_by must be a field-name string, builtin token, or callable; "
        f"received {type(color_by).__name__}",
        code="encoding_source_invalid",
        remedy="pass a string source name or a callable node -> value",
        argument="color_by",
    )


def _node_key(node: Any) -> str:
    """Stable per-draw key for a rendered record.

    Rolled Layer nodes key by ``layer_label``; per-pass Op nodes by their
    pass-qualified ``label``. The prepass and Phase B call this on the SAME
    record object, so the keying is consistent within one draw.
    """

    from ..data_classes.layer import Layer

    if isinstance(node, Layer):
        layer_label = getattr(node, "layer_label", None)
        if isinstance(layer_label, str):
            return layer_label
    from ..utils._multipass_access import get_multipass_attr

    label = get_multipass_attr(node, "label", None, multipass=None)
    if isinstance(label, str):
        return label
    layer_label = getattr(node, "layer_label", None)
    return layer_label if isinstance(layer_label, str) else str(id(node))


def _is_rolled_multipass(node: Any) -> bool:
    """Return whether ``node`` is a rolled multi-pass aggregate Layer."""

    from ..data_classes.layer import Layer

    return isinstance(node, Layer) and len(node.ops) > 1


def is_record_derived_image_node(trace: Trace, node: Any) -> bool:
    """THE closed image-origin predicate for pre-user image nodes (memo 2.4(i)).

    Covers all three record-derived image mechanisms: the visualizer_path
    branch, the annotation-image branch, and the raw-input montage input
    node. Image nodes are excluded from channel encoding AND from the
    normalization domain. Any future record-derived image producer must
    extend THIS predicate and its test pair, never add a local check
    elsewhere (classification drift between the trace-bearing prepass and
    the NodeSpec funnel is this design's named recurring risk).
    """

    visualizer_path = getattr(node, "visualizer_path", None)
    if isinstance(visualizer_path, str) and visualizer_path.lower().endswith(".png"):
        return True
    from ._render_nodes import _annotation_image_path_for_node

    if _annotation_image_path_for_node(trace, node) is not None:
        return True
    # Raw-input visual branch (montage / preview): input nodes whose label is
    # decorated from trace.raw_input by the post-user raw merge.
    return bool(getattr(node, "is_input", False)) and getattr(trace, "raw_input", None) is not None


def _coerce_scalar(state: EncodingState, node: Any, value: Any) -> float | None:
    """Apply the closed value/type rules (memo 2.2) to one resolved value."""

    if value is None:
        state.note(NOTE_NA_UNENCODED)
        return None
    if isinstance(value, bool):
        raise _encoding_error(
            f"color_by source {state.spec.display_name!r} produced a bool on node "
            f"{_node_key(node)!r}; a truth value is not an encodable magnitude",
            code="encoding_value_invalid",
            remedy="encode a numeric field, or map the bool to a number in a callable",
            argument="color_by",
        )
    if isinstance(value, (int, float)):
        as_float = float(value)
        if not math.isfinite(as_float):
            state.note(NOTE_NA_UNENCODED)
            return None
        return as_float
    numel = getattr(value, "numel", None)
    item = getattr(value, "item", None)
    if callable(numel) and callable(item):
        if numel() == 1:
            # Documented: float(x.item()) forces a device sync the user
            # opted into by passing a tensor-returning source.
            return _coerce_scalar(state, node, item())
        raise _encoding_error(
            f"color_by source {state.spec.display_name!r} produced a non-scalar "
            f"tensor on node {_node_key(node)!r}",
            code="encoding_value_invalid",
            remedy="reduce the tensor to one element (e.g. .mean()) in the callable",
            argument="color_by",
        )
    raise _encoding_error(
        f"color_by source {state.spec.display_name!r} produced "
        f"{type(value).__name__!r} on node {_node_key(node)!r}; the color channel "
        "accepts python ints/floats and 1-element tensors",
        code="encoding_value_invalid",
        remedy="pick a numeric source or convert the value in a callable",
        argument="color_by",
    )


def _varying_marker(node: Any) -> dict[str, Any]:
    """Return the reconciler's variation marker for a rolled Layer."""

    annotations = getattr(node, "annotations", None)
    if isinstance(annotations, dict):
        marker = annotations.get("varying_across_passes")
        if isinstance(marker, dict):
            return marker
    return {}


def _mirror_backed_op_field(field_name: str) -> str | None:
    """Return the Op field backing a mirrored Layer name, if any."""

    from ..data_classes._layer_spec import _LAYER_MIRROR_SPEC

    entry = _LAYER_MIRROR_SPEC.get(field_name)
    return entry[0] if entry is not None else None


def _rolled_reconciled(state: EncodingState, node: Any, field_name: str) -> float | None:
    """Reconciled family: marker-varying -> unencode; unmarked -> exact encode."""

    if field_name in _varying_marker(node):
        state.note(NOTE_VARIES)
        return None
    return _coerce_scalar(state, node, getattr(node, field_name, None))


def _rolled_summed(state: EncodingState, node: Any, field_name: str) -> float | None:
    """Summed family: exact aggregate, mandatory aggregation legend line."""

    value = _coerce_scalar(state, node, getattr(node, field_name, None))
    if value is not None:
        wording = _AGGREGATE_LEGEND_WORDING.get(field_name, "total across passes")
        line = f"{field_name}: {wording} on rolled nodes"
        if line not in state.aggregation_lines:
            state.aggregation_lines.append(line)
    return value


def _rolled_mirrored(state: EncodingState, node: Any, field_name: str) -> float | None:
    """Mirrored per-call numeric: never certified single-valued -> unencode."""

    del node, field_name
    state.note(NOTE_FIRST_PASS_ONLY)
    return None


def _rolled_derived(state: EncodingState, node: Any, field_name: str) -> float | None:
    """Derived composite: inherit the most conservative input verdict."""

    marker = _varying_marker(node)
    for input_name in _DERIVED_COMPOSITE_INPUTS[field_name]:
        if LAYER_SOURCE_ROWS.get(input_name) == ROW_MIRRORED_PER_CALL:
            state.note(NOTE_FIRST_PASS_ONLY)
            return None
        if input_name in marker:
            state.note(NOTE_VARIES)
            return None
    return _coerce_scalar(state, node, getattr(node, field_name, None))


def _rolled_structural(state: EncodingState, node: Any, field_name: str) -> float | None:
    """Structurally uniform: encode, with a defensive op-backed equality check.

    The structural premise is the live code's own declaration; a
    counterexample degrades honestly instead of painting a dishonest uniform
    value.
    """

    op_field = _mirror_backed_op_field(field_name)
    if op_field is not None:
        per_pass = [getattr(node.ops.get(index), op_field, None) for index in sorted(node.ops)]
        if len({repr(value) for value in per_pass}) > 1:
            state.note(NOTE_VARIES)
            return None
    return _coerce_scalar(state, node, getattr(node, field_name, None))


def _rolled_per_pass(state: EncodingState, node: Any, field_name: str) -> float | None:
    """Per-pass attribute: multipass-safe read degrades to honest n/a."""

    from ..utils._multipass_access import get_multipass_attr

    value = get_multipass_attr(node, field_name, None, multipass=None)
    if value is None:
        state.note(NOTE_NA_UNENCODED)
        return None
    return _coerce_scalar(state, node, value)


def _rolled_wrong_type_read(state: EncodingState, node: Any, field_name: str) -> float | None:
    """Shape / non-numeric rows: read and let the wrong-type rule speak."""

    from ..utils._multipass_access import get_multipass_attr

    return _coerce_scalar(state, node, get_multipass_attr(node, field_name, None, multipass=None))


_ROLLED_ROW_HANDLERS = {
    ROW_RECONCILED: _rolled_reconciled,
    ROW_SUMMED: _rolled_summed,
    ROW_MIRRORED_PER_CALL: _rolled_mirrored,
    ROW_DERIVED: _rolled_derived,
    ROW_STRUCTURAL: _rolled_structural,
    ROW_PER_PASS: _rolled_per_pass,
    ROW_SHAPE: _rolled_wrong_type_read,
    ROW_NON_NUMERIC: _rolled_wrong_type_read,
}


def _resolve_field_on_rolled(state: EncodingState, node: Any, field_name: str) -> float | None:
    """Resolve a FIELD source on a rolled multi-pass Layer per the allowlist."""

    row = LAYER_SOURCE_ROWS.get(field_name)
    if row is None:
        raise _encoding_error(
            f"color_by source {field_name!r} has no declared rolled-aggregate "
            f"semantics row and cannot resolve on rolled multi-pass node "
            f"{_node_key(node)!r}",
            code="encoding_source_invalid",
            remedy=(
                "unroll the graph (vis_mode='unrolled'), use a callable that "
                "asserts its own aggregate semantics, or classify the field in "
                "the rolled-aggregate allowlist"
            ),
            argument="color_by",
        )
    return _ROLLED_ROW_HANDLERS[row](state, node, field_name)


def _resolve_source_value(state: EncodingState, trace: Trace, node: Any) -> float | None:
    """Resolve one node's raw channel value (Phase A, exactly once per node)."""

    spec = state.spec
    if spec.source_kind == "callable":
        try:
            value = spec.source(node)
        except Exception as error:
            raise _encoding_error(
                f"color_by callable {spec.display_name!r} raised on node "
                f"{_node_key(node)!r}: {error}",
                code="encoding_callable_error",
                remedy=(
                    "fix the callable; per-pass reads off rolled aggregates trip "
                    "the multipass tripwire -- read Layer.ops for per-pass truth"
                ),
                argument="color_by",
            ) from error
        state.note(NOTE_CALLABLE)
        return _coerce_scalar(state, node, value)

    if spec.source_kind == "builtin":
        token = spec.source
        if token in _BUILTIN_PAYLOAD_SOURCES:
            from .overlays import builtin_overlay_value

            value = builtin_overlay_value(node, token)
            if value is None:
                state.note(NOTE_NA_UNENCODED)
                return None
            return _coerce_scalar(state, node, value)
        field_name = _BUILTIN_FIELD_SOURCES[token]
    else:
        field_name = spec.source

    if _is_rolled_multipass(node):
        return _resolve_field_on_rolled(state, node, field_name)
    from ..utils._multipass_access import get_multipass_attr

    value = get_multipass_attr(node, field_name, None, multipass=None)
    if value is None:
        state.note(NOTE_NA_UNENCODED)
        return None
    return _coerce_scalar(state, node, value)


def populate_encoding_state(state: EncodingState, trace: Trace, universe: Any) -> None:
    """PHASE A: collect values over the visible-node universe and normalize.

    Runs exactly once per draw (``build_render_ir`` calls it before any
    per-node spec resolution). User callables are invoked exactly once per
    visible eligible node here; Phase B only reads the precomputed map.
    """

    if state.populated:
        return
    state.populated = True

    raw_values: dict[str, float] = {}
    for unit in universe.units:
        emission = unit.emission
        if emission.kind != "raw_op" or emission.node is None:
            continue
        node = emission.node
        if is_record_derived_image_node(trace, node):
            continue
        state.eligible_count += 1
        value = _resolve_source_value(state, trace, node)
        if value is None:
            continue
        raw_values[_node_key(node)] = value

    if not raw_values:
        state.note(NOTE_NA_UNENCODED)
        return

    low = min(raw_values.values())
    high = max(raw_values.values())
    state.domain = (low, high)
    state.values = raw_values
    start, end = state.ramp
    if low == high:
        state.note(NOTE_CONSTANT)
        state.colors = {key: interpolate_hex(start, end, 0.5) for key in raw_values}
        return
    span = high - low
    state.colors = {
        key: interpolate_hex(start, end, (value - low) / span) for key, value in raw_values.items()
    }


def _format_domain_value(state: EncodingState, value: float) -> str:
    """Format a domain endpoint for the legend (builtin-aware)."""

    if state.spec.source_kind == "builtin":
        from .overlays import format_overlay_value

        formatted = format_overlay_value(state.spec.source, value)
        return formatted.split(": ", 1)[-1]
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.4g}"


def add_channel_legend_to_graphviz(
    dot: graphviz.Digraph, theme: VisualizationTheme, state: EncodingState
) -> None:
    """Emit the channel disclosure legend (memo 2.3 disclosure contract).

    Every legend drawn states the active channel's transform ("linear
    min-max"), its source, min/mid/max swatch rows with formatted values,
    and any rolled-aggregate / unencoded notes. All text routes through the
    NodeSpec choke point (escaped like every node label).
    """

    from ._render_leaf import _node_spec_to_graphviz_args
    from .node_spec import NodeSpec
    from .themes import apply_theme_to_spec

    with dot.subgraph(name="cluster_torchlens_encoding_legend") as legend:
        legend.attr(
            label="TorchLens encoding",
            labelloc="t",
            color=theme.default_border,
            fontcolor=theme.default_font,
            style="rounded",
        )
        start, end = state.ramp
        rows: list[NodeSpec] = []
        title_lines = [f"color_by: {state.spec.display_name}", "linear min-max"]
        for line in state.aggregation_lines:
            title_lines.append(line)
        for note in state.notes:
            title_lines.append(note)
        rows.append(NodeSpec(lines=title_lines, shape="box", style="filled,rounded"))
        if state.domain is not None:
            low, high = state.domain
            mid = (low + high) / 2.0
            for tag, fraction, value in (
                ("min", 0.0, low),
                ("mid", 0.5, mid),
                ("max", 1.0, high),
            ):
                rows.append(
                    NodeSpec(
                        lines=[f"{tag}: {_format_domain_value(state, value)}"],
                        shape="box",
                        fillcolor=interpolate_hex(start, end, fraction),
                    )
                )
        for index, spec in enumerate(rows):
            node_args = _node_spec_to_graphviz_args(apply_theme_to_spec(spec, theme))
            node_args["name"] = f"tl_encoding_legend_{index}"
            legend.node(**node_args)


#: Notice emitted when an active channel forces the dot engine where AUTO
#: would have chosen the rank backend by cost (memo 2.1). Forcing dot on a
#: >threshold-cost graph is the case the rank backend exists for -- expect
#: layout time, bounded by the render timeout.
ENCODING_FORCES_DOT_NOTICE = (
    "An active encoding channel ({channel}) requires the Graphviz dot layout, "
    "overriding the automatic rank-layout choice for this graph "
    "(estimated cost={cost} > threshold={threshold}). Expect longer layout "
    "time, bounded by the render timeout."
)


def attach_encoding_state(request: Any, channel_spec: EncodingChannelSpec, theme: Any) -> Any:
    """Return ``request`` with a fresh per-draw :class:`EncodingState` attached."""

    from dataclasses import replace

    return replace(
        request,
        encoding=EncodingState(spec=channel_spec, dark_theme=theme.name == "dark"),
    )


def resolve_encoding_engine(requested_engine: str, resolved_engine: str, layout_cost: int) -> str:
    """Apply the engine-resolution fence for an ACTIVE channel (memo 2.1).

    v1 encoding channels are dot-layout-only. EXPLICIT ``layout="rank"``
    refuses typed HERE (the earliest point the conflict is decidable); AUTO
    forces dot -- exactly mirroring the ``show_containers`` precedent -- with
    a notice when the force overrides what AUTO would have chosen by cost.
    """

    if requested_engine == "rank":
        raise _encoding_error(
            "encoding channels (color_by) require the Graphviz dot layout; "
            "explicit layout='rank' cannot render them",
            code="encoding_requires_dot_layout",
            remedy="pass layout='dot' or layout='auto', or drop color_by",
            argument="layout",
        )
    if resolved_engine == "rank":
        import warnings

        from ..utils.display import user_stacklevel
        from ._rank_layout_internal import layout as _rank_layout

        warnings.warn(
            ENCODING_FORCES_DOT_NOTICE.format(
                channel="color_by",
                cost=layout_cost,
                threshold=_rank_layout.RANK_LAYOUT_COST_THRESHOLD,
            ),
            stacklevel=user_stacklevel(),
        )
    return "dot"


def channel_wrapped_node_spec_fn(
    encoding: EncodingState,
    node: Any,
    node_spec_fn: Any,
) -> Any:
    """Wrap ``node_spec_fn`` with the channel's per-node fill transform.

    PHASE B application site: the precomputed fill applies in the C3 slot
    (after the node-style preset, before the user callback, which still sees
    and may override it). Keyed on the rendered node itself: unrolled
    per-pass Op nodes encode their OWN pass value even though the user
    callback receives the aggregate Layer.
    """

    channel_fill = encoding.fillcolor_for(node)
    if channel_fill is None:
        return node_spec_fn

    def channel_then_user(layer_log: Any, spec: Any) -> Any:
        spec = spec.replace(fillcolor=channel_fill)
        if node_spec_fn is None:
            return spec
        result = node_spec_fn(layer_log, spec)
        return spec if result is None else result

    return channel_then_user


def raise_encoding_dagua_refusal() -> None:
    """Refuse an active channel on the dagua renderer (never a silent drop).

    An active channel silently dropped by the alternate label path would be
    a dishonest no-op; channels are Graphviz-dot-only in v1.
    """

    raise _encoding_error(
        "encoding channels (color_by) are not supported by the dagua renderer",
        code="encoding_requires_dot_layout",
        remedy="use the graphviz renderer, or drop color_by",
        argument="vis_renderer",
    )


def maybe_add_channel_legend(dot: Any, theme: Any, request: Any) -> None:
    """Emit the channel disclosure legend per the tri-state visibility rule.

    None (AUTO) or True with an active channel -> channel legend; explicit
    False is honored (a deliberate act; the docs state the encoding is then
    undisclosed).
    """

    if request.encoding is not None and request.show_legend is not False:
        add_channel_legend_to_graphviz(dot, theme, request.encoding)
