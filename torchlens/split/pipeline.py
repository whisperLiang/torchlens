"""Named phases of the backend-neutral split preparation pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from hashlib import sha256
from typing import Any

from ..backends import resolve_backend_spec
from .adapters.base import SegmentBundle
from .errors import SplitUnsupportedError
from .graph import SplitTraceGraph, split_graph_from_trace
from .ir import SplitGraphIR, SplitModelProfile, SplitRequest
from .planner import SplitPlan, plan_split
from .program import (
    ReplayProgram,
    ReplaySegment,
    SplitCapabilityReport,
    build_capability_report,
    lower_replay_program,
)
from .shape_program import ShapeProgram, compile_shape_program


_MAX_EXHAUSTIVE_SHAPE_WITNESSES = 32


def _escape_pointer(value: str) -> str:
    """Escape one RFC 6901 JSON Pointer component."""

    return value.replace("~", "~0").replace("/", "~1")


def _resize_witness_tree(
    value: Any,
    path: str,
    *,
    axes: Mapping[str, int],
    batch_size: int,
    adapter: Any,
) -> Any:
    """Clone an input tree while resizing declared batch tensor leaves."""

    if adapter.is_tensor(value):
        if path in axes:
            return adapter.resize_batch(value, axes[path], batch_size)
        return adapter.clone(value)
    if isinstance(value, Mapping):
        resized = {
            key: _resize_witness_tree(
                item,
                f"{path}/{_escape_pointer(str(key))}",
                axes=axes,
                batch_size=batch_size,
                adapter=adapter,
            )
            for key, item in value.items()
        }
        if isinstance(value, dict):
            return type(value)(resized)
        return resized
    if isinstance(value, (list, tuple)):
        items = [
            _resize_witness_tree(
                item,
                f"{path}/{index}",
                axes=axes,
                batch_size=batch_size,
                adapter=adapter,
            )
            for index, item in enumerate(value)
        ]
        return type(value)(*items) if hasattr(type(value), "_fields") else type(value)(items)
    if is_dataclass(value) and not isinstance(value, type):
        updates = {
            item.name: _resize_witness_tree(
                getattr(value, item.name),
                f"{path}/{_escape_pointer(item.name)}",
                axes=axes,
                batch_size=batch_size,
                adapter=adapter,
            )
            for item in fields(value)
        }
        return replace(value, **updates)
    return value


def _witness_node_signature(node: Any) -> tuple[Any, ...]:
    """Return topology metadata that must remain stable across batch probes."""

    func_id = getattr(node.args_template, "func_id", None)
    return (
        node.op_type,
        node.module_path,
        node.output_container_path,
        node.is_input,
        node.is_output,
        node.is_buffer,
        len(node.parents),
        getattr(func_id, "namespace", None),
        getattr(func_id, "qualname", None),
        getattr(func_id, "dispatch_kind", None),
    )


def _align_shape_witness(
    graph: SplitTraceGraph,
    witness: SplitTraceGraph,
    *,
    batch_size: int,
) -> dict[str, tuple[int, ...] | None]:
    """Align one witness graph by ordered call/value topology."""

    if len(graph.nodes) != len(witness.nodes):
        raise SplitUnsupportedError(
            f"Dynamic batch {batch_size} changed graph topology from "
            f"{len(graph.nodes)} to {len(witness.nodes)} values."
        )
    shapes: dict[str, tuple[int, ...] | None] = {}
    graph_order = graph.order_by_id
    witness_order = witness.order_by_id
    for index, (base_node, witness_node) in enumerate(zip(graph.nodes, witness.nodes)):
        if _witness_node_signature(base_node) != _witness_node_signature(witness_node):
            raise SplitUnsupportedError(
                f"Dynamic batch {batch_size} changed graph topology at value {index}: "
                f"{base_node.op_type!r} != {witness_node.op_type!r}."
            )
        base_parent_indexes = tuple(
            graph_order[parent.canonical_id]
            for label in base_node.parents
            if (parent := graph.node_for_label(label)) is not None
        )
        witness_parent_indexes = tuple(
            witness_order[parent.canonical_id]
            for label in witness_node.parents
            if (parent := witness.node_for_label(label)) is not None
        )
        if base_parent_indexes != witness_parent_indexes:
            raise SplitUnsupportedError(
                f"Dynamic batch {batch_size} changed parent-call topology at value "
                f"{index}: {base_parent_indexes!r} != {witness_parent_indexes!r}."
            )
        shapes[base_node.canonical_id] = witness_node.output_shape
    return shapes


def _capture_shape_witnesses(
    model: Any,
    inputs: tuple[Any, ...],
    input_kwargs: dict[str, Any] | None,
    request: SplitRequest,
    graph: SplitTraceGraph,
    shape_program: ShapeProgram,
    adapter: Any,
) -> dict[int, dict[str, tuple[int, ...] | None]]:
    """Capture safe alternate-batch shape evidence for unresolved Torch relations."""

    if adapter.name != "torch" or not shape_program.unresolved:
        return {}
    low, high = shape_program.dynamic_batch
    batches = tuple(
        batch
        for batch in range(low, high + 1)
        if batch != shape_program.traced_batch_size
    )
    if not batches:
        return {}
    if len(batches) > _MAX_EXHAUSTIVE_SHAPE_WITNESSES:
        raise SplitUnsupportedError(
            "Dynamic shape ambiguity requires exhaustive witnesses across the declared "
            f"batch range, but {len(batches)} alternate batches exceed the safe limit "
            f"of {_MAX_EXHAUSTIVE_SHAPE_WITNESSES}. Narrow dynamic_batch or make the "
            "shape relation explicit."
        )

    from .._capture_state_helpers import _model_for_validation_replay
    from ..utils.rng import log_current_rng_states, set_rng_from_saved_states

    initial_rng = log_current_rng_states()
    witnesses: dict[int, dict[str, tuple[int, ...] | None]] = {}
    try:
        for batch_size in batches:
            set_rng_from_saved_states(initial_rng)
            witness_model, _plain_snapshot, copied = _model_for_validation_replay(model)
            if not copied:
                raise SplitUnsupportedError(
                    "Dynamic shape ambiguity requires a shape witness, but the model "
                    "could not be copied without mutating user state."
                )
            witness_inputs = tuple(
                _resize_witness_tree(
                    value,
                    f"/args/{index}",
                    axes=shape_program.input_batch_axes,
                    batch_size=batch_size,
                    adapter=adapter,
                )
                for index, value in enumerate(inputs)
            )
            witness_kwargs = {
                key: _resize_witness_tree(
                    value,
                    f"/kwargs/{_escape_pointer(str(key))}",
                    axes=shape_program.input_batch_axes,
                    batch_size=batch_size,
                    adapter=adapter,
                )
                for key, value in (input_kwargs or {}).items()
            }
            witness_capture = capture_model(
                witness_model,
                witness_inputs,
                request,
                input_kwargs=witness_kwargs,
            )
            witness_graph = split_graph_from_trace(
                witness_capture,
                batch_symbol=request.batch_symbol,
                dynamic_batch=request.dynamic_batch,
            )
            witnesses[batch_size] = _align_shape_witness(
                graph,
                witness_graph,
                batch_size=batch_size,
            )
            cleanup = getattr(witness_capture, "cleanup", None)
            if callable(cleanup):
                cleanup()
    finally:
        set_rng_from_saved_states(initial_rng)
    return witnesses


def capture_model(
    model: Any,
    inputs: tuple[Any, ...],
    spec: SplitRequest,
    *,
    input_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Capture one complete model execution with the native backend."""

    backend_spec = resolve_backend_spec(spec.backend, model, inputs, input_kwargs)
    from ..user_funcs import trace

    common: dict[str, Any] = {
        "input_kwargs": input_kwargs,
        "layers_to_save": "all",
        "keep_orphans": True,
        "backend": str(backend_spec.name),
    }
    if str(backend_spec.name) == "torch":
        common.update(
            {
                "intervention_ready": True,
                "capture_container_structure": True,
                "save_arg_values": True,
                "save_rng_states": True,
                "detach_saved_activations": False,
                "backward_ready": spec.trainable,
            }
        )
    if str(backend_spec.name) != "torch":
        return trace(model, inputs, **common)

    from .._capture_state_helpers import (
        _clone_state_dict_with_metadata,
        _ModuleTreePlainAttrSnapshot,
    )
    from ..utils.rng import log_current_rng_states, set_rng_from_saved_states

    rng_state = log_current_rng_states()
    state_dict = (
        _clone_state_dict_with_metadata(model)
        if hasattr(model, "state_dict") and hasattr(model, "load_state_dict")
        else None
    )
    training_modes = tuple(
        (module, bool(module.training))
        for module in (model.modules() if hasattr(model, "modules") else ())
    )
    plain_attrs = _ModuleTreePlainAttrSnapshot(
        model,
        ignored_names=frozenset({"_tl", "forward"}),
    )
    try:
        return trace(model, inputs, **common)
    finally:
        try:
            if state_dict is not None:
                model.load_state_dict(state_dict)
            for module, training in training_modes:
                module.train(training)
            plain_attrs.restore_changed_attrs()
        finally:
            set_rng_from_saved_states(rng_state)


def normalize_to_split_ir(
    capture: Any,
    spec: SplitRequest,
    *,
    inputs: tuple[Any, ...] = (),
    input_kwargs: dict[str, Any] | None = None,
    adapter: Any | None = None,
    plan: SplitPlan | None = None,
    model_profile: SplitModelProfile | None = None,
    model: Any | None = None,
) -> tuple[SplitTraceGraph, SplitGraphIR]:
    """Normalize a backend capture into the runtime graph and portable Split IR."""

    graph = (
        capture
        if isinstance(capture, SplitTraceGraph)
        else split_graph_from_trace(
            capture,
            batch_symbol=spec.batch_symbol,
            dynamic_batch=spec.dynamic_batch,
        )
    )
    if spec.dynamic_batch is not None:
        if adapter is None:
            from .adapters import resolve_split_adapter

            adapter = resolve_split_adapter(graph.backend)
        shape_program = compile_shape_program(
            graph,
            inputs,
            input_kwargs,
            spec,
            adapter=adapter,
        )
        if shape_program is not None and shape_program.unresolved and model is not None:
            witnesses = _capture_shape_witnesses(
                model,
                inputs,
                input_kwargs,
                spec,
                graph,
                shape_program,
                adapter,
            )
            if witnesses:
                shape_program = compile_shape_program(
                    graph,
                    inputs,
                    input_kwargs,
                    spec,
                    adapter=adapter,
                    shape_witnesses=witnesses,
                )
        if shape_program is not None:
            graph_hash = sha256(
                f"{graph.graph_shape_hash or ''}:{shape_program.fingerprint}".encode("utf-8")
            ).hexdigest()
            graph = replace(
                graph,
                graph_shape_hash=graph_hash,
                traced_batch_size=shape_program.traced_batch_size,
                shape_program=shape_program,
            )
    resolved_plan = plan if plan is not None else plan_split(graph, spec)
    return graph, SplitGraphIR.from_trace_graph(
        graph,
        plan=resolved_plan,
        profile_hash=None if model_profile is None else model_profile.profile_hash,
    )


def analyze_split_capabilities(
    adapter: Any,
    graph: SplitTraceGraph,
    plan: SplitPlan,
    spec: SplitRequest,
    *,
    prefix_program: ReplayProgram,
    suffix_program: ReplayProgram,
    graph_ir: SplitGraphIR | None = None,
    model_profile: SplitModelProfile | None = None,
    features: dict[str, Any] | None = None,
) -> SplitCapabilityReport:
    """Analyze replay, dynamic-shape, training, and cache capabilities."""

    return build_capability_report(
        adapter,
        graph,
        plan,
        spec,
        prefix_program=prefix_program,
        suffix_program=suffix_program,
        graph_ir=graph_ir,
        model_profile=model_profile,
        features=features,
    )


def lower_split_program(
    graph: SplitTraceGraph,
    plan: SplitPlan,
    spec: SplitRequest,
    *,
    adapter: Any,
    segment: ReplaySegment,
) -> ReplayProgram:
    """Lower one IR segment through the selected backend adapter."""

    return lower_replay_program(graph, plan, spec, segment=segment, adapter=adapter)


def execute_split_runtime(
    adapter: Any,
    graph: SplitTraceGraph,
    plan: SplitPlan,
    spec: SplitRequest,
) -> SegmentBundle:
    """Build executable backend recipes from a lowered split plan."""

    return adapter.build_segments(graph, plan, spec)


__all__ = [
    "analyze_split_capabilities",
    "capture_model",
    "execute_split_runtime",
    "lower_split_program",
    "normalize_to_split_ir",
]
