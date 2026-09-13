"""Named phases of the backend-neutral split preparation pipeline."""

from __future__ import annotations

import json
import sys
import types
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass, replace
from hashlib import sha256
from typing import Any, Literal

from ..backends import resolve_backend_spec
from ..backends.registry import JAX_BACKEND_NAME, TINYGRAD_BACKEND_NAME, TORCH_BACKEND_NAME
from .adapters.base import SegmentBundle
from .batch_probe import BatchProbeResult
from .batching import (
    FALLBACK_TRACE_BATCH,
    BatchSpec,
    canonical_batch_for,
    clone_dataclass_fields,
    rebatch_inputs,
    resolve_batch_spec,
    witness_probe_sizes,
)
from .errors import SplitUnsupportedError
from .graph import SplitTraceGraph, project_symbolic_shapes, split_graph_from_trace
from .ir import SplitGraphIR, SplitModelProfile, SplitRequest
from .placement import PlacementPlan
from .planner import SplitPlan, plan_split
from .program import (
    ReplayProgram,
    ReplaySegment,
    SplitCapabilityReport,
    build_capability_report,
    lower_replay_program,
)
from .shape_program import ShapeProgram, compile_shape_program


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
            f"Batch witness {batch_size} changed graph topology from "
            f"{len(graph.nodes)} to {len(witness.nodes)} values."
        )
    shapes: dict[str, tuple[int, ...] | None] = {}
    graph_order = graph.order_by_id
    witness_order = witness.order_by_id
    for index, (base_node, witness_node) in enumerate(zip(graph.nodes, witness.nodes)):
        if _witness_node_signature(base_node) != _witness_node_signature(witness_node):
            raise SplitUnsupportedError(
                f"Batch witness {batch_size} changed graph topology at value {index}: "
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
                f"Batch witness {batch_size} changed parent-call topology at value "
                f"{index}: {base_parent_indexes!r} != {witness_parent_indexes!r}."
            )
        shapes[base_node.canonical_id] = witness_node.output_shape
    return shapes


def _probe_batch_replay(
    model: Any,
    inputs: tuple[Any, ...],
    input_kwargs: dict[str, Any] | None,
    request: SplitRequest,
    graph: SplitTraceGraph,
    shape_program: ShapeProgram,
    adapter: Any,
    *,
    random_seed: int | None = None,
) -> ShapeProgram:
    """Use one B=2 capture to infer shapes and compare generated replay outputs.

    Passing this sample authorizes empirical extrapolation, not a proof about
    untested Python branches. Failures retain the captured-batch runtime.
    """

    traced_batch = shape_program.traced_batch_size
    probes = witness_probe_sizes(traced_batch)
    if not probes or not shape_program.input_batch_axes:
        return replace(
            shape_program,
            batch_probe=BatchProbeResult(
                traced_batch,
                None,
                "unavailable",
                "Inputs have no batch axes; replay requires the captured input shapes."
                if not shape_program.input_batch_axes
                else "No independent B=2 probe for this capture.",
            ),
        )

    witness_capture = None
    candidate = shape_program
    alignment_note = None
    phase = "construct probe"
    try:
        with _probe_state_scope(model, adapter):
            if adapter.name == "torch" and not isinstance(model, types.FunctionType):
                from .._capture_state_helpers import _model_for_validation_replay

                witness_model, _snapshot, copied = _model_for_validation_replay(model)
                if not copied:
                    raise SplitUnsupportedError("Cannot copy the model for an isolated B=2 probe.")
            else:
                witness_model = deepcopy(model)
                if witness_model is model and not isinstance(model, types.FunctionType):
                    raise SplitUnsupportedError("The model copy aliases the original model.")
                if adapter.name == "tf":
                    from ._tf_capture import preserve_probe_layer_names

                    preserve_probe_layer_names(model, witness_model)
            witness_inputs, witness_kwargs = rebatch_inputs(
                inputs,
                input_kwargs,
                axes=shape_program.input_batch_axes,
                batch_size=2,
                adapter=adapter,
            )
            # Capture may modify its inputs. Replay uses independent, identical
            # sample values, and never reruns the original Python model.
            replay_inputs, replay_kwargs = rebatch_inputs(
                witness_inputs,
                witness_kwargs,
                axes=shape_program.input_batch_axes,
                batch_size=2,
                adapter=adapter,
            )
            outputs: list[Any] = []

            def observe_output(output: Any) -> None:
                """Retain the native forward result independently of trace normalization."""

                outputs.append(_snapshot_probe_output(adapter, output))

            phase = "capture B=2"
            witness_capture = capture_model(
                witness_model,
                witness_inputs,
                request,
                input_kwargs=witness_kwargs,
                output_observer=observe_output,
                random_seed=random_seed,
            )
            if not outputs:
                raise SplitUnsupportedError("The backend did not expose the native probe output.")
            phase = "compare topology"
            witness_graph = split_graph_from_trace(witness_capture)
            try:
                witness_shapes = _align_shape_witness(graph, witness_graph, batch_size=2)
            except SplitUnsupportedError as exc:
                if adapter.name != "tinygrad":
                    raise
                # Lazy UOp optimization can fold singleton shape operations at
                # B=1. Do not invent cross-graph shape correspondences: retain
                # semantic shape lowering and still require the B=2 replay and
                # exact output structure/shape/dtype plus numeric comparison.
                witness_shapes = None
                alignment_note = f"Shape witness alignment unavailable: {exc}"
            if witness_shapes is not None:
                phase = "infer sampled shapes"
                candidate = compile_shape_program(
                    graph,
                    inputs,
                    input_kwargs,
                    request,
                    adapter=adapter,
                    batch_axes=dict(shape_program.input_batch_axes),
                    inference_mode=shape_program.inference_mode,
                    shape_witnesses={2: witness_shapes},
                )
            phase = "replay B=2"
            probe_graph = project_symbolic_shapes(
                replace(graph, traced_batch_size=traced_batch), candidate
            )
            if adapter.name == "torch":
                probe_graph = _with_probe_rng_states(probe_graph, witness_graph)
            # Probe the same split without creating optimizer-owned placement
            # replicas or consuming live training state.
            probe_request = replace(
                request,
                placement=PlacementPlan(),
                features=replace(request.features, training=False, live_param_sources=False),
            )
            probe_plan = plan_split(probe_graph, probe_request)
            segments = execute_split_runtime(adapter, probe_graph, probe_plan, probe_request)
            boundary = segments.prefix(
                *replay_inputs, input_kwargs=replay_kwargs, detach_boundary=True
            )
            actual = segments.suffix(boundary)
            phase = "compare outputs"
            mismatch = _probe_output_mismatch(adapter, outputs[-1], actual)
            if mismatch is not None:
                raise SplitUnsupportedError(mismatch)
        result = BatchProbeResult(traced_batch, 2, "passed", alignment_note)
    except Exception as exc:  # noqa: BLE001 - probe failures restrict capability, not preparation
        result = BatchProbeResult(
            traced_batch,
            2,
            "unavailable" if phase == "construct probe" else "failed",
            f"{phase}: {type(exc).__name__}: {exc}",
        )
    finally:
        # Trace.cleanup currently assumes Torch parameter .grad attributes.
        # Native preview captures are temporary locals and release normally.
        if witness_capture is not None and adapter.name == "torch":
            cleanup = getattr(witness_capture, "cleanup", None)
            if callable(cleanup):
                cleanup()
    return replace(candidate, batch_probe=result)


@dataclass(frozen=True)
class _ProbeOpState:
    """Read capture metadata unchanged except for an isolated probe's RNG state."""

    captured_op: Any
    func_rng_states: dict[str, Any] | None

    def __getattr__(self, name: str) -> Any:
        """Delegate all other metadata to the retained capture's operation."""

        return getattr(self.captured_op, name)


def _with_probe_rng_states(graph: SplitTraceGraph, witness: SplitTraceGraph) -> SplitTraceGraph:
    """Align temporary replay with the B=2 oracle's per-call random stream.

    Matching only the initial capture seed is insufficient: earlier random
    operations may consume a different number of draws at B=2. Topology has
    already been checked, so each replay call can use its aligned witness's
    state while preserving every B=1 callable, argument and captured Op.
    """

    return replace(
        graph,
        nodes=tuple(
            replace(
                node,
                op=_ProbeOpState(node.op, getattr(probe.op, "func_rng_states", None)),
            )
            for node, probe in zip(graph.nodes, witness.nodes, strict=True)
        ),
    )


@contextmanager
def _probe_state_scope(model: Any, adapter: Any) -> Iterator[None]:
    """Restore supported model state and shared RNGs around the complete probe."""

    from ..utils.rng import log_current_rng_states, set_rng_from_saved_states

    rng_state = log_current_rng_states()
    try:
        if adapter.name == "torch":
            with _torch_capture_state(model):
                yield
        elif adapter.name == "tinygrad":
            from ._tinygrad_state import tinygrad_capture_state

            with tinygrad_capture_state(model):
                yield
        elif isinstance(model, (types.FunctionType, types.MethodType)):
            from ._callable_state import callable_capture_state

            with callable_capture_state(model, adapter):
                yield
        else:
            yield
    finally:
        set_rng_from_saved_states(rng_state)


def _snapshot_probe_output(adapter: Any, value: Any) -> Any:
    """Copy tensors and containers without retaining a second autograd graph."""

    from .._state import pause_logging

    with pause_logging():
        if adapter.is_tensor(value):
            return adapter.clone(adapter.detach(value))
        if isinstance(value, dict):
            return type(value)(
                (key, _snapshot_probe_output(adapter, item)) for key, item in value.items()
            )
        if isinstance(value, (tuple, list)):
            items = [_snapshot_probe_output(adapter, item) for item in value]
            return type(value)(*items) if hasattr(type(value), "_fields") else type(value)(items)
        if is_dataclass(value) and not isinstance(value, type):
            return clone_dataclass_fields(
                value,
                {
                    item.name: _snapshot_probe_output(adapter, getattr(value, item.name))
                    for item in fields(value)
                },
            )
        return deepcopy(value)


def _probe_output_mismatch(
    adapter: Any, expected: Any, actual: Any, path: str = "output"
) -> str | None:
    """Compare exact output structure, shape and dtype before numeric tolerance."""

    if adapter.is_tensor(expected) or adapter.is_tensor(actual):
        if not (adapter.is_tensor(expected) and adapter.is_tensor(actual)):
            return f"{path}: tensor/non-tensor mismatch"
        if adapter.shape(expected) != adapter.shape(actual):
            return f"{path}: shape mismatch {adapter.shape(expected)} != {adapter.shape(actual)}"
        if adapter.dtype_name(expected) != adapter.dtype_name(actual):
            return f"{path}: dtype mismatch"
        dtype = str(adapter.dtype_name(expected))
        if "bool" in dtype or "int" in dtype:
            # Integer labels/masks need exact equality; TF's allclose-style
            # subtraction is not defined for boolean tensors.
            left = expected.tolist() if hasattr(expected, "tolist") else expected.numpy().tolist()
            right = actual.tolist() if hasattr(actual, "tolist") else actual.numpy().tolist()
            return None if left == right else f"{path}: numeric mismatch"
        if not adapter.allclose(expected, actual, atol=1e-5, rtol=1e-4):
            return f"{path}: numeric mismatch"
        return None
    if type(expected) is not type(actual):
        return f"{path}: output container/type mismatch"
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            return f"{path}: mapping keys mismatch"
        pairs = [(key, expected[key], actual[key]) for key in expected]
    elif isinstance(expected, (tuple, list)):
        if len(expected) != len(actual):
            return f"{path}: container length mismatch"
        pairs = [(index, left, right) for index, (left, right) in enumerate(zip(expected, actual))]
    elif is_dataclass(expected) and not isinstance(expected, type):
        pairs = [
            (item.name, getattr(expected, item.name), getattr(actual, item.name))
            for item in fields(expected)
        ]
    else:
        return None if expected == actual else f"{path}: literal mismatch"
    for key, left, right in pairs:
        mismatch = _probe_output_mismatch(adapter, left, right, f"{path}/{key}")
        if mismatch is not None:
            return mismatch
    return None


def capture_model(
    model: Any,
    inputs: tuple[Any, ...],
    spec: SplitRequest,
    *,
    input_kwargs: dict[str, Any] | None = None,
    output_observer: Callable[[Any], None] | None = None,
    random_seed: int | None = None,
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
    if output_observer is not None:
        common["output_transform"] = output_observer
    if random_seed is not None:
        common["random_seed"] = random_seed
    if str(backend_spec.name) == TORCH_BACKEND_NAME:
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
    if str(backend_spec.name) == TINYGRAD_BACKEND_NAME:
        from ._tinygrad_state import tinygrad_capture_state

        with tinygrad_capture_state(model):
            return trace(model, inputs, **common)
    if str(backend_spec.name) in {"tf", "tensorflow"}:
        from ._tf_capture import batch_stable_keras_add

        with batch_stable_keras_add():
            return trace(model, inputs, **common)
    if str(backend_spec.name) == JAX_BACKEND_NAME:
        from ._jax_capture import batch_stable_matmul

        with batch_stable_matmul():
            return trace(model, inputs, **common)
    if str(backend_spec.name) != TORCH_BACKEND_NAME:
        return trace(model, inputs, **common)

    with _torch_capture_state(model):
        return trace(model, inputs, **common)


@contextmanager
def _torch_capture_state(model: Any) -> Iterator[None]:
    """Restore Torch state and RNG around internal capture or replay execution."""

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
    registered_state = tuple(
        (
            module,
            dict(getattr(module, "_parameters", {})),
            dict(getattr(module, "_buffers", {})),
            dict(getattr(module, "_modules", {})),
        )
        for module in (model.modules() if hasattr(model, "modules") else ())
    )
    plain_attrs = _ModuleTreePlainAttrSnapshot(
        model,
        ignored_names=frozenset({"_tl", "forward"}),
    )
    try:
        # Split capture can encounter modules (notably transformer activation
        # helpers) that stored a torch builtin before TorchLens installed its
        # wrappers. Prepare first so the wrapper epoch ledger is populated,
        # then bind only direct stale function attrs for this transaction.
        # Ordinary ``tl.trace`` does not pass through this split-only scope.
        import torch

        if isinstance(model, torch.nn.Module):
            from ..backends.torch._held_refs import scoped_held_torch_function_refs
            from ..backends.torch.wrappers import wrap_torch

            # ``trace`` performs model preparation after entering this state
            # scope. Install wrappers here first so stale direct references can
            # be rebound before that preparation and the forward; the native
            # preparation call remains the single owner of model metadata.
            wrap_torch()
            with scoped_held_torch_function_refs(model):
                yield
        else:
            yield
    finally:
        active_error = sys.exc_info()[1]
        cleanup_errors: list[BaseException] = []

        def attempt_restore(action: Callable[[], None]) -> None:
            """Run one cleanup action while allowing later actions to proceed."""

            try:
                action()
            except BaseException as exc:  # noqa: BLE001 - cleanup must be best effort
                cleanup_errors.append(exc)

        def restore_registered_state() -> None:
            """Restore parameter, buffer, and child-module registrations first."""

            for module, parameters, buffers, children in reversed(registered_state):
                module._parameters.clear()
                module._parameters.update(parameters)
                module._buffers.clear()
                module._buffers.update(buffers)
                module._modules.clear()
                module._modules.update(children)

        def restore_state_dict() -> None:
            """Restore captured tensor values after the module tree is restored."""

            if state_dict is not None:
                model.load_state_dict(state_dict)

        def restore_training_modes() -> None:
            """Restore every module's original training mode."""

            for module, training in training_modes:
                module.train(training)

        attempt_restore(restore_registered_state)
        attempt_restore(restore_state_dict)
        attempt_restore(restore_training_modes)
        attempt_restore(plain_attrs.restore_changed_attrs)
        attempt_restore(lambda: set_rng_from_saved_states(rng_state))
        if active_error is None and cleanup_errors:
            raise cleanup_errors[0]


def capture_canonical_model(
    model: Any,
    inputs: tuple[Any, ...],
    spec: SplitRequest,
    *,
    input_kwargs: dict[str, Any] | None = None,
    adapter: Any,
) -> tuple[Any, tuple[Any, ...], dict[str, Any], BatchSpec]:
    """Capture the model, normalizing only declared or inferred batch axes.

    The canonical batch is ``B=1``; ``B=2`` is used only when a ``B=1``
    capture genuinely fails.  A large example batch is never used just
    because the caller supplied one. With no batch axes, inputs retain their
    original shapes and no canonical rebatching is attempted.

    Returns
    -------
    tuple
        ``(capture, canonical_inputs, canonical_kwargs, batch_spec)``.
    """

    batch_spec = resolve_batch_spec(
        inputs,
        input_kwargs,
        adapter=adapter,
        explicit_axes=spec.features.batch_axes,
    )
    if not batch_spec.axes:
        capture = capture_model(model, inputs, spec, input_kwargs=input_kwargs)
        return capture, inputs, dict(input_kwargs or {}), batch_spec

    canonical = canonical_batch_for(batch_spec)
    attempts: list[int] = [canonical]
    if FALLBACK_TRACE_BATCH not in attempts:
        attempts.append(FALLBACK_TRACE_BATCH)
    errors: dict[int, Exception] = {}
    for index, attempt in enumerate(attempts):
        is_last = index == len(attempts) - 1
        if batch_spec.user_batch_size == attempt:
            canonical_inputs, canonical_kwargs = inputs, dict(input_kwargs or {})
        else:
            try:
                canonical_inputs, canonical_kwargs = rebatch_inputs(
                    inputs,
                    input_kwargs,
                    axes=batch_spec.axes,
                    batch_size=attempt,
                    adapter=adapter,
                )
            except Exception as exc:  # noqa: BLE001 - rebatching is backend-owned
                raise SplitUnsupportedError(
                    f"Cannot construct canonical batch B={attempt} with backend "
                    f"{adapter.name!r}: {exc}"
                ) from exc
        try:
            capture = capture_model(model, canonical_inputs, spec, input_kwargs=canonical_kwargs)
        except Exception as exc:  # noqa: BLE001 - a model may genuinely reject a small batch
            # The last attempt propagates verbatim: a capture failure there is
            # the real diagnosis, not "no small batch worked".
            if is_last:
                raise
            errors[attempt] = exc
            continue
        return (
            capture,
            canonical_inputs,
            canonical_kwargs,
            BatchSpec(
                axes=batch_spec.axes,
                inference=batch_spec.inference,
                user_batch_size=batch_spec.user_batch_size,
                canonical_batch_size=attempt,
            ),
        )
    raise SplitUnsupportedError(
        f"Canonical split capture failed at every small batch {tuple(attempts)!r}: "
        f"{ {batch: repr(exc) for batch, exc in errors.items()} }."
    )


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
    batch_spec: BatchSpec | None = None,
) -> tuple[SplitTraceGraph, SplitGraphIR]:
    """Normalize a backend capture into the runtime graph and portable Split IR.

    The retained B=1 graph is reused after one empirical B=2 replay check.
    A failed or unavailable check keeps the graph runnable at its capture batch.
    """

    graph = capture if isinstance(capture, SplitTraceGraph) else split_graph_from_trace(capture)
    if adapter is None:
        from .adapters import resolve_split_adapter

        adapter = resolve_split_adapter(graph.backend)
    resolved_axes = None if batch_spec is None else dict(batch_spec.axes)
    inference_mode: Literal["explicit", "conservative_auto"] | None = None
    if batch_spec is not None:
        inference_mode = "explicit" if batch_spec.inference == "explicit" else "conservative_auto"
    shape_program = compile_shape_program(
        graph,
        inputs,
        input_kwargs,
        spec,
        adapter=adapter,
        batch_axes=resolved_axes,
        inference_mode=inference_mode,
    )
    if shape_program is not None and model is not None:
        shape_program = _probe_batch_replay(
            model,
            inputs,
            input_kwargs,
            spec,
            graph,
            shape_program,
            adapter,
            random_seed=getattr(capture, "random_seed", None) if adapter.name == "torch" else None,
        )
    if shape_program is not None:
        if shape_program.batch_probe is not None:
            probe = shape_program.batch_probe
            # Include authorization, not environment-dependent error text or
            # concrete runtime B, in cache and graph identity.
            stamp = json.dumps(
                ("single_probe_v1", probe.status, probe.traced_batch_size, probe.probe_batch_size)
            )
            shape_program = replace(
                shape_program,
                fingerprint=sha256(f"{shape_program.fingerprint}:{stamp}".encode()).hexdigest(),
            )
        graph_hash = sha256(
            f"{graph.graph_shape_hash or ''}:{shape_program.fingerprint}".encode()
        ).hexdigest()
        graph = replace(
            graph,
            graph_shape_hash=graph_hash,
            traced_batch_size=shape_program.traced_batch_size,
        )
        graph = project_symbolic_shapes(graph, shape_program)
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
    "capture_canonical_model",
    "capture_model",
    "execute_split_runtime",
    "lower_split_program",
    "normalize_to_split_ir",
]
