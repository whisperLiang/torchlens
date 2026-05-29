"""Public API for TorchLens native split runtimes."""

from __future__ import annotations

from typing import Any

import torch

from ..user_funcs import log_forward_pass

from .codegen import build_segments
from .planner import plan_split
from .runtime import SplitRuntime
from .shape import infer_traced_batch_size
from .spec import SplitSpec
from .trace_graph import trace_graph_from_model_log


def prepare_split(
    model: torch.nn.Module,
    example_inputs: Any,
    spec: SplitSpec,
    *,
    example_kwargs: dict[str, Any] | None = None,
) -> SplitRuntime:
    """Prepare a native TorchLens split replay/train runtime."""

    inputs = _as_input_tuple(example_inputs)
    kwargs = dict(example_kwargs or {})
    model_log = log_forward_pass(
        model,
        inputs,
        kwargs,
        vis_opt="none",
        layers_to_save="all",
        keep_unsaved_layers=True,
        detach_saved_tensors=False,
        save_function_args=True,
        intervention_ready=True,
    )
    traced_batch_size = infer_traced_batch_size(inputs)
    graph = trace_graph_from_model_log(
        model_log,
        traced_batch_size=traced_batch_size,
        batch_symbol=spec.batch_symbol,
        dynamic_batch=spec.dynamic_batch,
    )
    plan = plan_split(graph, spec)
    segments = build_segments(graph, plan, mode=spec.mode)
    return SplitRuntime(
        model=model,
        trace_graph=graph,
        split_spec=spec,
        plan=plan,
        segments=segments,
    )


def prepare_split_replay(
    model: torch.nn.Module,
    example_inputs: Any,
    spec: SplitSpec,
    *,
    example_kwargs: dict[str, Any] | None = None,
) -> SplitRuntime:
    """Prepare an inference-style split replay runtime."""

    replay_spec = SplitSpec(
        boundary=spec.boundary,
        batch_symbol=spec.batch_symbol,
        dynamic_batch=spec.dynamic_batch,
        trainable=False,
        trace_batch_mode=spec.trace_batch_mode,
        device_policy=spec.device_policy,
        mode=spec.mode,
    )
    return prepare_split(model, example_inputs, replay_spec, example_kwargs=example_kwargs)


def _as_input_tuple(example_inputs: Any) -> tuple[Any, ...]:
    if isinstance(example_inputs, tuple):
        return example_inputs
    return (example_inputs,)


__all__ = ["prepare_split", "prepare_split_replay"]
