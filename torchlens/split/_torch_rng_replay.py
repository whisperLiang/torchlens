"""Preserve native generator/device semantics for placed stochastic Torch calls."""

from __future__ import annotations

from collections.abc import Callable
from types import BuiltinFunctionType, MethodDescriptorType, WrapperDescriptorType
from typing import Any

from .. import _state
from ..utils._torch_compat import get_variable_functions_class
from ..utils._torch_symbols import torch_attr
from ..utils.rng import execute_with_restored_rng_autocast
from .errors import SplitErrorContext, SplitUnsupportedError


def stochastic_targets() -> dict[int, str]:
    """Resolve exact native identities; custom callables never inherit a name-based policy."""
    import torch

    originals, _ = _state.wrap_epoch_ledgers()
    names = (
        "rand",
        "randn",
        "rand_like",
        "randn_like",
        "randint",
        "randint_like",
        "randperm",
        "bernoulli",
        "normal",
        "poisson",
        "multinomial",
        "dropout",
        "feature_dropout",
        "alpha_dropout",
        "feature_alpha_dropout",
        "native_dropout",
        "_fused_dropout",
    )
    variable_functions = get_variable_functions_class()
    targets: dict[int, str] = {}
    native_types = (BuiltinFunctionType, MethodDescriptorType, WrapperDescriptorType)
    for name in names:
        for candidate in (torch_attr(name), getattr(variable_functions, name, None)):
            native = originals.get(id(candidate), candidate)
            if type(native) in native_types:
                targets[id(native)] = name
    attention = torch.nn.functional.scaled_dot_product_attention
    native_attention = originals.get(id(attention), attention)
    if type(native_attention) in native_types:
        targets[id(native_attention)] = "attention"
    return targets


def _map_tensors(value: Any, transform: Callable[[Any], Any]) -> Any:
    """Transport native tensor trees while retaining autograd copy edges."""
    import torch

    if isinstance(value, torch.Tensor):
        if type(value) not in (torch.Tensor, torch.nn.Parameter):
            raise ValueError("Stochastic device routing requires ordinary native tensors.")
        return transform(value)
    if isinstance(value, tuple):
        return tuple(_map_tensors(item, transform) for item in value)
    if isinstance(value, list):
        return [_map_tensors(item, transform) for item in value]
    if isinstance(value, dict):
        return {key: _map_tensors(item, transform) for key, item in value.items()}
    return value


def _uses_randomness(kind: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    """Avoid device round trips when a known stochastic operator has no active draws."""
    if kind in {"dropout", "feature_dropout", "alpha_dropout", "feature_alpha_dropout"}:
        probability = args[1] if len(args) > 1 else kwargs.get("p", 0.5)
        training = args[2] if len(args) > 2 else kwargs.get("train", True)
        return probability != 0 and training is not False
    if kind == "attention":
        return (args[4] if len(args) > 4 else kwargs.get("dropout_p", 0.0)) != 0
    return True


def route_stochastic_call(
    node: Any,
    kind: str | None,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    destination: Any,
) -> tuple[bool, Any]:
    """Execute recognized random calls on their captured device, then transport outputs.

    The original native operator and saved RNG/autocast context are retained.
    This routes the stochastic call only; other segment calls keep their placement.
    """
    if kind is None or destination is None or not _uses_randomness(kind, args, kwargs):
        return False, None
    import torch

    origin = getattr(node.op, "device_ref", None) or getattr(node.output_ref, "device", None)
    if origin is None:
        raise SplitUnsupportedError(
            "Stochastic replay requires the captured generator device.",
            context=SplitErrorContext(
                backend="torch", split_point="", reason="rng_device_unavailable"
            ),
        )
    source, target = torch.device(str(origin)), torch.device(destination)
    if target.type == "cuda" and target.index is None:
        target = torch.device("cuda", torch.cuda.current_device())
    if source == target:
        return False, None
    if kwargs.get("out") is not None or kwargs.get("generator") is not None:
        raise SplitUnsupportedError(
            "Cross-device random out= and explicit generators require an independent state contract.",
            context=SplitErrorContext(
                backend="torch", split_point="", reason="rng_device_state_unsupported"
            ),
        )
    rng_states = getattr(node.op, "func_rng_states", None)
    has_device_state = (
        source.type == "cpu"
        and "torch" in (rng_states or {})
        or source.type == "cuda"
        and (
            len((rng_states or {}).get("torch_cuda_all", ())) > (source.index or 0)
            or source.index in (None, 0)
            and "torch_cuda" in (rng_states or {})
        )
    )
    if not has_device_state:
        raise SplitUnsupportedError(
            "Cross-device stochastic replay requires saved per-call RNG state.",
            context=SplitErrorContext(
                backend="torch", split_point="", reason="rng_state_unavailable"
            ),
        )
    with _state.pause_logging(), torch.device(source):
        moved_args = _map_tensors(args, lambda value: value.to(source))
        moved_kwargs = _map_tensors(kwargs, lambda value: value.to(source))
        if "device" in moved_kwargs:
            moved_kwargs["device"] = source
        result = execute_with_restored_rng_autocast(
            node.target,
            moved_args,
            moved_kwargs,
            rng_states=rng_states,
            autocast_state=getattr(node.op, "func_autocast_state", None),
        )
        return True, _map_tensors(result, lambda value: value.to(target))
