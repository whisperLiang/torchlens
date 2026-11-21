"""Prepare and execute one recorded TensorFlow native call."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import partial
from numbers import Integral
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from ... import _state
from ._native_op_specs import _MISSING, _NATIVE_OP_SPECS, _AttrRule, _NativeOpSpec

if TYPE_CHECKING:
    from .op_callback_capture import TFInputCapture, TFOpCapture

_TF_REPLAY_PAUSED = ContextVar("torchlens_tf_replay_paused", default=False)


@contextmanager
def _pause_tf_replay() -> Iterator[None]:
    """Suppress TorchLens capture while running its own native replay."""

    token = _TF_REPLAY_PAUSED.set(True)
    try:
        with _state.pause_logging():
            yield
    finally:
        _TF_REPLAY_PAUSED.reset(token)


@dataclass(frozen=True, slots=True)
class _PreparedNativeCall:
    """Bound TensorFlow wrapper and static attributes for one captured call."""

    op_type: str
    raw_fn: Callable[..., Any]
    input_names: tuple[str, ...]
    list_input_count: int
    output_count: int

    def _execute(self, inputs: Sequence[Any]) -> Any:
        """Bind runtime tensors by the prepared ABI inside a replay scope."""

        if self.list_input_count:
            count = self.list_input_count
            inputs = (inputs[:count], *inputs[count:])
        return self.raw_fn(**dict(zip(self.input_names, inputs, strict=True)))

    def _outputs(self, result: Any) -> Sequence[Any]:
        """Check the native output count and return its ordered slots."""

        outputs = result if isinstance(result, (tuple, list)) else (result,)
        if len(outputs) != self.output_count:
            raise ValueError(
                f"{self.op_type} expects {self.output_count} outputs, got {len(outputs)}"
            )
        return outputs


def _prepare_native_call(
    tf: Any, capture: TFOpCapture
) -> tuple[_PreparedNativeCall, tuple[TFInputCapture, ...]]:
    """Check recorded inputs, attributes and output slots and bind the native wrapper."""

    spec = _NATIVE_OP_SPECS.get(capture.op_type)
    raw_fn = getattr(tf.raw_ops, capture.op_type, None)
    if spec is None or not callable(raw_fn):
        raise ValueError(f"TensorFlow raw op {capture.op_type!r} has no replay implementation")
    ordered_inputs = _ordered_inputs(capture)
    list_count = _list_input_count(spec, capture, len(ordered_inputs))
    attrs = _native_attrs(tf, capture, spec)
    count = attrs[spec.output_count_attr] if spec.output_count_attr else 1
    if capture._output_count != count or not 0 <= capture.output_index < count:
        raise ValueError(
            f"{capture.op_type} has invalid TensorFlow call outputs: "
            f"expected {count}, recorded {capture._output_count}, slot {capture.output_index}"
        )
    if attrs:
        raw_fn = partial(raw_fn, **attrs)
    return (
        _PreparedNativeCall(capture.op_type, raw_fn, spec.inputs, list_count, count),
        ordered_inputs,
    )


def _ordered_inputs(capture: TFOpCapture) -> tuple[TFInputCapture, ...]:
    """Bind callback operands by index and reject duplicate or missing slots."""

    if all(item.input_index == index for index, item in enumerate(capture.inputs)):
        return capture.inputs
    ordered = tuple(sorted(capture.inputs, key=lambda item: item.input_index))
    if any(item.input_index != index for index, item in enumerate(ordered)):
        raise ValueError(f"{capture.op_type} has invalid TensorFlow input slots")
    return ordered


def _list_input_count(spec: _NativeOpSpec, capture: TFOpCapture, count: int) -> int:
    """Validate the flattened input arity and determine any list binding size."""

    if not spec.min_list_inputs:
        if count != len(spec.inputs):
            raise ValueError(f"{capture.op_type} expects {len(spec.inputs)} inputs, got {count}")
        return 0
    list_count = count - len(spec.inputs) + 1
    if list_count < spec.min_list_inputs:
        min_count = len(spec.inputs) + spec.min_list_inputs - 1
        raise ValueError(f"{capture.op_type} expects at least {min_count} inputs, got {count}")
    if "N" in capture.attrs and _as_int(capture.attrs["N"]) != list_count:
        raise ValueError(f"{capture.op_type} has inconsistent TensorFlow list input count")
    return list_count


def _native_attrs(tf: Any, capture: TFOpCapture, spec: _NativeOpSpec) -> dict[str, Any]:
    """Decode specified attributes and leave optional defaults to the native wrapper."""

    attrs: dict[str, Any] = {}
    for rule in spec.attrs:
        raw = _captured_attr(capture.attrs, rule)
        if raw is _MISSING or raw is None:
            if rule.default is _MISSING:
                problem = "is missing" if raw is _MISSING else "is None"
                raise ValueError(f"{capture.op_type} required attribute {rule.name!r} {problem}")
            continue
        try:
            value = _normalize_attr(tf, rule, raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{capture.op_type} invalid attribute {rule.name}={raw!r}: {exc}"
            ) from exc
        if value != rule.default:
            attrs[rule.name] = value
    return attrs


def _captured_attr(attrs: Mapping[str, Any], rule: _AttrRule) -> Any:
    """Resolve callback aliases before default handling."""

    for name in rule.capture_names or (rule.name,):
        if name in attrs:
            return attrs[name]
    return _MISSING


def _normalize_attr(tf: Any, rule: _AttrRule, raw: Any) -> Any:
    """Return one native attribute in the wrapper's supported Python form."""

    if rule.kind == "dtype":
        value = tf.dtypes.as_dtype(raw).base_dtype
    else:
        value = _ATTR_CONVERTERS[rule.kind](raw)
    if rule.choices and getattr(value, "name", value) not in rule.choices:
        raise ValueError(f"expected one of {rule.choices}")
    if rule.min_items and len(value) < rule.min_items:
        raise ValueError(f"expected at least {rule.min_items} entries")
    return value


def _as_int(value: Any) -> int:
    """Accept Python and NumPy integral attributes with no lossy coercion."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("expected an integer")
    return int(value)


def _as_ints(value: Any) -> tuple[int, ...]:
    """Freeze a native integer list without rewriting its values."""

    if not isinstance(value, (list, tuple)):
        raise TypeError("expected an integer list")
    return tuple(_as_int(item) for item in value)


def _as_bool(value: Any) -> bool:
    """Require a native Boolean attribute."""

    if not isinstance(value, bool):
        raise TypeError("expected a bool")
    return value


def _as_text(value: Any) -> str:
    """Decode native text and reject non-text values."""

    if isinstance(value, bytes):
        return value.decode("utf-8")
    if not isinstance(value, str):
        raise TypeError("expected a string or UTF-8 bytes")
    return value


_ATTR_CONVERTERS = MappingProxyType(
    {"str": _as_text, "int": _as_int, "bool": _as_bool, "ints": _as_ints}
)


def _native_output_slots(captures: Sequence[TFOpCapture]) -> tuple[int, ...]:
    """Return the position of each slot in one complete native invocation."""

    leader = captures[0]
    count = len(captures)
    slots = {item.output_index: index for index, item in enumerate(captures)}
    if slots.keys() != set(range(count)) or any(
        item._call_id != leader._call_id
        or item.op_type != leader.op_type
        or item._output_count != count
        or item.inputs is not leader.inputs
        or item.attrs is not leader.attrs
        for item in captures
    ):
        raise ValueError(f"{leader.op_type} has incomplete or invalid TensorFlow call outputs")
    return tuple(slots[index] for index in range(count))


def _import_tensorflow() -> Any:
    """Import TensorFlow only when preparing or replaying a TensorFlow capture."""

    import tensorflow as tf

    return tf
