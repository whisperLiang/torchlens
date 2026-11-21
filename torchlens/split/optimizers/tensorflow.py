"""Keras optimizer state mapped by native variable resource and slot name."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ...backends.tf._tensor_probe import _tensor_ref_key
from ..adapters import tf as _adapter
from .base import (
    _GLOBAL,
    OptimizerFactory,
    OptimizerState,
    OptimizerStateTensor,
    _config_tree,
    _refuse,
)

_TYPES = ("SGD", "Adam", "AdamW")
_SCHEDULES = (
    "CosineDecay",
    "CosineDecayRestarts",
    "ExponentialDecay",
    "InverseTimeDecay",
    "PiecewiseConstantDecay",
    "PolynomialDecay",
)


def _keras_types() -> dict[str, Any]:
    """Resolve exact standard classes in the active Keras-on-TensorFlow installation."""

    keras = _adapter._tf().keras
    if not hasattr(keras, "Variable") or keras.backend.backend() != "tensorflow":
        return {}
    return {name: getattr(keras.optimizers, name) for name in _TYPES}


def _schedules() -> dict[str, Any]:
    """Allow only stateless, serializable built-in learning-rate schedules."""

    namespace = _adapter._tf().keras.optimizers.schedules
    return {name: cls for name in _SCHEDULES if (cls := getattr(namespace, name, None)) is not None}


def _resource(value: Any) -> object:
    """Return a verified native optimizer or model variable resource identity."""

    key = _tensor_ref_key(_adapter._tf_variable_resource(value))
    if key is None:
        _refuse(
            "keras_slot_layout_unsupported",
            "An optimizer variable has no native resource identity.",
            backend="tf",
        )
    return key


def _keras_config(optimizer: Any) -> dict[str, Any]:
    """Validate options before asking Keras to serialize its optimizer."""

    unsupported = (
        (getattr(optimizer, "use_ema", False), "EMA"),
        (
            getattr(optimizer, "gradient_accumulation_steps", None) is not None,
            "gradient accumulation",
        ),
        (getattr(optimizer, "global_clipnorm", None) is not None, "global gradient clipping"),
        (getattr(optimizer, "_exclude_from_weight_decay", None), "weight-decay exclusions"),
        (getattr(optimizer, "_exclude_from_weight_decay_pattern", None), "weight-decay exclusions"),
        (
            getattr(getattr(optimizer, "_distribution_strategy", None), "num_replicas_in_sync", 1)
            > 1,
            "distributed optimizer state",
        ),
    )
    for active, feature in unsupported:
        if active:
            _refuse(
                "keras_optimizer_mode_unsupported",
                f"Keras {feature} state is unsupported.",
                backend="tf",
            )
    schedule = getattr(optimizer, "_learning_rate", None)
    if (
        not callable(getattr(schedule, "assign", None))
        and type(schedule) not in _schedules().values()
    ):
        _refuse(
            "keras_learning_rate_unsupported",
            "Keras learning rate requires a native variable or built-in schedule.",
            backend="tf",
        )
    config = optimizer.get_config()
    config.pop("name", None)
    return config


def _slot_layout(kind: str, config: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the named slots required by this standard optimizer configuration."""

    if kind == "SGD":
        return ("momentum",) if config["momentum"] else ()
    return (
        ("momentum", "velocity", "velocity_hat") if config["amsgrad"] else ("momentum", "velocity")
    )


def _slot_variables(optimizer: Any, kind: str, names: tuple[str, ...]) -> dict[str, Any]:
    """Probe the attested Keras 3 slot-list layouts used by native update_step."""

    attrs = {"momentum": "_momentums", "velocity": "_velocities", "velocity_hat": "_velocity_hats"}
    if kind == "SGD" and hasattr(optimizer, "momentums"):
        attrs["momentum"] = "momentums"
    return {name: getattr(optimizer, attrs[name], None) for name in names}


def _variable_index(optimizer: Any) -> tuple[list[object], Any]:
    """Probe the native Keras 3 model-variable list and its lookup method."""

    variables = getattr(optimizer, "_trainable_variables", None)
    lookup = getattr(optimizer, "_get_variable_index", None)
    if variables is None or not callable(lookup):
        _refuse(
            "keras_slot_layout_unsupported",
            "Keras has no verified variable-index layout.",
            backend="tf",
        )
    return [_resource(value) for value in variables], lookup


def _positions(optimizer: Any, parameters: Mapping[str, Any]) -> dict[str, int]:
    """Prove a built Keras optimizer owns exactly the supplied native resources."""

    known, lookup = _variable_index(optimizer)
    desired = {state_id: _resource(value) for state_id, value in parameters.items()}
    if len(known) != len(desired) or set(known) != set(desired.values()):
        _refuse(
            "optimizer_parameter_ownership",
            "Keras optimizer parameter ownership differs from the segment.",
            backend="tf",
        )
    try:
        positions = {key: lookup(value) for key, value in parameters.items()}
    except (KeyError, TypeError) as exc:
        _refuse(
            "keras_slot_layout_unsupported",
            "Keras could not resolve a bound variable index.",
            backend="tf",
            cause=exc,
        )
    if any(
        not 0 <= index < len(known) or known[index] != desired[key]
        for key, index in positions.items()
    ):
        _refuse(
            "keras_slot_layout_unsupported",
            "Keras optimizer variable index disagrees with its native resource.",
            backend="tf",
        )
    return positions


def _global_variables(optimizer: Any) -> dict[str, Any]:
    """Bind public iterations and the mutable learning-rate resource when present."""

    rate = optimizer._learning_rate
    values = {"iterations": optimizer.iterations}
    if callable(getattr(rate, "assign", None)):
        values["learning_rate"] = rate
    return values


def _check_native_slot(parameter: Any, value: Any, name: str) -> None:
    """Check shape, dtype and Keras's colocation guarantee for one native slot."""

    tf = _adapter._tf()
    if tuple(parameter.shape) != tuple(value.shape):
        _refuse(
            "optimizer_slot_shape",
            f"Keras slot {name!r} shape differs from its parameter.",
            backend="tf",
        )
    if tf.as_dtype(parameter.dtype) != tf.as_dtype(value.dtype):
        _refuse(
            "optimizer_slot_dtype",
            f"Keras slot {name!r} dtype differs from its parameter.",
            backend="tf",
        )
    if _adapter._tf_gradient_source(value).device != _adapter._tf_gradient_source(parameter).device:
        _refuse(
            "optimizer_slot_device",
            f"Keras slot {name!r} is not colocated with its parameter.",
            backend="tf",
        )


def _native_slots(
    optimizer: Any, kind: str, names: tuple[str, ...], parameters: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Resolve built Keras slots by parameter key, with complete state coverage."""

    if not optimizer.built:
        return {state_id: {} for state_id in parameters}
    positions = _positions(optimizer, parameters)
    lists = _slot_variables(optimizer, kind, names)
    for name, values in lists.items():
        if not isinstance(values, (tuple, list)) or len(values) != len(parameters):
            _refuse(
                "keras_slot_layout_unsupported",
                f"Keras slot layout for {name!r} is unavailable.",
                backend="tf",
            )
    slots = {
        state_id: {name: values[index] for name, values in lists.items()}
        for state_id, index in positions.items()
    }
    for state_id, values in slots.items():
        for name, value in values.items():
            _check_native_slot(parameters[state_id], value, name)
    return slots


def _keras_state_variables(
    optimizer: Any, kind: str, names: tuple[str, ...], parameters: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Refuse any extra optimizer state or alias to a live model variable."""

    globals_ = _global_variables(optimizer)
    slots = _native_slots(optimizer, kind, names, parameters)
    keys = [_resource(value) for value in globals_.values()]
    keys.extend(_resource(value) for items in slots.values() for value in items.values())
    parameters_keys = {_resource(value) for value in parameters.values()}
    if len(keys) != len(set(keys)) or parameters_keys.intersection(keys):
        _refuse(
            "keras_slot_layout_unsupported",
            "Keras optimizer state has aliased resources.",
            backend="tf",
        )
    if set(keys) != {_resource(value) for value in optimizer.variables}:
        _refuse(
            "keras_slot_layout_unsupported",
            "Keras optimizer has unmapped native variables.",
            backend="tf",
        )
    return globals_, slots


def _slot_tensor(state_id: str, name: str, value: Any) -> OptimizerStateTensor:
    """Read one native value into portable CPU state."""

    return OptimizerStateTensor.from_array(state_id, name, np.asarray(value.numpy()))


def _keras_constructor(kind: str, config: Mapping[str, Any]) -> Any:
    """Build a native optimizer using closed, verified schedule deserialization."""

    cls = _keras_types()[kind]
    options = _config_tree(config, frozen=False)
    rate = options["learning_rate"]
    if isinstance(rate, dict):
        options["learning_rate"] = _schedules()[rate["class_name"]].from_config(rate["config"])
        return cls(**options)
    return cls.from_config(options)


def _global_slot_names(rate: Any) -> tuple[str, ...]:
    """Resolve which mutable scalars the configured learning-rate form requires."""

    if isinstance(rate, Mapping):
        if rate.get("class_name") not in _schedules() or not isinstance(
            rate.get("config"), Mapping
        ):
            _refuse(
                "keras_learning_rate_unsupported",
                "Invalid Keras learning-rate schedule.",
                backend="tf",
            )
        return ("iterations",)
    return ("iterations", "learning_rate")


def _validate_global(state: OptimizerState) -> None:
    """Require complete scalar global state and a matching learning-rate form."""

    rate = state.config.get("learning_rate")
    tensors = {slot.slot_name: slot for slot in state.global_state}
    shapes = {name: slot.shape for name, slot in tensors.items()}
    if shapes != dict.fromkeys(_global_slot_names(rate), ()):
        _refuse(
            "optimizer_global_state", "Invalid Keras optimizer global state slots.", backend="tf"
        )
    iterations = tensors["iterations"]
    if iterations.dtype not in {"int32", "int64"} or iterations.payload.item() < 0:
        _refuse("optimizer_global_state", "Invalid Keras optimizer iterations.", backend="tf")
    if "learning_rate" in tensors:
        learning_rate = tensors["learning_rate"]
        if learning_rate.payload.dtype.kind != "f" or learning_rate.payload.item() != rate:
            _refuse(
                "optimizer_global_state",
                "Keras learning rate and optimizer config disagree.",
                backend="tf",
            )


def _assign(value: Any, tensor: OptimizerStateTensor) -> None:
    """Assign one slot of a fresh optimizer and attest the native write."""

    tf = _adapter._tf()
    dtype = tf.as_dtype(value.dtype)
    if tuple(value.shape) != tensor.shape or dtype.name != tensor.dtype:
        _refuse(
            "optimizer_slot_shape_dtype",
            f"Keras target slot {tensor.slot_name!r} has a different shape or dtype.",
            backend="tf",
        )
    value.assign(tensor.payload.view(dtype.as_numpy_dtype))
    if _slot_tensor(tensor.state_id, tensor.slot_name, value).checksum != tensor.checksum:
        _refuse(
            "optimizer_restore_verification",
            f"Keras state verification failed for {tensor.slot_name!r}.",
            backend="tf",
        )


class _KerasOptimizerAdapter:
    """Native TensorFlow Keras support for SGD, Adam and AdamW."""

    name = "keras"

    def supports(self, optimizer: Any, /) -> bool:
        """Accept exact Keras types; subclasses need an explicitly registered adapter."""

        return type(optimizer) in _keras_types().values()

    def export_state(self, optimizer: Any, parameters: Mapping[str, Any], /) -> OptimizerState:
        """Copy validated resource-indexed slots and mutable global state."""

        kind = type(optimizer).__name__
        config = _keras_config(optimizer)
        names = _slot_layout(kind, config)
        global_vars, native = _keras_state_variables(optimizer, kind, names, parameters)
        globals_ = tuple(
            OptimizerStateTensor.from_array(
                _GLOBAL,
                name,
                np.asarray(config[name], dtype=_adapter._tf().as_dtype(value.dtype).as_numpy_dtype),
            )
            if name == "learning_rate"
            else _slot_tensor(_GLOBAL, name, value)
            for name, value in sorted(global_vars.items())
        )
        state = OptimizerState(
            kind,
            config,
            globals_,
            dict.fromkeys(parameters, names),
            {
                key: tuple(_slot_tensor(key, name, value) for name, value in slots.items())
                for key, slots in native.items()
            },
        )
        _validate_global(state)
        return state

    def merge_config(
        self, sources: Mapping[str, OptimizerState], routes: Mapping[str, str], /
    ) -> Mapping[str, Any]:
        """Require all contributing Keras optimizers to agree on update semantics."""

        config = next(iter(sources.values())).config
        if any(state.config != config for state in sources.values()):
            _refuse(
                "optimizer_config_incompatible",
                "Keras source optimizer configurations differ.",
                backend="tf",
            )
        return config

    def validate_restore(self, state: OptimizerState, parameters: Mapping[str, Any], /) -> None:
        """Check slot names and target geometry without constructing an optimizer."""

        _validate_global(state)
        names = _slot_layout(state.optimizer_type, state.config)
        for key, variable in parameters.items():
            if state.expected_slots[key] != names:
                _refuse(
                    "optimizer_slot_inventory", f"Invalid slot schema for {key!r}.", backend="tf"
                )
            for tensor in state.slots[key]:
                if tensor.shape != tuple(variable.shape):
                    _refuse(
                        "optimizer_slot_shape", f"Slot shape mismatch for {key!r}.", backend="tf"
                    )
                if tensor.dtype != _adapter._tf().as_dtype(variable.dtype).name:
                    _refuse(
                        "optimizer_slot_dtype", f"Slot dtype mismatch for {key!r}.", backend="tf"
                    )

    def build_optimizer(
        self,
        state: OptimizerState,
        parameters: Mapping[str, Any],
        factory: OptimizerFactory | None,
        /,
    ) -> Any:
        """Build a fresh optimizer around the actual target variables."""

        tf = _adapter._tf()
        device = _adapter._tf_gradient_source(next(iter(parameters.values()))).device
        with tf.device(device):
            optimizer: Any = (
                _keras_constructor(state.optimizer_type, state.config)
                if factory is None
                else factory(parameters, _config_tree(state.config, frozen=False))
            )
            if type(optimizer) is not _keras_types().get(state.optimizer_type):
                _refuse(
                    "optimizer_factory_type",
                    "Factory returned an incompatible Keras optimizer type.",
                    backend="tf",
                )
            if optimizer.built or int(optimizer.iterations.numpy()) != 0:
                _refuse(
                    "optimizer_factory_not_fresh",
                    "Keras factories must return a fresh, unbuilt optimizer.",
                    backend="tf",
                )
            if _keras_config(optimizer) != _config_tree(state.config, frozen=False):
                _refuse(
                    "optimizer_config_incompatible",
                    "Factory returned an incompatible Keras optimizer configuration.",
                    backend="tf",
                )
            _keras_state_variables(optimizer, state.optimizer_type, (), parameters)
            optimizer.build(list(parameters.values()))
        return optimizer

    def restore_state(
        self, optimizer: Any, state: OptimizerState, parameters: Mapping[str, Any], /
    ) -> None:
        """Assign the prepared Keras resources and verify all migrated values."""

        names = _slot_layout(state.optimizer_type, state.config)
        globals_, native = _keras_state_variables(
            optimizer, state.optimizer_type, names, parameters
        )
        for tensor in state.global_state:
            _assign(globals_[tensor.slot_name], tensor)
        for key, tensors in state.slots.items():
            for tensor in tensors:
                _assign(native[key][tensor.slot_name], tensor)


_ADAPTER = _KerasOptimizerAdapter()
