"""Native PyTorch optimizer state keyed by effective split parameter identity."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..._utils._numpy_bridge import tensor_from_numpy, tensor_to_numpy
from ..._utils._torch_symbols import torch_dtype_named
from ..adapters import torch as _adapter
from .base import OptimizerFactory, OptimizerState, OptimizerStateTensor, _config_tree, _refuse

_TYPES = ("SGD", "Adam", "AdamW")


def _torch_types() -> dict[str, type]:
    """Resolve the supported exact native optimizer classes."""

    namespace = _adapter._torch().optim
    return {name: getattr(namespace, name) for name in _TYPES}


def _scalar_tensor_option(value: Any) -> dict[str, Any]:
    """Describe one detached numerical option such as a tensor learning rate."""

    if value.ndim or value.requires_grad or not value.is_floating_point():
        _refuse(
            "optimizer_config_unsupported",
            "Optimizer tensor options must be detached floating scalars.",
            backend="torch",
        )
    return {
        "tensor_dtype": str(value.dtype).removeprefix("torch."),
        "value": value.item(),
        "device": "cpu" if value.device.type == "cpu" else "parameter",
    }


def _export_option(value: Any) -> Any:
    """Encode plain options and scalar tensor learning rates without references."""

    if isinstance(value, _adapter._torch().Tensor):
        return _scalar_tensor_option(value)
    if isinstance(value, (list, tuple)):
        return tuple(_export_option(item) for item in value)
    return value


def _new_scalar_tensor(option: Mapping[str, Any], device: Any) -> Any:
    """Make independent tensor option storage using its serialized dtype and device policy."""

    torch = _adapter._torch()
    dtype = torch_dtype_named(option["tensor_dtype"])
    if dtype is None or option.get("device") not in {"cpu", "parameter"}:
        _refuse("optimizer_config_unsupported", "Invalid tensor optimizer option.", backend="torch")
    return torch.tensor(
        option["value"], dtype=dtype, device="cpu" if option["device"] == "cpu" else device
    )


def _import_option(option: Any, device: Any) -> Any:
    """Reconstruct portable group options before calling the native constructor."""

    if isinstance(option, Mapping) and "tensor_dtype" in option:
        return _new_scalar_tensor(option, device)
    if isinstance(option, (list, tuple)):
        return tuple(_import_option(item, device) for item in option)
    return option


def _parameter_ids(parameters: Mapping[str, Any]) -> dict[int, str]:
    """Index distinct live tensors by transient identity within one optimizer call."""

    identities = {id(value): key for key, value in parameters.items()}
    if len(identities) != len(parameters):
        _refuse(
            "optimizer_parameter_ownership",
            "Torch parameter bindings contain a duplicate tensor.",
            backend="torch",
        )
    return identities


def _record_group(group: dict[str, Any], identities: Mapping[int, str]) -> dict[str, Any]:
    """Encode a group using the actual live tensor in each parameter position."""

    keys = []
    for value in group["params"]:
        key = identities.get(id(value))
        if key is None:
            _refuse(
                "optimizer_parameter_ownership",
                "Torch optimizer owns a tensor outside its segment.",
                backend="torch",
            )
        keys.append(key)
    options = {
        key: _export_option(value)
        for key, value in group.items()
        if key not in {"params", "param_names"}
    }
    return {
        "parameters": tuple(keys),
        "options": options,
        "names": tuple(group.get("param_names", ())),
    }


def _group_options(
    config: Mapping[str, Any], parameters: Mapping[str, Any]
) -> dict[str, Mapping[str, Any]]:
    """Validate exact parameter group membership and return per-parameter options."""

    options: dict[str, Mapping[str, Any]] = {}
    for group in config["groups"]:
        keys = group["parameters"]
        names = group["names"]
        if names and len(names) != len(keys):
            _refuse(
                "optimizer_parameter_inventory",
                "Torch param_names differs from its parameter group.",
                backend="torch",
            )
        for key in keys:
            if key in options:
                _refuse(
                    "optimizer_parameter_ownership",
                    f"Duplicate Torch optimizer ownership for {key!r}.",
                    backend="torch",
                )
            options[key] = group["options"]
    if options.keys() != parameters.keys():
        _refuse(
            "optimizer_parameter_ownership",
            "Torch optimizer parameter groups differ from the segment.",
            backend="torch",
        )
    return options


def _slot_layout(kind: str, options: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the complete native tensor state schema after a parameter's first step."""

    if options.get("differentiable"):
        _refuse(
            "optimizer_config_unsupported",
            "Differentiable optimizer history is unsupported.",
            backend="torch",
        )
    if kind == "SGD":
        return ("momentum_buffer",) if options["momentum"] else ()
    adam = ("exp_avg", "exp_avg_sq", "step")
    return (*adam, "max_exp_avg_sq") if options["amsgrad"] else adam


def _slot_device(name: str, parameter: Any, options: Mapping[str, Any]) -> Any:
    """Match the native Adam step placement and parameter-local moment placement."""

    if name == "step" and not (options.get("capturable") or options.get("fused")):
        return _adapter._torch().device("cpu")
    return parameter.device


def _slot_tensor(state_id: str, name: str, value: Any) -> OptimizerStateTensor:
    """Copy a dense native state tensor through TorchLens's NumPy compatibility path."""

    torch = _adapter._torch()
    if not isinstance(value, torch.Tensor) or value.layout != torch.strided or value.requires_grad:
        _refuse(
            "optimizer_slot_unsupported",
            f"Torch slot {name!r} must be a dense detached tensor.",
            backend="torch",
        )
    host = value.detach().resolve_conj().resolve_neg().cpu()
    dtype = "bfloat16" if host.dtype == torch.bfloat16 else None
    if dtype is not None:
        host = host.view(torch.int16)
    try:
        return OptimizerStateTensor.from_array(state_id, name, tensor_to_numpy(host), dtype=dtype)
    except TypeError as exc:
        _refuse(
            "optimizer_slot_dtype",
            f"Unsupported Torch slot dtype for {name!r}.",
            backend="torch",
            cause=exc,
        )


def _check_slot(slot: OptimizerStateTensor, parameter: Any) -> None:
    """Validate scalar step counters and parameter-shaped native history."""

    if slot.slot_name == "step":
        _check_step(slot)
    elif slot.shape != tuple(parameter.shape):
        _refuse(
            "optimizer_slot_shape", f"Slot shape mismatch for {slot.state_id!r}.", backend="torch"
        )
    elif slot.dtype != str(parameter.dtype).removeprefix("torch."):
        _refuse(
            "optimizer_slot_dtype", f"Slot dtype mismatch for {slot.state_id!r}.", backend="torch"
        )


def _check_step(slot: OptimizerStateTensor) -> None:
    """Validate the native Adam step counter independently of parameter geometry."""

    if slot.shape != () or slot.dtype not in {"float32", "float64"}:
        _refuse(
            "optimizer_slot_shape_dtype",
            "Torch step must be a float32 or float64 scalar.",
            backend="torch",
        )
    value = slot.payload.item()
    if not value.is_integer() or value < 0:
        _refuse(
            "optimizer_slot_value",
            "Torch optimizer step must be a nonnegative integer.",
            backend="torch",
        )


def _export_slots(
    optimizer: Any, parameters: Mapping[str, Any], options: Mapping[str, Mapping[str, Any]]
) -> dict[str, tuple[OptimizerStateTensor, ...]]:
    """Copy and check every per-parameter slot without using state_dict integer IDs."""

    output = {}
    for state_id, param in parameters.items():
        slots = []
        for name, value in optimizer.state.get(param, {}).items():
            tensor = _slot_tensor(state_id, name, value)
            if value.device != _slot_device(name, param, options[state_id]):
                _refuse(
                    "optimizer_slot_device",
                    f"Torch slot {name!r} has incompatible device placement.",
                    backend="torch",
                )
            _check_slot(tensor, param)
            slots.append(tensor)
        output[state_id] = tuple(slots)
    return output


def _selected_group(group: Mapping[str, Any], selected: set[str]) -> dict[str, Any]:
    """Keep the order and native metadata of parameters remaining in a group."""

    positions = tuple(i for i, key in enumerate(group["parameters"]) if key in selected)
    return {
        "parameters": tuple(group["parameters"][i] for i in positions),
        "options": group["options"],
        "names": tuple(group["names"][i] for i in positions) if group["names"] else (),
    }


def _native_group(group: Mapping[str, Any], parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct the group with effective target parameter objects."""

    values = [parameters[key] for key in group["parameters"]]
    device = values[0].device if values else next(iter(parameters.values())).device
    config = {key: _import_option(value, device) for key, value in group["options"].items()}
    config["params"] = values
    if group["names"]:
        config["param_names"] = list(group["names"])
    return config


def _native_slot(slot: OptimizerStateTensor, parameter: Any, options: Mapping[str, Any]) -> Any:
    """Make independent native storage for a target slot and attest its contents."""

    torch = _adapter._torch()
    tensor = tensor_from_numpy(slot.payload.astype(slot.payload.dtype.newbyteorder("="), copy=True))
    if slot.dtype == "bfloat16":
        tensor = tensor.view(torch.bfloat16)
    tensor = tensor.to(_slot_device(slot.slot_name, parameter, options))
    if _slot_tensor(slot.state_id, slot.slot_name, tensor).checksum != slot.checksum:
        _refuse(
            "optimizer_restore_verification",
            f"Torch state verification failed for {slot.slot_name!r}.",
            backend="torch",
        )
    return tensor


def _new_torch_optimizer(state: OptimizerState, parameters: Mapping[str, Any]) -> Any:
    """Reconstruct native groups and constructor defaults around target tensors."""

    cls = _torch_types()[state.optimizer_type]
    device = next(iter(parameters.values())).device
    defaults = {
        key: _import_option(value, device) for key, value in state.config["defaults"].items()
    }
    # AdamW supplies decoupling itself. Some Torch releases also store that fact in defaults.
    if state.optimizer_type == "AdamW":
        defaults.pop("decoupled_weight_decay", None)
    groups = [_native_group(group, parameters) for group in state.config["groups"]]
    return cls(groups, **defaults)


class _TorchOptimizerAdapter:
    """Native Torch support for SGD, Adam and AdamW with parameter groups."""

    name = "torch"

    def supports(self, optimizer: Any, /) -> bool:
        """Accept exact supported native classes."""

        return type(optimizer) in _torch_types().values()

    def export_state(self, optimizer: Any, parameters: Mapping[str, Any], /) -> OptimizerState:
        """Verify optimizer ownership and copy complete named native state."""

        ids = _parameter_ids(parameters)
        if any(id(param) not in ids for param in optimizer.state):
            _refuse(
                "optimizer_parameter_ownership",
                "Torch optimizer contains state outside its parameter groups.",
                backend="torch",
            )
        config = {
            "defaults": {key: _export_option(value) for key, value in optimizer.defaults.items()},
            "groups": tuple(_record_group(group, ids) for group in optimizer.param_groups),
        }
        kind = type(optimizer).__name__
        options = _group_options(config, parameters)
        expected = {key: _slot_layout(kind, group) for key, group in options.items()}
        return OptimizerState(
            kind, config, (), expected, _export_slots(optimizer, parameters, options)
        )

    def merge_config(
        self, sources: Mapping[str, OptimizerState], routes: Mapping[str, str], /
    ) -> Mapping[str, Any]:
        """Preserve group-local options while matching source constructor defaults."""

        first = next(iter(sources.values())).config
        defaults = first["defaults"]
        if any(state.config["defaults"] != defaults for state in sources.values()):
            _refuse(
                "optimizer_config_incompatible",
                "Torch source optimizer defaults differ.",
                backend="torch",
            )
        groups = []
        for owner, state in sources.items():
            selected = {key for key, source in routes.items() if source == owner}
            for group in state.config["groups"]:
                narrowed = _selected_group(group, selected)
                if narrowed["parameters"] or not group["parameters"]:
                    groups.append(narrowed)
        return {"defaults": defaults, "groups": tuple(groups)}

    def validate_restore(self, state: OptimizerState, parameters: Mapping[str, Any], /) -> None:
        """Check target groups and individual Torch state geometry."""

        if state.global_state or state.optimizer_type not in _torch_types():
            _refuse(
                "optimizer_global_state",
                "Invalid global state or optimizer type for Torch.",
                backend="torch",
            )
        options = _group_options(state.config, parameters)
        for key, parameter in parameters.items():
            if state.expected_slots[key] != _slot_layout(state.optimizer_type, options[key]):
                _refuse(
                    "optimizer_slot_inventory",
                    f"Invalid Torch slot schema for {key!r}.",
                    backend="torch",
                )
            if options[key].get("capturable") and parameter.device.type == "cpu":
                _refuse(
                    "optimizer_slot_device",
                    "Capturable Torch optimizers require an accelerator.",
                    backend="torch",
                )
            for slot in state.slots[key]:
                _check_slot(slot, parameter)

    def build_optimizer(
        self,
        state: OptimizerState,
        parameters: Mapping[str, Any],
        factory: OptimizerFactory | None,
        /,
    ) -> Any:
        """Build native groups around the exact effective parameters for this target."""

        optimizer: Any = (
            _new_torch_optimizer(state, parameters)
            if factory is None
            else factory(parameters, _config_tree(state.config, frozen=False))
        )
        if not self.supports(optimizer) or type(optimizer).__name__ != state.optimizer_type:
            _refuse(
                "optimizer_factory_type",
                "Factory returned an incompatible Torch optimizer type.",
                backend="torch",
            )
        if optimizer.state:
            _refuse(
                "optimizer_factory_not_fresh",
                "Torch factories must return a fresh optimizer without history.",
                backend="torch",
            )
        if self.export_state(optimizer, parameters).config != state.config:
            _refuse(
                "optimizer_config_incompatible",
                "Factory returned incompatible Torch parameter groups or defaults.",
                backend="torch",
            )
        return optimizer

    def restore_state(
        self, optimizer: Any, state: OptimizerState, parameters: Mapping[str, Any], /
    ) -> None:
        """Populate independent native state storage for the newly built optimizer."""

        options = _group_options(state.config, parameters)
        optimizer.state.update(
            {
                parameters[key]: {
                    slot.slot_name: _native_slot(slot, parameters[key], options[key])
                    for slot in slots
                }
                for key, slots in state.slots.items()
                if slots
            }
        )


_ADAPTER = _TorchOptimizerAdapter()
