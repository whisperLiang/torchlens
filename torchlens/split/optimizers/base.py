"""Portable optimizer records and the native optimizer adapter contract."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, NoReturn, Protocol

import numpy as np
from numpy.typing import NDArray

from ..errors import SplitErrorContext, SplitUnsupportedError

if TYPE_CHECKING:
    from ..ir import GraphPartition

OptimizerFactory = Callable[[Mapping[str, Any], Mapping[str, Any]], Any]
_Action = Literal["keep", "transfer", "initialize", "unsupported"]
_ParameterSpec = tuple[str, tuple[int, ...], str]
_SCHEMA_VERSION = 1
_GLOBAL = "optimizer"


def _refuse(
    reason: str,
    message: str,
    backend: str = "",
    partition: str = "",
    *,
    cause: Exception | None = None,
) -> NoReturn:
    """Raise a coded migration refusal with its specific optimizer-state reason."""

    raise SplitUnsupportedError(
        message,
        context=SplitErrorContext(backend, partition, reason),
        code=SplitUnsupportedError.code,
    ) from cause


def _config_tree(value: Any, *, frozen: bool) -> Any:
    """Copy safe configuration data without retaining opaque objects or callables."""

    if value is None or type(value) in (str, bool, int, float):
        return value
    if isinstance(value, Mapping) and all(type(key) is str for key in value):
        values = {key: _config_tree(item, frozen=frozen) for key, item in value.items()}
        return MappingProxyType(values) if frozen else values
    if isinstance(value, (list, tuple)):
        items = tuple(_config_tree(item, frozen=frozen) for item in value)
        return items if frozen else list(items)
    _refuse("optimizer_config_unsupported", f"Nonportable optimizer config value {type(value)!r}.")


def _json_bytes(value: Any) -> bytes:
    """Encode metadata deterministically for integrity comparisons."""

    return json.dumps(
        _config_tree(value, frozen=False), sort_keys=True, separators=(",", ":")
    ).encode()


def _tensor_checksum(
    state_id: str, slot_name: str, shape: tuple[int, ...], dtype: str, payload: NDArray[Any]
) -> str:
    """Hash the slot's logical identity, array metadata, and canonical bytes."""

    digest = sha256(_json_bytes((state_id, slot_name, shape, dtype)))
    digest.update(payload.reshape(-1).view(np.uint8).data)
    return digest.hexdigest()


@dataclass(frozen=True, eq=False)
class OptimizerStateTensor:
    """Independent CPU data and SHA256 for one named state tensor.

    Examples
    --------
    >>> import numpy as np
    >>> slot = tl.split.OptimizerStateTensor.from_array('parameter:weight', 'momentum', np.zeros(3))
    >>> slot.shape, slot.dtype, slot.payload.flags.writeable
    ((3,), 'float64', False)
    """

    state_id: str
    slot_name: str
    shape: tuple[int, ...]
    dtype: str
    payload: NDArray[Any] = field(repr=False)
    checksum: str

    @classmethod
    def from_array(
        cls, state_id: str, slot_name: str, payload: NDArray[Any], /, *, dtype: str | None = None
    ) -> OptimizerStateTensor:
        """Copy native state into immutable canonical host storage.

        Examples
        --------
        >>> import numpy as np
        >>> tl.split.OptimizerStateTensor.from_array('optimizer', 'iterations', np.array(3)).shape
        ()
        """

        logical_dtype = payload.dtype.name if dtype is None else dtype
        layout = _slot_dtype(logical_dtype)
        if logical_dtype == "bfloat16" and payload.dtype.name == "bfloat16":
            payload = payload.view(np.int16)
        if payload.dtype.name != layout.name:
            _refuse("optimizer_slot_dtype", f"Optimizer slot {slot_name!r} has an invalid dtype.")
        canonical = payload.astype(layout, copy=False)
        array = np.frombuffer(canonical.tobytes(order="C"), dtype=layout).reshape(payload.shape)
        checksum = _tensor_checksum(state_id, slot_name, array.shape, logical_dtype, array)
        return cls(state_id, slot_name, array.shape, logical_dtype, array, checksum)

    def _validate(self) -> None:
        """Reject invalid array metadata or modified payload bytes."""

        array = self.payload
        if not isinstance(array, np.ndarray) or array.shape != self.shape:
            _refuse("optimizer_slot_shape", f"Shape mismatch for slot {self.slot_name!r}.")
        if array.dtype != _slot_dtype(self.dtype) or not array.flags.c_contiguous:
            _refuse(
                "optimizer_slot_dtype", f"Dtype or storage mismatch for slot {self.slot_name!r}."
            )
        if (
            _tensor_checksum(self.state_id, self.slot_name, self.shape, self.dtype, array)
            != self.checksum
        ):
            _refuse("optimizer_slot_checksum", f"Checksum mismatch for slot {self.slot_name!r}.")


def _slot_dtype(dtype: str) -> np.dtype[Any]:
    """Resolve canonical numeric storage, including bit-exact bfloat16."""

    try:
        storage = np.dtype("int16" if dtype == "bfloat16" else dtype)
    except TypeError as exc:
        _refuse("optimizer_slot_dtype", f"Unknown optimizer tensor dtype {dtype!r}.", cause=exc)
    if storage.kind not in "biufc":
        _refuse("optimizer_slot_dtype", f"Unsupported optimizer tensor dtype {dtype!r}.")
    return storage.newbyteorder("<")


@dataclass(frozen=True)
class OptimizerState:
    """Interchange record for one optimizer, indexed by stable parameter ID.

    Examples
    --------
    >>> from torchlens.split.optimizers import OptimizerState
    >>> state = OptimizerState('SGD', {'momentum': 0.0}, (), {'parameter:weight': ()}, {})
    >>> state.config['momentum']
    0.0
    """

    optimizer_type: str
    config: Mapping[str, Any]
    global_state: tuple[OptimizerStateTensor, ...]
    expected_slots: Mapping[str, tuple[str, ...]]
    slots: Mapping[str, tuple[OptimizerStateTensor, ...]]

    def __post_init__(self) -> None:
        """Freeze the adapter's metadata and state collections."""

        object.__setattr__(self, "config", _config_tree(self.config, frozen=True))
        for name in ("expected_slots", "slots"):
            values = MappingProxyType(
                {key: tuple(value) for key, value in getattr(self, name).items()}
            )
            object.__setattr__(self, name, values)
        object.__setattr__(self, "global_state", tuple(self.global_state))


@dataclass(frozen=True)
class OptimizerStateMigration:
    """Action and slot names for one parameter's optimizer transition.

    Examples
    --------
    >>> row = tl.split.OptimizerStateMigration('parameter:weight', 'suffix', 'prefix', 'transfer', ())
    >>> row.target_optimizer
    'prefix'
    """

    state_id: str
    source_optimizer: str
    target_optimizer: str
    action: _Action
    slot_names: tuple[str, ...]


@dataclass(frozen=True)
class OptimizerStateMigrationPlan:
    """Read-only plan with per-parameter decisions and structured refusals.

    Examples
    --------
    >>> plan = tl.split.OptimizerStateMigrationPlan('tf', 'graph', 'cut-a', 'cut-b', ())
    >>> plan.supported
    True
    """

    backend: str
    graph_id: str
    source_partition: str
    target_partition: str
    entries: tuple[OptimizerStateMigration, ...]
    unsupported_reasons: tuple[str, ...] = ()
    _partition: GraphPartition | None = field(default=None, repr=False)
    _source_checksum: str = field(default="", repr=False)

    @property
    def supported(self) -> bool:
        """Return whether every optimizer transition can be represented safely."""

        return not self.unsupported_reasons and all(
            row.action != "unsupported" for row in self.entries
        )


@dataclass(frozen=True)
class OptimizerStateSnapshot:
    """A schema-versioned, device-independent snapshot for a target cut.

    Examples
    --------
    >>> runtime = tl.split.prepare_graph(
    ...     nn.Linear(3, 2), torch.ones(2, 3), features=tl.split.SplitFeatures(training=True))
    >>> cut = runtime.compute_dag().to_partition(())
    >>> source = runtime.materialize(runtime.analyze(cut))
    >>> opts = {'suffix': torch.optim.SGD(source.model.parameters(), lr=0.1)}
    >>> plan = source.plan_optimizer_state_migration(cut, optimizers=opts)
    >>> snapshot = source.export_optimizer_state(optimizers=opts, plan=plan)
    >>> snapshot.schema_version, tuple(snapshot.optimizers)
    (1, ('suffix',))
    """

    schema_version: int
    backend: str
    graph_id: str
    source_partition: str
    target_partition: str
    entries: tuple[OptimizerStateMigration, ...]
    optimizers: Mapping[str, OptimizerState] = field(repr=False)
    adapter_names: Mapping[str, str]
    parameter_specs: Mapping[str, _ParameterSpec]
    checksum: str


@dataclass(frozen=True)
class OptimizerStateMigrationResult:
    """Fresh native optimizers and counts of restored or initialized state.

    Examples
    --------
    >>> result = tl.split.OptimizerStateMigrationResult({}, (), 0, 0, {'unsupported': 0})
    >>> result.restored_slots
    0
    """

    optimizers: Mapping[str, Any] = field(repr=False)
    restored_parameters: tuple[str, ...]
    restored_slots: int
    initialized_slots: int
    migration_summary: Mapping[str, int]


class OptimizerStateAdapter(Protocol):
    """Inspect, construct, and restore native optimizer state by stable ID.

    Implementations must build fresh optimizers and only write to their own
    detached state. Factories receive a stable-ID mapping of target parameters
    and a plain copy of the validated optimizer configuration.

    Examples
    --------
    >>> adapter = tl.split.optimizers.get_optimizer_adapter('torch', torch.optim.SGD(model.parameters(), lr=0.1))
    >>> adapter.name
    'torch'
    """

    name: str

    def supports(self, optimizer: Any, /) -> bool:
        """Return whether this adapter owns the exact native optimizer type."""
        ...

    def export_state(self, optimizer: Any, parameters: Mapping[str, Any], /) -> OptimizerState:
        """Validate optimizer ownership and copy its portable state without mutation."""
        ...

    def merge_config(
        self, sources: Mapping[str, OptimizerState], routes: Mapping[str, str], /
    ) -> Mapping[str, Any]:
        """Return compatible configuration for the target parameter routes."""
        ...

    def validate_restore(self, state: OptimizerState, parameters: Mapping[str, Any], /) -> None:
        """Preflight the complete state schema against the effective target variables."""
        ...

    def build_optimizer(
        self,
        state: OptimizerState,
        parameters: Mapping[str, Any],
        factory: OptimizerFactory | None,
        /,
    ) -> Any:
        """Construct a fresh target optimizer using the validated configuration."""
        ...

    def restore_state(
        self, optimizer: Any, state: OptimizerState, parameters: Mapping[str, Any], /
    ) -> None:
        """Assign the prepared optimizer's state and verify the native result."""
        ...
