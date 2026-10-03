"""Functional parameter bindings for native MLX split segments."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from typing import Any

from ...backends.mlx._call_state import mlx_rng_scope, snapshot_mlx_rng
from ...backends.mlx.containers import iter_arrays_with_paths
from ...backends.mlx.validation import MLXOpCapture
from ..errors import SplitUnsupportedError
from ..state import SegmentState, StateEntry
from ._mlx_coordination import MlxStateCoordinator, snapshot_mlx_values


class MlxSegmentState(SegmentState):
    """Own replaceable training parameters while retaining stable capture identities."""

    def __init__(self, *, own_parameters: bool, **kwargs: Any) -> None:
        """Create a state binding with an optional private training parameter tree."""

        super().__init__(**kwargs)
        self.own_parameters = own_parameters
        self.version = 0

    def resolve(
        self,
        value: Any,
        *,
        trainable: bool | None = None,
        shareable: bool = False,
        source_id: int | None = None,
    ) -> Any:
        """Resolve state, owning training arrays so updates never write the source model."""

        resolved = super().resolve(
            value, trainable=trainable, shareable=shareable, source_id=source_id
        )
        if self.own_parameters and trainable and self.adapter.is_tensor(value):
            key = id(value) if source_id is None else source_id
            entry = self._entries[key]
            if entry.ownership != "owned":
                resolved = self.adapter.replicate_state(
                    resolved, self.placement.device, trainable=True
                )
                self._entries[key] = replace(entry, ownership="owned", value=resolved)
        return resolved


class MlxStateBinding:
    """Bind array literals by identity and expose exactly the segment's trainable leaves."""

    def __init__(
        self,
        *,
        graph: Any,
        node_ids: frozenset[str],
        adapter: Any,
        placement: Any,
        own_parameters: bool,
    ) -> None:
        """Inventory captured parameter identities before resolving array literals."""

        self.adapter = adapter
        self.state = MlxSegmentState(
            adapter=adapter, placement=placement, own_parameters=own_parameters
        )
        self._parameters: dict[int, tuple[str, bool]] = {}
        self._source_ids = {
            source: canonical
            for node in graph.nodes
            if isinstance(node.target, MLXOpCapture)
            for source, canonical in node.target.source_ids
        }
        self._overrides: dict[int, Any] = {}
        self._local_state: dict[int, Any] | None = None
        self._local_updates: dict[int, Any] = {}
        self._rng_override: dict[str, tuple[Any, ...]] | None = None
        self.rng_journal: dict[str, tuple[Any, ...]] = {}
        self.coordinator = MlxStateCoordinator(self)
        self._nodes = tuple(node for node in graph.nodes if node.canonical_id in node_ids)
        for node in graph.nodes:
            for param in node.param_refs:
                value = getattr(param, "_param_ref", None)
                if adapter.is_tensor(value):
                    self._parameters.setdefault(
                        self.source_id(value), (str(param.address), bool(param.is_trainable))
                    )
            capture = node.target
            if not isinstance(capture, MLXOpCapture) or capture.module_ref is None:
                continue
            module = capture.module_ref
            trainable_ids = {
                id(value)
                for value, _path in iter_arrays_with_paths(
                    module.trainable_parameters(), adapter.is_tensor
                )
            }
            for value, path in iter_arrays_with_paths(module.parameters(), adapter.is_tensor):
                address = ".".join((node.module_path, *(str(part) for part in path))).removeprefix(
                    "self."
                )
                self._parameters.setdefault(
                    self.source_id(value), (address, id(value) in trainable_ids)
                )

    def source_id(self, value: Any) -> int:
        """Resolve replaceable module arrays to their first capture-time state identity."""

        return self._source_ids.get(id(value), id(value))

    def resolve(self, value: Any) -> Any:
        """Resolve one unlabeled captured array, applying a scoped AD substitution first."""

        source_id = self.source_id(value)
        if source_id in self._overrides:
            return self._overrides[source_id]
        if self._local_state is not None and source_id in self._local_state:
            return self._local_state[source_id]
        _name, trainable = self._parameters.get(source_id, ("", False))
        value = self.coordinator.values.get(source_id, value)
        resolved = self.state.resolve(value, trainable=trainable, source_id=source_id)
        self.coordinator.values.setdefault(source_id, resolved)
        return resolved

    def _literal_values(self) -> Iterator[tuple[str, Any]]:
        """Yield captured state occurrences without materializing their bindings."""

        for node in self._nodes:
            if node.is_input:
                continue
            capture = node.target
            if isinstance(capture, MLXOpCapture):
                leaves = iter_arrays_with_paths(
                    (capture.args, capture.kwargs), self.adapter.is_tensor
                )
                for index, (value, _path) in enumerate(leaves):
                    yield f"{node.canonical_id}:literal:{index}", value
            elif not node.parents and not node.is_output:
                value = getattr(node.op, "out", None)
                if self.adapter.is_tensor(value):
                    yield f"{node.canonical_id}:source", value

    def bound_values(self) -> dict[str, Any]:
        """Expose state by stable node/template occurrence for recutting and placement."""

        return {key: self.resolve(value) for key, value in self._literal_values()}

    def trainable_source_ids(self) -> frozenset[int]:
        """Return consumed trainable identities without freezing lazy segment state."""

        return frozenset(
            self.source_id(value)
            for _key, value in self._literal_values()
            if self._parameters.get(self.source_id(value), ("", False))[1]
        )

    def parameters(self) -> dict[str, Any]:
        """Return unique, consumed trainable arrays keyed by their capture addresses."""

        import mlx.core as mx

        self.bound_values()
        return {
            self._parameters[entry.source_id][0]: entry.value
            for entry in self.state.entries()
            if entry.source_id in self._parameters
            and entry.trainable
            and mx.issubdtype(entry.value.dtype, mx.floating)
        }

    @contextmanager
    def using_parameters(self, values: dict[str, Any]) -> Iterator[None]:
        """Temporarily substitute functional gradient roots without committing state."""

        previous = self._overrides
        self._overrides = {
            source_id: values[name]
            for source_id, (name, _trainable) in self._parameters.items()
            if name in values
        }
        try:
            yield
        finally:
            self._overrides = previous

    def update_parameters(self, values: dict[str, Any]) -> None:
        """Validate an optimizer's entire result before committing replaceable arrays."""

        parameters = self.parameters()
        if values.keys() != parameters.keys():
            raise SplitUnsupportedError("MLX optimizer returned a different parameter inventory.")
        for name, value in values.items():
            if (
                not self.adapter.is_tensor(value)
                or value.shape != parameters[name].shape
                or value.dtype != parameters[name].dtype
            ):
                raise SplitUnsupportedError(
                    f"MLX optimizer changed parameter {name!r} shape or dtype."
                )
        staged: dict[int, StateEntry] = {}
        for entry in self.state.entries():
            name = self._parameters.get(entry.source_id, ("", False))[0]
            if name in values:
                staged[entry.source_id] = replace(entry, ownership="owned", value=values[name])
        self.coordinator.publish(
            {key: entry.value for key, entry in staged.items()}, parameters=True
        )

    @property
    def is_replaying(self) -> bool:
        """Return whether a prefix VJP is recomputing a saved forward."""

        return self._rng_override is not None

    def begin_forward(self) -> dict[int, Any]:
        """Save pre-forward state and begin a fresh random-call journal."""

        if self.is_replaying:
            return {}
        self.bound_values()
        self.rng_journal = {}
        return snapshot_mlx_values({entry.source_id: entry.value for entry in self.state.entries()})

    @contextmanager
    def using_random_journal(self, rng: dict[str, tuple[Any, ...]]) -> Iterator[None]:
        """Match independent probe keys while retaining ordinary shared-buffer commits."""

        previous = self._rng_override
        self._rng_override = rng
        try:
            yield
        finally:
            self._rng_override = previous

    @contextmanager
    def random_call(self, capture: MLXOpCapture) -> Iterator[None]:
        """Advance native randomness once, or replay a saved call without advancing it."""

        if capture.rng_state is None:
            yield
            return
        key = capture.labels_raw[0]
        if self._rng_override is not None:
            with mlx_rng_scope(self._rng_override[key]):
                yield
        else:
            self.rng_journal[key] = snapshot_mlx_rng()
            yield

    def module_arrays(self, capture: MLXOpCapture, args: tuple[Any, ...]) -> list[Any]:
        """Pair captured module state paths with their concrete pre-call values."""

        if capture.module_ref is None or not args:
            return []
        result = []
        for source, path in iter_arrays_with_paths(capture.module_ref, self.adapter.is_tensor):
            value = args[0]
            for part in path:
                value = value[part]
            result.append((path, self.source_id(source), value))
        return result

    def commit_module(self, before: list[Any], module: Any) -> None:
        """Route native buffer replacements into logical state rather than the source model."""

        updates = {}
        for path, source_id, previous in before:
            value = module
            for part in path:
                value = value[part]
            if self.adapter.is_tensor(value) and value is not previous:
                updates[source_id] = value
        if self._local_state is not None:
            self._local_state.update(updates)
            self._local_updates.update(updates)
        else:
            self.coordinator.publish(updates)

    @contextmanager
    def functional_forward(self) -> Iterator[dict[int, Any]]:
        """Stage suffix buffer updates until loss and gradients have evaluated successfully."""

        self.bound_values()
        previous, previous_updates = self._local_state, self._local_updates
        self._local_state = {entry.source_id: entry.value for entry in self.state.entries()}
        self._local_updates = {}
        try:
            yield self._local_updates
        finally:
            self._local_state, self._local_updates = previous, previous_updates

    @contextmanager
    def replaying(self, values: dict[int, Any], rng: dict[str, tuple[Any, ...]]) -> Iterator[None]:
        """Recompute saved state and masks locally, discarding all repeated buffer updates."""

        previous, previous_rng = self._local_state, self._rng_override
        previous_updates = self._local_updates
        self._local_state, self._rng_override = dict(values), rng
        self._local_updates = {}
        try:
            yield
        finally:
            self._local_state, self._rng_override = previous, previous_rng
            self._local_updates = previous_updates


__all__ = ["MlxStateBinding"]
