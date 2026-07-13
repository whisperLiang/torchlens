"""Backend-neutral intermediate representation for split programs.

The objects in this module deliberately contain only metadata that can be
shared by split backends.  A backend may keep an opaque :class:`BackendHandle`
next to an IR node while lowering, but that handle is never included in the
portable representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping

if TYPE_CHECKING:
    from .graph import SplitTraceGraph, SplitTraceNode
    from .planner import SplitPlan


class SplitVerificationStatus(str, Enum):
    """Verification level for a lowered split component."""

    EXACT = "exact"
    REGION_VERIFIED = "region_verified"
    UNVERIFIED = "unverified"
    UNSUPPORTED = "unsupported"
    ENVIRONMENT_UNAVAILABLE = "environment_unavailable"


@dataclass(frozen=True)
class BackendHandle:
    """Opaque backend-owned replay handle.

    ``payload`` is intentionally excluded from :meth:`as_dict`; it may be a
    callable, a JAX jaxpr, a TensorFlow graph object, or a lazy UOp graph.
    """

    backend: str
    kind: str
    location: str | None = None
    payload: Any = field(default=None, repr=False, compare=False)

    def as_dict(self) -> dict[str, Any]:
        """Return portable handle metadata without the executable payload."""

        return {"backend": self.backend, "kind": self.kind, "location": self.location}


@dataclass(frozen=True)
class ShapeConstraint:
    """One symbolic shape relation used during replay."""

    constraint_id: str
    kind: Literal[
        "batch_axis",
        "broadcast",
        "reshape",
        "flatten",
        "concat",
        "attention",
        "index",
        "shape_producing",
        "opaque",
        "equal",
        "product",
        "range",
        "axis",
    ]
    value_ids: tuple[str, ...] = ()
    axes: tuple[int, ...] = ()
    expression: str | None = None
    description: str | None = None
    lhs: Any | None = None
    rhs: Any | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-like constraint metadata."""

        return {
            "constraint_id": self.constraint_id,
            "kind": self.kind,
            "value_ids": self.value_ids,
            "axes": self.axes,
            "expression": self.expression,
            "description": self.description,
            "lhs": self.lhs,
            "rhs": self.rhs,
        }


@dataclass(frozen=True)
class ValueIR:
    """A value flowing through a backend-neutral split graph."""

    value_id: str
    kind: Literal[
        "input",
        "output",
        "parameter",
        "buffer",
        "constant",
        "boundary",
        "intermediate",
    ]
    shape: Any | None = None
    dtype: str | None = None
    device: str | None = None
    requires_grad: bool | None = None
    alias_group: str | None = None
    source_id: str | None = None
    module_path: str | None = None
    provenance: str | None = None
    backend_handle: BackendHandle | None = field(default=None, repr=False, compare=False)

    def as_dict(self) -> dict[str, Any]:
        """Return portable value metadata."""

        return {
            "value_id": self.value_id,
            "kind": self.kind,
            "shape": self.shape,
            "dtype": self.dtype,
            "device": self.device,
            "requires_grad": self.requires_grad,
            "alias_group": self.alias_group,
            "source_id": self.source_id,
            "module_path": self.module_path,
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class OpIR:
    """A normal replayable operation and its value dependencies."""

    node_id: str
    op_type: str
    input_value_ids: tuple[str, ...]
    output_value_ids: tuple[str, ...]
    module_path: str | None
    backend_location: str | None
    replayable: bool
    trainable: bool
    dynamic_shape: bool
    verification: SplitVerificationStatus
    source: str | None = None
    backend_handle: BackendHandle | None = field(default=None, repr=False, compare=False)

    def as_dict(self) -> dict[str, Any]:
        """Return portable operation metadata."""

        return {
            "node_id": self.node_id,
            "op_type": self.op_type,
            "input_value_ids": self.input_value_ids,
            "output_value_ids": self.output_value_ids,
            "module_path": self.module_path,
            "backend_location": self.backend_location,
            "replayable": self.replayable,
            "trainable": self.trainable,
            "dynamic_shape": self.dynamic_shape,
            "verification": self.verification.value,
            "source": self.source,
        }


@dataclass(frozen=True)
class RegionIR:
    """An opaque or structured backend region such as JAX control flow."""

    node_id: str
    region_kind: str
    input_value_ids: tuple[str, ...]
    output_value_ids: tuple[str, ...]
    module_path: str | None
    replayable: bool
    trainable: bool
    dynamic_shape: bool
    verification: SplitVerificationStatus
    source: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    backend_handle: BackendHandle | None = field(default=None, repr=False, compare=False)

    def as_dict(self) -> dict[str, Any]:
        """Return portable region metadata."""

        return {
            "node_id": self.node_id,
            "region_kind": self.region_kind,
            "input_value_ids": self.input_value_ids,
            "output_value_ids": self.output_value_ids,
            "module_path": self.module_path,
            "replayable": self.replayable,
            "trainable": self.trainable,
            "dynamic_shape": self.dynamic_shape,
            "verification": self.verification.value,
            "source": self.source,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class StateIR:
    """Parameter, buffer, RNG, mutable-state, and alias declarations."""

    parameter_value_ids: tuple[str, ...] = ()
    buffer_value_ids: tuple[str, ...] = ()
    rng_value_ids: tuple[str, ...] = ()
    mutable_value_ids: tuple[str, ...] = ()
    alias_groups: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return portable state metadata."""

        return {
            "parameter_value_ids": self.parameter_value_ids,
            "buffer_value_ids": self.buffer_value_ids,
            "rng_value_ids": self.rng_value_ids,
            "mutable_value_ids": self.mutable_value_ids,
            "alias_groups": dict(self.alias_groups),
        }


BoundaryRole = Literal[
    "input",
    "output",
    "primary",
    "skip",
    "passthrough",
    "multi_scale_feature",
    "index",
    "shape_value",
]


@dataclass(frozen=True)
class BoundarySchema:
    """Stable ABI metadata for one split boundary value."""

    value_id: str
    container_path: tuple[Any, ...]
    role: BoundaryRole
    shape: Any | None
    dtype: str | None
    device: str | None = None
    requires_grad: bool | None = None
    alias_group: str | None = None
    source_kind: str = "boundary"
    label: str = ""
    backend: str = ""
    module_path: str | None = None
    op_type: str = ""
    output_index: int | None = None
    device_policy: str = "runtime"

    @property
    def canonical_id(self) -> str:
        """Return the stable boundary ID used by replay recipes."""

        return self.value_id

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-like boundary ABI metadata."""

        return {
            "value_id": self.value_id,
            "container_path": self.container_path,
            "role": self.role,
            "shape": self.shape,
            "dtype": self.dtype,
            "device": self.device,
            "requires_grad": self.requires_grad,
            "alias_group": self.alias_group,
            "source_kind": self.source_kind,
            "label": self.label,
            "backend": self.backend,
            "module_path": self.module_path,
            "op_type": self.op_type,
            "output_index": self.output_index,
            "device_policy": self.device_policy,
        }


@dataclass(frozen=True)
class SplitGraphIR:
    """Normalized graph shared by planners, capability analysis, and lowerers."""

    backend: str
    graph_hash: str | None
    profile_hash: str | None
    values: tuple[ValueIR, ...]
    ops: tuple[OpIR, ...]
    regions: tuple[RegionIR, ...]
    state: StateIR
    shape_constraints: tuple[ShapeConstraint, ...]
    boundary_schema: tuple[BoundarySchema, ...]
    input_value_ids: tuple[str, ...]
    output_value_ids: tuple[str, ...]

    @property
    def value_by_id(self) -> dict[str, ValueIR]:
        """Return values keyed by stable value ID."""

        return {value.value_id: value for value in self.values}

    @property
    def op_by_id(self) -> dict[str, OpIR | RegionIR]:
        """Return operations and regions keyed by stable node ID."""

        result: dict[str, OpIR | RegionIR] = {}
        for operation in self.ops:
            result[operation.node_id] = operation
        for region in self.regions:
            result[region.node_id] = region
        return result

    def as_dict(self) -> dict[str, Any]:
        """Return the portable schema; backend handles are omitted."""

        return {
            "backend": self.backend,
            "graph_hash": self.graph_hash,
            "profile_hash": self.profile_hash,
            "values": tuple(value.as_dict() for value in self.values),
            "ops": tuple(op.as_dict() for op in self.ops),
            "regions": tuple(region.as_dict() for region in self.regions),
            "state": self.state.as_dict(),
            "shape_constraints": tuple(
                constraint.as_dict() for constraint in self.shape_constraints
            ),
            "boundary_schema": tuple(item.as_dict() for item in self.boundary_schema),
            "input_value_ids": self.input_value_ids,
            "output_value_ids": self.output_value_ids,
        }

    @classmethod
    def from_trace_graph(
        cls,
        graph: "SplitTraceGraph",
        *,
        plan: "SplitPlan | None" = None,
        profile_hash: str | None = None,
    ) -> "SplitGraphIR":
        """Normalize an existing backend capture projection into Split IR.

        The current capture layer already records the backend-specific handle
        on each :class:`SplitTraceNode`; this method gives it a stable value and
        operation identity without serializing that handle.
        """

        values: list[ValueIR] = []
        ops: list[OpIR] = []
        regions: list[RegionIR] = []
        parameter_ids: list[str] = []
        buffer_ids: list[str] = []
        aliases: dict[str, list[str]] = {}
        constraints: list[ShapeConstraint] = []
        boundary_schema: list[BoundarySchema] = []

        for node in graph.nodes:
            value_id = _value_id(node)
            value_kind = _value_kind(node)
            source_id = _state_source_id(node)
            alias_group = _alias_group(node)
            values.append(
                ValueIR(
                    value_id=value_id,
                    kind=value_kind,
                    shape=_shape_tuple(node.symbolic_output_shape or node.output_shape),
                    dtype=node.dtype,
                    requires_grad=node.requires_grad,
                    alias_group=alias_group,
                    source_id=source_id,
                    module_path=node.module_path,
                    provenance=node.raw_label or node.label,
                    backend_handle=BackendHandle(
                        backend=graph.backend,
                        kind=type(node.target).__name__ if node.target is not None else "source",
                        location=node.raw_label or node.label,
                        payload=node.target,
                    ),
                )
            )
            if alias_group is not None:
                aliases.setdefault(alias_group, []).append(value_id)
            if node.is_param_source:
                parameter_ids.append(value_id)
            if node.is_buffer:
                buffer_ids.append(value_id)

        node_by_id = graph.node_by_id
        verification_rank = {
            SplitVerificationStatus.EXACT: 0,
            SplitVerificationStatus.REGION_VERIFIED: 1,
            SplitVerificationStatus.UNVERIFIED: 2,
            SplitVerificationStatus.ENVIRONMENT_UNAVAILABLE: 3,
            SplitVerificationStatus.UNSUPPORTED: 4,
        }
        for call in graph.replay_calls:
            members = tuple(node_by_id[node_id] for node_id in call.output_node_ids)
            node = next((member for member in members if member.target is not None), members[0])
            verification = max(
                (_verification_for_node(member) for member in members),
                key=verification_rank.__getitem__,
            )
            input_value_ids = tuple(
                dict.fromkeys(
                    _value_id_for_parent(graph, parent)
                    for member in members
                    for parent in member.parents
                )
            )
            output_value_ids = tuple(_value_id(member) for member in members)
            handle = BackendHandle(
                backend=graph.backend,
                kind=type(node.target).__name__ if node.target is not None else "missing",
                location=node.raw_label or node.label,
                payload=node.target,
            )
            if _is_region_node(node):
                regions.append(
                    RegionIR(
                        node_id=call.call_id,
                        region_kind=str(getattr(node.target, "primitive", node.op_type)),
                        input_value_ids=input_value_ids,
                        output_value_ids=output_value_ids,
                        module_path=node.module_path,
                        replayable=verification != SplitVerificationStatus.UNSUPPORTED,
                        trainable=verification != SplitVerificationStatus.UNSUPPORTED,
                        dynamic_shape=any(
                            member.symbolic_output_shape is not None for member in members
                        ),
                        verification=verification,
                        source=node.raw_label or node.label,
                        metadata=dict(getattr(node.target, "region_metadata", {}) or {}),
                        backend_handle=handle,
                    )
                )
            else:
                ops.append(
                    OpIR(
                        node_id=call.call_id,
                        op_type=node.op_type,
                        input_value_ids=input_value_ids,
                        output_value_ids=output_value_ids,
                        module_path=node.module_path,
                        backend_location=node.raw_label or node.label,
                        replayable=verification != SplitVerificationStatus.UNSUPPORTED,
                        trainable=verification != SplitVerificationStatus.UNSUPPORTED,
                        dynamic_shape=any(
                            member.symbolic_output_shape is not None for member in members
                        ),
                        verification=verification,
                        source=node.raw_label or node.label,
                        backend_handle=handle,
                    )
                )
        if graph.shape_program is None:
            for node in graph.compute_nodes:
                if node.symbolic_output_shape is None:
                    continue
                constraints.append(
                    ShapeConstraint(
                        constraint_id=f"shape:{node.canonical_id}",
                        kind=_shape_constraint_kind(node),
                        value_ids=(_value_id(node),),
                        expression="*".join(str(dim) for dim in node.symbolic_output_shape),
                        description=f"symbolic shape for {node.label}",
                    )
                )

        if graph.shape_program is not None:
            constraints.extend(
                ShapeConstraint(
                    constraint_id=constraint.constraint_id,
                    kind="batch_axis" if constraint.kind == "axis" else constraint.kind,
                    value_ids=constraint.value_ids,
                    expression=None,
                    description=constraint.description,
                    lhs=None if constraint.lhs is None else constraint.lhs.as_dict(),
                    rhs=None if constraint.rhs is None else constraint.rhs.as_dict(),
                )
                for constraint in graph.shape_program.constraints
            )

        if plan is not None:
            for key, spec in plan.boundary_spec.items():
                boundary_schema.append(_boundary_schema_from_spec(key, spec))

        return cls(
            backend=graph.backend,
            graph_hash=graph.graph_shape_hash,
            profile_hash=profile_hash,
            values=tuple(values),
            ops=tuple(ops),
            regions=tuple(regions),
            state=StateIR(
                parameter_value_ids=tuple(parameter_ids),
                buffer_value_ids=tuple(buffer_ids),
                alias_groups={key: tuple(items) for key, items in aliases.items()},
            ),
            shape_constraints=tuple(constraints),
            boundary_schema=tuple(boundary_schema),
            input_value_ids=tuple(
                _value_id_for_node_id(node_id) for node_id in graph.input_node_ids
            ),
            output_value_ids=tuple(
                _value_id_for_node_id(node_id) for node_id in graph.output_node_ids
            ),
        )


@dataclass(frozen=True)
class SplitFeatures:
    """Requested split features independent of a backend implementation."""

    replay: bool = True
    dynamic_batch: tuple[int, int] | None = None
    training: bool = False
    boundary_cache: bool = False
    batch_axes: Mapping[str, int] = field(default_factory=dict)
    cross_device: bool = False
    live_param_sources: bool | None = None

    def __post_init__(self) -> None:
        """Validate feature ranges."""

        if self.dynamic_batch is not None:
            low, high = self.dynamic_batch
            if low <= 0 or high < low:
                raise ValueError("dynamic_batch must be a positive inclusive range")
        for path, axis in self.batch_axes.items():
            if not isinstance(path, str) or not (
                path.startswith("/args/") or path.startswith("/kwargs/")
            ):
                raise ValueError("batch_axes keys must be JSON Pointers rooted at /args or /kwargs")
            if not isinstance(axis, int):
                raise TypeError("batch_axes values must be integer axis indexes")


@dataclass(frozen=True)
class SplitPoint:
    """Typed split point replacing stringly-typed boundary specifications."""

    kind: Literal["after", "before", "percent"]
    target: str | float

    def __post_init__(self) -> None:
        """Validate the typed point before it reaches the planner."""

        if self.kind not in {"after", "before", "percent"}:
            raise ValueError("SplitPoint.kind must be 'after', 'before', or 'percent'")
        if self.kind == "percent":
            try:
                value = float(self.target)
            except (TypeError, ValueError) as exc:
                raise ValueError("percent split target must be numeric") from exc
            if not 0 < value < 100:
                raise ValueError("percent split target must be strictly between 0 and 100")
        elif not isinstance(self.target, str) or not self.target.strip():
            raise ValueError("named split point target must be a non-empty string")

    def as_boundary(self) -> str:
        """Return the internal planner spelling for this point."""

        if self.kind == "percent":
            return f"percent:{self.target}"
        return f"{self.kind}:{self.target}"


def after(target: str) -> SplitPoint:
    """Create an ``after`` split point."""

    return SplitPoint("after", target)


def before(target: str) -> SplitPoint:
    """Create a ``before`` split point."""

    return SplitPoint("before", target)


def percent(value: float) -> SplitPoint:
    """Create a percentage split point."""

    return SplitPoint("percent", value)


@dataclass(frozen=True)
class SplitRequest:
    """Public v2 request for preparing a split runtime."""

    point: SplitPoint
    backend: str | None = None
    model_profile: str | "SplitModelProfile" | None = None
    features: SplitFeatures = field(default_factory=SplitFeatures)
    validation: Literal["strict", "permissive"] = "strict"
    device_policy: Literal["runtime"] = "runtime"
    batch_symbol: str = "B"

    def __post_init__(self) -> None:
        """Validate request-level policy."""

        if self.validation not in {"strict", "permissive"}:
            raise ValueError("validation must be 'strict' or 'permissive'")
        if self.device_policy != "runtime":
            raise ValueError("device_policy must be 'runtime'")
        if not self.batch_symbol:
            raise ValueError("batch_symbol must be non-empty")

    @property
    def boundary(self) -> str:
        """Return the planner spelling of the typed split point."""

        return self.point.as_boundary()

    @property
    def dynamic_batch(self) -> tuple[int, int] | None:
        """Return the requested dynamic batch range."""

        return self.features.dynamic_batch

    @property
    def trainable(self) -> bool:
        """Return whether split training was requested."""

        return self.features.training

    @property
    def use_live_param_sources(self) -> bool | None:
        """Return the optional live parameter-source policy."""

        return self.features.live_param_sources


SplitModelLoader = Callable[..., Any]


@dataclass(frozen=True)
class SplitModelProfile:
    """Pinned metadata describing a real-model test/load profile."""

    id: str
    backend: str
    family: str
    loader: SplitModelLoader | None = field(default=None, repr=False, compare=False)
    input_factory: SplitModelLoader | None = field(default=None, repr=False, compare=False)
    checkpoint_source: str | None = None
    checkpoint_revision: str | None = None
    checkpoint_sha256: str | None = None
    expected_capabilities: Mapping[str, str] = field(default_factory=dict)
    test_tier: Literal["pr", "nightly"] = "pr"

    def __post_init__(self) -> None:
        """Validate pinned checkpoint metadata when a source is declared."""

        if not self.id.strip() or not self.backend.strip() or not self.family.strip():
            raise ValueError("model profile id, backend, and family must be non-empty")
        if self.test_tier not in {"pr", "nightly"}:
            raise ValueError("model profile test_tier must be 'pr' or 'nightly'")
        if self.checkpoint_source is not None and (
            not self.checkpoint_revision or not self.checkpoint_sha256
        ):
            raise ValueError(
                "checkpoint_source requires a fixed checkpoint_revision and checkpoint_sha256"
            )
        if self.checkpoint_sha256 is not None:
            digest = self.checkpoint_sha256.lower()
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError("checkpoint_sha256 must be a 64-character hexadecimal digest")

    @property
    def profile_hash(self) -> str:
        """Return a stable hash of public profile metadata."""

        payload = "|".join(
            (
                self.id,
                self.backend,
                self.family,
                self.checkpoint_source or "",
                self.checkpoint_revision or "",
                self.checkpoint_sha256 or "",
            )
        )
        return sha256(payload.encode("utf-8")).hexdigest()


ModelProfile = SplitModelProfile


def _value_id(node: "SplitTraceNode") -> str:
    """Return a stable value ID for a trace node."""

    return _value_id_for_node_id(node.canonical_id)


def _value_id_for_node_id(node_id: str) -> str:
    """Return a stable value ID for a node ID."""

    return f"value:{node_id}"


def _value_id_for_parent(graph: "SplitTraceGraph", parent: str) -> str:
    """Resolve a parent alias to a stable value ID."""

    node = graph.node_for_label(parent)
    return _value_id(node) if node is not None else f"value:{parent}"


def _value_kind(
    node: "SplitTraceNode",
) -> Literal["input", "output", "parameter", "buffer", "constant", "boundary", "intermediate"]:
    """Classify a trace node value."""

    if node.is_input:
        return "input"
    if node.is_output:
        return "output"
    if node.is_buffer:
        return "buffer"
    if node.is_param_source:
        return "parameter"
    if node.target is None:
        return "constant"
    return "intermediate"


def _state_source_id(node: "SplitTraceNode") -> str | None:
    """Build a stable state source ID without using object identity."""

    if not (node.is_param_source or node.is_buffer or node.param_refs):
        return None
    path = node.module_path or node.label
    return f"state:{sha256(path.encode('utf-8')).hexdigest()[:16]}"


def _alias_group(node: "SplitTraceNode") -> str | None:
    """Return an alias group based on stable source metadata."""

    source = _state_source_id(node)
    return source


def _is_region_node(node: "SplitTraceNode") -> bool:
    """Return whether a node is represented by an opaque backend region."""

    return type(node.target).__name__ == "JaxRegionCapture"


def _verification_for_node(node: "SplitTraceNode") -> SplitVerificationStatus:
    """Assign a conservative verification level to a trace node."""

    if node.target is None and not (node.is_input or node.is_output or node.is_buffer):
        return SplitVerificationStatus.UNSUPPORTED
    if _is_region_node(node):
        return SplitVerificationStatus.REGION_VERIFIED
    return SplitVerificationStatus.EXACT


def _shape_constraint_kind(
    node: "SplitTraceNode",
) -> Literal[
    "batch_axis",
    "broadcast",
    "reshape",
    "flatten",
    "concat",
    "attention",
    "index",
    "shape_producing",
    "opaque",
]:
    """Infer a conservative shape-constraint category from an op name."""

    text = node.op_type.lower()
    if "reshape" in text or "view" in text:
        return "reshape"
    if "flatten" in text:
        return "flatten"
    if "concat" in text or "cat" in text:
        return "concat"
    if "broadcast" in text or "expand" in text:
        return "broadcast"
    if "attn" in text or "attention" in text:
        return "attention"
    if "index" in text or "gather" in text:
        return "index"
    if "shape" in text or "size" in text:
        return "shape_producing"
    return "batch_axis"


def _boundary_schema_from_spec(key: str, spec: BoundarySchema) -> BoundarySchema:
    """Convert the existing planner boundary spec to the v2 ABI."""

    del key
    return spec


def _shape_tuple(shape: Any) -> tuple[Any, ...] | None:
    """Convert concrete or symbolic shape objects to portable tuples."""

    if shape is None:
        return None
    as_tuple = getattr(shape, "as_tuple", None)
    if callable(as_tuple):
        shape = as_tuple()
    try:
        return tuple(shape)
    except TypeError:
        return None


__all__ = [
    "BackendHandle",
    "BoundaryRole",
    "BoundarySchema",
    "ModelProfile",
    "OpIR",
    "RegionIR",
    "ShapeConstraint",
    "SplitFeatures",
    "SplitGraphIR",
    "SplitModelProfile",
    "SplitPoint",
    "SplitRequest",
    "SplitVerificationStatus",
    "StateIR",
    "ValueIR",
    "after",
    "before",
    "percent",
]
