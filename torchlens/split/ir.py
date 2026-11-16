"""Backend-neutral intermediate representation for split programs.

The objects in this module deliberately contain only metadata that can be
shared by split backends.  A backend may keep an opaque :class:`BackendHandle`
next to an IR node while lowering, but that handle is never included in the
portable representation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Literal

from torchlens.errors import ArgumentTypeError
from torchlens.split.errors import SplitRequestError

from .placement import DevicePlacement, PlacementPlan

if TYPE_CHECKING:
    from .graph import SplitTraceGraph, SplitTraceNode
    from .planner import SplitPlan


class SplitVerificationStatus(str, Enum):
    """Verification level for a lowered split component.

    Examples
    --------
    >>> tl.SplitVerificationStatus.EXACT.value
    'exact'
    """

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

    Examples
    --------
    >>> handle = tl.split.BackendHandle('torch', 'callable', payload=torch.relu)
    >>> handle.as_dict()
    {'backend': 'torch', 'kind': 'callable', 'location': None}
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
    """One symbolic shape relation used during replay.

    Examples
    --------
    >>> constraint = tl.split.ShapeConstraint('batch', 'batch_axis', ('v:1',), (0,))
    >>> constraint.as_dict()['kind']
    'batch_axis'
    """

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
    """A value flowing through a backend-neutral split graph.

    Examples
    --------
    >>> value = tl.split.ValueIR('v:1', 'input', shape=(1, 3), dtype='torch.float32')
    >>> value.value_id, value.kind
    ('v:1', 'input')
    """

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
    """A normal replayable operation and its value dependencies.

    Examples
    --------
    >>> graph = tl.split.prepare_graph(torch.nn.Linear(3, 2).eval(), torch.ones(1, 3))
    >>> operation = graph.graph_ir.ops[0]
    >>> operation.replayable, bool(operation.output_value_ids)
    (True, True)
    """

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
    """An opaque or structured backend region such as JAX control flow.

    Examples
    --------
    >>> region = tl.split.RegionIR('region:1', 'native', ('v:1',), ('v:2',),
    ...     None, True, False, False, tl.SplitVerificationStatus.EXACT)
    >>> region.region_kind, region.output_value_ids
    ('native', ('v:2',))
    """

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
    """Parameter, buffer, RNG, mutable-state, and alias declarations.

    Examples
    --------
    >>> state = tl.split.StateIR(parameter_value_ids=('parameter:weight',))
    >>> state.parameter_value_ids, state.buffer_value_ids
    (('parameter:weight',), ())
    """

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
    "passthrough",
    "multi_scale_feature",
    "index",
    "shape_value",
    "activation",
]


@dataclass(frozen=True)
class BoundarySchema:
    """Stable ABI metadata for one split boundary value.

    Examples
    --------
    >>> schema = tl.BoundarySchema('v:1', (), 'activation', (2, 3), 'torch.float32')
    >>> schema.value_id, schema.shape
    ('v:1', (2, 3))
    """

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
    """Normalized graph shared by planners, capability analysis, and lowerers.

    Examples
    --------
    >>> graph = tl.split.prepare_graph(torch.nn.Linear(3, 2).eval(), torch.ones(1, 3))
    >>> graph.graph_ir.backend, bool(graph.graph_ir.ops)
    ('torch', True)
    """

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
        graph: SplitTraceGraph,
        *,
        plan: SplitPlan | None = None,
        profile_hash: str | None = None,
    ) -> SplitGraphIR:
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
            alias_group = _state_source_id(node)
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
            boundary_schema.extend(plan.boundary_spec.values())

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
    """Requested split features independent of a backend implementation.

    ``batch_axes`` declares batch-axis SEMANTICS (which axis of which input
    leaf is the batch), never a set of allowed batch values. ``None`` selects
    conservative auto-inference; ``{}`` explicitly declares no batch axes.
    Rank-zero and rank-one inputs are not auto-inferred as batched. A runtime batch
    is accepted after an empirical B=2 replay probe, subject to shape guards.
    Untested batches are extrapolations, not universally verified executions.

    ``retain_trace`` controls diagnostic capture retention, not replay values.
    ``None`` uses a compact Torch inference runtime and retains complete traces
    for training and other backends. ``True`` retains the full diagnostic Trace.
    Explicit ``False`` supports Torch inference and training. Compact training
    retains live state and replay recipes, but never an ordinary activation
    archive or a reusable captured autograd graph.

    Examples
    --------
    >>> features = tl.SplitFeatures(training=True, retain_trace=False)
    >>> features.training, features.retain_trace
    (True, False)
    """

    replay: bool = True
    training: bool = False
    boundary_cache: bool = False
    batch_axes: Mapping[str, int] | None = None
    cross_device: bool = False
    live_param_sources: bool | None = None
    retain_trace: bool | None = None

    def __post_init__(self) -> None:
        """Validate declared batch-axis semantics."""

        if self.retain_trace is not None and not isinstance(self.retain_trace, bool):
            raise ArgumentTypeError(
                "retain_trace must be a bool or None",
                code="split_argument_type",
                remedy="set retain_trace to True, False, or None",
            )
        for path, axis in (self.batch_axes or {}).items():
            if not isinstance(path, str) or not (
                path.startswith("/args/") or path.startswith("/kwargs/")
            ):
                raise SplitRequestError(
                    "batch_axes keys must be JSON Pointers rooted at /args or /kwargs",
                    code=SplitRequestError.code,
                )
            if not isinstance(axis, int):
                raise ArgumentTypeError(
                    "batch_axes values must be integer axis indexes",
                    code="split_argument_type",
                    remedy="use integer axis indexes as batch_axes values",
                )

    def as_dict(self) -> dict[str, Any]:
        """Serialize requested feature flags for capability diagnostics."""

        return {
            "replay": self.replay,
            "training": self.training,
            "boundary_cache": self.boundary_cache,
            "batch_axes": None if self.batch_axes is None else dict(self.batch_axes),
            "cross_device": self.cross_device,
            "live_param_sources": self.live_param_sources,
            "retain_trace": self.retain_trace,
        }


@dataclass(frozen=True)
class GraphPartition:
    """An explicit set of canonical compute-node IDs assigned to the prefix.

    Inputs are implicitly on the prefix and output reconstruction on the suffix.
    The planner checks dependency closure and keeps every native call atomic.
    IDs refer to ``SplitTraceNode.canonical_id``, including pass qualification;
    display labels, module aliases and partial native calls are not accepted.

    Examples
    --------
    >>> partition = tl.split.GraphPartition(frozenset({'v:2', 'v:5'}))
    >>> sorted(partition.prefix_node_ids)
    ['v:2', 'v:5']
    """

    prefix_node_ids: frozenset[str]

    def __post_init__(self) -> None:
        """Freeze the selection and reject invalid ID representations."""

        if isinstance(self.prefix_node_ids, (str, bytes)):
            raise ArgumentTypeError(
                "prefix_node_ids must be a collection of canonical node IDs",
                code="split_argument_type",
                remedy="use a collection of non-empty canonical node ID strings",
            )
        ids = frozenset(self.prefix_node_ids)
        if any(not isinstance(node_id, str) or not node_id for node_id in ids):
            raise ArgumentTypeError(
                "prefix_node_ids must contain non-empty canonical node ID strings",
                code="split_argument_type",
                remedy="use a collection of non-empty canonical node ID strings",
            )
        object.__setattr__(self, "prefix_node_ids", ids)

    def as_boundary(self) -> str:
        """Return a stable identity independent of iteration order or a target node."""

        digest = sha256(repr(tuple(sorted(self.prefix_node_ids))).encode("utf-8")).hexdigest()
        return f"partition:{digest}"


@dataclass(frozen=True)
class SplitRequest:
    """Typed request for preparing a Split Runtime.

    Examples
    --------
    >>> request = tl.SplitRequest(tl.split.GraphPartition(frozenset()))
    >>> request.partition.prefix_node_ids
    frozenset()
    """

    partition: GraphPartition
    backend: str | None = None
    model_profile: str | SplitModelProfile | None = None
    features: SplitFeatures = field(default_factory=SplitFeatures)
    validation: Literal["strict", "permissive"] = "strict"
    device_policy: Literal["runtime"] = "runtime"
    batch_symbol: str = "B"
    placement: PlacementPlan = field(default_factory=PlacementPlan)

    def __post_init__(self) -> None:
        """Validate request-level policy."""

        if not isinstance(self.partition, GraphPartition):
            raise ArgumentTypeError(
                "partition must be a GraphPartition",
                code="split_argument_type",
                remedy="pass a GraphPartition created by dag.to_partition",
            )
        if self.validation not in {"strict", "permissive"}:
            raise SplitRequestError(
                "validation must be 'strict' or 'permissive'",
                code=SplitRequestError.code,
            )
        if self.device_policy != "runtime":
            raise SplitRequestError(
                "device_policy must be 'runtime'",
                code=SplitRequestError.code,
            )
        if not self.batch_symbol:
            raise SplitRequestError(
                "batch_symbol must be non-empty",
                code=SplitRequestError.code,
            )
        if not isinstance(self.placement, PlacementPlan):
            raise ArgumentTypeError(
                "placement must be a PlacementPlan",
                code="split_argument_type",
                remedy="pass a PlacementPlan instance as placement",
            )

    @property
    def boundary(self) -> str:
        """Return the planner spelling of the canonical partition."""

        return self.partition.as_boundary()

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
    """Pinned metadata describing a real-model test/load profile.

    Examples
    --------
    >>> profile = tl.SplitModelProfile('doc_example', 'torch', 'mlp')
    >>> profile.id, profile.backend
    ('doc_example', 'torch')
    """

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
            raise SplitRequestError(
                "model profile id, backend, and family must be non-empty",
                code=SplitRequestError.code,
            )
        if self.test_tier not in {"pr", "nightly"}:
            raise SplitRequestError(
                "model profile test_tier must be 'pr' or 'nightly'",
                code=SplitRequestError.code,
            )
        if self.checkpoint_source is not None and (
            not self.checkpoint_revision or not self.checkpoint_sha256
        ):
            raise SplitRequestError(
                "checkpoint_source requires a fixed checkpoint_revision and checkpoint_sha256",
                code=SplitRequestError.code,
            )
        if self.checkpoint_sha256 is not None:
            digest = self.checkpoint_sha256.lower()
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise SplitRequestError(
                    "checkpoint_sha256 must be a 64-character hexadecimal digest",
                    code=SplitRequestError.code,
                )

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


def _value_id(node: SplitTraceNode) -> str:
    """Return a stable value ID for a trace node."""

    return _value_id_for_node_id(node.canonical_id)


def _value_id_for_node_id(node_id: str) -> str:
    """Return a stable value ID for a node ID."""

    return f"value:{node_id}"


def _value_id_for_parent(graph: SplitTraceGraph, parent: str) -> str:
    """Resolve a parent alias to a stable value ID."""

    node = graph.node_for_label(parent)
    return _value_id(node) if node is not None else f"value:{parent}"


def _value_kind(
    node: SplitTraceNode,
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


def _state_source_id(node: SplitTraceNode) -> str | None:
    """Build a stable state source ID without using object identity."""

    if not (node.is_param_source or node.is_buffer or node.param_refs):
        return None
    path = node.module_path or node.label
    return f"state:{sha256(path.encode('utf-8')).hexdigest()[:16]}"


def _is_region_node(node: SplitTraceNode) -> bool:
    """Return whether a node is represented by an opaque backend region."""

    return type(node.target).__name__ == "JaxRegionCapture"


def _verification_for_node(node: SplitTraceNode) -> SplitVerificationStatus:
    """Assign a conservative verification level to a trace node."""

    if node.target is None and not (node.is_input or node.is_output or node.is_buffer):
        return SplitVerificationStatus.UNSUPPORTED
    if _is_region_node(node):
        return SplitVerificationStatus.REGION_VERIFIED
    return SplitVerificationStatus.EXACT


_OpShapeKind = Literal[
    "batch_axis",
    "broadcast",
    "reshape",
    "flatten",
    "concat",
    "attention",
    "index",
    "shape_producing",
    "opaque",
]


def _shape_constraint_kind(node: SplitTraceNode) -> _OpShapeKind:
    """Infer a conservative shape-constraint category from an op name."""
    text = node.op_type.lower()
    categories: tuple[tuple[tuple[str, ...], _OpShapeKind], ...] = (
        (("reshape", "view"), "reshape"),
        (("flatten",), "flatten"),
        (("concat", "cat"), "concat"),
        (("broadcast", "expand"), "broadcast"),
        (("attn", "attention"), "attention"),
        (("index", "gather"), "index"),
        (("shape", "size"), "shape_producing"),
    )
    for markers, kind in categories:
        if any(marker in text for marker in markers):
            return kind
    return "batch_axis"


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


__all__ = (
    "BackendHandle",
    "BoundaryRole",
    "BoundarySchema",
    "DevicePlacement",
    "GraphPartition",
    "OpIR",
    "PlacementPlan",
    "RegionIR",
    "ShapeConstraint",
    "SplitFeatures",
    "SplitGraphIR",
    "SplitModelProfile",
    "SplitRequest",
    "SplitVerificationStatus",
    "StateIR",
    "ValueIR",
)
