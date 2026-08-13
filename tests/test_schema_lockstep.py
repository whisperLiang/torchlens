"""Field-catalog lockstep ENFORCEMENT (grind r2 row 20 / matrix R11).

The recurring incident class this module exists to end: a field catalog, its
owning record class, the generated artifacts derived from it, and the version
authorities that gate its persistence drift apart between efforts. Round 81
found ~52 such drifts by hand. Hand passes do not scale and do not stay found,
so this module is the MECHANISM instead:

1. **Closure.** ``test_every_field_catalog_is_registered`` derives the catalog
   universe from :mod:`torchlens.constants` at runtime and demands every member
   be registered here. A new ``*_FIELD_ORDER`` cannot be added without a
   lockstep entry, so the mechanism cannot be bypassed by omission -- the exact
   hole that left ``FUNC_CALL_LOCATION_FIELD_ORDER`` outside
   ``tests/test_record_field_policy.py::RECORD_CASES``.
2. **Generated == declared.** Every checked-in generated artifact is
   regenerated in-process and diffed, and the artifact universe is itself
   derived from the tree (by generated-file header marker), so a new generated
   file also cannot skip registration.
3. **Runtime == declared.** Every attribute a live captured record actually
   carries must be declared in that record's ``FIELD_POLICY``. This derivation
   is SOURCE-FREE (no ``inspect.getsource``), so unlike the older
   ``test_internals.py::TestFieldOrderSync`` checks it cannot go red for
   environment reasons (stale bytecode, zipped/installed source, missing
   ``.py`` files) while the declarations are actually fine.
4. **Red-capability.** Each checker is a plain function over its two sides, and
   ``TestMechanismIsRedCapable`` plants drift into each one and proves the
   checker reports it. A lockstep gate nobody has proved can fail is not a
   gate.

Everything here is smoke-tier: the whole module is declaration arithmetic plus
one small capture.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import constants
from torchlens._io import FieldPolicy
from torchlens.data_classes.backward_pass import BackwardPass
from torchlens.data_classes.buffer import Buffer
from torchlens.data_classes.field_policy import RecordFieldPolicy, field_order_from_policy
from torchlens.data_classes.func_call_location import FuncCallLocation
from torchlens.data_classes.grad_fn import GradFn
from torchlens.data_classes.grad_fn_call import GradFnCall
from torchlens.data_classes.layer import Layer
from torchlens.data_classes.module import Module, ModuleCall
from torchlens.data_classes.op import Op
from torchlens.data_classes.param import Param
from torchlens.data_classes.trace import Trace

pytestmark = pytest.mark.smoke

_REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Authority 1: field catalogs (``*_FIELD_ORDER``) vs their owning FIELD_POLICY.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Catalog:
    """One registered field catalog and the authority it must track.

    Parameters
    ----------
    constant:
        Name of the catalog list in :mod:`torchlens.constants`.
    owner:
        Record class whose ``FIELD_POLICY`` generates the catalog, or ``None``
        for a pure alias.
    alias_of:
        Name of the catalog this one is a compatibility alias for, or ``None``
        for a primary catalog. An alias must be the SAME list object, not a
        copy, so the two spellings cannot diverge.
    """

    constant: str
    owner: type[Any] | None = None
    alias_of: str | None = None


#: Every field catalog in ``torchlens.constants``, with its authority.
#: ``test_every_field_catalog_is_registered`` keeps this exhaustive.
CATALOGS: tuple[Catalog, ...] = (
    Catalog("MODEL_LOG_FIELD_ORDER", owner=Trace),
    Catalog("LAYER_PASS_LOG_FIELD_ORDER", owner=Op),
    Catalog("LAYER_LOG_FIELD_ORDER", owner=Layer),
    Catalog("PARAM_LOG_FIELD_ORDER", owner=Param),
    Catalog("BUFFER_LOG_FIELD_ORDER", owner=Buffer),
    Catalog("GRAD_FN_LOG_FIELD_ORDER", owner=GradFn),
    Catalog("GRAD_FN_PASS_LOG_FIELD_ORDER", owner=GradFnCall),
    Catalog("MODULE_PASS_LOG_FIELD_ORDER", owner=ModuleCall),
    Catalog("MODULE_LOG_FIELD_ORDER", owner=Module),
    Catalog("BACKWARD_PASS_FIELD_ORDER", owner=BackwardPass),
    Catalog("FUNC_CALL_LOCATION_FIELD_ORDER", owner=FuncCallLocation),
    # Historical spellings retained for callers; same object by construction.
    Catalog("OP_LOG_FIELD_ORDER", alias_of="LAYER_PASS_LOG_FIELD_ORDER"),
    Catalog("TENSOR_LOG_FIELD_ORDER", alias_of="LAYER_PASS_LOG_FIELD_ORDER"),
)

_CATALOG_SUFFIX = "FIELD_ORDER"


def declared_catalog_names(module: Any = constants) -> set[str]:
    """Return the field-catalog universe derived from a constants module.

    Parameters
    ----------
    module:
        Module to scan (injectable so the closure checker can be drift-planted).

    Returns
    -------
    set[str]
        Names of every public list constant whose name ends in ``FIELD_ORDER``.
    """

    return {
        name
        for name, value in vars(module).items()
        if name.endswith(_CATALOG_SUFFIX) and isinstance(value, list)
    }


def catalog_registration_gaps(
    declared: set[str],
    registered: set[str],
) -> tuple[set[str], set[str]]:
    """Return catalogs missing from, and phantom in, the lockstep registry.

    Parameters
    ----------
    declared:
        Catalog names derived from the code.
    registered:
        Catalog names carried by :data:`CATALOGS`.

    Returns
    -------
    tuple[set[str], set[str]]
        ``(unregistered, phantom)``.
    """

    return declared - registered, registered - declared


def test_every_field_catalog_is_registered() -> None:
    """Every ``*_FIELD_ORDER`` in constants has a lockstep registry entry."""

    unregistered, phantom = catalog_registration_gaps(
        declared_catalog_names(), {catalog.constant for catalog in CATALOGS}
    )
    assert not unregistered, (
        "field catalogs with no lockstep entry (add them to CATALOGS in "
        f"tests/test_schema_lockstep.py): {sorted(unregistered)}"
    )
    assert not phantom, f"registered catalogs that no longer exist: {sorted(phantom)}"


def policy_catalog_diff(
    policy: dict[str, RecordFieldPolicy],
    catalog: list[str],
) -> tuple[list[str], list[str]]:
    """Return the generated-vs-declared difference for one catalog.

    Parameters
    ----------
    policy:
        Owning record's ``FIELD_POLICY`` table.
    catalog:
        Checked-in ordered field list.

    Returns
    -------
    tuple[list[str], list[str]]
        ``(generated, declared)`` -- equal when the catalog is in lockstep.
    """

    return field_order_from_policy(policy), list(catalog)


_PRIMARY_CATALOGS = tuple(c for c in CATALOGS if c.owner is not None)
_ALIAS_CATALOGS = tuple(c for c in CATALOGS if c.alias_of is not None)


@pytest.mark.parametrize("catalog", _PRIMARY_CATALOGS, ids=lambda c: c.constant)
def test_catalog_is_generated_from_its_owning_field_policy(catalog: Catalog) -> None:
    """The checked-in catalog equals the view generated from ``FIELD_POLICY``."""

    assert catalog.owner is not None
    generated, declared = policy_catalog_diff(
        catalog.owner.FIELD_POLICY, getattr(constants, catalog.constant)
    )
    assert generated == declared, (
        f"{catalog.constant} drifted from {catalog.owner.__name__}.FIELD_POLICY; "
        f"only in policy: {sorted(set(generated) - set(declared))}; "
        f"only in catalog: {sorted(set(declared) - set(generated))}"
    )
    assert len(declared) == len(set(declared)), f"{catalog.constant} has duplicates"


@pytest.mark.parametrize("catalog", _ALIAS_CATALOGS, ids=lambda c: c.constant)
def test_alias_catalogs_share_the_primary_object(catalog: Catalog) -> None:
    """An alias catalog IS its primary, so the spellings cannot diverge."""

    assert catalog.alias_of is not None
    assert getattr(constants, catalog.constant) is getattr(constants, catalog.alias_of)


# ---------------------------------------------------------------------------
# Authority 2: private-named ordered fields (the OL#47 disclosure class).
# ---------------------------------------------------------------------------

#: Tier A -- private-named fields that sit in a PUBLIC ``FIELD_ORDER`` while
#: being ``FieldPolicy.DROP`` (ordered, yet deliberately non-portable and
#: session-only). This is the most confusing combination on the record surface
#: and the one the package docs claim to enumerate, so each entry states WHY.
#: OL#47 -- ``_fast_run_session`` landing here undocumented -- is the incident
#: this ledger closes as a class.
PRIVATE_ORDERED_DROP_FIELDS: dict[str, dict[str, str]] = {
    "Trace": {
        "_runnable": "sparse-runnable state container; rebuilt at load, never portable itself",
        "_fast_run_session": "session-time guarded-static-loop handle (tl.Trace.run(fast=True))",
        "_transform": "capture-time input transform callable; opaque, session-only",
        "_output_transform": "capture-time output transform callable; opaque, session-only",
        "_visualizer_dir": "per-session visualizer scratch directory path",
        "_out_dedup_mode": "session dedup strategy for retained activation payloads",
        "_out_identity_cache": "session identity cache backing out dedup",
        "_out_hash_cache": "session hash cache backing out dedup",
        "_code_context_cache": "session source-context cache; rebuilt from source on demand",
        "_source_model_ref": "weak reference to the captured model; never portable",
        "_intervention_spec": "live intervention spec object; persisted by its own saver",
        "_warned_direct_write": "once-per-trace warn sentinel; a fresh load must warn again",
        "_warned_mutate_in_place": "once-per-trace warn sentinel; a fresh load must warn again",
        "_last_hook_handle_ids": "session hook-handle bookkeeping for teardown",
    },
    "Op": {
        "_construction_done": "construction-phase latch read by the direct-write guard",
    },
}

#: Tier B -- every other private-named ordered field. These are ordinary
#: persisted internals (KEEP/BLOB policy), so they need a reviewed one-line
#: registration rather than prose. Pinning the set still makes a newly ordered
#: private field a deliberate diff instead of a silent surface change.
PRIVATE_ORDERED_PERSISTED_FIELDS: frozenset[str] = frozenset(
    {
        "Trace._tracing_finished",
        "Trace._capture_outcome",
        "Trace._layers_logged",
        "Trace._layers_saved",
        "Trace._replay_arg_version_data_complete",
        "Trace._grad_op_nums_to_save",
        "Trace._activation_transform_repr",
        "Trace._source_code_blob",
        "Trace._has_direct_writes",
        "Trace._spec_revision",
        "Trace._out_recipe_revision",
        "Trace._append_sequence_id",
        "Trace._layer_nums_to_save",
        "Trace._raw_to_final_layer_labels",
        "Trace._raw_to_final_parent_layer_labels",
        "Trace._raw_to_final_op_labels",
        "Trace._final_to_raw_layer_labels",
        "Trace._lookup_keys_to_layer_num_dict",
        "Trace._layer_num_to_lookup_keys_dict",
        "Trace._ambiguous_lookup_keys",
        "Trace._containers",
        "Trace._annotation_blobs",
        "Trace._buffer_persistence",
        "Trace._orphan_labels",
        "Trace._orphan_logs",
        "Trace._phase_timings",
        "Trace._grad_fn_param_refs",
        "Op._label_raw",
        "Op._layer_label_raw",
        "Op._tracing_finished",
        "Op._param_barcodes",
        "Op._param_logs",
        "Op._edge_uses",
        "Op._address_normalized",
        "Layer._param_barcodes",
        "Layer._param_logs",
        "Param._derived_grad_record_path",
        "GradFnCall._time_started",
        "GradFnCall._time_finished",
    }
)


def _private_ordered_fields_by_tier(
    catalogs: tuple[Catalog, ...],
) -> tuple[set[str], set[str]]:
    """Return live private-named ordered fields split by portable policy.

    Parameters
    ----------
    catalogs:
        Registered primary catalogs.

    Returns
    -------
    tuple[set[str], set[str]]
        ``(drop_fields, persisted_fields)`` as ``"Class._field"`` strings.
    """

    drop: set[str] = set()
    persisted: set[str] = set()
    for catalog in catalogs:
        if catalog.owner is None:
            continue
        policy = catalog.owner.FIELD_POLICY
        for name in getattr(constants, catalog.constant):
            if not name.startswith("_"):
                continue
            key = f"{catalog.owner.__name__}.{name}"
            if policy[name].portable_policy is FieldPolicy.DROP:
                drop.add(key)
            else:
                persisted.add(key)
    return drop, persisted


def private_ordered_field_gaps(
    catalogs: tuple[Catalog, ...],
    ledger: dict[str, dict[str, str]],
) -> tuple[set[str], set[str]]:
    """Return unledgered and phantom private-named ordered DROP fields.

    Parameters
    ----------
    catalogs:
        Registered primary catalogs.
    ledger:
        Declared tier-A ledger keyed by record class name.

    Returns
    -------
    tuple[set[str], set[str]]
        ``(unledgered, phantom)`` as ``"Class._field"`` strings.
    """

    live, _ = _private_ordered_fields_by_tier(catalogs)
    declared = {f"{cls_name}.{field}" for cls_name, fields in ledger.items() for field in fields}
    return live - declared, declared - live


def test_private_named_ordered_drop_fields_are_ledgered() -> None:
    """An ordered private DROP field must state why it is on the surface."""

    unledgered, phantom = private_ordered_field_gaps(_PRIMARY_CATALOGS, PRIVATE_ORDERED_DROP_FIELDS)
    assert not unledgered, (
        "private-named FieldPolicy.DROP fields in a public FIELD_ORDER with no ledger "
        f"entry (document them in PRIVATE_ORDERED_DROP_FIELDS): {sorted(unledgered)}"
    )
    assert not phantom, f"ledgered private ordered fields that no longer exist: {sorted(phantom)}"


def test_private_named_ordered_persisted_fields_are_registered() -> None:
    """Ordering a private persisted field stays a reviewed one-line diff."""

    _, live = _private_ordered_fields_by_tier(_PRIMARY_CATALOGS)
    assert live == PRIVATE_ORDERED_PERSISTED_FIELDS, (
        "private-named persisted ordered fields changed; only live: "
        f"{sorted(live - PRIVATE_ORDERED_PERSISTED_FIELDS)}; only registered: "
        f"{sorted(PRIVATE_ORDERED_PERSISTED_FIELDS - live)}"
    )


def test_private_ordered_field_reasons_are_nonempty() -> None:
    """Every tier-A ledger entry carries a real reason, not a placeholder."""

    for cls_name, fields in PRIVATE_ORDERED_DROP_FIELDS.items():
        for field, reason in fields.items():
            assert len(reason.strip()) >= 20, f"{cls_name}.{field} needs a real reason"


# ---------------------------------------------------------------------------
# Authority 3: generated artifacts vs their generators.
# ---------------------------------------------------------------------------


def _render_schema_bindings() -> str:
    """Return a fresh rendering of the storage-bindings module.

    Returns
    -------
    str
        Generated source text.
    """

    from tools.generate_record_schema import _collect, _render

    return _render(_collect())


def _render_op_record_manifest() -> str:
    """Return a fresh rendering of the op-record cell-source manifest.

    Returns
    -------
    str
        Generated source text.
    """

    from tools.generate_op_record_manifest import generate

    return generate()


@dataclass(frozen=True)
class GeneratedArtifact:
    """One checked-in generated module and its in-process renderer.

    Parameters
    ----------
    path:
        Repo-relative path of the generated module.
    render:
        Callable returning the freshly generated source text.
    regenerate_command:
        Command a developer runs to refresh the artifact.
    """

    path: str
    render: Callable[[], str]
    regenerate_command: str


#: Every generated module under ``torchlens/``. Kept exhaustive by
#: ``test_every_generated_module_is_registered``.
GENERATED_ARTIFACTS: tuple[GeneratedArtifact, ...] = (
    GeneratedArtifact(
        "torchlens/data_classes/_schema_bindings.py",
        _render_schema_bindings,
        "python tools/generate_record_schema.py",
    ),
    GeneratedArtifact(
        "torchlens/ir/op_record_manifest.py",
        _render_op_record_manifest,
        "python -m tools.generate_op_record_manifest",
    ),
)

#: Header marker every generated module carries on its first line.
_GENERATED_MARKER = "GENERATED"


def _module_paths() -> Iterator[Path]:
    """Yield every Python module in the shipped package.

    Yields
    ------
    Path
        Absolute path of one package module.
    """

    yield from sorted((_REPO_ROOT / "torchlens").rglob("*.py"))


def generated_module_paths(marker: str = _GENERATED_MARKER) -> set[str]:
    """Return repo-relative paths of modules declaring themselves generated.

    A generated module announces itself on its docstring's FIRST line; that
    keeps the scan from matching prose deeper in a hand-written file.

    Parameters
    ----------
    marker:
        Header token that marks a module as generated.

    Returns
    -------
    set[str]
        Repo-relative POSIX paths.
    """

    found: set[str] = set()
    for path in _module_paths():
        with path.open(encoding="utf-8") as handle:
            first_line = handle.readline()
        if marker in first_line:
            found.add(path.relative_to(_REPO_ROOT).as_posix())
    return found


def artifact_registration_gaps(
    found: set[str],
    registered: set[str],
) -> tuple[set[str], set[str]]:
    """Return generated modules missing from, and phantom in, the registry.

    Parameters
    ----------
    found:
        Generated-module paths discovered in the tree.
    registered:
        Paths carried by :data:`GENERATED_ARTIFACTS`.

    Returns
    -------
    tuple[set[str], set[str]]
        ``(unregistered, phantom)``.
    """

    return found - registered, registered - found


def test_every_generated_module_is_registered() -> None:
    """Every self-declared generated module has a regenerate-and-diff entry."""

    unregistered, phantom = artifact_registration_gaps(
        generated_module_paths(), {artifact.path for artifact in GENERATED_ARTIFACTS}
    )
    assert not unregistered, (
        "generated modules with no lockstep entry (register them in "
        f"GENERATED_ARTIFACTS): {sorted(unregistered)}"
    )
    assert not phantom, f"registered generated modules that no longer exist: {sorted(phantom)}"


@pytest.mark.parametrize("artifact", GENERATED_ARTIFACTS, ids=lambda a: a.path)
def test_generated_artifact_is_current(artifact: GeneratedArtifact) -> None:
    """The checked-in generated module matches a fresh in-process generation.

    Deliberately in-process rather than a subprocess: the same comparison at a
    fraction of the cost, which is what lets it live in the smoke tier where
    drift is caught the same minute it lands.
    """

    checked_in = (_REPO_ROOT / artifact.path).read_text(encoding="utf-8")
    assert checked_in == artifact.render(), (
        f"{artifact.path} is stale -- run: {artifact.regenerate_command}"
    )


# ---------------------------------------------------------------------------
# Authority 4: live record attributes vs their declared policy.
# ---------------------------------------------------------------------------

#: Row-facade plumbing every kind-table-backed record carries. These are not
#: fields: they are the two-word ``(core, row)`` handle the facade reads
#: through (see ``torchlens/_trace_core/record_rows.py``).
FACADE_PLUMBING_ATTRS = frozenset({"_tl_core", "_tl_row"})


class _LockstepModel(nn.Module):
    """Small model populating params, buffers, modules, and grad_fns."""

    def __init__(self) -> None:
        """Initialize the covered layer families."""

        super().__init__()
        self.linear = nn.Linear(3, 3)
        self.bn = nn.BatchNorm1d(3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a scalar so one backward pass covers the grad families.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Scalar output.
        """

        return torch.relu(self.bn(self.linear(x))).sum()


@pytest.fixture(scope="module")
def lockstep_trace() -> Iterator[Trace]:
    """Yield one populated trace shared by the runtime-declaration checks.

    Yields
    ------
    Trace
        Trace with every record family populated and one backward pass.
    """

    model = _LockstepModel().eval()
    trace = tl.trace(
        model,
        torch.randn(2, 3, requires_grad=True),
        capture=tl.options.CaptureOptions(save_grads="all"),
    )
    trace.log_backward(trace[trace.output_layers[0]].out)
    try:
        yield trace
    finally:
        trace.cleanup()


def _live_records(trace: Trace) -> dict[str, Any]:
    """Return one representative live instance per record family.

    Parameters
    ----------
    trace:
        Populated trace.

    Returns
    -------
    dict[str, Any]
        Record class name -> representative instance.
    """

    grad_fn = next(record for record in trace.grad_fns if record.calls)
    return {
        "Trace": trace,
        "Op": next(iter(trace.ops)),
        "Layer": next(iter(trace.layers)),
        "Param": next(iter(trace.params)),
        "Buffer": next(iter(trace.buffers)),
        "GradFn": grad_fn,
        "GradFnCall": next(iter(grad_fn.calls.values())),
        "ModuleCall": next(iter(trace.module_calls)),
        "Module": next(iter(trace.modules)),
        "BackwardPass": next(iter(trace.backward_passes)),
    }


def undeclared_runtime_attributes(
    instance_attrs: set[str],
    policy: dict[str, RecordFieldPolicy],
    allowed: frozenset[str] = FACADE_PLUMBING_ATTRS,
) -> set[str]:
    """Return attributes a live record carries but never declared.

    Parameters
    ----------
    instance_attrs:
        Attribute names present in the instance ``__dict__``.
    policy:
        The record class's declared ``FIELD_POLICY`` table.
    allowed:
        Facade plumbing that legitimately owns no declared field.

    Returns
    -------
    set[str]
        Undeclared attribute names.
    """

    return instance_attrs - set(policy) - allowed


_RECORD_NAMES = (
    "Trace",
    "Op",
    "Layer",
    "Param",
    "Buffer",
    "GradFn",
    "GradFnCall",
    "ModuleCall",
    "Module",
    "BackwardPass",
)


@pytest.mark.parametrize("record_name", _RECORD_NAMES)
def test_live_record_attributes_are_all_declared(lockstep_trace: Trace, record_name: str) -> None:
    """Every attribute a captured record carries is declared in FIELD_POLICY.

    The derivation is runtime-only -- no source reading -- so this check is
    immune to the environment failure modes (stale bytecode, source-less
    installs) that make the source-introspecting field-order tests
    environment-sensitive.
    """

    record = _live_records(lockstep_trace)[record_name]
    undeclared = undeclared_runtime_attributes(
        set(vars(record)) if hasattr(record, "__dict__") else set(),
        type(record).FIELD_POLICY,
    )
    assert not undeclared, (
        f"{record_name} carries undeclared attributes at runtime "
        f"(add them to FIELD_POLICY): {sorted(undeclared)}"
    )


def test_facade_plumbing_allowance_stays_minimal() -> None:
    """The undeclared-attribute allowance stays the two facade handles.

    A growing allowance is how a real drift gets excused, so the allowance
    itself is pinned.
    """

    assert FACADE_PLUMBING_ATTRS == {"_tl_core", "_tl_row"}


# ---------------------------------------------------------------------------
# Authority 5: persistence version authorities (SF-40 drift plant).
# ---------------------------------------------------------------------------

#: Reviewed pins for the persistence version authorities. These are three
#: INDEPENDENT counters (a merged root manifest is a different discriminated
#: object from a rank core's manifest -- see ``torchlens/merged/_enums.py``), so
#: this is deliberately NOT an equality assertion between them: it is a pin
#: apiece, with the co-change list a bump must honor.
#:
#: Bumping ``TLSPEC_VERSION``: update the pin, the manifest schema, and the
#: load-path tests. Bumping ``MIN_TLSPEC_VERSION`` (the rehydration floor):
#: also update the floor statement in ``CLAUDE.md`` and
#: ``tests/test_rehydration_floor.py``. Bumping ``MERGED_TLSPEC_VERSION``:
#: also update ``docs/reference/merged_trace_contract.md``, whose stated value
#: is checked against the code below.
VERSION_AUTHORITY_PINS: dict[str, int] = {
    "TLSPEC_VERSION": 7,
    "MIN_TLSPEC_VERSION": 6,
    "MERGED_TLSPEC_VERSION": 7,
}


def _version_authority_values() -> dict[str, int]:
    """Return the live value of each persistence version authority.

    Returns
    -------
    dict[str, int]
        Authority name -> value.
    """

    from torchlens._io import MIN_TLSPEC_VERSION, TLSPEC_VERSION
    from torchlens.merged._enums import MERGED_TLSPEC_VERSION

    return {
        "TLSPEC_VERSION": TLSPEC_VERSION,
        "MIN_TLSPEC_VERSION": MIN_TLSPEC_VERSION,
        "MERGED_TLSPEC_VERSION": MERGED_TLSPEC_VERSION,
    }


def version_pin_drift(live: dict[str, int], pins: dict[str, int]) -> dict[str, tuple[int, int]]:
    """Return authorities whose live value left its reviewed pin.

    Parameters
    ----------
    live:
        Live authority values.
    pins:
        Reviewed pinned values.

    Returns
    -------
    dict[str, tuple[int, int]]
        Authority -> ``(live, pinned)`` for every mismatch.
    """

    return {
        name: (value, pins[name])
        for name, value in live.items()
        if name in pins and value != pins[name]
    }


def test_version_authorities_match_their_reviewed_pins() -> None:
    """A persistence version bump is a reviewed diff, never a silent one."""

    live = _version_authority_values()
    assert set(live) == set(VERSION_AUTHORITY_PINS)
    drift = version_pin_drift(live, VERSION_AUTHORITY_PINS)
    assert not drift, (
        "persistence version authority moved without updating its lockstep pin "
        f"and co-change list: {drift}"
    )
    assert live["MIN_TLSPEC_VERSION"] <= live["TLSPEC_VERSION"]


def documented_merged_versions(text: str) -> set[int]:
    """Return every merged ``tlspec_version`` value stated in contract prose.

    Parameters
    ----------
    text:
        Contract document text.

    Returns
    -------
    set[int]
        Stated version numbers.
    """

    return {int(match) for match in re.findall(r"tlspec_version[:`\s]+(\d+)", text)}


def test_merged_contract_doc_states_the_shipped_version() -> None:
    """The merged contract doc's stated version tracks the code constant."""

    contract = _REPO_ROOT / "docs" / "reference" / "merged_trace_contract.md"
    stated = documented_merged_versions(contract.read_text(encoding="utf-8"))
    assert stated, "merged contract must state its tlspec_version"
    live = _version_authority_values()["MERGED_TLSPEC_VERSION"]
    assert stated == {live}, (
        f"docs/reference/merged_trace_contract.md states tlspec_version {sorted(stated)} "
        f"but MERGED_TLSPEC_VERSION is {live}"
    )


# ---------------------------------------------------------------------------
# The mechanism must be able to go RED. Each checker gets a planted drift.
# ---------------------------------------------------------------------------


class TestMechanismIsRedCapable:
    """Plant drift into each checker and prove it is reported.

    Without these, a lockstep gate could be silently vacuous -- a registry that
    matches itself, a diff that compares a value with itself. Each test below
    is the negative control for one checker above.
    """

    def test_catalog_registration_closure_detects_an_unregistered_catalog(self) -> None:
        """A new catalog with no registry entry is reported."""

        class _FakeConstants:
            NEW_THING_FIELD_ORDER = ["a", "b"]
            MODEL_LOG_FIELD_ORDER = ["c"]
            NOT_A_CATALOG = 3

        declared = declared_catalog_names(_FakeConstants)
        assert declared == {"NEW_THING_FIELD_ORDER", "MODEL_LOG_FIELD_ORDER"}
        unregistered, phantom = catalog_registration_gaps(declared, {"MODEL_LOG_FIELD_ORDER"})
        assert unregistered == {"NEW_THING_FIELD_ORDER"}
        assert not phantom

    def test_catalog_registration_closure_detects_a_phantom_entry(self) -> None:
        """A registry entry for a deleted catalog is reported."""

        unregistered, phantom = catalog_registration_gaps({"A_FIELD_ORDER"}, {"GONE_FIELD_ORDER"})
        assert unregistered == {"A_FIELD_ORDER"}
        assert phantom == {"GONE_FIELD_ORDER"}

    def test_policy_catalog_diff_detects_a_dropped_field(self) -> None:
        """A field present in the policy but missing from the catalog differs."""

        generated, declared = policy_catalog_diff(
            Trace.FIELD_POLICY, constants.MODEL_LOG_FIELD_ORDER[:-1]
        )
        assert generated != declared

    def test_policy_catalog_diff_detects_a_reordering(self) -> None:
        """Order is part of the contract, so a swap must differ too."""

        reordered = list(constants.MODEL_LOG_FIELD_ORDER)
        reordered[0], reordered[1] = reordered[1], reordered[0]
        generated, declared = policy_catalog_diff(Trace.FIELD_POLICY, reordered)
        assert generated != declared

    def test_private_ordered_ledger_detects_an_unledgered_field(self) -> None:
        """A private ordered field missing from the ledger is reported."""

        pruned = {
            cls_name: {k: v for k, v in fields.items() if k != "_runnable"}
            for cls_name, fields in PRIVATE_ORDERED_DROP_FIELDS.items()
        }
        unledgered, _ = private_ordered_field_gaps(_PRIMARY_CATALOGS, pruned)
        assert unledgered == {"Trace._runnable"}

    def test_private_ordered_ledger_detects_a_phantom_entry(self) -> None:
        """A ledger entry with no live ordered field is reported."""

        padded = {cls: dict(fields) for cls, fields in PRIVATE_ORDERED_DROP_FIELDS.items()}
        padded["Trace"]["_never_existed"] = "planted"
        _, phantom = private_ordered_field_gaps(_PRIMARY_CATALOGS, padded)
        assert phantom == {"Trace._never_existed"}

    def test_private_ordered_tiers_partition_the_live_set(self) -> None:
        """The two tiers are disjoint and together cover every private field."""

        drop, persisted = _private_ordered_fields_by_tier(_PRIMARY_CATALOGS)
        assert drop and persisted
        assert not drop & persisted

    def test_generated_artifact_closure_detects_an_unregistered_module(self) -> None:
        """A generated module absent from the registry is reported."""

        unregistered, phantom = artifact_registration_gaps(
            {"torchlens/x/_gen.py", "torchlens/ir/op_record_manifest.py"},
            {"torchlens/ir/op_record_manifest.py"},
        )
        assert unregistered == {"torchlens/x/_gen.py"}
        assert not phantom

    def test_generated_artifact_scan_finds_the_known_generated_modules(self) -> None:
        """The header scan is not vacuous: it finds the real generated files."""

        found = generated_module_paths()
        assert "torchlens/data_classes/_schema_bindings.py" in found
        assert "torchlens/ir/op_record_manifest.py" in found

    def test_generated_artifact_diff_detects_a_mutated_artifact(self) -> None:
        """A byte-level edit to a generated module is reported."""

        artifact = GENERATED_ARTIFACTS[0]
        checked_in = (_REPO_ROOT / artifact.path).read_text(encoding="utf-8")
        assert checked_in + "# tampered\n" != artifact.render()

    def test_runtime_declaration_checker_detects_an_undeclared_attribute(self) -> None:
        """An attribute outside the policy and the allowance is reported."""

        undeclared = undeclared_runtime_attributes(
            {"label", "_tl_core", "smuggled_field"}, dict(Op.FIELD_POLICY)
        )
        assert undeclared == {"smuggled_field"}

    def test_runtime_declaration_checker_honors_only_the_named_allowance(self) -> None:
        """The allowance excuses exactly the facade handles, nothing more."""

        assert undeclared_runtime_attributes({"_tl_row"}, {}) == set()
        assert undeclared_runtime_attributes({"_tl_rows"}, {}) == {"_tl_rows"}

    def test_version_pin_checker_detects_a_bump(self) -> None:
        """A version bump without a pin update is reported."""

        drift = version_pin_drift({"TLSPEC_VERSION": 8}, {"TLSPEC_VERSION": 7})
        assert drift == {"TLSPEC_VERSION": (8, 7)}
        assert not version_pin_drift({"TLSPEC_VERSION": 7}, {"TLSPEC_VERSION": 7})

    def test_merged_doc_parser_reads_stated_versions(self) -> None:
        """The doc parser finds prose and code-fence spellings, not noise."""

        assert documented_merged_versions("carries `tlspec_version: 7`,") == {7}
        assert documented_merged_versions("# tlspec_version: 9, descriptor") == {9}
        assert documented_merged_versions("nothing here") == set()


def test_field_policy_entries_declare_a_portable_policy() -> None:
    """Every declared field carries a real portable policy value.

    Cheap catch for a half-added field: present in the table, but with a
    policy value that is not a member of the closed vocabulary.
    """

    for catalog in _PRIMARY_CATALOGS:
        assert catalog.owner is not None
        for name, item in catalog.owner.FIELD_POLICY.items():
            assert isinstance(item.portable_policy, FieldPolicy), (
                f"{catalog.owner.__name__}.{name} has a non-FieldPolicy portable policy"
            )
