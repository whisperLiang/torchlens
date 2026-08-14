"""Input-boundary walking: one inert authority, witnessed arity, bounded descent (B3 L4).

Every case below was a live defect on main:

* classification probed ``_fields`` through the LIVE instance (``hasattr``), so a hostile
  container's ``__getattribute__`` executed during ``snapshot_input_boundary`` -- the exact
  thing the r71 C contract promises never happens -- and could STEER the kind: a genuine
  namedtuple subclass whose hook raised ``AttributeError`` for ``"_fields"`` classified as
  a plain ``sequence``, so no declared-schema gate was ever consulted and its hidden state
  escaped judgment;
* the capture walker resolved fields by raw MRO while the runtime binding walker used a
  live ``getattr``, so the same container was field-addressable to one and zero-field to
  the other (untyped ``AttributeError`` at runtime, or two different path keyings);
* physical namedtuple arity was never witnessed: snapshots of instances with different
  hidden positional payloads compared EQUAL, and a zero-field namedtuple silently dropped
  its children INCLUDING TENSORS with no refusal and no witness gap;
* neither walker had a cycle guard or depth bound, so a self-referential container raised
  ``RecursionError`` from internals (not a diagnosable refusal);
* an unset ``init=False`` dataclass field killed every intervention-ready capture with an
  untyped ``AttributeError``;
* registered-container ``aux`` was compared type-blind, so a ``mode=True`` capture re-run
  with the ``mode=1`` twin reported VERIFIED end-to-end.
"""

from __future__ import annotations

import collections
import dataclasses
import enum
from typing import Any

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._input_walk import (
    classify_input_container,
    declares_namedtuple_fields,
    empty_input_container_kind,
    namedtuple_arity_mismatch,
    snapshot_input_boundary,
    unset_declared_fields,
    walk_input_boundary,
)
from torchlens._runnable_witness_contracts import _container_field_names

pytestmark = pytest.mark.smoke

_OneField = collections.namedtuple("_OneField", "x")
_ZeroField = collections.namedtuple("_ZeroField", "")


class _PropFields(tuple):
    """Tuple subclass whose ``_fields`` is a PROPERTY (a live hook, not a schema)."""

    probes: list[str] = []

    @property
    def _fields(self) -> tuple[str, ...]:
        """Record the probe and hand back a forged schema."""

        _PropFields.probes.append("_fields")
        return ("a", "b")


class _ListFields(tuple):
    """Tuple subclass whose ``_fields`` is a list, not a tuple of names."""

    _fields = ["a", "b"]


class _HidingNamedtuple(_OneField):
    """Real namedtuple subclass whose hook hides ``_fields`` from a live read."""

    def __getattribute__(self, name: str) -> Any:
        """Hide the declared schema from any live probe."""

        if name == "_fields":
            raise AttributeError(name)
        return object.__getattribute__(self, name)


def _reasons(snapshot: dict[str, Any]) -> list[str]:
    """Refusal reasons of one boundary snapshot, in order."""

    return [refusal["reason"] for refusal in snapshot["refusals"]]


def test_classification_never_touches_the_instance():
    """A ``_fields`` property must not run during classify or snapshot."""

    value = _PropFields((torch.zeros(1), torch.zeros(1)))
    _PropFields.probes.clear()
    assert classify_input_container(value) == "namedtuple"
    snapshot_input_boundary(value)
    assert _PropFields.probes == []


def test_a_hiding_hook_cannot_steer_classification():
    """A namedtuple whose hook hides ``_fields`` still classifies as a namedtuple."""

    value = tuple.__new__(_HidingNamedtuple, (torch.zeros(1),))
    assert declares_namedtuple_fields(value)
    assert classify_input_container(value) == "namedtuple"


def test_malformed_fields_refuse_instead_of_reading_as_zero_field():
    """A list-valued ``_fields`` is a non-total schema, not an empty one."""

    value = _ListFields((torch.zeros(1), torch.zeros(1)))
    assert namedtuple_arity_mismatch(value)
    assert "namedtuple_schema_not_total" in _reasons(snapshot_input_boundary(value))


def test_hidden_positional_payload_refuses_typed():
    """Physical arity beyond the declared schema must refuse, never pass silently."""

    value = tuple.__new__(_OneField, (torch.zeros(1), "capture_flag"))
    snapshot = snapshot_input_boundary(value)
    assert "namedtuple_schema_not_total" in _reasons(snapshot)
    root = snapshot["nodes"][0]
    assert root["kind"] == "namedtuple"
    assert root["size"] == 2  # PHYSICAL arity, not len(_fields)
    assert root["fields"] == ["x"]


def test_zero_field_namedtuple_carrying_tensors_is_not_empty():
    """A zero-field namedtuple with children must not classify as an EMPTY container."""

    value = tuple.__new__(_ZeroField, (torch.zeros(1), "steer"))
    assert empty_input_container_kind(value) is None
    assert classify_input_container(value) == "namedtuple"
    assert "namedtuple_schema_not_total" in _reasons(snapshot_input_boundary(value))
    # A GENUINELY empty namedtuple still reads as empty.
    assert empty_input_container_kind(_ZeroField()) == "namedtuple"


def test_empty_container_kind_alias_delegates_to_the_one_authority():
    """The ``_io.runnable`` spelling must be the inert implementation, not a twin."""

    from torchlens._io.runnable import empty_container_kind

    value = tuple.__new__(_ZeroField, (torch.zeros(1),))
    assert empty_container_kind(value) is empty_input_container_kind(value)
    assert empty_container_kind({}) == "mapping"
    assert empty_container_kind([]) == "sequence"


def test_both_walkers_resolve_fields_through_one_authority():
    """The runtime binding walker must agree with the capture walker, and never crash."""

    prop = _PropFields((torch.zeros(1), torch.zeros(1)))
    listed = _ListFields((torch.zeros(1), torch.zeros(1)))
    assert _container_field_names(prop) == ()
    assert _container_field_names(listed) == ()
    assert _container_field_names(_OneField(torch.zeros(1))) == ("x",)


def test_self_referential_container_refuses_instead_of_recursing():
    """A cycle must become a typed refusal, never a ``RecursionError``."""

    tree: dict[str, Any] = {"x": torch.zeros(1)}
    tree["self"] = tree
    snapshot = snapshot_input_boundary(tree)
    assert "input_container_cycle" in _reasons(snapshot)


def test_over_deep_container_refuses_instead_of_recursing():
    """A nest past the declared bound must become a typed refusal."""

    value: Any = torch.zeros(1)
    for _ in range(3000):
        value = [value]
    snapshot = snapshot_input_boundary(value)
    assert "input_container_too_deep" in _reasons(snapshot)


def test_cycle_ceilings_the_subtree_in_the_value_walker():
    """The literal/site walker must ceiling a cycle through its opaque channel."""

    tree: dict[str, Any] = {"x": torch.zeros(1)}
    tree["self"] = tree
    opaque: list[tuple[Any, ...]] = []
    tensors: list[tuple[Any, ...]] = []
    walk_input_boundary(
        tree,
        key_component=lambda key: key,
        on_tensor=lambda _value, path: tensors.append(path),
        on_opaque_key_subtree=lambda _value, path: opaque.append(path),
    )
    assert tensors == [("x",)]
    assert opaque == [("self",)]


def test_repeated_sibling_container_is_still_witnessed_twice():
    """The fence tracks ANCESTORS only: a shared container must not be dropped."""

    shared = {"t": torch.zeros(1)}
    tensors: list[tuple[Any, ...]] = []
    walk_input_boundary(
        {"a": shared, "b": shared},
        key_component=lambda key: key,
        on_tensor=lambda _value, path: tensors.append(path),
    )
    assert sorted(tensors) == [("a", "t"), ("b", "t")]


def test_unset_init_false_dataclass_field_no_longer_kills_the_capture():
    """A legal lazily-populated dataclass field must not raise from internals."""

    @dataclasses.dataclass
    class _Box:
        """Dataclass with a declared-but-unset scratch field."""

        x: Any
        cache: Any = dataclasses.field(init=False)

    class _Model(nn.Module):
        """Reads only the set field."""

        def forward(self, box: Any) -> torch.Tensor:
            """Double the tensor field."""

            return box.x * 2

    box = _Box(torch.randn(1, 4))
    assert unset_declared_fields(box) == ("cache",)
    snapshot = snapshot_input_boundary(box)
    assert "unset_declared_field" in _reasons(snapshot)
    assert snapshot["nodes"][0]["unset_fields"] == ["cache"]
    # The whole point: an intervention-ready capture completes instead of raising.
    trace = tl.trace(_Model(), box, intervention_ready=True)
    assert trace.num_ops >= 1


def test_registered_aux_is_type_strict():
    """``True``/``1`` aux twins must not compare equal; a semantic aux must refuse."""

    @dataclasses.dataclass
    class _Wrap:
        """Registered container whose aux carries a mode flag."""

        value: Any
        mode: Any

    tl.register_container(
        _Wrap,
        lambda wrap: ((wrap.value,), wrap.mode),
        lambda children, aux: _Wrap(children[0], aux),
        state_complete=True,
    )
    tensor = torch.zeros(1)
    bool_aux = snapshot_input_boundary(_Wrap(tensor, True))
    int_aux = snapshot_input_boundary(_Wrap(tensor, 1))
    assert bool_aux["nodes"] != int_aux["nodes"]
    assert not _reasons(bool_aux)

    # A NaN aux must compare equal to itself instead of false-diverging.
    nan_a = snapshot_input_boundary(_Wrap(tensor, float("nan")))
    nan_b = snapshot_input_boundary(_Wrap(tensor, float("nan")))
    assert nan_a["nodes"] == nan_b["nodes"]

    class _Mode(enum.IntEnum):
        """Semantic aux the declared schema cannot carry."""

        FAST = 1

    assert "registered_aux_unsafe" in _reasons(snapshot_input_boundary(_Wrap(tensor, _Mode.FAST)))


def test_foreign_grafted_getattribute_is_uninspectable():
    """A grafted foreign C slot wrapper must fail closed, not pass the inertness gate."""

    from torchlens._input_walk import _declared_schema_uninspectable

    @dataclasses.dataclass
    class _Sneak:
        """Dataclass with ``type.__getattribute__`` grafted on."""

        x: Any

    _Sneak.__getattribute__ = type.__getattribute__  # type: ignore[method-assign]
    assert _declared_schema_uninspectable(_Sneak.__new__(_Sneak)) is True


def test_ordinary_containers_stay_inspectable():
    """The tightened gate must not reject the containers real users pass."""

    from torchlens._input_walk import _declared_schema_uninspectable

    @dataclasses.dataclass
    class _Cfg:
        """Plain dataclass input."""

        x: Any

    assert _declared_schema_uninspectable(_Cfg(torch.zeros(1))) is False
    assert _declared_schema_uninspectable(_OneField(torch.zeros(1))) is False
    assert _declared_schema_uninspectable({}) is False
    assert _declared_schema_uninspectable([]) is False


def test_mapping_ordered_key_fact_comes_from_the_child_traversal():
    """The persisted key order must be the order the model actually iterates."""

    class _Skew(dict):
        """Mapping whose ``keys()`` order contradicts ``__iter__``/``items()``."""

        def keys(self):  # type: ignore[override]
            """Return the keys in reversed order."""

            return reversed(list(dict.keys(self)))

    value = _Skew({"a": torch.zeros(1), "b": torch.zeros(1)})
    root = snapshot_input_boundary({"cfg": value})["nodes"][1]
    assert root["kind"] == "mapping"
    assert root["keys"] == ["a", "b"]
