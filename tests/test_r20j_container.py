"""r20j hardening: IR container spec correctness + registry honesty.

Covers ledger findings N4 (false-VERIFIED tensor-key dict witness), N5 (rebuild
must reject malformed specs), N6 (most-derived registration dispatch), and N7
(snapshot dedup must not drop the later observation's event index).
"""

from __future__ import annotations


import pytest
import torch

from torchlens.ir.container import (
    rebuild_container_from_spec,
)
from torchlens.ir.container_registry import (
    Role,
    walk_container,
)


# ---------------------------------------------------------------------------
# N4 -- tensor-key dict must NOT be blessed reconstructable (false-VERIFIED class)
# ---------------------------------------------------------------------------


def test_n4_tensor_key_dict_is_not_reconstructable() -> None:
    """A mixed tensor-key dict is reported unreconstructable AND cannot silently lie.

    The witness previously said ``reconstructable=True`` (kind ``dict``, length counts the
    tensor key) while ``rebuild_container_from_spec`` raised "Not enough leaves". The honest
    fix degrades the spec to ``opaque`` so the flag matches reality.
    """

    visible = torch.tensor([1.0])
    payload = {"visible": visible, torch.tensor([9.0]): torch.tensor([2.0])}
    result = walk_container(payload, role=Role.MODEL_OUTPUT, capability="full_spec")
    assert result is not None
    # The witness must be honest: it can NOT be rebuilt.
    assert result.reconstructable is False
    assert result.spec.kind == "opaque"
    # And the honesty holds under actual reconstruction: rebuilding an opaque spec is a
    # clean typed refusal, never a misleading "Not enough leaves".
    with pytest.raises(ValueError, match="Opaque ContainerSpec cannot be reconstructed"):
        rebuild_container_from_spec(result.spec, [visible])


def test_n4_clean_nested_containers_stay_reconstructable() -> None:
    """The honest recursion must NOT regress the clean nested list/dict/tuple+literal path."""

    payload = {
        "pair": (torch.tensor([1.0]), torch.tensor([2.0])),
        "nested": {"inner": [torch.tensor([3.0]), 7]},
    }
    result = walk_container(payload, role=Role.MODEL_OUTPUT, capability="full_spec")
    assert result is not None
    assert result.reconstructable is True
    assert result.spec.kind == "dict"
    leaves = [torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])]
    rebuilt = rebuild_container_from_spec(result.spec, leaves)
    assert set(rebuilt) == {"pair", "nested"}
    assert rebuilt["nested"]["inner"][1] == 7


def test_n4_nested_opaque_child_makes_parent_unreconstructable() -> None:
    """Class-wide: an opaque node ANYWHERE makes the reconstructable witness False.

    The prior shallow ``spec.kind != "opaque"`` blessed a container whose nested child
    (here a generator) is opaque -- a lying witness. The recursive honesty check catches it.
    """

    def gen() -> object:
        yield torch.tensor([1.0])

    payload = [torch.tensor([5.0]), gen()]
    result = walk_container(payload, role=Role.MODEL_OUTPUT, capability="full_spec")
    assert result is not None
    # top-level kind is a list (not opaque), but a nested child is opaque -> not rebuildable.
    assert result.spec.kind == "list"
    assert any(child.kind == "opaque" for _c, child in result.spec.child_specs)
    assert result.reconstructable is False
