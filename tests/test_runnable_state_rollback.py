"""Run-preparation staging must publish NOTHING on failure (grind b3-l1, F-R07)."""

from __future__ import annotations

import pytest
import torch

from torchlens import _runnable_state
from torchlens._runnable_state import PreparedRunnableState, _apply_state_metadata_facts
from torchlens.errors import StateBindingError
from torchlens.runnable import StateSource

pytestmark = pytest.mark.smoke


class _Binding:
    """State-binding stand-in carrying the recorded metadata facts."""

    def __init__(self, name: str, requires_grad: bool) -> None:
        """Record the declared name and capture-time trainable bit."""

        self.state_dict_name = name
        self.captured_grad_fn = False
        self.captured_requires_grad = requires_grad


class _Slot:
    """Tensor-slot stand-in with an owning state binding."""

    def __init__(self, slot_id: str, binding: _Binding) -> None:
        """Attach the binding to the slot identity."""

        self.slot_id = slot_id
        self.state_binding = binding


class _Descriptor:
    """Descriptor stand-in exposing only the state-slot surface."""

    def __init__(self, slots: list[_Slot]) -> None:
        """Hold the slot list the staging loop walks."""

        self.tensor_slots = slots


def test_state_metadata_staging_rolls_back_flipped_bits_on_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mid-loop metadata refusal restores every already-flipped staged bit.

    Staged slot values can be the trace-persisted clones themselves, so a
    slot-N failure that left slots 1..N-1 flipped would leak mutated
    ``requires_grad`` bits into embedded/staged user state across future runs,
    contradicting the publish-nothing-on-failure staging contract.
    """

    float_value = torch.zeros(2)
    int_value = torch.zeros(2, dtype=torch.int64)
    assert not float_value.requires_grad
    slots = [
        _Slot("s1", _Binding("layer.weight", requires_grad=True)),
        _Slot("s2", _Binding("layer.counter", requires_grad=True)),
    ]
    descriptor = _Descriptor(slots)
    binding_facts = {
        "layer.weight": {"grad_fn": False, "requires_grad": True},
        "layer.counter": {"grad_fn": False, "requires_grad": True},
    }
    monkeypatch.setattr(
        _runnable_state, "recorded_state_metadata_facts", lambda _desc: binding_facts
    )
    prepared = PreparedRunnableState(
        slot_values={"s1": float_value, "s2": int_value},
        state_source=StateSource.USER_STATE_DICT,
        initializer_policy_version=None,
        seed=None,
        random_filled_slot_ids=(),
    )

    with pytest.raises(StateBindingError):
        _apply_state_metadata_facts(descriptor, prepared)  # type: ignore[arg-type]

    # The int slot refused (integers cannot require grad); the float slot's
    # already-applied flip must have been rolled back, not published.
    assert not float_value.requires_grad
    assert not int_value.requires_grad
