"""Validation releases completed oracle work before retaining captured payloads."""

import weakref
from typing import Any

import pytest
import torch

import torchlens as tl
import torchlens._user_public_impls as user_impls
import torchlens.user_funcs as user_funcs
from torchlens.validation import validate_forward_pass


@pytest.mark.smoke
@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("cached_forward", [False, True])
def test_ground_truth_model_is_released_before_validation_capture(
    monkeypatch: pytest.MonkeyPatch, prepared: bool, cached_forward: bool
) -> None:
    """An oracle model copy must not compete with the later full-payload capture."""

    model = torch.nn.Linear(2, 2)
    if cached_forward:
        model.cached_forward = model.forward
    inputs = torch.ones(1, 2)
    if prepared:
        tl.trace(model, inputs).cleanup()
    oracle_refs: list[weakref.ReferenceType[torch.nn.Module]] = []
    parameter_refs: list[weakref.ReferenceType[torch.nn.Parameter]] = []
    capture_calls: list[bool] = []
    make_oracle = user_impls._model_for_ground_truth_validation
    capture = user_funcs._run_model_and_save_specified_outs

    def track_oracle(*args: Any, **kwargs: Any) -> Any:
        """Observe the isolated oracle without owning it."""
        oracle, snapshot = make_oracle(*args, **kwargs)
        oracle_refs.append(weakref.ref(oracle))
        parameter_refs.extend(weakref.ref(parameter) for parameter in oracle.parameters())
        return oracle, snapshot

    def capture_after_oracle(*args: Any, **kwargs: Any) -> Any:
        """Admit capture only after the oracle's model storage is releasable."""
        assert oracle_refs and oracle_refs[-1]() is None
        assert parameter_refs and all(ref() is None for ref in parameter_refs)
        capture_calls.append(True)
        return capture(*args, **kwargs)

    monkeypatch.setattr(user_impls, "_model_for_ground_truth_validation", track_oracle)
    monkeypatch.setattr(user_funcs, "_run_model_and_save_specified_outs", capture_after_oracle)
    assert validate_forward_pass(model, inputs)
    assert capture_calls
