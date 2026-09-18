"""Checkpoint invocation token pins (L9 memo 2.3, wave-2 subset).

Tokens are the WITNESS, site keys the GROUPER: one classified non-reentrant
checkpoint enter mints exactly one per-trace ordinal token; pack evidence is
count-only (the forward-side slot->op binding is NOT claimed); unpack
evidence points are backward-derived (fire brackets -> shipped user-op
pairing -> read-only L1 site keys). The projected summary lands on the
DROP-gated ``Trace.checkpoint_invocation_witness`` field. The typed
ambiguity REFUSAL is S2-authored (R-L9-1) and deliberately NOT shipped here;
no test below exercises an identity-read refusal. All spellings
DOCUMENTED-UNSTABLE.
"""

from __future__ import annotations

import inspect
import warnings
import weakref
from typing import Any

import pytest
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

import torchlens as tl
from torchlens import _state
from torchlens.backends.torch import backward as backward_mod
from torchlens.backends.torch._aten_capture import _activate_aten_recording_for_tests
from torchlens.ir.events import CheckpointInvocationObserved

pytestmark = pytest.mark.smoke


class _OneCheckpoint(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 8)
        self.b = nn.Linear(8, 8)
        self.c = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.a(x))
        h = checkpoint(self.b, h, use_reentrant=False)
        return self.c(torch.relu(h))


class _TwoIdenticalCheckpoints(nn.Module):
    """Two structurally identical checkpointed invocations (the ruling's core case)."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 8)
        self.b1 = nn.Linear(8, 8)
        self.b2 = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.a(x))
        h = checkpoint(self.b1, h, use_reentrant=False)
        return checkpoint(self.b2, h, use_reentrant=False)


def _captured(model: nn.Module, x: torch.Tensor, *, backward: bool = True) -> tl.Trace:
    torch.manual_seed(0)
    trace = tl.trace(model, x, backward_ready=True, save_mode="reference")
    if backward:
        trace.log_backward(trace.output_ops[0].out.sum())
    return trace


# ---------------------------------------------------------------------------
# Minting: one token per logical invocation; recompute never mints.
# ---------------------------------------------------------------------------


def test_plain_single_checkpoint_mints_exactly_one_token() -> None:
    trace = _captured(_OneCheckpoint(), torch.randn(3, 4))
    witness = trace.checkpoint_invocation_witness
    assert witness["token_count"] == 1
    assert witness["degrade_flags"] == []
    assert witness["verdict"] == "checkpoint_invocations_observed"
    (record,) = witness["tokens"].values()
    # Recompute non-minting: the backward ran (unpack evidence exists), the
    # _recomputation_hook entered, and the count stayed 1.
    assert record["pack_count"] > 0
    assert record["unpack_evidence_count"] > 0
    minted = [
        event for event in trace.backward_events if isinstance(event, CheckpointInvocationObserved)
    ]
    assert len(minted) == 1


def test_two_identical_invocations_mint_two_tokens() -> None:
    trace = _captured(_TwoIdenticalCheckpoints(), torch.randn(3, 4))
    witness = trace.checkpoint_invocation_witness
    assert witness["token_count"] == 2
    assert sorted(witness["tokens"]) == [1, 2]
    for record in witness["tokens"].values():
        assert record["unpack_evidence_count"] > 0


def test_forward_only_capture_has_token_with_zero_unpack_evidence() -> None:
    trace = _captured(_OneCheckpoint(), torch.randn(3, 4), backward=False)
    witness = trace.checkpoint_invocation_witness
    assert witness["token_count"] == 1
    (record,) = witness["tokens"].values()
    assert record["unpack_evidence_count"] == 0
    assert record["site_key_candidates"] == []


def test_backward_derived_site_candidates_resolve_under_armed_brackets() -> None:
    torch.manual_seed(0)
    with _activate_aten_recording_for_tests():
        trace = tl.trace(
            _OneCheckpoint(), torch.randn(3, 4), backward_ready=True, save_mode="reference"
        )
        trace.log_backward(trace.output_ops[0].out.sum())
    witness = trace.checkpoint_invocation_witness
    (record,) = witness["tokens"].values()
    # Unpack evidence inside witnessed fire brackets resolves through the
    # shipped user-op pairing to the checkpointed module's site key.
    assert record["site_key_candidates"], "armed brackets produced no site candidates"
    assert all(key.startswith("s1|") for key in record["site_key_candidates"])
    assert any(label is not None for _, label, _ in record["window_evidence"])


# ---------------------------------------------------------------------------
# Negative control + degrade classes.
# ---------------------------------------------------------------------------


class _PlainHooks(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autograd.graph.saved_tensors_hooks(lambda t: t, lambda t: t):
            h = self.fc(x)
        with torch.autograd.graph.save_on_cpu():
            return h * 2


def test_negative_control_plain_hooks_and_save_on_cpu() -> None:
    trace = _captured(_PlainHooks(), torch.randn(3, 4))
    witness = trace.checkpoint_invocation_witness
    assert witness["token_count"] == 0
    assert witness["degrade_flags"] == []
    assert witness["verdict"] == "no_checkpoint_invocation_observed"


class _Reentrant(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 8)
        self.b = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(checkpoint(lambda y: torch.relu(self.a(y)), x, use_reentrant=True))


def test_reentrant_sets_d5_and_never_reaches_affirmative_verdict() -> None:
    trace = _captured(_Reentrant(), torch.randn(3, 4, requires_grad=True))
    witness = trace.checkpoint_invocation_witness
    assert witness["token_count"] == 0
    assert "reentrant_node_discovered" in witness["degrade_flags"]
    assert witness["verdict"] == "evidence_incomplete"


class _PausedCheckpoint(nn.Module):
    """A checkpoint entered under paused logging: classifier condition (2) fails."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 8)
        self.b = nn.Linear(8, 8)
        self.c = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.a(x))
        with _state.pause_logging():
            h = checkpoint(self.b, h, use_reentrant=False)
        # Traced tail: the paused region must not produce the model output
        # (an all-paused tail would be an unattributable output by design).
        return self.c(h)


def test_unwitnessed_enter_sets_d6_instead_of_staying_silent() -> None:
    torch.manual_seed(0)
    # The paused-region output enters module c without graph provenance;
    # TorchLens discloses that honestly and the disclosure is expected here.
    with pytest.warns(UserWarning, match="no graph/source provenance"):
        trace = tl.trace(
            _PausedCheckpoint(), torch.randn(3, 4), backward_ready=True, save_mode="reference"
        )
    trace.log_backward(trace.output_ops[0].out.sum())
    witness = trace.checkpoint_invocation_witness
    assert witness["token_count"] == 0
    assert "unwitnessed_checkpoint_enter" in witness["degrade_flags"]
    assert witness["verdict"] == "evidence_incomplete"


def test_classifier_unavailable_sets_d1_and_no_tokens(monkeypatch) -> None:
    monkeypatch.setattr(backward_mod, "_resolve_checkpoint_hook_cls", lambda: None)
    trace = _captured(_OneCheckpoint(), torch.randn(3, 4))
    witness = trace.checkpoint_invocation_witness
    assert witness["token_count"] == 0
    assert "classifier_unavailable" in witness["degrade_flags"]
    assert witness["verdict"] == "evidence_incomplete"


def test_unmatched_backward_warn_sets_d4_and_warn_once_preserved() -> None:
    torch.manual_seed(0)
    trace = tl.trace(nn.Linear(4, 2), torch.randn(3, 4), backward_ready=True)
    foreign = torch.randn(3, requires_grad=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with trace.recording_backward():
            (foreign * 2).sum().backward()
    unmatched = [w for w in caught if "did not" in str(w.message)]
    assert len(unmatched) == 1
    witness = trace.checkpoint_invocation_witness
    assert "unmatched_backward_warn" in witness["degrade_flags"]
    assert witness["verdict"] == "evidence_incomplete"


def test_patch_unavailable_sets_d2(monkeypatch) -> None:
    trace = _captured(nn.Linear(4, 2), torch.randn(3, 4))
    monkeypatch.setattr(backward_mod, "_SAVED_TENSORS_HOOKS_INIT_PATCHED", False)
    backward_mod._refresh_checkpoint_witness(trace)
    witness = trace.checkpoint_invocation_witness
    assert "patch_unavailable" in witness["degrade_flags"]
    assert witness["verdict"] == "evidence_incomplete"


# ---------------------------------------------------------------------------
# Fresh-wrapper discipline: re-minting never stacks token layers.
# ---------------------------------------------------------------------------


def test_token_wrappers_unwrap_prior_layer_instead_of_stacking() -> None:
    trace = tl.trace(nn.Linear(4, 2), torch.randn(3, 4))
    state = backward_mod._checkpoint_token_state(trace)
    state["tokens"][1] = {"pack_count": 0, "unpack_evidence": []}
    state["tokens"][2] = {"pack_count": 0, "unpack_evidence": []}
    calls: list[str] = []

    def base(value):
        calls.append("base")
        return value

    first = backward_mod._token_bearing_pack_hook(trace, 1, base)
    second = backward_mod._token_bearing_pack_hook(
        trace, 2, getattr(first, "__tl_token_inner__", first)
    )
    assert second.__tl_token_inner__ is base
    second(torch.zeros(1))
    assert state["tokens"][1]["pack_count"] == 0, "stale token layer still counting"
    assert state["tokens"][2]["pack_count"] == 1
    assert calls == ["base"]


def test_checkpoint_hook_metadata_cleanup_reaches_every_wrapper_layer() -> None:
    """Native temporary hook metadata must disappear from retained closures."""
    trace = tl.trace(nn.Linear(4, 2), torch.randn(3, 4))

    def base(value: Any) -> Any:
        """Return a saved value unchanged."""
        return value

    class HookState:
        """Stand in for user hook state that can retain an autograd graph."""

    base._checkpoint_internal = True
    scoped = backward_mod._scoped_saved_tensors_hook(base)
    state = HookState()
    state_ref = weakref.ref(state)
    scoped._user_hooks = state
    first = backward_mod._token_bearing_pack_hook(trace, 1, scoped)
    second = backward_mod._token_bearing_pack_hook(trace, 2, first.__tl_token_inner__)
    assert first._checkpoint_internal is second._checkpoint_internal is True
    assert first._user_hooks is second._user_hooks is state

    # PyTorch 2.14 deletes this attribute before popping its native hook
    # stack. Copying it onto wrappers would keep stale graph references.
    del second._user_hooks
    del state
    assert state_ref() is None
    assert all(not hasattr(hook, "_user_hooks") for hook in (base, scoped, first, second))

    # Re-entry must see fresh state without reviving an earlier invocation.
    next_state = HookState()
    second._user_hooks = next_state
    assert base._user_hooks is first._user_hooks is next_state
    del second._user_hooks


def test_nested_checkpoint_preserves_user_hooks_gradients_and_stack() -> None:
    """Nested checkpoint hooks retain native metadata and leave no TLS leak."""
    from torch.utils import checkpoint as checkpoint_module

    create_selective_checkpoint_contexts = getattr(
        checkpoint_module, "create_selective_checkpoint_contexts", None
    )
    if create_selective_checkpoint_contexts is None:
        pytest.skip("This torch build does not expose selective checkpoint contexts")

    packed: list[tuple[int, ...]] = []
    checkpoint_options = (
        {"respect_saved_tensors_hooks": True}
        if "respect_saved_tensors_hooks" in inspect.signature(checkpoint).parameters
        else {}
    )

    def pack(value: torch.Tensor) -> torch.Tensor:
        """Record a genuine user hook invocation and offload a detached copy."""
        packed.append(tuple(value.shape))
        return value.detach().clone()

    class Nested(nn.Module):
        """Exercise selective checkpoint's search through internal hook layers."""

        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def inner(self, x: torch.Tensor) -> torch.Tensor:
            return checkpoint(
                lambda value: self.lin(value).relu(),
                x,
                use_reentrant=False,
                context_fn=lambda: create_selective_checkpoint_contexts(
                    [torch.ops.aten.addmm.default]
                ),
                **checkpoint_options,
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
                return checkpoint(self.inner, x, use_reentrant=False)

    model = Nested()
    x = torch.randn(3, 4, requires_grad=True)
    expected = model(x)
    expected_grads = torch.autograd.grad(expected.sum(), (x, *model.parameters()))
    packed.clear()
    trace = tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(backward_ready=True),
        save_mode="reference",
    )
    output = trace.output_ops[0].out
    actual_grads = torch.autograd.grad(output.sum(), (x, *model.parameters()))
    torch.testing.assert_close(output, expected)
    for actual, target in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual, target)
    assert packed
    assert trace.checkpoint_invocation_witness["token_count"] == 2
    # torch.func refuses any leaked saved-tensor hook, even for an unrelated
    # function. Check it in the same process after both checkpoint contexts.
    torch.testing.assert_close(torch.func.grad(lambda value: value.square().sum())(x), 2 * x)


# ---------------------------------------------------------------------------
# Wave-2 schema pins: the witness never rides ordinary v7 saves.
# ---------------------------------------------------------------------------


def test_witness_rides_ordinary_save_at_v8(tmp_path) -> None:
    """tlspec v8: the checkpoint witness persists on a plain save/load."""

    trace = _captured(_OneCheckpoint(), torch.randn(3, 4))
    assert trace.checkpoint_invocation_witness["token_count"] == 1
    path = tmp_path / "ckpt_plain.tlspec"
    tl.save(trace, str(path))
    loaded = tl.load(str(path))
    assert loaded.checkpoint_invocation_witness is not None
    assert loaded.checkpoint_invocation_witness["token_count"] == 1
