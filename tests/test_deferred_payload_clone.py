"""Deferred activation-payload clones (clone-on-write) behavior contract.

Eligible plain captures save payload ALIASES instead of eager clones; the
torch wrapper materializes a pending alias onto exclusive fresh storage
before any wrapped call writes its storage. These tests pin the contract:

  * saved payload bytes/metadata are identical to eager-clone behavior,
    including across in-place ops, ``out=`` destinations, ``__setitem__``,
    ``.data`` writes, ``inplace=True`` conveniences, and train-mode
    batch-norm buffer side effects;
  * post-hoc in-place edits of one saved activation never bleed into
    another (eager isolation semantics);
  * a user-retained live activation mutated after capture never corrupts
    the saved value;
  * the version-counter belt refuses loudly when a storage was mutated
    through a path interception never saw.
"""

import contextlib

import pytest
import torch
import torch.nn as nn

import torchlens as tl
import torchlens.backends.torch.wrappers as _wrappers
from torchlens.utils import tensor_utils as _tu


@contextlib.contextmanager
def _payload_clone_mode(defer: bool):
    """Force deferred or eager payload clones for the duration of the block."""
    prev_tu, prev_w = _tu._DEFER_ENABLED, _wrappers._COW_ENABLED
    _tu._DEFER_ENABLED = defer
    _wrappers._COW_ENABLED = defer
    try:
        yield
    finally:
        _tu._DEFER_ENABLED = prev_tu
        _wrappers._COW_ENABLED = prev_w


class _InplaceZoo(nn.Module):
    """Every wrapped mutation surface the interceptor must cover."""

    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(8)
        self.lin = nn.Linear(8, 8)
        self.drop = nn.Dropout(p=0.5, inplace=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.lin(x)
        y = self.bn(y)  # train mode: buffer side effects (state stays eager)
        y = self.relu(y)  # in-place relu on the bn output storage
        z = y * 2
        z += 1  # augmented-assignment dunder
        w = torch.empty_like(z)
        torch.add(z.detach(), 3, out=w)  # out= destination
        w[0, 0:2] = 5.0  # __setitem__
        v = w.clone()
        v.data.mul_(2)  # write through a .data alias
        u = self.drop(v)  # F.dropout(..., inplace=True) under the hood
        q = u.reshape(2, 4, 2)  # view chain: payloads share one storage
        q2 = q.permute(0, 2, 1)
        self.stash = z  # user-retained live activation
        return q2.contiguous().sum(dim=-1)


def _zoo_trace(defer: bool, seed: int = 7, grad: bool = False):
    with _payload_clone_mode(defer):
        torch.manual_seed(seed)
        model = _InplaceZoo().train()
        torch.manual_seed(seed)
        x = torch.randn(2, 8)
        if grad:
            log = tl.trace(model, x, random_seed=99)
        else:
            with torch.no_grad():
                log = tl.trace(model, x, random_seed=99)
    return model, log


def _tensor_payload_labels(log):
    return [
        k
        for k in log.layer_labels
        if isinstance(log[k].out, torch.Tensor)
        # empty_like output is uninitialized memory: nondeterministic across
        # runs under eager cloning too, so it carries no byte contract.
        and not k.startswith("empty")
    ]


def _assert_payloads_identical(log_eager, log_defer):
    labels_e = list(log_eager.layer_labels)
    labels_d = list(log_defer.layer_labels)
    assert labels_e == labels_d
    for k in _tensor_payload_labels(log_eager):
        a, b = log_eager[k].out, log_defer[k].out
        assert a.dtype == b.dtype, k
        assert a.shape == b.shape, k
        assert a.stride() == b.stride(), k
        assert a.requires_grad == b.requires_grad, k
        assert (a.grad_fn is None) == (b.grad_fn is None), k
        if a.numel():
            ac = a.detach().contiguous()
            bc = b.detach().contiguous()
            assert ac.numpy().tobytes() == bc.numpy().tobytes(), k


@pytest.mark.smoke
def test_deferred_payloads_byte_identical_no_grad():
    _, log_eager = _zoo_trace(defer=False)
    _, log_defer = _zoo_trace(defer=True)
    _assert_payloads_identical(log_eager, log_defer)


@pytest.mark.smoke
def test_deferred_payloads_byte_identical_grad_mode():
    # Grad-enabled captures defer only no-autograd payloads; the public
    # surface (values AND grad_fn presence) must be unchanged either way.
    _, log_eager = _zoo_trace(defer=False, grad=True)
    _, log_defer = _zoo_trace(defer=True, grad=True)
    _assert_payloads_identical(log_eager, log_defer)


@pytest.mark.smoke
def test_post_hoc_inplace_edit_isolation():
    _, log = _zoo_trace(defer=True)
    labels = _tensor_payload_labels(log)
    snapshot = {k: log[k].out.clone() for k in labels}
    victim = next(k for k in labels if log[k].out.is_floating_point() and log[k].out.numel() > 0)
    log[victim].out.add_(1234.5)
    for k in labels:
        if k == victim:
            continue
        assert torch.equal(snapshot[k], log[k].out), k


@pytest.mark.smoke
def test_retained_live_activation_mutation_does_not_corrupt_saved():
    model, log = _zoo_trace(defer=True)
    labels = _tensor_payload_labels(log)
    saved = {k: log[k].out.clone() for k in labels}
    # The model kept a live handle to a mid-forward activation; a post-capture
    # in-place write through it must be intercepted before the bytes move.
    model.stash.add_(999.0)
    for k in labels:
        assert torch.equal(saved[k], log[k].out), k


@pytest.mark.smoke
def test_validation_tripwire_green_on_deferred_capture():
    with _payload_clone_mode(True):
        torch.manual_seed(7)
        model = _InplaceZoo().train()
        torch.manual_seed(7)
        x = torch.randn(2, 8)
        assert tl.validate(model, x, scope="forward", random_seed=99) is True


@pytest.mark.smoke
def test_version_belt_refuses_unintercepted_mutation():
    x = torch.randn(4)
    _tu.arm_deferred_payload_window(frozenset())
    try:
        alias = _tu.safe_copy(x, detach_tensor=True, save_mode="copy")
    finally:
        _tu.disarm_deferred_payload_window()
    assert alias.data_ptr() == x.data_ptr()  # genuinely deferred
    # Simulate a mutation path the wrapper never sees: the raw dispatcher
    # surface bypasses TorchLens wrapping entirely but still bumps the shared
    # version counter, which is exactly what the belt exists to catch.
    torch.ops.aten.relu_(x)
    key, entry = next(
        (k, e) for k, entries in _tu._DEFER_PENDING.items() for e in entries if e.ref() is alias
    )
    with pytest.raises(RuntimeError, match="deferred-clone tripwire"):
        _tu._belt_check_pending_alias(entry, alias)
    _tu._DEFER_PENDING.pop(key, None)


@pytest.mark.smoke
def test_kill_switch_restores_eager_clones():
    import gc

    gc.collect()  # earlier tests' (cyclic) traces may still pin live aliases
    baseline_live = sum(
        1 for entries in _tu._DEFER_PENDING.values() for e in entries if e.ref() is not None
    )
    with _payload_clone_mode(False):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(inplace=True))
        with torch.no_grad():
            log = tl.trace(model, torch.randn(2, 4))
        pending_live = sum(
            1 for entries in _tu._DEFER_PENDING.values() for e in entries if e.ref() is not None
        )
        assert pending_live == baseline_live
        assert isinstance(log[log.layer_labels[-1]].out, torch.Tensor)
