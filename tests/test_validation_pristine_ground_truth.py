"""Wrap-state-independent ground truth (R75-1, b9-fable round-2 probe).

Once torch is wrapped, the phase-0 ground-truth forward, the per-op replay,
and backward validation's stock-autograd pass all observed through the SAME
installed wrapper closures, so a TorchLens-induced numeric distortion (a
planted 0.1% ``tanh`` distortion in the wrapper layer) validated CLEAN in a
wrapped process and was caught only in a fresh one. The ground-truth
oracles now run with the wrappers removed (restored from the pre-decoration
originals ledger) and reinstalled afterwards.

The probe needs a process whose FIRST wrap installs the distorted wrapper,
so the red-capable cases run in a subprocess with a fresh interpreter.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import _state
from torchlens.validation._pristine import pristine_torch_oracle

_PROBE_SCRIPT = textwrap.dedent(
    """
    import torch
    import torch.nn as nn
    import torchlens as tl
    import torchlens.backends.torch.wrappers as wrappers

    # Plant the b9-fable distortion BEFORE the first wrap: the wrapper built
    # for tanh silently multiplies its result by 1.001 using pristine
    # pre-wrap references. This models a wrapper-layer capture bug.
    pristine_mul = torch.mul
    real_decorator = wrappers.torch_func_decorator

    def planted_decorator(func, func_name, property_accessor=None):
        if func_name == "tanh":
            orig = func

            def distorted(*args, **kwargs):
                return pristine_mul(orig(*args, **kwargs), 1.001)

            distorted.__name__ = getattr(orig, "__name__", "tanh")
            distorted.__qualname__ = getattr(orig, "__qualname__", "tanh")
            return real_decorator(distorted, func_name, property_accessor)
        return real_decorator(func, func_name, property_accessor)

    wrappers.torch_func_decorator = planted_decorator

    class TanhModel(nn.Module):
        def forward(self, x):
            return torch.tanh(x) + 1.0

    model = TanhModel().eval()
    x = torch.randn(4, 8)

    # Step 1: run a capture first, so the process is in the WRAPPED state
    # (the probe's capture-then-validate arm -- the one that validated
    # clean pre-fix).
    trace = tl.trace(model, x.clone())

    # Step 2: validate the same model in the wrapped process.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        verdict = tl.validate(model, x.clone(), scope="forward", validate_metadata=False)
    print(f"VERDICT={verdict}")
    """
)

_CLEAN_SCRIPT = _PROBE_SCRIPT.replace(
    "wrappers.torch_func_decorator = planted_decorator",
    "# no plant: positive control",
)


def _run_probe(script: str) -> str:
    """Run a probe script in a fresh interpreter and return its stdout."""

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return result.stdout


@pytest.mark.heavy
def test_planted_wrapper_distortion_fails_validation_in_a_wrapped_process() -> None:
    """The b9-fable probe, now caught: wrapped-process validation FAILS.

    Red-capable: pre-fix this exact script printed VERDICT=True (the
    distorted tanh corrupted ground truth, capture, and replay
    identically), and only a fresh-process validation caught the bug.
    """

    stdout = _run_probe(_PROBE_SCRIPT)
    assert "VERDICT=False" in stdout, stdout


@pytest.mark.heavy
def test_clean_wrapped_process_validation_still_passes() -> None:
    """Positive control: no plant, same wrapped-process flow, verdict True."""

    stdout = _run_probe(_CLEAN_SCRIPT)
    assert "VERDICT=True" in stdout, stdout


def test_pristine_oracle_unwraps_and_restores_wrap_state() -> None:
    """Inside the context torch is pristine; after it, wrappers return."""

    model = nn.Sequential(nn.Linear(4, 4), nn.Tanh()).eval()
    tl.trace(model, torch.randn(2, 4))
    assert _state._is_decorated
    wrapped_tanh = torch.tanh
    with pristine_torch_oracle() as pristine:
        assert pristine
        assert not _state._is_decorated
        assert torch.tanh is not wrapped_tanh
    assert _state._is_decorated
    # The reinstalled wrapper intercepts again (identity may differ; the
    # ledger keyed restore is what matters).
    assert id(torch.tanh) in _state._decorated_to_orig


def test_pristine_oracle_is_noop_when_never_wrapped() -> None:
    """A never-wrapped process yields pristine=True without touching state."""

    script = textwrap.dedent(
        """
        import torch
        from torchlens import _state
        from torchlens.validation._pristine import pristine_torch_oracle

        assert not _state._is_decorated
        before = torch.tanh
        with pristine_torch_oracle() as pristine:
            assert pristine
            assert torch.tanh is before
        assert not _state._is_decorated
        print("NOOP-OK")
        """
    )
    assert "NOOP-OK" in _run_probe(script)


def test_backward_validation_runs_pristine_stock_pass() -> None:
    """Backward validation's stock pass also settles under the pristine root.

    Functional pin: in a wrapped process, validate_backward_pass still
    passes on an honest model (the unwrap/rewrap cycle is transparent), and
    the wrappers are back afterwards.
    """

    from torchlens.validation.backward import validate_backward_pass

    model = nn.Sequential(nn.Linear(4, 4), nn.Tanh()).eval()
    tl.trace(model, torch.randn(2, 4))
    assert _state._is_decorated
    assert validate_backward_pass(
        model,
        torch.randn(2, 4),
        random_seed=3,
        validate_metadata=False,
    )
    assert _state._is_decorated
