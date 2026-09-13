"""Split capture binds known held Torch functions without changing caller state."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest
import torch

from torchlens.utils._subprocess import run_bounded_subprocess


@pytest.mark.heavy
@pytest.mark.parametrize("activation", ["gelu", "silu"])
def test_fresh_split_captures_held_activation_once_per_batch(
    activation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pre-wrap activation references produce real ops in both batch captures.

    Parameters
    ----------
    activation:
        Torch functional activation held before TorchLens installs its wrappers.
    """

    source = textwrap.dedent(
        """
        import sys
        import torch
        import torch.nn.functional as functional

        calls = []
        activation_name = sys.argv[1]
        held = getattr(functional, activation_name)

        class HeldActivation(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.act = held

            def forward(self, values):
                assert torch.overrides._get_current_function_mode() is None
                return self.act(values)

        class WindowBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(4, 8)
                self.activation = HeldActivation()
                self.fc2 = torch.nn.Linear(8, 4)

            def forward(self, values):
                calls.append(int(values.shape[0]))
                windows = values.reshape(values.shape[0] * 4, 3, 4)
                return self.fc2(self.activation(self.fc1(windows))) + windows

        torch.manual_seed(0)
        model = WindowBlock().eval()
        assert model.activation.act is held
        # This environment may lazily load Triton from Torch's Dynamo path;
        # initialize that native extension before TorchLens installs wrappers.
        from torchlens.utils._torch_compat import get_dynamo_optimized_module_type

        get_dynamo_optimized_module_type(force_probe=True)
        import torchlens as tl
        example = torch.randn(2, 4, 3, 4)

        for index in range(2):
            runtime = tl.split.prepare(
                model, example, tl.split.SplitRequest(point=tl.split.percent(50))
            )
            assert calls == [1, 2] * (index + 1), calls
            assert model.activation.act is held
            assert runtime.batch_validation["status"] == "passed", runtime.batch_validation
            assert runtime.trace.rescue_rerun is None
            kinds = [node.op_type for node in runtime.trace_graph.compute_nodes]
            assert activation_name in kinds, kinds
            assert "internalsource" not in kinds, kinds

        for batch in (1, 2, 3):
            values = torch.randn(batch, 4, 3, 4)
            with torch.no_grad():
                expected = model(values)
                actual = runtime.replay(values)
            assert actual.shape == (batch * 4, 3, 4)
            torch.testing.assert_close(actual, expected)
            assert model.activation.act is held
        print("held-activation-split-ok")
        """
    )
    monkeypatch.setenv("DEBUG", "0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    result = run_bounded_subprocess(
        [sys.executable, "-c", source, activation],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, (
        f"held {activation} subprocess failed ({result.returncode})\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "held-activation-split-ok" in result.stdout


@pytest.mark.smoke
@pytest.mark.parametrize("fail", [False, True])
def test_held_function_scope_restores_direct_reference(fail: bool) -> None:
    """Scope teardown preserves the original function even when capture raises.

    Parameters
    ----------
    fail:
        Whether the scope body exits by raising a sentinel exception.
    """

    from torchlens.backends.torch._held_refs import scoped_held_torch_function_refs

    model = torch.nn.Module()
    held = torch._C._nn.gelu
    model.act = held
    original_container = [held]
    model.held_list = original_container
    values = torch.tensor([-1.0, 0.0, 2.0])
    expected = held(values)
    error = RuntimeError("held-function capture failed")
    try:
        with scoped_held_torch_function_refs(model):
            assert model.act is torch.nn.functional.gelu
            assert model.act is not held
            assert model.held_list is original_container and model.held_list[0] is held
            torch.testing.assert_close(model.act(values), expected)
            if fail:
                raise error
    except RuntimeError as caught:
        assert fail and caught is error
    else:
        assert not fail
    assert model.act is held
    assert model.held_list is original_container and model.held_list[0] is held
    torch.testing.assert_close(model.act(values), expected)


@pytest.mark.smoke
def test_held_function_scope_keeps_user_reassignment() -> None:
    """A forward's deliberate reassignment is not overwritten by scope teardown."""

    from torchlens.backends.torch._held_refs import scoped_held_torch_function_refs

    model = torch.nn.Module()
    model.act = torch._C._nn.gelu
    replacement = torch.nn.functional.relu
    with scoped_held_torch_function_refs(model):
        assert model.act is torch.nn.functional.gelu
        model.act = replacement
    assert model.act is replacement


@pytest.mark.smoke
def test_held_function_scope_ignores_foreign_namespace_patches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only an installed wrapper counterpart can replace a held known function.

    Parameters
    ----------
    monkeypatch:
        Restore a simulated third-party namespace patch after the check.
    """

    from torchlens.backends.torch._held_refs import scoped_held_torch_function_refs

    model = torch.nn.Module()
    held = torch._C._nn.gelu
    model.act = held

    def custom(values: torch.Tensor) -> torch.Tensor:
        """Return inputs unchanged without being a registered Torch function.

        Parameters
        ----------
        values:
            Input tensor to return.

        Returns
        -------
        torch.Tensor
            The unchanged input tensor.
        """

        return values

    model.custom = custom
    monkeypatch.setattr(torch.nn.functional, "gelu", custom)
    with scoped_held_torch_function_refs(model):
        assert model.act is held
        assert model.custom is custom
    assert model.act is held
    assert model.custom is custom
