"""Forward-only execution, input, and render fixtures for the real-model suite."""

import os
from collections.abc import Iterator
from os.path import join as opj
from typing import Any

import pytest
import torch

VIS_OUTPUT_DIR = opj(os.environ["TORCHLENS_TEST_OUTPUTS_DIR"], "visualizations")
_REAL_MODEL_BATCH_SIZE = 2


def _assert_native_graph_preprocessing_refused(
    model: torch.nn.Module, inputs: tuple[Any, ...]
) -> None:
    """Require numeric agreement and honest refusal of unwrapped PyG native graph ops.

    Parameters
    ----------
    model:
        Model with native neighbor/triplet preprocessing outside the eager inventory.
    inputs:
        Original model inputs used for the independent output comparison.
    """
    from torchlens import trace as trace_fn
    from torchlens._user_public_impls import _validate_forward_pass_torch
    from torchlens.validation.diagnostics import get_validation_failure
    from torchlens.validation.invariants import MetadataInvariantError, check_metadata_invariants

    with pytest.warns(
        UserWarning, match="TorchLens found tensor arguments with no graph/source provenance"
    ):
        captured = trace_fn(model, inputs)
    try:
        torch.testing.assert_close(
            captured[captured.output_layers[0]].out, model(*inputs), rtol=0, atol=0
        )
        with pytest.raises(MetadataInvariantError, match="graph_connectivity.*unattributed"):
            check_metadata_invariants(captured)
    finally:
        captured.cleanup()

    observed = []

    def observe(trace: Any) -> None:
        """Keep native output and metadata checks while refusing a completeness claim."""
        torch.testing.assert_close(
            trace[trace.output_layers[0]].out, model(*inputs), rtol=0, atol=0
        )
        failure = get_validation_failure(trace)
        assert failure is not None and failure.check == "bfs_completeness"
        assert failure.extra["dispatch_op_count"] > failure.extra["captured_op_count"]
        assert any(row["reason"] == "unowned_dispatch" for row in trace.completeness_diagnostics)
        assert trace.validation_replay_status.state in {"failed", "unverified"}
        observed.append(failure.check)

    with pytest.warns(
        UserWarning, match="TorchLens found tensor arguments with no graph/source provenance"
    ):
        # Test the independent dispatch backstop after separately proving that
        # metadata validation rejects the unowned native computation.
        assert not _validate_forward_pass_torch(
            model, inputs, validate_metadata=False, _trace_observer=observe
        )
    assert observed == ["bfs_completeness"]


@pytest.fixture(name="default_input1")
def _reference_input1(default_input1: torch.Tensor) -> torch.Tensor:
    """Keep two samples for model/BatchNorm coverage without retaining six payloads.

    Parameters
    ----------
    default_input1:
        The parent conftest's image fixture.

    Returns
    -------
    torch.Tensor
        The same images and resolution with the reference batch size.
    """
    return default_input1[:_REAL_MODEL_BATCH_SIZE]


@pytest.fixture(name="default_input2")
def _reference_input2(default_input2: torch.Tensor) -> torch.Tensor:
    """Return the second image fixture at the same reference batch size.

    Parameters
    ----------
    default_input2:
        The parent conftest's independent image fixture.

    Returns
    -------
    torch.Tensor
        The second set of reference images.
    """
    return default_input2[:_REAL_MODEL_BATCH_SIZE]


@pytest.fixture(autouse=True)
def _forward_only_reference_kernels(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Avoid retaining unused autograd graphs and use reproducible SDPA replay.

    These tests validate forward values and render graphs; none runs backward.
    The math kernel also avoids CPU flash-attention rounding differences when
    replay reconstructs an attention argument with a different memory stride.
    HF's scripted helpers are imported as eager Python so their internal tensor
    operations belong to the eager graph this file tests.

    Parameters
    ----------
    monkeypatch:
        Restore JIT's compilation switch after each forward-only test.

    Yields
    ------
    None
        Forward-only execution with the reference attention implementation.
    """
    monkeypatch.setattr(torch.jit._state._enabled, "enabled", False)
    from torchvision.models.detection.image_list import ImageList

    from torchlens.ir.container import _CONTAINER_REGISTRY, RegisteredContainer

    # Detection transforms return this custom tensor holder. Declare its exact
    # state with the existing container adapter instead of guessing BFS paths.
    monkeypatch.setitem(
        _CONTAINER_REGISTRY,
        ImageList,
        RegisteredContainer(
            lambda value: ([value.tensors], value.image_sizes),
            lambda sizes, children: ImageList(children[0], sizes),
            state_complete=True,
        ),
    )
    # Reduced-precision matmul can choose BF16 for a strided batched input
    # but FP32 for its contiguous replay clone. Use one numeric reference.
    prior_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        with torch.no_grad(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            yield
    finally:
        torch.set_float32_matmul_precision(prior_precision)


def _clear_render_pdf(*path_parts: str) -> None:
    """Remove any stale PDF artifact before a render so it cannot mask a no-op."""
    import os

    path = opj(VIS_OUTPUT_DIR, *path_parts) + ".pdf"
    if os.path.exists(path):
        os.remove(path)


def _assert_render_pdf(*path_parts: str) -> None:
    """Assert a ``show_model_graph`` run wrote a fresh non-empty PDF artifact.

    Cheap render-output check for tests that render but otherwise assert nothing
    (or skip forward-pass validation). A no-op renderer leaves no file (or an
    empty one), which this catches. ``path_parts`` are the ``vis_outpath`` parts
    (without extension); the default ``vis_fileformat`` is ``pdf``. Pair with
    :func:`_clear_render_pdf` before the render so a stale file cannot mask a no-op.
    """
    import os

    path = opj(VIS_OUTPUT_DIR, *path_parts) + ".pdf"
    assert os.path.exists(path), f"expected visualization artifact at {path}"
    with open(path, "rb") as fh:
        head = fh.read(4)
    assert head == b"%PDF", f"expected {path} to be a non-empty PDF artifact"
