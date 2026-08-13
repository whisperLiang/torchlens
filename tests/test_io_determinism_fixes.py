"""Regression tests for portable artifact determinism and round-trip fidelity."""

from __future__ import annotations

import json
from pathlib import Path

import torch

import torchlens as tl


class _LinearModel(torch.nn.Module):
    """Tiny parameterized model used by deterministic artifact tests."""

    def __init__(self) -> None:
        """Initialize one deterministic linear layer."""

        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Apply the test layer.

        Parameters
        ----------
        value:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Layer output.
        """

        return self.linear(value)


class _ComplexModel(torch.nn.Module):
    """Tiny complex-valued model used by payload transport tests."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Return a supported complex64 activation.

        Parameters
        ----------
        value:
            Complex input tensor.

        Returns
        -------
        torch.Tensor
            Complex activation.
        """

        return value * (1 + 0j)


def _save_seeded_trace(path: Path, *, random_seed: int) -> tl.Trace:
    """Capture, save, and reload one trace with a chosen capture seed.

    Parameters
    ----------
    path:
        Destination bundle path.
    random_seed:
        Capture seed controlling the live parameter barcode.

    Returns
    -------
    tl.Trace
        Reloaded portable trace.
    """

    torch.manual_seed(0)
    trace = tl.trace(_LinearModel(), torch.ones(1, 2), random_seed=random_seed)
    tl.save(trace, path)
    return tl.load(path)


def test_save_scrub_remaps_process_local_identity_tokens(tmp_path: Path) -> None:
    """Equivalent traces persist dense identities independent of live ids and barcodes."""

    first_path = tmp_path / "first.tlspec"
    second_path = tmp_path / "second.tlspec"
    first = _save_seeded_trace(first_path, random_seed=1)
    second = _save_seeded_trace(second_path, random_seed=2)

    first_param_ops = [op for op in first.layer_list if op.uses_params]
    second_param_ops = [op for op in second.layer_list if op.uses_params]
    assert [op._param_barcodes for op in first_param_ops] == [
        op._param_barcodes for op in second_param_ops
    ]
    assert [op.equivalence_class for op in first_param_ops] == [
        op.equivalence_class for op in second_param_ops
    ]
    assert first.model_object_id == second.model_object_id == 1
    assert first.input_object_id == second.input_object_id == 1

    first_manifest = json.loads((first_path / "manifest.json").read_text(encoding="utf-8"))
    second_manifest = json.loads((second_path / "manifest.json").read_text(encoding="utf-8"))
    assert [site["op_kind"] for site in first_manifest["sites"]] == [
        site["op_kind"] for site in second_manifest["sites"]
    ]


def test_save_scrub_remaps_autograd_identity_joins(tmp_path: Path) -> None:
    """Autograd ids become dense while all persisted graph joins remain valid."""

    value = torch.ones(1, 2, requires_grad=True)
    trace = tl.trace(_LinearModel(), value, backward_ready=True)
    trace.log_backward(trace.output_ops[0].out.sum())
    path = tmp_path / "backward.tlspec"
    tl.save(trace, path)

    loaded = tl.load(path)
    expected_ids = list(range(1, len(loaded.grad_fn_order) + 1))
    assert loaded.grad_fn_order == expected_ids
    assert list(loaded.grad_fn_logs) == expected_ids
    assert set(loaded.backward_root_grad_fn_object_ids) <= set(expected_ids)
    assert all(
        grad_fn.grad_fn_object_id == grad_fn_id
        and set(grad_fn.next_grad_fn_ids) <= set(expected_ids)
        for grad_fn_id, grad_fn in loaded.grad_fn_logs.items()
    )


def test_bundle_writer_resolves_lazy_conjugate_payloads(tmp_path: Path) -> None:
    """Saved tensor bytes represent the logical value, not lazy physical storage."""

    trace = tl.trace(
        _ComplexModel(),
        torch.tensor([1 + 2j], dtype=torch.complex64),
        layers_to_save="all",
        activation_transform=lambda value: value.conj(),
    )
    expected = trace.output_ops[0].transformed_out.clone()
    path = tmp_path / "conjugate.tlspec"
    tl.save(trace, path)

    loaded = tl.load(path)
    assert torch.equal(loaded.output_ops[0].transformed_out, expected)
