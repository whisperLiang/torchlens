"""Validation helpers for TorchLens split runtimes."""

from __future__ import annotations

from typing import Any

import torch

from .boundary import ReplayBoundary


def canonical_boundary_signature(runtime: Any) -> tuple[tuple[Any, ...], ...]:
    """Return device-neutral canonical boundary signature."""

    rows = []
    for label, spec in runtime.boundary_spec.items():
        rows.append((label, spec.canonical_id, spec.shape, spec.dtype, spec.role))
    return tuple(rows)


def assert_canonical_abi_equivalent(left: Any, right: Any) -> None:
    """Assert two runtimes have the same device-neutral boundary ABI."""

    if canonical_boundary_signature(left) != canonical_boundary_signature(right):
        raise AssertionError("Split canonical boundary ABI differs.")


def validate_dynamic_batches(
    runtime: Any,
    input_factory: Any,
    batch_sizes: tuple[int, ...] = (1, 2, 4, 8),
    *,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> None:
    """Validate split replay equivalence for multiple runtime batch sizes."""

    for batch_size in batch_sizes:
        inputs = input_factory(batch_size)
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        if not runtime.validate_equivalence(runtime.model, inputs, atol=atol, rtol=rtol):
            raise AssertionError(f"Split replay mismatch for batch size {batch_size}.")


def boundary_liveness_zero_probe(
    runtime: Any,
    boundary: ReplayBoundary,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> dict[str, bool]:
    """Probe whether each boundary tensor affects suffix output."""

    baseline = runtime.run_suffix(boundary)
    live: dict[str, bool] = {}
    for tensor_id in boundary.tensors:
        candidate = boundary.clone()
        candidate.tensors[tensor_id].zero_()
        output = runtime.run_suffix(candidate)
        live[tensor_id] = not _tree_allclose(baseline, output, atol=atol, rtol=rtol)
    return live


def _tree_allclose(left: Any, right: Any, *, atol: float, rtol: float) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return bool(torch.allclose(left, right, atol=atol, rtol=rtol))
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)) and len(left) == len(right):
        return all(_tree_allclose(a, b, atol=atol, rtol=rtol) for a, b in zip(left, right, strict=True))
    if isinstance(left, dict) and isinstance(right, dict) and set(left) == set(right):
        return all(_tree_allclose(left[key], right[key], atol=atol, rtol=rtol) for key in left)
    return left == right


__all__ = [
    "assert_canonical_abi_equivalent",
    "boundary_liveness_zero_probe",
    "canonical_boundary_signature",
    "validate_dynamic_batches",
]
