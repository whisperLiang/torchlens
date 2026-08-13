"""Capture-fidelity census harness skeleton (merge-ranks C0 -> C2).

The census is the relaxation gate of the merge-ranks tier: per candidate
topology, run the workload twice -- BARE under a reference dispatch logger
(ground-truth aten stream) and under FULL TorchLens capture -- and require
(design-merge-ranks-c v5, 5.2):

1. bit-identical model outputs bare-vs-captured (non-perturbation, including
   the copies the active save policy takes);
2. every ground-truth dispatched op accounted for in the trace;
3. every op in any of the five enumerated collective namespaces carrying its
   own boundary record, a typed unobserved fallback, or an explicit discharge
   link, with the no-double-tick seq invariant;
4. total plane-S/plane-P linkage.

C0 lands this SKELETON: the reference dispatch logger, the bare-vs-captured
runner, and criterion 1. Criteria 2-4 are C2 work (they need plane-P capture
to exist) and raise ``NotImplementedError`` so a future green census can
never be vacuous about which criteria actually ran.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch.utils._python_dispatch import TorchDispatchMode


class ReferenceDispatchLogger(TorchDispatchMode):
    """Ground-truth aten stream recorder for the bare census leg."""

    def __init__(self) -> None:
        super().__init__()
        self.ops: list[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # noqa: ANN001
        self.ops.append(str(func))
        return func(*args, **kwargs)


@dataclass
class CensusResult:
    """Outcome of one census run for one topology/workload."""

    outputs_bit_identical: bool
    ground_truth_ops: list[str] = field(default_factory=list)
    criteria_run: tuple[int, ...] = (1,)
    failures: list[str] = field(default_factory=list)

    @property
    def green(self) -> bool:
        """Green ONLY for the criteria that actually ran; never vacuous."""

        return not self.failures


def _flatten_tensors(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        found: list[torch.Tensor] = []
        for item in value:
            found.extend(_flatten_tensors(item))
        return found
    if isinstance(value, dict):
        found = []
        for item in value.values():
            found.extend(_flatten_tensors(item))
        return found
    return []


def run_census_criterion_1(
    model_factory: Callable[[], torch.nn.Module],
    input_factory: Callable[[], Any],
    seed: int = 1234,
    capture_kwargs: dict[str, Any] | None = None,
) -> CensusResult:
    """Run the bare-vs-captured bit-identity leg (criterion 1).

    Parameters
    ----------
    model_factory / input_factory:
        Deterministic constructors; called once per leg under the same seed
        so both legs see identical parameters and inputs.
    seed:
        RNG seed applied before each leg.
    capture_kwargs:
        Extra ``tl.trace`` kwargs (e.g. a representative ``save=`` + witness
        configuration -- the census must prove non-perturbation INCLUDING the
        copies the save policy takes).

    Returns
    -------
    CensusResult
        Criterion-1 verdict plus the recorded ground-truth aten stream (kept
        for the C2 criteria).
    """

    import torchlens as tl

    torch.manual_seed(seed)
    bare_model = model_factory()
    bare_input = input_factory()
    logger = ReferenceDispatchLogger()
    with torch.no_grad(), logger:
        bare_out = bare_model(bare_input)

    torch.manual_seed(seed)
    captured_model = model_factory()
    captured_input = input_factory()
    log = tl.trace(captured_model, captured_input, **(capture_kwargs or {}))

    failures: list[str] = []
    bare_tensors = _flatten_tensors(bare_out)
    captured_tensors = [op.out for op in log.output_ops]
    if len(bare_tensors) != len(captured_tensors):
        failures.append(
            f"output arity differs: bare {len(bare_tensors)} vs captured {len(captured_tensors)}"
        )
    else:
        for index, (bare_tensor, captured_tensor) in enumerate(zip(bare_tensors, captured_tensors)):
            if not torch.equal(bare_tensor, captured_tensor):
                failures.append(f"output {index} not bit-identical")
    return CensusResult(
        outputs_bit_identical=not failures,
        ground_truth_ops=logger.ops,
        criteria_run=(1,),
        failures=failures,
    )


def run_census_criterion_2(*_args: Any, **_kwargs: Any) -> None:
    """Plane-P completeness: every ground-truth op accounted for. C2 work."""

    raise NotImplementedError(
        "census criterion 2 requires plane-P dispatcher capture (merge-ranks C2)"
    )


def run_census_criterion_3(*_args: Any, **_kwargs: Any) -> None:
    """Collective-namespace accounting + no-double-tick. C2 work."""

    raise NotImplementedError(
        "census criterion 3 requires plane-P dispatcher capture (merge-ranks C2)"
    )


def run_census_criterion_4(*_args: Any, **_kwargs: Any) -> None:
    """Total plane-S/plane-P linkage. C2 work."""

    raise NotImplementedError(
        "census criterion 4 requires plane-P dispatcher capture (merge-ranks C2)"
    )
