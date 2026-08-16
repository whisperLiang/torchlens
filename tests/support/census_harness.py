"""Capture-fidelity census harness (merge-ranks C0 -> C2/C3; wave-0 build).

The census is the relaxation gate of the merge-ranks tier: per candidate
topology, run the workload twice -- BARE under the reference observers
(ground-truth streams) and under FULL TorchLens capture -- and require
(design-merge-ranks-c v5, 5.2 + the L8 census plan sec 1.1):

1. K1 non-perturbation: bit-identical outputs AND final parameter/buffer
   bytes AND torch/python RNG engine states, clean-bare vs captured;
2. K2 plane-P completeness: every mode-visible op (any namespace) accounted
   for in the trace;
3. K3 collective + completion accounting over the UNION of the mode stream
   and the dispatcher-interposition inventory, with the no-double-tick seq
   invariant;
4. K4 total plane-S/plane-P linkage.

Ground truth is DUAL-CHANNEL: a ``TorchDispatchMode`` provably does NOT see
ACT-triggered ``wait_tensor`` (plan probe P2), so the bare leg runs the
mode logger AND dispatcher-level ``torch.library.Library(..., "IMPL")``
interposition counters over completion ops. K1's baseline is a THIRD, fully
clean bare run; a harness self-check pins that the instrumented bare leg
matches the clean leg bit-identically, which simultaneously proves the
counters non-perturbing.

Wave 0 (this build) lands: the row registry, the widened ``CensusResult``
with the full-conjunction ``row_green`` gate, the dual-channel bare leg,
criterion 1 (widened), the refusal-row runner, and the report generator.
Criteria 2-4 are C2 work (they need plane-P capture) and raise
``NotImplementedError`` so a green can never be vacuous about which criteria
ran. The wave-0 per-row product is named "ZI baseline + criterion-1 green"
and is NEVER called row green (plan rule 1.1(1) + phasing note).
"""

from __future__ import annotations

import contextlib
import json
import random
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode

# --------------------------------------------------------------------------
# Typed harness failure strings (harness-internal vocabulary, NOT public API;
# plan rule 1.1(1): floor and ZI misses land in ``failures`` so no green path
# can bypass them even if a caller reads ``failures`` alone).
# --------------------------------------------------------------------------
CONTENT_FLOOR_MISSING = "content_floor_missing"
COMPLETION_FLOOR_MISSING = "completion_floor_missing"
GEOMETRY_FLOOR_MISSING = "geometry_floor_missing"
ZI_GATE_FAILED = "zi_gate_failed"

#: The full-criteria conjunction row_green requires (plan rule 1.1(1)).
FULL_CRITERIA: tuple[int, ...] = (1, 2, 3, 4)

#: The honest wave-0 per-row product name (plan 1.2 phasing note + 1.4).
WAVE0_PRODUCT_NAME = "ZI baseline + criterion-1 green"

#: Product name for STAYS-REFUSED rows asserted red-by-typed-refusal.
REFUSAL_PRODUCT_NAME = "red-by-typed-refusal (refusal stays)"

#: Completion ops the dispatcher-interposition channel counts in wave 0.
#: ``wait_tensor`` is the op the mode provably cannot see (probe P2/P3a);
#: the five-namespace breadth arrives with the wave-1 criterion-3 bodies.
#: Declared here so the covered set is never a silent cap.
INTERPOSED_COMPLETION_OPS: tuple[str, ...] = ("_c10d_functional::wait_tensor",)

#: The class of ops invisible to BOTH ground-truth channels, disclosed
#: verbatim in every census report (plan 1.1, NOT-COVERED line).
NOT_COVERED_LINE = (
    "NOT COVERED: an op invisible to the TorchDispatchMode AND outside the "
    f"interposed completion-op set {INTERPOSED_COMPLETION_OPS!r} is invisible "
    "to BOTH ground-truth channels; the census cannot claim it."
)


class ReferenceDispatchLogger(TorchDispatchMode):
    """Ground-truth op-stream recorder for the bare census leg.

    Records ``str(func)`` for EVERY dispatched op -- any namespace, not aten
    alone -- so K2's any-namespace accounting is what it already measures.
    """

    def __init__(self) -> None:
        super().__init__()
        self.ops: list[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # noqa: ANN001
        self.ops.append(str(func))
        return func(*args, **kwargs)


# --------------------------------------------------------------------------
# Row registry: the plan-1.2 topology matrix. Rows are DEFINITIONS with their
# criteria obligations; execution wave and gating ride each spec so the report
# generator can emit honest NOT-RUN disclosures for everything that did not run.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CensusRowSpec:
    """One row of the census topology matrix (plan sec 1.2)."""

    row_id: str
    group: str
    title: str
    world: str
    disposition: str
    wave: int
    runnable_wave0: bool
    gated_on: tuple[str, ...] = ()
    floors: tuple[str, ...] = ()
    cuda_only: bool = False


def _rows(*specs: CensusRowSpec) -> dict[str, CensusRowSpec]:
    return {spec.row_id: spec for spec in specs}


CENSUS_ROWS: dict[str, CensusRowSpec] = _rows(
    # Group A -- controls / already-relaxed scope.
    CensusRowSpec(
        "A1",
        "A",
        "dense single-process, no dist init, armed ON",
        "1proc",
        "must stay green forever (zero-interference anchor)",
        0,
        True,
        floors=("content",),
    ),
    CensusRowSpec(
        "A2",
        "A",
        "dense explicit c10d collectives (sync + async_op + coalescing)",
        "W2",
        "shipped C1 scope; full-criteria row-green required before any C2 relaxation PR",
        0,
        True,
        floors=("content", "completion"),
    ),
    CensusRowSpec(
        "A3",
        "A",
        "plain p2p send/recv + isend/irecv incl. tags",
        "W2",
        "never a CAPTURE-refusal kind (merge is C3 -- D1)",
        0,
        True,
        floors=("content",),
    ),
    # Group B -- C2 relaxation candidates (eligibility evidence; the narrowing
    # itself additionally requires the D-L8-CAP ruling).
    CensusRowSpec(
        "B1",
        "B",
        "plain DTensor: distribute_tensor/module, Shard+Replicate, redistribute",
        "W2",
        "RELAX if green (gated on D-L8-CAP)",
        1,
        False,
        gated_on=("wave-1 criteria 2-4", "D-L8-CAP"),
        floors=("content", "completion", "geometry"),
    ),
    CensusRowSpec(
        "B2",
        "B",
        "ColwiseParallel via parallelize_module",
        "W2",
        "RELAX if green (gated on D-L8-CAP); historical vacuous row",
        1,
        False,
        gated_on=("wave-1 criteria 2-4", "D-L8-CAP"),
        floors=("content", "completion", "geometry"),
    ),
    CensusRowSpec(
        "B3",
        "B",
        "RowwiseParallel",
        "W2",
        "RELAX if green (gated on D-L8-CAP)",
        1,
        False,
        gated_on=("wave-1 criteria 2-4", "D-L8-CAP"),
        floors=("content", "completion", "geometry"),
    ),
    CensusRowSpec(
        "B4",
        "B",
        "SequenceParallel (forward reduce_scatter witness-pin source)",
        "W2",
        "RELAX if green (gated on D-L8-CAP)",
        1,
        False,
        gated_on=("wave-1 criteria 2-4", "D-L8-CAP"),
        floors=("content", "completion", "geometry"),
    ),
    CensusRowSpec(
        "B5",
        "B",
        "Colwise+Rowwise MLP pair (canonical TP block)",
        "W2",
        "RELAX if green (gated on D-L8-CAP); composition is its own row",
        1,
        False,
        gated_on=("wave-1 criteria 2-4", "D-L8-CAP"),
        floors=("content", "completion", "geometry"),
    ),
    CensusRowSpec(
        "B6",
        "B",
        "FSDP2 fully_shard, forward (the c10d:: discharge row)",
        "W2",
        "RELAX if green (gated on D-L8-CAP)",
        1,
        False,
        gated_on=("wave-1 criteria 2-4", "D-L8-CAP"),
        floors=("content", "completion", "geometry"),
    ),
    CensusRowSpec(
        "B7",
        "B",
        "shard_dim_alltoall (redistribute Shard(0)->Shard(1))",
        "W2",
        "rides B1",
        1,
        False,
        gated_on=("wave-1 criteria 2-4", "D-L8-CAP"),
        floors=("content", "completion", "geometry"),
    ),
    CensusRowSpec(
        "B8",
        "B",
        "TP + FSDP2 2D on a 2x2 mesh",
        "W4",
        "RELAX only if independently green; runs ONLY if B2/B3/B6 green",
        1,
        False,
        gated_on=("B2", "B3", "B6", "D-L8-CAP"),
        floors=("content", "completion", "geometry"),
    ),
    # Group C -- refusal parity (asserted red-by-typed-refusal, never skipped).
    CensusRowSpec(
        "C1r",
        "C",
        "ShardedTensor (legacy)",
        "1proc",
        "refuses via monolithic dtensor finding with ShardedTensor variant tag",
        0,
        True,
    ),
    CensusRowSpec(
        "C2r",
        "C",
        "PrepareModuleInput/Output",
        "1proc",
        "refuses via monolithic tensor_parallel",
        0,
        True,
    ),
    CensusRowSpec(
        "C3r",
        "C",
        "synthetic TP-ish hook not attributable to an identified style",
        "1proc",
        "FAILS CLOSED typed (S3 fail-closed leg)",
        0,
        True,
    ),
    CensusRowSpec(
        "C4r",
        "C",
        "PP composed with TP/FSDP2 (3D)",
        "1proc",
        "monolithic refusal today; pp_composition_unsupported lands with the C2 split (F5)",
        0,
        True,
    ),
    CensusRowSpec(
        "C5r",
        "C",
        "merge_ranks over a member with distributed_scope == rank_local_shard",
        "n/a",
        "trivially red-by-construction pre-relaxation (marker not shipped); "
        "asserted via marker absence until the first C2 capture relaxation",
        0,
        True,
    ),
    # Group D -- C3 schedule rows (wave 2).
    CensusRowSpec(
        "D1",
        "D",
        "hand-rolled determinate-peer pipeline (plain p2p)",
        "W2",
        "capture-legal today; MERGE relaxation = D9 (default: refusal stays)",
        2,
        False,
        gated_on=("wave-2", "D9 for the merge-side relaxation"),
        floors=("content",),
    ),
    CensusRowSpec(
        "D2",
        "D",
        "torch.distributed.pipelining, 2-stage GPipe inference schedule",
        "W2",
        "RELAX pipeline_parallel for this schedule class if green + D-L8-CAP",
        2,
        False,
        gated_on=("wave-2", "S4", "D-L8-CAP"),
        floors=("content", "completion"),
    ),
    CensusRowSpec(
        "D3",
        "D",
        "interleaved 1F1B training schedule",
        "W2",
        "RELAX only for census-proven schedule classes, under D-L8-CAP",
        2,
        False,
        gated_on=("wave-2", "S4", "D-L8-CAP"),
        floors=("content", "completion"),
    ),
    # Group N -- census self-honesty (each pins a criterion's red).
    CensusRowSpec(
        "N1",
        "N",
        "model factory perturbs between legs",
        "1proc",
        "K1 red (not bit-identical)",
        0,
        True,
    ),
    CensusRowSpec(
        "N2",
        "N",
        "compute-capture hole simulation (drop an aten op class)",
        "1proc",
        "K2 red; needs the wave-1 criterion-2 body",
        1,
        False,
        gated_on=("wave-1 criterion 2",),
    ),
    CensusRowSpec(
        "N3a",
        "N",
        "synthetic public-call + c10d:: double-tick construction",
        "W2",
        "K3 red (seq invariant); needs the wave-1 criterion-3 body",
        1,
        False,
        gated_on=("wave-1 criterion 3",),
    ),
    CensusRowSpec(
        "N3b",
        "N",
        "fake op injected into an allowlisted namespace at runtime",
        "1proc",
        "uncaptured_collective_op; capture ceilinged",
        0,
        True,
    ),
    CensusRowSpec(
        "N3c",
        "N",
        "funcol destination read before observed completion",
        "1proc",
        "read_of_inflight_destination present; witness not_present",
        0,
        True,
    ),
    CensusRowSpec(
        "N3d",
        "N",
        "dispatcher-interposition counter suppressed while async traffic runs",
        "1proc",
        "completion-event floor missed -> row UNVERIFIABLE",
        0,
        True,
    ),
    CensusRowSpec(
        "N4",
        "N",
        "orphan interior record injected",
        "1proc",
        "K4 red; needs the wave-1 criterion-4 body",
        1,
        False,
        gated_on=("wave-1 criterion 4",),
    ),
    CensusRowSpec(
        "N5",
        "N",
        "replicate-only DTensor capture (params AND inputs)",
        "1proc",
        "distributed_scope marker ABSENT asserted (over-labeling red)",
        1,
        False,
        gated_on=("wave-1 marker substrate",),
    ),
    # CUDA/NCCL annex -- NOT-RUN-DISCLOSED on this box (no NVIDIA GPU).
    CensusRowSpec(
        "CUDA-p2p-tags",
        "CUDA",
        "live NCCL tag-ignoring p2p pairing",
        "GPU",
        "non-gating implementation-regression leg (design 6.3)",
        3,
        False,
        cuda_only=True,
    ),
    CensusRowSpec(
        "CUDA-abort-recreate",
        "CUDA",
        "NCCL abort/recreate lifecycle",
        "GPU",
        "non-gating implementation-regression leg (design 6.3)",
        3,
        False,
        cuda_only=True,
    ),
    CensusRowSpec(
        "CUDA-coalescing-work",
        "CUDA",
        "NCCL coalescing Work-set shapes",
        "GPU",
        "non-gating implementation-regression leg (design 6.3)",
        3,
        False,
        cuda_only=True,
    ),
    CensusRowSpec(
        "CUDA-digest-witness-cost",
        "CUDA",
        "CUDA digest-sync witness cost",
        "GPU",
        "non-gating implementation-regression leg (design 6.3)",
        3,
        False,
        cuda_only=True,
    ),
    CensusRowSpec(
        "CUDA-multigpu-realism",
        "CUDA",
        "multi-GPU FSDP2/TP realism + interleaved schedules at scale",
        "GPU",
        "non-gating implementation-regression leg (design 6.3)",
        3,
        False,
        cuda_only=True,
    ),
)


# --------------------------------------------------------------------------
# The widened census result.
# --------------------------------------------------------------------------


@dataclass
class CensusResult:
    """Outcome of one census run for one topology/workload row."""

    outputs_bit_identical: bool
    ground_truth_ops: list[str] = field(default_factory=list)
    criteria_run: tuple[int, ...] = (1,)
    failures: list[str] = field(default_factory=list)
    row_id: str = ""
    world: str = ""
    floors: dict[str, str] = field(default_factory=dict)
    zi_gate_passed: bool | None = None
    completion_events: dict[str, int] = field(default_factory=dict)
    interposition_channel: str = "absent"
    refusal_kinds: tuple[str, ...] = ()
    not_run: list[str] = field(default_factory=list)

    @property
    def green(self) -> bool:
        """Green ONLY for the criteria that actually ran; never vacuous.

        Per-criterion bookkeeping only (the shipped scoped property). The
        release-grade verdict is :attr:`row_green`.
        """

        return not self.failures

    @property
    def floors_met(self) -> bool:
        """True when no applicable floor is missing."""

        return all(verdict != "missing" for verdict in self.floors.values())

    @property
    def row_green(self) -> bool:
        """The FULL rule-1.1(1) conjunction; the only release-grade green.

        ``row_green = (criteria_run == (1,2,3,4)) AND (not failures) AND
        floors_met AND zi_gate_passed``. Floor and ZI misses are ALSO typed
        entries in ``failures``, so no green path bypasses them.
        """

        return (
            self.criteria_run == FULL_CRITERIA
            and not self.failures
            and self.floors_met
            and self.zi_gate_passed is True
        )

    @property
    def product_name(self) -> str:
        """The honest per-row product label the report carries."""

        if self.refusal_kinds:
            return REFUSAL_PRODUCT_NAME if self.green else "red (refusal missing)"
        if self.row_green:
            return "row green"
        if self.criteria_run == (1,) and self.green and self.zi_gate_passed is True:
            return WAVE0_PRODUCT_NAME
        return "red" if self.failures else "partial (not row green)"


# --------------------------------------------------------------------------
# Dual-channel ground truth: dispatcher-level interposition counters.
# --------------------------------------------------------------------------


class InterpositionTeardownError(RuntimeError):
    """A leaked interposition registration ticked after teardown."""


def _run_wait_tensor_probe() -> None:
    """Issue one tiny funcol collective and materialize it (ticks wait_tensor)."""

    import torch.distributed as dist
    import torch.distributed._functional_collectives as funcol

    probe = funcol.all_reduce(torch.ones(1), "sum", dist.group.WORLD)
    _ = probe + 0.0


@contextlib.contextmanager
def dispatcher_interposition_counters(
    *, verify_teardown: bool = True, suppress: bool = False
) -> Iterator[dict[str, int]]:
    """Count completion ops at the dispatcher level (plane-W's mechanism).

    Registers ``torch.library.Library("_c10d_functional", "IMPL")`` wrappers
    for :data:`INTERPOSED_COMPLETION_OPS` at the CPU key, redispatching below
    via ``ExcludeDispatchKeyGuard`` (plan probe P3a). The registration is
    PROCESS-GLOBAL, so this context manager owns its lifetime: deregistered on
    exit via ``Library._destroy()``, with a teardown ASSERTION (a post-teardown
    probe op must not tick any counter). A leaked registration raises
    :class:`InterpositionTeardownError` -- it would perturb the very run it
    certifies.

    Parameters
    ----------
    verify_teardown:
        Run the post-teardown probe (requires an initialized process group).
    suppress:
        N3d red construction: register NOTHING while claiming the channel ran,
        so the completion-event floor is provably load-bearing.
    """

    counters = dict.fromkeys(INTERPOSED_COMPLETION_OPS, 0)
    if suppress:
        yield counters
        return

    library = torch.library.Library("_c10d_functional", "IMPL")

    def _counted_wait_tensor(tensor: torch.Tensor) -> torch.Tensor:
        counters["_c10d_functional::wait_tensor"] += 1
        op = torch.ops._c10d_functional.wait_tensor.default
        with torch._C._ExcludeDispatchKeyGuard(torch._C.DispatchKeySet(torch._C.DispatchKey.CPU)):
            return op(tensor)

    library.impl("wait_tensor", _counted_wait_tensor, "CPU")
    try:
        yield counters
    finally:
        library._destroy()
        if verify_teardown:
            import torch.distributed as dist

            if not (dist.is_available() and dist.is_initialized()):
                raise InterpositionTeardownError(
                    "teardown probe needs an initialized process group; hold the "
                    "world open through interposition teardown"
                )
            snapshot = dict(counters)
            _run_wait_tensor_probe()
            if counters != snapshot:
                raise InterpositionTeardownError(
                    f"interposition counter ticked after _destroy(): {counters} != {snapshot}"
                )


# --------------------------------------------------------------------------
# K1 state digests + RAM preflight.
# --------------------------------------------------------------------------


def _tensor_bytes_digest(tensor: torch.Tensor) -> str:
    detached = tensor.detach().contiguous().cpu().reshape(-1)
    return sha256(detached.view(torch.uint8).numpy().tobytes()).hexdigest()


def model_state_digest(model: torch.nn.Module) -> str:
    """Digest of every named parameter + buffer's exact bytes, order-stable."""

    hasher = sha256()
    for name, param in sorted(model.named_parameters()):
        hasher.update(name.encode())
        hasher.update(_tensor_bytes_digest(param).encode())
    for name, buffer in sorted(model.named_buffers()):
        hasher.update(name.encode())
        hasher.update(_tensor_bytes_digest(buffer).encode())
    return hasher.hexdigest()


def rng_state_digest() -> str:
    """Digest of the torch + python RNG engine states."""

    hasher = sha256()
    hasher.update(torch.get_rng_state().numpy().tobytes())
    hasher.update(repr(random.getstate()).encode())
    return hasher.hexdigest()


def preflight_spawn_world(world_size: int, min_available_gb: float = 4.0) -> tuple[bool, float]:
    """RAM preflight for single-host spawn worlds (CPU gloo sims).

    Returns ``(ok, available_gb)``; callers skip (never crash the box) when
    not ok. Sims are sequential per row on this box by design.
    """

    del world_size
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return True, float("inf")
    available_kb = 0
    for line in meminfo.read_text().splitlines():
        if line.startswith("MemAvailable:"):
            available_kb = int(line.split()[1])
            break
    available_gb = available_kb / (1024 * 1024)
    return available_gb >= min_available_gb, available_gb


# --------------------------------------------------------------------------
# Criterion 1 (widened) + the wave-0 row runner.
# --------------------------------------------------------------------------


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


def _compare_outputs(
    reference: list[torch.Tensor], candidate: list[torch.Tensor], label: str
) -> list[str]:
    failures: list[str] = []
    if len(reference) != len(candidate):
        failures.append(f"output arity differs ({label}): {len(reference)} vs {len(candidate)}")
        return failures
    for index, (ref, cand) in enumerate(zip(reference, candidate)):
        if not torch.equal(ref, cand):
            failures.append(f"output {index} not bit-identical ({label})")
    return failures


def run_census_criterion_1(
    model_factory: Callable[[], torch.nn.Module],
    input_factory: Callable[[], Any],
    seed: int = 1234,
    capture_kwargs: dict[str, Any] | None = None,
    interposition: bool = False,
    suppress_interposition: bool = False,
) -> CensusResult:
    """Run the K1 bit-identity leg with the dual-channel bare ground truth.

    Three legs, each freshly seeded: (0) CLEAN bare -- no instrumentation of
    any kind, the K1 baseline; (1) INSTRUMENTED bare -- the reference dispatch
    mode plus (optionally) the dispatcher-interposition counters, self-checked
    bit-identical against the clean leg (proving the observers non-perturbing);
    (2) CAPTURED -- full TorchLens capture, interposition asserted absent.

    K1 compares clean-bare vs captured on outputs AND final parameter/buffer
    bytes AND torch/python RNG engine states (v5's S2 widening applied to the
    census criterion itself).

    Parameters
    ----------
    model_factory / input_factory:
        Deterministic constructors; called once per leg under the same seed.
    seed:
        RNG seed applied before each leg.
    capture_kwargs:
        Extra ``tl.trace`` kwargs (a representative ``save=`` + witness
        configuration -- non-perturbation must be proven INCLUDING the copies
        the save policy takes).
    interposition:
        Run the dispatcher-interposition channel on the instrumented bare leg
        (requires an initialized process group for the teardown probe).
    suppress_interposition:
        N3d: claim the channel while registering nothing (red construction).
    """

    import torchlens as tl

    failures: list[str] = []
    completion_events: dict[str, int] = {}
    interposition_channel = "absent"

    # Leg 0: CLEAN bare (K1 baseline; no instrumentation of any kind).
    torch.manual_seed(seed)
    random.seed(seed)
    clean_model = model_factory()
    clean_input = input_factory()
    with torch.no_grad():
        clean_out = clean_model(clean_input)
    clean_tensors = _flatten_tensors(clean_out)
    clean_state = model_state_digest(clean_model)
    clean_rng = rng_state_digest()

    # Leg 1: INSTRUMENTED bare (mode logger + optional interposition counters).
    torch.manual_seed(seed)
    random.seed(seed)
    bare_model = model_factory()
    bare_input = input_factory()
    logger = ReferenceDispatchLogger()
    if interposition or suppress_interposition:
        interposition_channel = "suppressed" if suppress_interposition else "dispatcher"
        with dispatcher_interposition_counters(suppress=suppress_interposition) as counters:
            with torch.no_grad(), logger:
                bare_out = bare_model(bare_input)
        completion_events = dict(counters)
    else:
        with torch.no_grad(), logger:
            bare_out = bare_model(bare_input)
    failures.extend(
        _compare_outputs(
            clean_tensors,
            _flatten_tensors(bare_out),
            "instrumented-bare vs clean-bare self-check",
        )
    )

    # Leg 2: CAPTURED (interposition asserted absent: never installed here,
    # and leg 1's teardown probe proved deregistration).
    torch.manual_seed(seed)
    random.seed(seed)
    captured_model = model_factory()
    captured_input = input_factory()
    log = tl.trace(captured_model, captured_input, **(capture_kwargs or {}))
    captured_tensors = [op.out for op in log.output_ops]
    failures.extend(_compare_outputs(clean_tensors, captured_tensors, "clean-bare vs captured"))
    if model_state_digest(captured_model) != clean_state:
        failures.append("final parameter/buffer bytes differ (clean-bare vs captured)")
    if rng_state_digest() != clean_rng:
        failures.append("RNG engine states differ (clean-bare vs captured)")

    return CensusResult(
        outputs_bit_identical=not failures,
        ground_truth_ops=logger.ops,
        criteria_run=(1,),
        failures=failures,
        completion_events=completion_events,
        interposition_channel=interposition_channel,
    )


def run_census_row(
    row_id: str,
    model_factory: Callable[[], torch.nn.Module],
    input_factory: Callable[[], Any],
    *,
    seed: int = 1234,
    capture_kwargs: dict[str, Any] | None = None,
    criteria: tuple[int, ...] = (1,),
    content_floor: Callable[[list[str]], bool] | None = None,
    completion_floor: Callable[[dict[str, int]], bool] | None = None,
    zi_gate_passed: bool | None = None,
    interposition: bool = False,
    suppress_interposition: bool = False,
) -> CensusResult:
    """Run one census row to its requested criteria with floor enforcement.

    Wave-0 rows request ``criteria=(1,)``; requesting 2-4 dispatches to the
    honestly-unimplemented criterion bodies (C2 work) and raises. Floors are
    evaluated over the row's ground-truth channels and every miss lands in
    ``failures`` as a typed harness entry, wired INTO ``row_green``.
    """

    spec = CENSUS_ROWS[row_id]
    for criterion in criteria:
        if criterion == 2:
            run_census_criterion_2()
        elif criterion == 3:
            run_census_criterion_3()
        elif criterion == 4:
            run_census_criterion_4()

    result = run_census_criterion_1(
        model_factory,
        input_factory,
        seed=seed,
        capture_kwargs=capture_kwargs,
        interposition=interposition,
        suppress_interposition=suppress_interposition,
    )
    result.row_id = row_id
    result.world = spec.world
    result.zi_gate_passed = zi_gate_passed

    if content_floor is not None:
        met = bool(content_floor(result.ground_truth_ops))
        result.floors["content"] = "met" if met else "missing"
        if not met:
            result.failures.append(CONTENT_FLOOR_MISSING)
    if completion_floor is not None:
        met = bool(completion_floor(result.completion_events))
        result.floors["completion"] = "met" if met else "missing"
        if not met:
            result.failures.append(COMPLETION_FLOOR_MISSING)
    for floor_name in spec.floors:
        # Registry floors the wave-0 run could not evaluate (geometry needs
        # captured dual-geometry records; c10d Work-level completion accounting
        # is criterion 3) are DISCLOSED, never silently green and never
        # silently dropped.
        result.floors.setdefault(floor_name, "not_evaluated_wave0")
    if zi_gate_passed is False:
        result.failures.append(ZI_GATE_FAILED)
    result.outputs_bit_identical = not result.failures
    return result


def run_refusal_row(
    row_id: str,
    trigger: Callable[[], Any],
    expected_kinds: tuple[str, ...],
) -> CensusResult:
    """Assert a STAYS-REFUSED row red-by-typed-refusal (never skipped).

    Runs ``trigger`` (a capture attempt) and requires it to raise the typed
    distributed refusal carrying every kind in ``expected_kinds``. The result
    is red-by-construction for ``row_green`` (criteria never ran) and its
    ``green`` says only that the refusal fired as pinned.
    """

    from torchlens._distributed import DistributedCaptureUnsupportedError

    spec = CENSUS_ROWS[row_id]
    failures: list[str] = []
    seen_kinds: tuple[str, ...] = ()
    try:
        trigger()
        failures.append(f"refusal did not fire (expected kinds {expected_kinds!r})")
    except DistributedCaptureUnsupportedError as refusal:
        seen_kinds = tuple(finding.kind for finding in refusal.fields["findings"])
        for kind in expected_kinds:
            if kind not in seen_kinds:
                failures.append(f"expected refusal kind {kind!r} missing from {seen_kinds!r}")
    return CensusResult(
        outputs_bit_identical=False,
        criteria_run=(),
        failures=failures,
        row_id=row_id,
        world=spec.world,
        refusal_kinds=seen_kinds,
    )


# --------------------------------------------------------------------------
# Criteria 2-4: honestly unimplemented until the C2 recording lane (wave 1).
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# The census REPORT (the artifact D-rulings consume; plan sec 1.4).
# --------------------------------------------------------------------------


class CensusReportError(RuntimeError):
    """The report generator refused a dishonest row claim."""


def _row_report_entry(result: CensusResult) -> dict[str, Any]:
    if result.row_id not in CENSUS_ROWS:
        raise CensusReportError(f"unknown census row {result.row_id!r}")
    # The generator recomputes the verdict from the result's own fields; it
    # never trusts a caller-supplied label. Any claim of row-green below the
    # full conjunction is unrepresentable by construction, and the floor
    # verdict is asserted PER ROW, not just the criteria tuple.
    product = result.product_name
    if product == "row green" and (
        result.criteria_run != FULL_CRITERIA
        or result.failures
        or not result.floors_met
        or result.zi_gate_passed is not True
    ):  # pragma: no cover - unreachable by construction, kept as a tripwire
        raise CensusReportError(f"row {result.row_id}: green claim below the full conjunction")
    return {
        "row": result.row_id,
        "world": result.world,
        "criteria_run": list(result.criteria_run),
        "product": product,
        "row_green": result.row_green,
        "floors": dict(result.floors),
        "zi_gate_passed": result.zi_gate_passed,
        "failures": list(result.failures),
        "refusal_kinds": list(result.refusal_kinds),
        "completion_events": dict(result.completion_events),
        "interposition_channel": result.interposition_channel,
        "not_run": list(result.not_run),
    }


def generate_census_report(
    results: list[CensusResult],
    *,
    torch_version: str,
    repo_sha: str,
    md_path: Path | str | None = None,
    json_path: Path | str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Emit the census report: one row per topology + NOT-RUN disclosures.

    Enforces plan rule 1.1(1) mechanically: the ``row green`` label appears
    ONLY for the full conjunction; wave-0 emissions carry the per-row product
    name "ZI baseline + criterion-1 green"; STAYS-REFUSED rows carry the
    refusal product name; registry rows without a result land in the NOT-RUN
    table with their gating; the NOT-COVERED line is always present.
    """

    entries = [_row_report_entry(result) for result in results]
    ran_ids = {entry["row"] for entry in entries}

    def _not_run_reason(spec: CensusRowSpec) -> str:
        if spec.cuda_only:
            return "NOT RUN: no CUDA device on this box (lspci-verified); non-gating GPU annex"
        if spec.runnable_wave0:
            return (
                "asserted red in the pytest census suite (no CensusResult artifact); "
                "see tests/test_distributed_census_topologies.py"
            )
        return f"NOT RUN: wave-{spec.wave} row; gated on {', '.join(spec.gated_on) or 'n/a'}"

    not_run_rows = [
        {"row": spec.row_id, "world": spec.world, "reason": _not_run_reason(spec)}
        for spec in CENSUS_ROWS.values()
        if spec.row_id not in ran_ids
    ]
    payload: dict[str, Any] = {
        "torch_version": torch_version,
        "repo_sha": repo_sha,
        "rows": entries,
        "not_run": not_run_rows,
        "not_covered": NOT_COVERED_LINE,
        "interposed_completion_ops": list(INTERPOSED_COMPLETION_OPS),
    }

    lines = [
        f"# Distributed capture-fidelity census report ({repo_sha})",
        "",
        f"torch: {torch_version}  |  repo: {repo_sha}",
        "",
        "| row | world | criteria run | product | floors | failures |",
        "|---|---|---|---|---|---|",
    ]
    for entry in entries:
        floors = ", ".join(f"{k}={v}" for k, v in entry["floors"].items()) or "-"
        fails = "; ".join(entry["failures"]) or "-"
        criteria = ",".join(str(c) for c in entry["criteria_run"]) or "-"
        lines.append(
            f"| {entry['row']} | {entry['world']} | {criteria} "
            f"| {entry['product']} | {floors} | {fails} |"
        )
    lines += ["", "## NOT-RUN disclosures", ""]
    for row in not_run_rows:
        lines.append(f"- {row['row']} ({row['world']}): {row['reason']}")
    lines += ["", NOT_COVERED_LINE, ""]
    md_text = "\n".join(lines)

    if md_path is not None:
        Path(md_path).write_text(md_text)
    if json_path is not None:
        Path(json_path).write_text(json.dumps(payload, indent=2, sort_keys=True))
    return md_text, payload
