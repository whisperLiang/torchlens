"""Postprocess dependency derivation: declared contracts -> execution order.

The pipeline order is DERIVED, not hand-maintained (design-ppdag-v3): each
step's ``PostprocessStepContract`` declares its op-column writes/reads,
trace-state tokens, row effects, and barrier; ``derive_edges`` orients every
conflict by the frozen ``LEGACY_STEP_RANK`` (never by registry position),
and ``execution_order`` runs Kahn's algorithm with a min-heap keyed by rank.
Ranks are unique, so no secondary tie-break exists.

Day-1 identity theorem (verified independently by both design reviewers):
every edge rule orients rank-ascending, so the graph is acyclic by
construction and the rank-keyed Kahn emits exactly rank order — which R1
pins to the registry order. The derivation therefore detects ARTIFACT DRIFT
(registry, rank, or declarations disagreeing), never reorder legality; the
semantic direction authority is the reason-bearing ``PINNED_ORDER_PAIRS``
corpus (checks K1 at import in ``_contracts``, K2 test-side).

Import-time structural checks (plain ``raise``, never ``assert`` —
``python -O`` strips asserts): the 7.1 family below. The read-before-write
findings channel is a TEST REPORT over the recording matrix, never an
import gate — import must not depend on recordings.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Iterator

from ._contracts import (
    CAPTURE_BASELINE_TOKENS,
    LEGACY_STEP_RANK,
    POSTPROCESS_STEP_CONTRACTS,
    PostprocessStepContract,
)

#: The registry order: the canonical hand-established step sequence the
#: executor runs. R1 pins it strictly rank-ascending; R2 pins the derived
#: order equal to it. Step "0" is the fenced prologue (producer lane) and
#: deliberately absent.
REGISTRY_ORDER: tuple[str, ...] = (
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "11",
    "11.5",
    "11.75",
    "12",
    "13",
    "14",
    "15",
    "15.5",
    "16",
    "16.5",
    "17",
    "17.5",
    "18",
    "19",
    "20",
)


@dataclass(frozen=True)
class DerivedEdge:
    """One derived ordering edge with its provenance.

    ``kind`` is one of ``raw`` / ``ww`` / ``war`` / ``row_barrier`` /
    ``token_raw`` / ``token_ww`` / ``token_war`` / ``barrier``; ``carrier``
    names the column or token (empty for structural edges).
    """

    src: str
    dst: str
    kind: str
    carrier: str


def _registry_contracts() -> dict[str, PostprocessStepContract]:
    """Return the registry-step contracts (step "0" excluded)."""

    return {
        step: contract
        for step, contract in POSTPROCESS_STEP_CONTRACTS.items()
        if step != "0"
    }


def _op_touching(contract: PostprocessStepContract) -> bool:
    """Whether a step reads or writes ANY op column (edge rule 4 operand).

    Row-set membership is an implicit input of every per-op iteration and
    an implicit precondition of every cell write, so row-effect steps pin
    two-sidedly against every op-touching step — including each other.
    """

    return bool(
        contract.writes
        or contract.reads
        or contract.placeholder_probes
        or contract.row_effects
    )


def derive_edges() -> list[DerivedEdge]:
    """Derive every ordering edge from the declared contracts.

    All rules orient by ``LEGACY_STEP_RANK`` (design-ppdag-v3 §2.3):

    1. RAW: reader depends on EVERY lower-rank writer of the column
       (refine-in-place semantics).
    2. WW: co-writers order pairwise by rank.
    3. WAR: a lower-rank reader pins before every higher-rank writer.
       Declared placeholder probes generate WAR edges too — a probe's
       correctness depends on observing "not yet set", so the probing step
       stays pinned before the writer (probes are exempt from FINDINGS,
       never from edges).
    4. Row structure: a step with row_effects is a two-sided barrier with
       respect to every op-touching step.
    5. Tokens: RAW/WW/WAR over ``r:<t>``/``w:<t>``.
    6. Barrier: a ``barrier=True`` step orders against every other step.
    """

    contracts = _registry_contracts()
    rank = LEGACY_STEP_RANK
    edges: list[DerivedEdge] = []

    def _directed(low: str, high: str) -> tuple[str, str] | None:
        if rank[low] < rank[high]:
            return (low, high)
        if rank[high] < rank[low]:
            return (high, low)
        return None

    # Rules 1-3: column conflicts.
    writers: dict[str, list[str]] = {}
    readers: dict[str, list[str]] = {}
    for step, contract in contracts.items():
        for column in contract.writes:
            writers.setdefault(column, []).append(step)
        for column in contract.reads | contract.placeholder_probes:
            readers.setdefault(column, []).append(step)
    for column, column_writers in writers.items():
        by_rank = sorted(column_writers, key=rank.__getitem__)
        for i, low in enumerate(by_rank):
            for high in by_rank[i + 1 :]:
                edges.append(DerivedEdge(low, high, "ww", column))
        for reader in readers.get(column, ()):  # rules 1 and 3
            for writer in column_writers:
                if writer == reader:
                    continue
                if rank[writer] < rank[reader]:
                    edges.append(DerivedEdge(writer, reader, "raw", column))
                else:
                    edges.append(DerivedEdge(reader, writer, "war", column))

    # Rule 5: token conflicts.
    token_writers: dict[str, list[str]] = {}
    token_readers: dict[str, list[str]] = {}
    for step, contract in contracts.items():
        for entry in contract.trace_state:
            prefix, _, token = entry.partition(":")
            target = token_writers if prefix == "w" else token_readers
            target.setdefault(token, []).append(step)
    for token, its_writers in token_writers.items():
        by_rank = sorted(its_writers, key=rank.__getitem__)
        for i, low in enumerate(by_rank):
            for high in by_rank[i + 1 :]:
                edges.append(DerivedEdge(low, high, "token_ww", token))
        for reader in token_readers.get(token, ()):
            for writer in its_writers:
                if writer == reader:
                    continue
                if rank[writer] < rank[reader]:
                    edges.append(DerivedEdge(writer, reader, "token_raw", token))
                else:
                    edges.append(DerivedEdge(reader, writer, "token_war", token))

    # Rule 4: two-sided row-structure barriers.
    row_steps = [s for s, c in contracts.items() if c.row_effects]
    op_steps = [s for s, c in contracts.items() if _op_touching(c)]
    for row_step in row_steps:
        for other in op_steps:
            if other == row_step:
                continue
            pair = _directed(row_step, other)
            if pair is not None:
                edges.append(DerivedEdge(pair[0], pair[1], "row_barrier", ""))

    # Rule 6: full barriers.
    barrier_steps = [s for s, c in contracts.items() if c.barrier]
    for barrier_step in barrier_steps:
        for other in contracts:
            if other == barrier_step:
                continue
            pair = _directed(barrier_step, other)
            if pair is not None:
                edges.append(DerivedEdge(pair[0], pair[1], "barrier", ""))

    return edges


def derived_pinnable_pairs() -> dict[tuple[str, str], set[str]]:
    """Return the RAW/WW pair corpus view: (producer, consumer) -> carriers.

    This is check K2's left-hand side (and the one-time corpus seed input):
    every derived producer->consumer fact, as step pairs with the column or
    token names that carry it. WAR pairs are excluded — a WAR edge is not a
    producer->consumer fact, but hoisting the reader past the writer under
    a coordinated reversal MANUFACTURES a new RAW pair, which lands here
    and turns K2 red until pinned with a reason.
    """

    pairs: dict[tuple[str, str], set[str]] = {}
    for edge in derive_edges():
        if edge.kind in ("raw", "ww", "token_raw", "token_ww"):
            pairs.setdefault((edge.src, edge.dst), set()).add(edge.carrier)
    return pairs


def execution_order() -> tuple[str, ...]:
    """Derive the execution order: Kahn with a min-heap keyed by rank.

    Ranks are unique integers, so no secondary key exists (the historical
    "registry-index tie-break" formulation is gone); R2 asserts the emitted
    order equals ``REGISTRY_ORDER``.
    """

    contracts = _registry_contracts()
    indegree: dict[str, int] = {step: 0 for step in contracts}
    successors: dict[str, set[str]] = {step: set() for step in contracts}
    for edge in derive_edges():
        if edge.dst not in successors[edge.src]:
            successors[edge.src].add(edge.dst)
            indegree[edge.dst] += 1
    heap = [
        (LEGACY_STEP_RANK[step], step)
        for step, degree in indegree.items()
        if degree == 0
    ]
    heapq.heapify(heap)
    order: list[str] = []
    while heap:
        _, step = heapq.heappop(heap)
        order.append(step)
        for successor in successors[step]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                heapq.heappush(heap, (LEGACY_STEP_RANK[successor], successor))
    if len(order) != len(contracts):
        cyclic = sorted(step for step, degree in indegree.items() if degree > 0)
        raise ValueError(
            f"Postprocess dependency derivation found a cycle involving "
            f"{cyclic}; declared contracts contradict LEGACY_STEP_RANK."
        )
    return tuple(order)


def _iter_structural_violations() -> Iterator[str]:
    """Yield every 7.1-family structural violation (import checks)."""

    contracts = _registry_contracts()
    # 7.1-1: registry <-> contracts bijection (step "0" exempted).
    registry_set = set(REGISTRY_ORDER)
    contract_set = set(contracts)
    if registry_set != contract_set:
        yield (
            f"registry/contract mismatch: contracts without registry entry "
            f"{sorted(contract_set - registry_set)}, registry steps without "
            f"contract {sorted(registry_set - contract_set)}"
        )
        return
    # 7.1-3 (R1): registry strictly rank-ascending.
    registry_ranks = [LEGACY_STEP_RANK[step] for step in REGISTRY_ORDER]
    for i in range(1, len(registry_ranks)):
        if registry_ranks[i - 1] >= registry_ranks[i]:
            yield (
                f"registry order contradicts LEGACY_STEP_RANK at steps "
                f"{REGISTRY_ORDER[i - 1]!r}, {REGISTRY_ORDER[i]!r} (R1)"
            )
    # 7.1-5: token read-before-write analogue.
    token_writer_ranks: dict[str, int] = {}
    for step, contract in contracts.items():
        for entry in contract.trace_state:
            prefix, _, token = entry.partition(":")
            if prefix == "w":
                current = token_writer_ranks.get(token)
                step_rank = LEGACY_STEP_RANK[step]
                if current is None or step_rank < current:
                    token_writer_ranks[token] = step_rank
    for step, contract in contracts.items():
        for entry in contract.trace_state:
            prefix, _, token = entry.partition(":")
            if prefix != "r" or token in CAPTURE_BASELINE_TOKENS:
                continue
            earliest = token_writer_ranks.get(token)
            if earliest is None or earliest >= LEGACY_STEP_RANK[step]:
                yield (
                    f"step {step} declares r:{token} with no lower-rank "
                    f"w:{token} and {token!r} is not capture-baseline "
                    "(7.1-5)"
                )
    # 7.1-4 (R2): the derived order reproduces the registry.
    derived = execution_order()
    if derived != REGISTRY_ORDER:
        yield (
            f"derived execution order diverges from the registry (R2): "
            f"derived={derived!r}"
        )


def _validate_derivation() -> None:
    """Run the 7.1 checks; refuse import on the first violation, by name."""

    violations = list(_iter_structural_violations())
    if violations:
        raise ValueError(
            "Postprocess DAG structural checks failed:\n- "
            + "\n- ".join(violations)
        )


_validate_derivation()
