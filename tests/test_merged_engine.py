"""Unit matrix for the C1 merge derivation over synthetic rank evidence.

Pure-function tests of ``torchlens.merged._engine.derive_merge`` (no process
groups, no torch.distributed init): the PRE-JOIN audit matrix (merge-side
rows of design-merge-ranks-c v5, 1.3), seq-delta alignment, relation and
cross-check conflicts, the totalized witness derivation, expected-ranks
widening, determinism, and the contract-doc lockstep gates.
"""

from __future__ import annotations

import json
from pathlib import Path
import re

import pytest

from torchlens.distributed._ledger import (
    GroupLifecycleEvent,
    GroupLifecycleLedger,
    membership_digest_for_ranks,
)
from torchlens.merged import (
    MERGE_FINDING_KINDS,
    BoundaryConsistency,
    MergeAlignment,
    MergedErrorCode,
    MergeValueStatus,
    derive_merge,
)
from torchlens.merged._errors import MergeInputError
from torchlens.merged._evidence import RankEvidence

WORLD = membership_digest_for_ranks([0, 1])


def seeded_ledger(digest: str = WORLD) -> GroupLifecycleLedger:
    ledger = GroupLifecycleLedger()
    ledger.append(GroupLifecycleEvent(0, "seed", digest, 0, "seeded", "seeded", 0))
    return ledger


def armed_ledger(digest: str = WORLD, generations: int = 1) -> GroupLifecycleLedger:
    ledger = GroupLifecycleLedger()
    index = 0
    for ordinal in range(generations):
        ledger.append(
            GroupLifecycleEvent(
                index, "create", digest, ordinal, "wrapped", "armed_before_any_group", ordinal
            )
        )
        index += 1
        if ordinal < generations - 1:
            ledger.append(
                GroupLifecycleEvent(
                    index, "destroy", digest, ordinal, "wrapped", "armed_before_any_group"
                )
            )
            index += 1
    return ledger


def boundary(
    rank: int,
    seq: int,
    *,
    kind: str = "all_reduce",
    digest: str = WORLD,
    ordinal: int = 0,
    channel: str = "coll",
    members: tuple[int, ...] = (0, 1),
    backend: str | None = "gloo",
    roles: list[dict] | None = None,
    witness_policy: str = "none",
    contribution_digests: list[str] | None = None,
    destination_digests: list[str] | None = None,
    reduce_op: str | None = "RedOpType.SUM",
    c10d_group_seq: int | None = None,
    async_op: bool = False,
) -> dict:
    if roles is None:
        roles = [
            {
                "role": "contribution_destination",
                "index": 0,
                "shape": [2, 4],
                "logical_shape": None,
                "placements": None,
            }
        ]
    return {
        "schema": "collective_boundary_v1",
        "kind": kind,
        "func": f"torch.distributed.{kind}",
        "correlation": {
            "membership_digest": digest,
            "lifetime_ordinal": ordinal,
            "channel": channel,
            "seq": seq,
        },
        "group": {
            "global_ranks": list(members),
            "size": len(members),
            "backend": backend,
            "my_global_rank": rank,
            "my_group_rank": members.index(rank) if rank in members else None,
            "coord_provenance": "test",
        },
        "reduce_op": reduce_op,
        "peer": None,
        "events": {
            "async_op": async_op,
            "completion_binding": "unobserved" if async_op else "issue_sync",
        },
        "roles": roles,
        "witness": {
            "policy_resolved": witness_policy,
            "contribution_digests": contribution_digests,
            "destination_digests": destination_digests,
            "not_present_reason": (
                "async_completion_unobserved"
                if async_op and witness_policy == "digest"
                else None
            ),
        },
        "lifetime_evidence": {
            "ordinal_source": "seeded",
            "install_epoch": "seeded",
            "arming_source": "explicit",
        },
        "c10d_group_seq": c10d_group_seq,
        "disclosures": [],
        "op_labels_raw": [f"{kind}_{seq}_raw_r{rank}"],
        "op_node": True,
    }


def evidence(rank: int, boundaries: list[dict], ledger=None, epoch: str = "seeded") -> RankEvidence:
    return RankEvidence(
        rank=rank,
        boundaries=tuple(boundaries),
        ledger=ledger if ledger is not None else seeded_ledger(),
        install_epoch=epoch,
        source=f"synthetic[{rank}]",
    )


def digest_kwargs(dest: str = "aa") -> dict:
    return {
        "witness_policy": "digest",
        "contribution_digests": ["cc"],
        "destination_digests": [dest],
    }


class TestDeltaAlignment:
    def test_differing_absolute_bases_join_by_delta(self):
        # Rank 0 armed earlier and ticked warmups: absolute seqs 5,6 vs 0,1.
        d = derive_merge(
            {
                0: evidence(0, [boundary(0, 5), boundary(0, 6)]),
                1: evidence(1, [boundary(1, 0), boundary(1, 1)]),
            }
        )
        assert d.stored_alignment is MergeAlignment.ALIGNED
        assert [j.key[3] for j in d.joins] == [0, 1]
        assert all(j.presence == (0, 1) for j in d.joins)
        assert {r.seq_abs for r in d.joins[0].per_rank.values()} == {5, 0}

    def test_three_rank_join(self):
        digest3 = membership_digest_for_ranks([0, 1, 2])

        def led():
            return seeded_ledger(digest3)

        cores = {
            rank: evidence(
                rank,
                [boundary(rank, 0, digest=digest3, members=(0, 1, 2))],
                ledger=led(),
            )
            for rank in (0, 1, 2)
        }
        d = derive_merge(cores)
        assert d.stored_alignment is MergeAlignment.ALIGNED
        (join,) = d.joins
        assert join.presence == (0, 1, 2)

    def test_missing_key_is_presence_gap_and_partial(self):
        d = derive_merge(
            {
                0: evidence(0, [boundary(0, 0), boundary(0, 1)]),
                1: evidence(1, [boundary(1, 0)]),
            }
        )
        assert d.stored_alignment is MergeAlignment.PARTIAL
        gaps = d.gap_findings
        assert len(gaps) == 1 and gaps[0].ranks == (1,)

    def test_member_rank_without_core_is_gap_on_every_join(self):
        # Membership records [0, 1] but only rank 0's core is presented.
        d = derive_merge({0: evidence(0, [boundary(0, 0)])})
        assert d.stored_alignment is MergeAlignment.PARTIAL
        (gap,) = d.gap_findings
        assert gap.ranks == (1,)


class TestAuditMatrix:
    """Merge-side rows of the v5 1.3 PRE-JOIN audit matrix."""

    def test_asymmetric_arming_conflicts_with_zero_gaps_and_zero_joins(self):
        # Sol's repro shape: rank 0 evidences {0: seeded (destroyed), 1: wrapped}
        # and captured the recreated group as uid (digest, 1); late-arming
        # rank 1 evidences {0: seeded} and captured it as (digest, 0). The
        # uids never join -- the audit must refuse BEFORE presence-gap
        # derivation, so the outcome is structural with ZERO gaps.
        led0 = GroupLifecycleLedger()
        led0.append(GroupLifecycleEvent(0, "seed", WORLD, 0, "seeded", "seeded", 0))
        led0.append(GroupLifecycleEvent(1, "destroy", WORLD, 0, "seeded", "seeded"))
        led0.append(GroupLifecycleEvent(2, "create", WORLD, 1, "wrapped", "seeded", 1))
        d = derive_merge(
            {
                0: evidence(0, [boundary(0, 0, ordinal=1)], ledger=led0),
                1: evidence(1, [boundary(1, 0, ordinal=0)], ledger=seeded_ledger()),
            }
        )
        assert d.stored_alignment is MergeAlignment.CONFLICTED
        assert [f.kind for f in d.findings] == ["group_lifetime_evidence_conflict"]
        assert d.joins == ()
        assert d.gap_findings == ()
        assert WORLD in d.conflicted_memberships

    def test_seed_discharge_positive_mixed_armed_and_seeded(self):
        # Rank 0 armed before any group (complete witness, ONE generation);
        # rank 1 seeded ordinal 0. The seed discharges: clean join, no
        # over-refusal of the legitimate mixed MPMD case.
        d = derive_merge(
            {
                0: evidence(
                    0,
                    [boundary(0, 0)],
                    ledger=armed_ledger(),
                    epoch="armed_before_any_group",
                ),
                1: evidence(1, [boundary(1, 0)], ledger=seeded_ledger()),
            }
        )
        assert d.stored_alignment is MergeAlignment.ALIGNED
        assert len(d.joins) == 1 and d.joins[0].presence == (0, 1)

    def test_seed_discharge_negative_two_generation_witness(self):
        # The complete witness evidences TWO generations: a seeded ordinal-0
        # entry from another rank is unprovable and the membership refuses.
        led0 = armed_ledger(generations=2)
        d = derive_merge(
            {
                0: evidence(
                    0,
                    [boundary(0, 0, ordinal=1)],
                    ledger=led0,
                    epoch="armed_before_any_group",
                ),
                1: evidence(1, [boundary(1, 0, ordinal=0)], ledger=seeded_ledger()),
            }
        )
        assert d.stored_alignment is MergeAlignment.CONFLICTED
        assert d.joins == () and d.gap_findings == ()

    def test_no_witness_identical_vectors_join(self):
        d = derive_merge(
            {
                0: evidence(0, [boundary(0, 0)], ledger=seeded_ledger()),
                1: evidence(1, [boundary(1, 0)], ledger=seeded_ledger()),
            }
        )
        assert d.stored_alignment is MergeAlignment.ALIGNED

    def test_complete_witness_disagreement_is_evidence_corruption(self):
        d = derive_merge(
            {
                0: evidence(
                    0,
                    [boundary(0, 0)],
                    ledger=armed_ledger(generations=1),
                    epoch="armed_before_any_group",
                ),
                1: evidence(
                    1,
                    [boundary(1, 0, ordinal=1)],
                    ledger=armed_ledger(generations=2),
                    epoch="armed_before_any_group",
                ),
            }
        )
        assert d.stored_alignment is MergeAlignment.CONFLICTED
        assert d.findings[0].kind == "group_lifetime_evidence_conflict"
        assert "complete witnesses disagree" in d.findings[0].detail


class TestScopeAndInputRefusals:
    def test_p2p_kind_refuses_typed(self):
        core = evidence(0, [boundary(0, 0, kind="send", channel="p2p/0->1", reduce_op=None)])
        with pytest.raises(MergeInputError) as excinfo:
            derive_merge({0: core})
        assert excinfo.value.fields["code"] == MergedErrorCode.MERGE_SCOPE_UNSUPPORTED.value

    def test_dtensor_dual_geometry_refuses_typed(self):
        roles = [
            {
                "role": "contribution_destination",
                "index": 0,
                "shape": [2, 4],
                "logical_shape": [4, 4],
                "placements": ["Shard(dim=0)"],
            }
        ]
        core = evidence(0, [boundary(0, 0, roles=roles)])
        with pytest.raises(MergeInputError) as excinfo:
            derive_merge({0: core})
        assert excinfo.value.fields["code"] == MergedErrorCode.MERGE_SCOPE_UNSUPPORTED.value


class TestRelationsAndCrossChecks:
    def test_kind_disagreement_at_joined_key_conflicts(self):
        d = derive_merge(
            {
                0: evidence(0, [boundary(0, 0, kind="all_reduce")]),
                1: evidence(1, [boundary(1, 0, kind="broadcast", reduce_op=None)]),
            }
        )
        assert d.stored_alignment is MergeAlignment.CONFLICTED
        assert any(f.kind == "relation_violation" for f in d.findings)

    def test_reduce_op_disagreement_conflicts(self):
        d = derive_merge(
            {
                0: evidence(0, [boundary(0, 0, reduce_op="RedOpType.SUM")]),
                1: evidence(1, [boundary(1, 0, reduce_op="RedOpType.MAX")]),
            }
        )
        assert d.stored_alignment is MergeAlignment.CONFLICTED

    def test_all_gather_wrong_destination_count_conflicts(self):
        roles = [
            {"role": "contribution", "index": 0, "shape": [2], "logical_shape": None, "placements": None},
            {"role": "destination", "index": 0, "shape": [2], "logical_shape": None, "placements": None},
        ]  # group of 2 but only ONE destination
        d = derive_merge(
            {
                0: evidence(0, [boundary(0, 0, kind="all_gather", roles=roles, reduce_op=None)]),
                1: evidence(1, [boundary(1, 0, kind="all_gather", roles=roles, reduce_op=None)]),
            }
        )
        assert d.stored_alignment is MergeAlignment.CONFLICTED

    def test_c10d_group_seq_delta_disagreement_conflicts(self):
        d = derive_merge(
            {
                0: evidence(
                    0,
                    [boundary(0, 0, c10d_group_seq=10), boundary(0, 1, c10d_group_seq=11)],
                ),
                1: evidence(
                    1,
                    [boundary(1, 0, c10d_group_seq=20), boundary(1, 1, c10d_group_seq=25)],
                ),
            }
        )
        assert d.stored_alignment is MergeAlignment.CONFLICTED
        assert any(f.kind == "correlation_delta_mismatch" for f in d.findings)

    def test_c10d_group_seq_absent_never_demotes(self):
        d = derive_merge(
            {
                0: evidence(0, [boundary(0, 0, c10d_group_seq=None)]),
                1: evidence(1, [boundary(1, 0, c10d_group_seq=7)]),
            }
        )
        assert d.stored_alignment is MergeAlignment.ALIGNED

    def test_interleaved_group_orders_are_an_order_contradiction(self):
        # Two generations of the same membership, both wrapped by complete
        # witnesses (identical lineage vectors -> audit-compatible), but the
        # ranks issue them in OPPOSITE local orders: the join graph has a
        # cycle and the merge is structurally conflicted.
        def led():
            ledger = GroupLifecycleLedger()
            ledger.append(
                GroupLifecycleEvent(0, "create", WORLD, 0, "wrapped", "armed_before_any_group", 0)
            )
            ledger.append(
                GroupLifecycleEvent(1, "create", WORLD, 1, "wrapped", "armed_before_any_group", 1)
            )
            return ledger

        d = derive_merge(
            {
                0: evidence(
                    0,
                    [boundary(0, 0, ordinal=0), boundary(0, 0, ordinal=1)],
                    ledger=led(),
                    epoch="armed_before_any_group",
                ),
                1: evidence(
                    1,
                    [boundary(1, 0, ordinal=1), boundary(1, 0, ordinal=0)],
                    ledger=led(),
                    epoch="armed_before_any_group",
                ),
            }
        )
        assert d.stored_alignment is MergeAlignment.CONFLICTED
        assert any(f.kind == "order_contradiction" for f in d.findings)


class TestWitnessDerivation:
    def test_matching_digests_attest(self):
        d = derive_merge(
            {
                0: evidence(0, [boundary(0, 0, **digest_kwargs())]),
                1: evidence(1, [boundary(1, 0, **digest_kwargs())]),
            }
        )
        assert d.joins[0].consistency is BoundaryConsistency.ATTESTED
        assert d.stored_value_status is MergeValueStatus.ATTESTED_COMPLETE

    def test_mismatch_demotes_value_status_only(self):
        d = derive_merge(
            {
                0: evidence(0, [boundary(0, 0, **digest_kwargs("aa"))]),
                1: evidence(1, [boundary(1, 0, **digest_kwargs("bb"))]),
            }
        )
        assert d.stored_alignment is MergeAlignment.ALIGNED  # never structural
        assert d.joins[0].consistency is BoundaryConsistency.MISMATCHED
        assert d.stored_value_status is MergeValueStatus.DIVERGENT
        assert any(f.kind == "value_divergence" for f in d.findings)

    def test_witness_level_none_is_not_present_and_unwitnessed(self):
        d = derive_merge(
            {
                0: evidence(0, [boundary(0, 0)]),
                1: evidence(1, [boundary(1, 0)]),
            }
        )
        assert d.joins[0].consistency is BoundaryConsistency.NOT_PRESENT
        assert d.stored_value_status is MergeValueStatus.UNWITNESSED

    def test_async_unobserved_completion_is_not_present_never_mismatched(self):
        # Async all_reduce: contribution digests exist (pre-reduce bytes,
        # rank-distinct), destination digests absent. Falling back to the
        # contribution digests would FABRICATE a mismatch; the verdict must
        # be not_present.
        kwargs = {
            "witness_policy": "digest",
            "contribution_digests": None,
            "destination_digests": None,
            "async_op": True,
        }
        d = derive_merge(
            {
                0: evidence(
                    0,
                    [
                        boundary(
                            0, 0, contribution_digests=["r0-pre"], **{k: v for k, v in kwargs.items() if k != "contribution_digests"}
                        )
                    ],
                ),
                1: evidence(
                    1,
                    [
                        boundary(
                            1, 0, contribution_digests=["r1-pre"], **{k: v for k, v in kwargs.items() if k != "contribution_digests"}
                        )
                    ],
                ),
            }
        )
        assert d.joins[0].consistency is BoundaryConsistency.NOT_PRESENT
        assert d.stored_value_status is MergeValueStatus.UNWITNESSED

    def test_reduce_is_not_applicable_at_every_level(self):
        roles_root = [
            {"role": "contribution_destination", "index": 0, "shape": [2], "logical_shape": None, "placements": None},
        ]
        roles_leaf = [
            {"role": "contribution", "index": 0, "shape": [2], "logical_shape": None, "placements": None},
        ]
        d = derive_merge(
            {
                0: evidence(
                    0,
                    [boundary(0, 0, kind="reduce", roles=roles_root, **digest_kwargs())],
                ),
                1: evidence(
                    1,
                    [boundary(1, 0, kind="reduce", roles=roles_leaf, witness_policy="digest", contribution_digests=["x"])],
                ),
            }
        )
        assert d.joins[0].consistency is BoundaryConsistency.NOT_APPLICABLE
        assert d.stored_value_status is MergeValueStatus.UNWITNESSED

    def test_unknown_backend_is_not_applicable(self):
        d = derive_merge(
            {
                0: evidence(0, [boundary(0, 0, backend="fancy_tpu", **digest_kwargs())]),
                1: evidence(1, [boundary(1, 0, backend="fancy_tpu", **digest_kwargs())]),
            }
        )
        assert d.joins[0].consistency is BoundaryConsistency.NOT_APPLICABLE

    def test_broadcast_root_contribution_witnesses_destinations(self):
        root_roles = [
            {"role": "contribution", "index": 0, "shape": [2], "logical_shape": None, "placements": None},
        ]
        leaf_roles = [
            {"role": "destination", "index": 0, "shape": [2], "logical_shape": None, "placements": None},
        ]
        d = derive_merge(
            {
                0: evidence(
                    0,
                    [
                        boundary(
                            0, 0, kind="broadcast", roles=root_roles, reduce_op=None,
                            witness_policy="digest", contribution_digests=["same"],
                        )
                    ],
                ),
                1: evidence(
                    1,
                    [
                        boundary(
                            1, 0, kind="broadcast", roles=leaf_roles, reduce_op=None,
                            witness_policy="digest", destination_digests=["same"],
                        )
                    ],
                ),
            }
        )
        assert d.joins[0].consistency is BoundaryConsistency.ATTESTED

    def test_partial_attestation_is_attested_partial(self):
        d = derive_merge(
            {
                0: evidence(0, [boundary(0, 0, **digest_kwargs()), boundary(0, 1)]),
                1: evidence(1, [boundary(1, 0, **digest_kwargs()), boundary(1, 1)]),
            }
        )
        assert d.stored_value_status is MergeValueStatus.ATTESTED_PARTIAL


class TestExpectedRanksWidenOnly:
    def test_declared_ranks_without_cores_are_gaps(self):
        d = derive_merge(
            {
                0: evidence(0, [boundary(0, 0)]),
                1: evidence(1, [boundary(1, 0)]),
            },
            expected_ranks=[0, 1, 2, 3],
        )
        assert d.stored_alignment is MergeAlignment.PARTIAL
        (gap,) = d.gap_findings
        assert gap.ranks == (2, 3)

    def test_narrow_declaration_never_removes_gaps(self):
        # Membership records [0, 1]; declaring only [0] must not erase the
        # gap for missing rank 1 (rank cores are gap-derivation authority).
        d = derive_merge({0: evidence(0, [boundary(0, 0)])}, expected_ranks=[0])
        assert d.stored_alignment is MergeAlignment.PARTIAL
        assert d.gap_findings[0].ranks == (1,)


class TestDeterminism:
    def test_identical_evidence_yields_byte_identical_payloads(self):
        def build():
            return {
                0: evidence(0, [boundary(0, 3, **digest_kwargs()), boundary(0, 4)]),
                1: evidence(1, [boundary(1, 0, **digest_kwargs()), boundary(1, 1)]),
            }

        left = json.dumps(derive_merge(build()).to_payload(), sort_keys=True)
        right = json.dumps(derive_merge(build()).to_payload(), sort_keys=True)
        assert left == right


class TestContractLockstep:
    """The contract document IS the spec: frozen tables track the code exactly."""

    DOC = Path(__file__).resolve().parents[1] / "docs" / "reference" / "merged_trace_contract.md"

    def test_error_code_table_matches_enum_exactly(self):
        doc = self.DOC.read_text()
        match = re.search(
            r"The exact `MergedErrorCode` values are:\n\n```text\n(.*?)```", doc, re.S
        )
        assert match is not None
        doc_codes = [line.strip() for line in match.group(1).strip().split("\n")]
        assert doc_codes == [member.value for member in MergedErrorCode]

    def test_finding_kind_table_matches_tuple_exactly(self):
        doc = self.DOC.read_text()
        match = re.search(
            r"The exact `MERGE_FINDING_KINDS` values are:\n\n```text\n(.*?)```", doc, re.S
        )
        assert match is not None
        doc_kinds = [line.strip() for line in match.group(1).strip().split("\n")]
        assert doc_kinds == list(MERGE_FINDING_KINDS)
