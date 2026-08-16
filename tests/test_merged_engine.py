"""Unit matrix for the C1 merge derivation over synthetic rank evidence.

Pure-function tests of ``torchlens.merged._engine.derive_merge`` (no process
groups, no torch.distributed init): the PRE-JOIN audit matrix (merge-side
rows of design-merge-ranks-c v5, 1.3), seq-delta alignment, relation and
cross-check conflicts, the totalized witness derivation, expected-ranks
widening, determinism, and the contract-doc lockstep gates.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from types import SimpleNamespace

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
from torchlens.merged._evidence import (
    P2P_KINDS,
    TENSORLESS_KINDS,
    RankEvidence,
    extract_rank_evidence,
)

pytestmark = pytest.mark.smoke

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
        "peer": (
            {"raw": {"tag": 0}, "canonical": {"src": 0, "dst": 1}} if kind in P2P_KINDS else None
        ),
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
                "async_completion_unobserved" if async_op and witness_policy == "digest" else None
            ),
        },
        "lifetime_evidence": {
            "ordinal_source": "seeded",
            "install_epoch": "seeded",
            "arming_source": "explicit",
        },
        "c10d_group_seq": c10d_group_seq,
        # Recorder-coherent derived fields: the parse chokepoint refuses
        # records whose disclosure/op_node surface contradicts the rest
        # (tensorless kinds emit no op node, hence no back-references).
        "disclosures": ["read_of_inflight_destination"] if async_op else [],
        "op_labels_raw": [] if kind in TENSORLESS_KINDS else [f"{kind}_{seq}_raw_r{rank}"],
        "op_node": kind not in TENSORLESS_KINDS,
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


def trace_for_boundaries(boundaries: list[dict], ledger: GroupLifecycleLedger) -> SimpleNamespace:
    """Build the minimal trace surface consumed by rank-evidence extraction.

    Parameters
    ----------
    boundaries:
        Synthetic collective-boundary journal.
    ledger:
        Matching group-lifecycle ledger.

    Returns
    -------
    SimpleNamespace
        Trace-shaped object carrying distributed annotations.
    """

    return SimpleNamespace(
        annotations={
            "distributed": {
                "boundaries": boundaries,
                "group_lifecycle_ledger": ledger.to_payload(),
                "install_epoch": "seeded",
            }
        }
    )


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

    def test_parse_refuses_rank_outside_recorded_membership(self) -> None:
        """A rank core cannot claim evidence for a group it does not belong to."""

        forged = boundary(5, 0, members=(0, 1), **digest_kwargs())
        with pytest.raises(MergeInputError) as excinfo:
            extract_rank_evidence(
                trace_for_boundaries([forged], seeded_ledger()),
                "forged-rank-5",
            )
        assert excinfo.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value

    def test_join_refuses_presence_outside_recorded_membership(self) -> None:
        """Direct engine callers receive the same presence-subset refusal."""

        with pytest.raises(MergeInputError) as excinfo:
            derive_merge(
                {
                    0: evidence(0, [boundary(0, 0, **digest_kwargs())]),
                    5: evidence(5, [boundary(5, 0, **digest_kwargs())]),
                }
            )
        assert excinfo.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value

    def test_parse_refuses_duplicate_rank_local_sequence(self) -> None:
        """Duplicate absolute sequence keys cannot overwrite a boundary silently."""

        duplicated = [boundary(0, 7), boundary(0, 7)]
        with pytest.raises(MergeInputError) as excinfo:
            extract_rank_evidence(
                trace_for_boundaries(duplicated, seeded_ledger()),
                "duplicate-seq-rank-0",
            )
        assert excinfo.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value

    def test_join_refuses_duplicate_rank_local_sequence(self) -> None:
        """Direct engine evidence cannot exploit duplicate-sequence overwrite."""

        with pytest.raises(MergeInputError) as excinfo:
            derive_merge(
                {
                    0: evidence(0, [boundary(0, 7), boundary(0, 7)]),
                    1: evidence(1, [boundary(1, 3)]),
                }
            )
        assert excinfo.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value


class TestBoundaryParseValidation:
    """Deep-hunt F2: roles and witness digest fields are validated typed at parse.

    Fail-before: a role entry without ``shape`` passed extraction and escaped
    ``derive_merge`` as a raw ``KeyError('shape')`` from both ``merge_ranks``
    and load rederivation; a bare-STRING digest field char-split through
    ``tuple(...)`` and two cores carrying the same garbage string rendered a
    fabricated ``attested_complete``.
    """

    def _extract(self, entry: dict) -> None:
        extract_rank_evidence(
            trace_for_boundaries([entry], seeded_ledger()),
            "forged-boundary",
        )

    def _assert_refuses(self, entry: dict) -> None:
        with pytest.raises(MergeInputError) as excinfo:
            self._extract(entry)
        assert excinfo.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value

    def test_role_entry_missing_shape_refuses_typed(self):
        self._assert_refuses(
            boundary(0, 0, roles=[{"role": "contribution_destination", "index": 0}])
        )

    def test_role_entry_not_a_mapping_refuses_typed(self):
        self._assert_refuses(boundary(0, 0, roles=["contribution"]))

    def test_roles_not_a_list_refuses_typed(self):
        entry = boundary(0, 0)
        entry["roles"] = {"role": "contribution"}
        self._assert_refuses(entry)

    def test_role_name_outside_vocabulary_refuses_typed(self):
        self._assert_refuses(
            boundary(0, 0, roles=[{"role": "spectator", "index": 0, "shape": [2]}])
        )

    def test_role_shape_with_non_integer_dim_refuses_typed(self):
        self._assert_refuses(
            boundary(0, 0, roles=[{"role": "contribution", "index": 0, "shape": [2, "x"]}])
        )

    def test_string_digest_field_refuses_typed(self):
        entry = boundary(0, 0, witness_policy="digest")
        entry["witness"]["contribution_digests"] = "ccdd"
        entry["witness"]["destination_digests"] = "aabb"
        self._assert_refuses(entry)

    def test_non_hex_digest_element_refuses_typed(self):
        entry = boundary(0, 0, witness_policy="digest")
        entry["witness"]["destination_digests"] = ["not-a-digest"]
        self._assert_refuses(entry)

    def test_non_string_op_label_refuses_typed(self):
        entry = boundary(0, 0)
        entry["op_labels_raw"] = ["fine", 7]
        self._assert_refuses(entry)

    def test_non_integer_my_group_rank_refuses_typed(self):
        entry = boundary(0, 0)
        entry["group"]["my_group_rank"] = "0"
        self._assert_refuses(entry)

    def test_non_string_backend_refuses_typed(self):
        entry = boundary(0, 0)
        entry["group"]["backend"] = 7
        self._assert_refuses(entry)

    def test_non_string_channel_refuses_typed(self):
        entry = boundary(0, 0)
        entry["correlation"]["channel"] = 0
        self._assert_refuses(entry)

    def test_unhashable_list_channel_refuses_typed(self):
        # A LIST channel used to escape the parse as a raw
        # ``TypeError: unhashable type: 'list'`` from the rank-local
        # correlation-key dedup set, not the promised typed refusal.
        entry = boundary(0, 0)
        entry["correlation"]["channel"] = ["coll"]
        self._assert_refuses(entry)

    def test_negative_seq_refuses_typed(self):
        self._assert_refuses(boundary(0, -1))

    def test_valid_sha256_digest_lists_still_parse(self):
        entry = boundary(
            0,
            0,
            witness_policy="digest",
            contribution_digests=["c" * 64],
            destination_digests=["a" * 64],
        )
        self._extract(entry)  # must not raise

    def test_parse_refuses_membership_digest_ranks_incoherence(self):
        """Deep-hunt F3: the digest must equal sha256(sorted(global_ranks)).

        Fail-before: two cores presenting the digest of a DIFFERENT membership
        ([5, 6, 7]) over global_ranks [0, 1] merged ALIGNED, rebinding one
        communicator's boundaries to another membership's digest, ordinal
        lineage, and audit row.
        """

        fake = membership_digest_for_ranks([5, 6, 7])
        self._assert_refuses(boundary(0, 0, digest=fake))

    def test_engine_refuses_membership_digest_ranks_incoherence(self):
        """Direct-engine evidence receives the same digest-coherence refusal."""

        fake = membership_digest_for_ranks([5, 6, 7])
        with pytest.raises(MergeInputError) as excinfo:
            derive_merge(
                {
                    0: evidence(0, [boundary(0, 0, digest=fake)], ledger=seeded_ledger(fake)),
                    1: evidence(1, [boundary(1, 0, digest=fake)], ledger=seeded_ledger(fake)),
                }
            )
        assert excinfo.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value

    def test_engine_belt_refuses_string_digests_typed(self):
        """Direct-engine evidence cannot fabricate ATTESTED via char-split.

        Fail-before: consistency rendered ``attested`` and the merge presented
        ``attested_complete`` from two identical garbage strings.
        """

        def forged(rank: int) -> dict:
            entry = boundary(rank, 0, witness_policy="digest")
            entry["witness"]["contribution_digests"] = "ccdd"
            entry["witness"]["destination_digests"] = "aabb"
            return entry

        with pytest.raises(MergeInputError) as excinfo:
            derive_merge({0: evidence(0, [forged(0)]), 1: evidence(1, [forged(1)])})
        assert excinfo.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value


class TestRolesDeletionVacuousTruth:
    """b6-opus-R18-1: uniform roles deletion must refuse, never render TOP.

    Fail-before: deleting the ``roles`` record from EVERY member of a join
    vacuously satisfied the set-of-shapes agreement (all ranks presented the
    same EMPTY shape set) and the merge rendered the top verdict --
    aligned / attested_complete with zero findings. Asymmetric deletion was
    caught; uniform corruption, the merge threat model, was the escape.
    """

    def _extract(self, entry: dict) -> None:
        extract_rank_evidence(
            trace_for_boundaries([entry], seeded_ledger()),
            "roles-tamper",
        )

    def _assert_refuses(self, entry: dict) -> None:
        with pytest.raises(MergeInputError) as excinfo:
            self._extract(entry)
        assert excinfo.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value

    def test_roles_key_deleted_refuses_typed(self):
        entry = boundary(0, 0)
        del entry["roles"]
        self._assert_refuses(entry)

    def test_tensor_kind_with_empty_roles_refuses_typed(self):
        self._assert_refuses(boundary(0, 0, roles=[]))

    def test_tensorless_kind_with_empty_roles_still_parses(self):
        entry = boundary(0, 0, kind="barrier", reduce_op=None, roles=[])
        self._extract(entry)

    def test_uniform_roles_deletion_refuses_on_every_rank_core(self):
        """The headline escape: BOTH rank cores tampered identically."""

        for rank in (0, 1):
            entry = boundary(rank, 0)
            del entry["roles"]
            with pytest.raises(MergeInputError) as excinfo:
                extract_rank_evidence(
                    trace_for_boundaries([entry], seeded_ledger()),
                    f"uniform-tamper[{rank}]",
                )
            assert excinfo.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value

    def test_engine_belt_flags_zero_role_entries_per_rank(self):
        """Direct ``derive_merge`` callers bypass evidence parse; the relation
        table still names every rank presenting zero roles for a
        tensor-carrying kind instead of agreeing on the empty shape set."""

        d = derive_merge(
            {
                0: evidence(0, [boundary(0, 0, roles=[])]),
                1: evidence(1, [boundary(1, 0, roles=[])]),
            }
        )
        zero_role = [
            f
            for f in d.findings
            if f.kind == "relation_violation" and "zero tensor roles" in f.detail
        ]
        assert {f.detail.split("rank ")[1][0] for f in zero_role} == {"0", "1"}

    def test_tensorless_kinds_mirror_capture_side_tensorless_flags(self):
        """The evidence vocabulary tracks ``CollectiveSite.tensorless`` exactly."""

        from torchlens.backends.torch.collectives import COLLECTIVE_SITES
        from torchlens.merged._evidence import TENSORLESS_KINDS

        assert {site.kind for site in COLLECTIVE_SITES if site.tensorless} == TENSORLESS_KINDS


class TestSweepFieldValidation:
    """p5 sibling sweep: the reduce-op and seq cross-check fields parse typed.

    Fail-before: uniform ``reduce_op`` deletion from every rank core vacuously
    satisfied the reduce-op agreement check (same escape class as roles
    deletion), and a tampered non-integer ``c10d_group_seq`` crashed the
    engine's delta arithmetic with a raw ``TypeError`` instead of the promised
    typed refusal.
    """

    def _assert_refuses(self, entry: dict) -> None:
        with pytest.raises(MergeInputError) as excinfo:
            extract_rank_evidence(
                trace_for_boundaries([entry], seeded_ledger()),
                "sweep-tamper",
            )
        assert excinfo.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value

    def test_reduce_op_deleted_on_reduce_kind_refuses_typed(self):
        entry = boundary(0, 0)  # all_reduce
        del entry["reduce_op"]
        self._assert_refuses(entry)

    def test_reduce_op_null_on_reduce_kind_refuses_typed(self):
        self._assert_refuses(boundary(0, 0, reduce_op=None))

    def test_reduce_op_null_on_non_reduce_kind_still_parses(self):
        extract_rank_evidence(
            trace_for_boundaries(
                [boundary(0, 0, kind="broadcast", reduce_op=None)], seeded_ledger()
            ),
            "clean-broadcast",
        )

    def test_non_integer_c10d_group_seq_refuses_typed(self):
        entry = boundary(0, 0)
        entry["c10d_group_seq"] = "5"
        self._assert_refuses(entry)

    def test_bool_c10d_group_seq_refuses_typed(self):
        entry = boundary(0, 0)
        entry["c10d_group_seq"] = True
        self._assert_refuses(entry)

    def test_reduce_op_kinds_mirror_capture_side_has_reduce_op_flags(self):
        """The evidence vocabulary tracks ``CollectiveSite.has_reduce_op`` exactly."""

        from torchlens.backends.torch.collectives import COLLECTIVE_SITES
        from torchlens.merged._evidence import REDUCE_OP_KINDS

        assert {site.kind for site in COLLECTIVE_SITES if site.has_reduce_op} == REDUCE_OP_KINDS


class TestWireVocabularyLockstep:
    """R49: the distributed->merged wire vocabulary cannot drift silently.

    The collective_boundary_v1 payload is WRITTEN by
    ``backends/torch/collectives.py`` (+ the lifecycle ledger) and READ by
    ``merged/_evidence.py``; both sides used to re-spell the closed
    vocabularies independently with zero drift gate, so a writer-side rename
    silently turned every future artifact unparseable (or, worse, unvalidated
    on the renamed axis). Declared residual: a NEW writer-side token is only
    caught at parse time; hoisting the writer's literals into one shared
    constant home is relayed to the capture lane.
    """

    def _writer_string_literals(self) -> set[str]:
        import inspect

        from torchlens.backends.torch import collectives

        tree = ast.parse(inspect.getsource(collectives))
        return {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }

    def test_boundary_schema_matches_writer(self):
        from torchlens.backends.torch import collectives
        from torchlens.merged import _evidence

        assert collectives.BOUNDARY_SCHEMA == _evidence.BOUNDARY_SCHEMA

    def test_install_epoch_vocabulary_matches_ledger_literal(self):
        from typing import get_args

        from torchlens.distributed._ledger import InstallEpoch
        from torchlens.merged._evidence import _INSTALL_EPOCHS

        assert set(get_args(InstallEpoch)) == set(_INSTALL_EPOCHS)

    def test_writer_spells_every_reader_vocabulary_token(self):
        from torchlens.merged import _evidence

        writer_literals = self._writer_string_literals()
        for vocab_name in (
            "_COMPLETION_BINDINGS",
            "_WITNESS_POLICIES",
            "_DISCLOSURE_TOKENS",
            "_NOT_PRESENT_REASONS",
        ):
            vocab = getattr(_evidence, vocab_name)
            missing = set(vocab) - writer_literals
            assert not missing, (
                f"reader vocabulary {vocab_name} member(s) {sorted(missing)} never "
                "appear in the writer module -- a writer-side rename drifted the wire"
            )

    def test_reader_vocabularies_are_pinned(self):
        from torchlens.merged import _evidence

        assert set(_evidence._COMPLETION_BINDINGS) == {"issue_sync", "unobserved"}
        assert set(_evidence._WITNESS_POLICIES) == {"none", "digest"}
        assert set(_evidence._DISCLOSURE_TOKENS) == {
            "read_of_inflight_destination",
            "c10d_group_seq_read_failed",
        }
        assert set(_evidence._NOT_PRESENT_REASONS) == {"async_completion_unobserved"}


class TestReleaseContract:
    """Contract section 8: release() refuses typed, never lies about members.

    Fail-before (opus R18 [W]): release() landed contradicting the contract
    doc ("no separate cleanup surface"), post-release member access raised
    bare ``KeyError`` from the emptied handle dict, and ``merged.ranks``
    presented a released presenter as a ZERO-MEMBER merge (empty mapping,
    ``len() == 0``) while ``rank_ids`` still listed the ranks -- a silent
    presence lie.
    """

    def _merged(self):
        from torchlens.merged import merge_ranks

        return merge_ranks(
            [
                trace_for_boundaries([boundary(0, 0)], seeded_ledger()),
                trace_for_boundaries([boundary(1, 0)], seeded_ledger()),
            ]
        )

    def _assert_released_refusal(self, call) -> None:
        from torchlens.merged._errors import MergedSurfaceUnsupportedError

        with pytest.raises(MergedSurfaceUnsupportedError) as excinfo:
            call()
        assert excinfo.value.fields["code"] == MergedErrorCode.MERGED_MEMBER_RELEASED.value

    def test_member_surfaces_refuse_typed_after_release(self, tmp_path):
        merged = self._merged()
        merged.release()
        self._assert_released_refusal(lambda: merged.ranks)
        self._assert_released_refusal(lambda: merged["anything"])
        self._assert_released_refusal(lambda: merged.super_op("anything"))
        self._assert_released_refusal(lambda: merged.save(tmp_path / "released"))

    def test_join_ops_refuses_typed_after_release(self):
        merged = self._merged()
        (join,) = merged.joins
        merged.release()
        self._assert_released_refusal(lambda: merged.join_ops(join))

    def test_release_never_presents_zero_members(self):
        """A released presenter must not read as an empty merge."""

        merged = self._merged()
        merged.release()
        with pytest.raises(Exception) as excinfo:
            len(merged.ranks)
        assert getattr(excinfo.value, "fields", {}).get("code") == (
            MergedErrorCode.MERGED_MEMBER_RELEASED.value
        )

    def test_verdicts_stay_readable_after_release(self):
        merged = self._merged()
        before = (merged.alignment, merged.value_status, merged.rank_ids)
        merged.release()
        assert (merged.alignment, merged.value_status, merged.rank_ids) == before
        assert merged.report.alignment is before[0]
        assert merged.joins and merged.gaps == merged._derivation.gap_findings
        assert isinstance(merged.findings, tuple)
        assert "MergedTrace" in repr(merged)
        assert "alignment" in merged.summary()

    def test_release_is_idempotent(self):
        merged = self._merged()
        merged.release()
        merged.release()
        self._assert_released_refusal(lambda: merged.ranks)

    def test_pre_release_member_access_unchanged(self):
        merged = self._merged()
        assert set(merged.ranks) == {0, 1}
        assert merged.ranks[0] is not None


class TestWitnessCompletionCoherence:
    """R18 fixwave-6: forged witness/completion/disclosure records refuse at parse.

    Fail-before (sol HIGH, 4th round): an async boundary
    (``completion_binding="unobserved"``) carrying FORGED ``destination_digests``
    -- bytes the recorder definitionally never observed -- rendered
    ``attested``/``attested_complete`` when the forgery matched across cores;
    ``policy_resolved="none"`` cores presenting digests attested the same way
    (opus+sol); and 15/17 disclosure-tamper arms (``async_op`` flips, stripped
    ``read_of_inflight_destination``, spurious tokens, ``op_node``/``peer``
    rewrites, per-boundary install-epoch promotion) passed parse untouched.
    Every axis now refuses typed at the one chokepoint merge time and load
    rederivation share.
    """

    def _extract(self, entry: dict) -> None:
        extract_rank_evidence(
            trace_for_boundaries([entry], seeded_ledger()),
            "coherence-tamper",
        )

    def _assert_refuses(self, entry: dict) -> None:
        with pytest.raises(MergeInputError) as excinfo:
            self._extract(entry)
        assert excinfo.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value

    # --- the headline forgery: async destination digests -------------------

    def test_forged_destination_digests_on_unobserved_completion_refuse(self):
        entry = boundary(
            0,
            0,
            async_op=True,
            witness_policy="digest",
            contribution_digests=["c" * 64],
            destination_digests=["a" * 64],
        )
        self._assert_refuses(entry)

    def test_honest_async_digest_record_still_parses(self):
        entry = boundary(
            0,
            0,
            async_op=True,
            witness_policy="digest",
            contribution_digests=["c" * 64],
            destination_digests=None,
        )
        self._extract(entry)  # must not raise

    # --- digests under witness policy "none" -------------------------------

    def test_contribution_digests_under_policy_none_refuse(self):
        entry = boundary(0, 0)
        entry["witness"]["contribution_digests"] = ["c" * 64]
        self._assert_refuses(entry)

    def test_destination_digests_under_policy_none_refuse(self):
        entry = boundary(0, 0)
        entry["witness"]["destination_digests"] = ["a" * 64]
        self._assert_refuses(entry)

    # --- events coherence ---------------------------------------------------

    def test_async_op_flag_contradicting_completion_binding_refuses(self):
        entry = boundary(0, 0, async_op=True)
        entry["events"]["async_op"] = False
        self._assert_refuses(entry)

    def test_sync_record_claiming_unobserved_binding_refuses(self):
        entry = boundary(0, 0)
        entry["events"]["completion_binding"] = "unobserved"
        self._assert_refuses(entry)

    def test_non_boolean_async_op_refuses(self):
        entry = boundary(0, 0)
        entry["events"]["async_op"] = "no"
        self._assert_refuses(entry)

    # --- disclosure coherence -----------------------------------------------

    def test_stripped_inflight_read_disclosure_refuses(self):
        entry = boundary(0, 0, async_op=True)
        entry["disclosures"] = []
        self._assert_refuses(entry)

    def test_spurious_inflight_read_disclosure_refuses(self):
        entry = boundary(0, 0)
        entry["disclosures"] = ["read_of_inflight_destination"]
        self._assert_refuses(entry)

    def test_unknown_disclosure_token_refuses(self):
        entry = boundary(0, 0)
        entry["disclosures"] = ["totally_fine_trust_me"]
        self._assert_refuses(entry)

    def test_group_seq_value_with_read_failed_disclosure_refuses(self):
        entry = boundary(0, 0, c10d_group_seq=7)
        entry["disclosures"] = ["c10d_group_seq_read_failed"]
        self._assert_refuses(entry)

    # --- not_present_reason coherence ----------------------------------------

    def test_not_present_reason_outside_vocabulary_refuses(self):
        entry = boundary(0, 0)
        entry["witness"]["not_present_reason"] = "because"
        self._assert_refuses(entry)

    def test_spurious_async_reason_on_sync_digest_record_refuses(self):
        entry = boundary(
            0,
            0,
            witness_policy="digest",
            contribution_digests=["c" * 64],
            destination_digests=["a" * 64],
        )
        entry["witness"]["not_present_reason"] = "async_completion_unobserved"
        self._assert_refuses(entry)

    def test_missing_async_reason_on_async_digest_record_refuses(self):
        entry = boundary(
            0,
            0,
            async_op=True,
            witness_policy="digest",
            contribution_digests=["c" * 64],
        )
        entry["witness"]["not_present_reason"] = None
        self._assert_refuses(entry)

    # --- op_node / peer / lifetime coherence ---------------------------------

    def test_op_node_false_on_tensor_kind_refuses(self):
        entry = boundary(0, 0)
        entry["op_node"] = False
        self._assert_refuses(entry)

    def test_op_node_true_on_tensorless_kind_refuses(self):
        entry = boundary(0, 0, kind="barrier", reduce_op=None, roles=[])
        entry["op_node"] = True
        self._assert_refuses(entry)

    def test_peer_record_on_collective_kind_refuses(self):
        entry = boundary(0, 0)
        entry["peer"] = {"canonical": {"src": 0, "dst": 1}}
        self._assert_refuses(entry)

    def test_missing_peer_record_on_p2p_kind_refuses(self):
        entry = boundary(0, 0, kind="send", channel="p2p/0->1", reduce_op=None)
        entry["peer"] = None
        self._assert_refuses(entry)

    def test_lifetime_epoch_outside_vocabulary_refuses(self):
        entry = boundary(0, 0)
        entry["lifetime_evidence"]["install_epoch"] = "definitely_complete"
        self._assert_refuses(entry)

    def test_boundary_epoch_promotion_against_record_epoch_refuses(self):
        entry = boundary(0, 0)
        entry["lifetime_evidence"]["install_epoch"] = "armed_before_any_group"
        self._assert_refuses(entry)

    # --- tensorless empty digest lists (honest recorder shape) ---------------

    def test_tensorless_empty_digest_lists_still_parse(self):
        """The recorder emits [] digest lists for barrier under policy digest."""

        entry = boundary(0, 0, kind="barrier", reduce_op=None, roles=[], witness_policy="digest")
        entry["witness"]["contribution_digests"] = []
        entry["witness"]["destination_digests"] = []
        self._extract(entry)  # must not raise

    def test_tensor_kind_empty_digest_list_still_refuses(self):
        entry = boundary(0, 0, witness_policy="digest")
        entry["witness"]["contribution_digests"] = []
        self._assert_refuses(entry)


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
            {
                "role": "contribution",
                "index": 0,
                "shape": [2],
                "logical_shape": None,
                "placements": None,
            },
            {
                "role": "destination",
                "index": 0,
                "shape": [2],
                "logical_shape": None,
                "placements": None,
            },
        ]  # group of 2 but only ONE destination
        d = derive_merge(
            {
                0: evidence(0, [boundary(0, 0, kind="all_gather", roles=roles, reduce_op=None)]),
                1: evidence(1, [boundary(1, 0, kind="all_gather", roles=roles, reduce_op=None)]),
            }
        )
        assert d.stored_alignment is MergeAlignment.CONFLICTED

    def test_backend_disagreement_conflicts_and_demotes_witness(self):
        """Deep-hunt F4: cross-rank backend disagreement is never silent.

        Fail-before: ``group_backend.setdefault`` was first-writer-wins in
        rank order -- rank 0 claiming "gloo" flipped the group into the
        witness-verdict backends and the join rendered ATTESTED under gloo
        contract semantics while rank 1 recorded an unknown backend.
        """

        d = derive_merge(
            {
                0: evidence(0, [boundary(0, 0, backend="gloo", **digest_kwargs())]),
                1: evidence(1, [boundary(1, 0, backend="mystery_backend", **digest_kwargs())]),
            }
        )
        assert d.stored_alignment is MergeAlignment.CONFLICTED
        assert any(f.kind == "relation_violation" and "backend" in f.detail for f in d.findings)
        # The disputed backend is demoted: never verdict-grade.
        assert d.joins[0].backend is None
        assert d.joins[0].consistency is BoundaryConsistency.NOT_APPLICABLE

    def test_backend_agreement_reports_no_finding(self):
        d = derive_merge(
            {
                0: evidence(0, [boundary(0, 0, **digest_kwargs())]),
                1: evidence(1, [boundary(1, 0, **digest_kwargs())]),
            }
        )
        assert d.stored_alignment is MergeAlignment.ALIGNED
        assert d.joins[0].backend == "gloo"

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

    def test_armed_rank_base_misalignment_is_a_correlation_conflict(self):
        """Deep-hunt F5: differing capture windows cannot fabricate a join.

        Rank 0 recorded absolute seqs {0, 1}; rank 1 recorded {1} only. Delta
        alignment paired rank 0's seq 0 with rank 1's seq 1 -- two DIFFERENT
        collectives presented as one honest correspondence (invisible under
        witness "none", and the c10d cross-check cancels constant offsets).
        Both ranks are armed before any group, so their counters tick on every
        issue and equal-seq is provable: the disagreement must conflict.
        """

        d = derive_merge(
            {
                0: evidence(
                    0,
                    [boundary(0, 0), boundary(0, 1)],
                    ledger=armed_ledger(),
                    epoch="armed_before_any_group",
                ),
                1: evidence(
                    1,
                    [boundary(1, 1)],
                    ledger=armed_ledger(),
                    epoch="armed_before_any_group",
                ),
            }
        )
        assert d.stored_alignment is MergeAlignment.CONFLICTED
        assert any(
            f.kind == "correlation_delta_mismatch" and "absolute issue sequences" in f.detail
            for f in d.findings
        )

    def test_armed_ranks_with_equal_absolute_seqs_stay_aligned(self):
        d = derive_merge(
            {
                0: evidence(
                    0,
                    [boundary(0, 3), boundary(0, 4)],
                    ledger=armed_ledger(),
                    epoch="armed_before_any_group",
                ),
                1: evidence(
                    1,
                    [boundary(1, 3), boundary(1, 4)],
                    ledger=armed_ledger(),
                    epoch="armed_before_any_group",
                ),
            }
        )
        assert d.stored_alignment is MergeAlignment.ALIGNED

    def test_seeded_rank_base_offsets_never_compared(self):
        # Mixed epochs: the seeded rank's absolute base is a rank-local fact
        # (arm-time histories differ); only armed-before-any-group ranks are
        # held to equal absolute seqs, so this stays an honest delta join.
        d = derive_merge(
            {
                0: evidence(
                    0,
                    [boundary(0, 0)],
                    ledger=armed_ledger(),
                    epoch="armed_before_any_group",
                ),
                1: evidence(1, [boundary(1, 7)], ledger=seeded_ledger()),
            }
        )
        assert d.stored_alignment is MergeAlignment.ALIGNED

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
                            0,
                            0,
                            contribution_digests=["r0-pre"],
                            **{k: v for k, v in kwargs.items() if k != "contribution_digests"},
                        )
                    ],
                ),
                1: evidence(
                    1,
                    [
                        boundary(
                            1,
                            0,
                            contribution_digests=["r1-pre"],
                            **{k: v for k, v in kwargs.items() if k != "contribution_digests"},
                        )
                    ],
                ),
            }
        )
        assert d.joins[0].consistency is BoundaryConsistency.NOT_PRESENT
        assert d.stored_value_status is MergeValueStatus.UNWITNESSED

    def test_reduce_is_not_applicable_at_every_level(self):
        roles_root = [
            {
                "role": "contribution_destination",
                "index": 0,
                "shape": [2],
                "logical_shape": None,
                "placements": None,
            },
        ]
        roles_leaf = [
            {
                "role": "contribution",
                "index": 0,
                "shape": [2],
                "logical_shape": None,
                "placements": None,
            },
        ]
        d = derive_merge(
            {
                0: evidence(
                    0,
                    [boundary(0, 0, kind="reduce", roles=roles_root, **digest_kwargs())],
                ),
                1: evidence(
                    1,
                    [
                        boundary(
                            1,
                            0,
                            kind="reduce",
                            roles=roles_leaf,
                            witness_policy="digest",
                            contribution_digests=["x"],
                        )
                    ],
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
            {
                "role": "contribution",
                "index": 0,
                "shape": [2],
                "logical_shape": None,
                "placements": None,
            },
        ]
        leaf_roles = [
            {
                "role": "destination",
                "index": 0,
                "shape": [2],
                "logical_shape": None,
                "placements": None,
            },
        ]
        d = derive_merge(
            {
                0: evidence(
                    0,
                    [
                        boundary(
                            0,
                            0,
                            kind="broadcast",
                            roles=root_roles,
                            reduce_op=None,
                            witness_policy="digest",
                            contribution_digests=["same"],
                        )
                    ],
                ),
                1: evidence(
                    1,
                    [
                        boundary(
                            1,
                            0,
                            kind="broadcast",
                            roles=leaf_roles,
                            reduce_op=None,
                            witness_policy="digest",
                            destination_digests=["same"],
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


class TestPresenterLookupNarrowing:
    """Deep-hunt F7: ``__getitem__``'s rank scan must not swallow core defects.

    Fail-before: ``except Exception`` read a rank core whose lookup raised
    ``RuntimeError`` as a MISS, so a defective core silently vanished and
    another rank's hit presented as an unambiguous single-rank result --
    ``super_op`` directly below was already narrowed (b5 R45-2) for exactly
    this reason.
    """

    class _BrokenTrace:
        def __getitem__(self, item):
            raise RuntimeError("corrupt core: internal invariant violated")

    class _GoodTrace:
        def __getitem__(self, item):
            return f"op<{item}>"

    class _MissTrace:
        def __getitem__(self, item):
            raise KeyError(item)

    def _merged(self, trace0, trace1):
        from torchlens.merged._presenter import MergedTrace, _RankHandle

        derivation = derive_merge(
            {0: evidence(0, [boundary(0, 0)]), 1: evidence(1, [boundary(1, 0)])}
        )
        return MergedTrace(
            derivation,
            {0: _RankHandle(0, trace=trace0), 1: _RankHandle(1, trace=trace1)},
        )

    def test_rank_core_defect_surfaces_from_getitem(self):
        merged = self._merged(self._BrokenTrace(), self._GoodTrace())
        with pytest.raises(RuntimeError, match="corrupt core"):
            merged["relu_1_2"]

    def test_lookup_miss_still_reads_as_a_miss(self):
        merged = self._merged(self._MissTrace(), self._GoodTrace())
        assert merged["relu_1_2"] == "op<relu_1_2>"


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

    def test_tree_hash_framing_matches_frozen_contract(self) -> None:
        """The tree-hash prose must name the implementation's unambiguous framing."""

        doc = self.DOC.read_text()
        assert "8-byte big-endian path length" in doc
        assert re.search(r"8-byte big-endian\s+file\s+size", doc)
        assert "32 raw SHA-256 bytes" in doc


def _hex64(char: str) -> str:
    """A syntactically valid lowercase SHA-256 hex digest for fixtures."""

    return char * 64


class TestOpLabelBackReferenceForgery:
    """R18-1(i): op-label back-references parse against the recorder's shape.

    Fail-before (b6 R18 merged-evidence forging family): ``op_labels_raw``
    was validated only as list-of-strings, so fabricated, EMPTY, or DUPLICATE
    label lists derived ``aligned``/``attested_complete`` with zero findings
    while the join table's op back-references lied. The recorder emits one
    label per logged boundary output tensor (>= 1 for every op-bearing kind,
    exactly 0 for tensorless kinds), so those shapes refuse at the one parse
    chokepoint merge time and load rederivation share.

    Declared residual (documented, not open): a FABRICATED but well-formed
    label list still parses -- the parse chokepoint never dereferences the
    member trace (synthetic evidence carriers and payload-free loads have no
    resolvable op table). It cannot improve any verdict (no verdict reads the
    labels) and every consumer that RESOLVES back-references
    (``MergedTrace.join_ops``) refuses typed on an unresolvable label; a
    forger who also reauthors the member's op table to match is the
    documented coherent-reauthoring boundary.
    """

    def _assert_refuses(self, entry: dict) -> None:
        with pytest.raises(MergeInputError) as excinfo:
            extract_rank_evidence(
                trace_for_boundaries([entry], seeded_ledger()),
                "op-label-tamper",
            )
        assert excinfo.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value

    def test_empty_op_labels_on_op_bearing_kind_refuses(self):
        entry = boundary(0, 0)
        entry["op_labels_raw"] = []
        self._assert_refuses(entry)

    def test_duplicate_op_labels_refuse(self):
        entry = boundary(0, 0)
        entry["op_labels_raw"] = ["allreduce_1_raw", "allreduce_1_raw"]
        self._assert_refuses(entry)

    def test_labels_on_tensorless_kind_refuse(self):
        entry = boundary(0, 0, kind="barrier", reduce_op=None, roles=[])
        entry["op_labels_raw"] = ["barrier_1_raw"]
        self._assert_refuses(entry)

    def test_tensorless_empty_labels_still_parse(self):
        entry = boundary(0, 0, kind="barrier", reduce_op=None, roles=[])
        extract_rank_evidence(
            trace_for_boundaries([entry], seeded_ledger()),
            "honest-barrier",
        )

    def test_fabricated_label_refuses_typed_at_join_ops_access(self):
        """The access-time half of the residual: resolution is fail-closed."""

        from torchlens.merged._presenter import MergedTrace, _RankHandle

        class _UnresolvingCore:
            _raw_to_final_op_labels: dict = {}

            def __getitem__(self, item):
                raise KeyError(item)

        derivation = derive_merge(
            {0: evidence(0, [boundary(0, 0)]), 1: evidence(1, [boundary(1, 0)])}
        )
        merged = MergedTrace(
            derivation,
            {
                0: _RankHandle(0, trace=_UnresolvingCore()),
                1: _RankHandle(1, trace=_UnresolvingCore()),
            },
        )
        (join,) = merged.joins
        with pytest.raises(MergeInputError) as excinfo:
            merged.join_ops(join)
        assert excinfo.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value


class TestRoleIndexValidation:
    """R18-1(ii): ``roles[].index`` parses typed (was unvalidated and unread)."""

    def _role(self, name: str, index) -> dict:
        return {
            "role": name,
            "index": index,
            "shape": [2],
            "logical_shape": None,
            "placements": None,
        }

    def _assert_refuses(self, roles: list[dict]) -> None:
        with pytest.raises(MergeInputError) as excinfo:
            extract_rank_evidence(
                trace_for_boundaries([boundary(0, 0, roles=roles)], seeded_ledger()),
                "role-index-tamper",
            )
        assert excinfo.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value

    def test_missing_index_refuses(self):
        role = self._role("contribution", 0)
        del role["index"]
        self._assert_refuses([role])

    def test_negative_index_refuses(self):
        self._assert_refuses([self._role("contribution", -1)])

    def test_boolean_index_refuses(self):
        self._assert_refuses([self._role("contribution", True)])

    def test_non_integer_index_refuses(self):
        self._assert_refuses([self._role("contribution", "0")])

    def test_duplicate_index_within_one_role_name_refuses(self):
        self._assert_refuses([self._role("contribution", 0), self._role("contribution", 0)])

    def test_positional_holes_across_role_names_still_parse(self):
        # NOT dense 0..n-1 by design: the recorder keys contribution roles by
        # INPUT position and destination roles by OUTPUT position, and a
        # tensor serving as both keeps its input position -- leaving an
        # honest hole in the destination positions.
        extract_rank_evidence(
            trace_for_boundaries(
                [
                    boundary(
                        0,
                        0,
                        roles=[
                            self._role("contribution_destination", 0),
                            self._role("destination", 1),
                        ],
                    )
                ],
                seeded_ledger(),
            ),
            "honest-role-holes",
        )


class TestGroupRecordForgery:
    """R18-2: the group record's size / group-rank redundancies are closed.

    Fail-before: ``group["size"]`` was written by the recorder but never
    validated; ``my_group_rank`` was never tied to the membership position it
    definitionally equals, so a permuted value silently rebound slice-witness
    pairings and an out-of-range value crashed the engine's group-rank list
    indexing as a raw IndexError; and selectively STRIPPING ``my_group_rank``
    from a slice-witnessed join deleted its ``value_divergence`` finding.
    """

    def _assert_refuses(self, boundaries: list[dict]) -> None:
        with pytest.raises(MergeInputError) as excinfo:
            extract_rank_evidence(
                trace_for_boundaries(boundaries, seeded_ledger()),
                "group-record-tamper",
            )
        assert excinfo.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value

    def test_size_deleted_refuses(self):
        entry = boundary(0, 0)
        del entry["group"]["size"]
        self._assert_refuses([entry])

    def test_size_mismatch_refuses(self):
        entry = boundary(0, 0)
        entry["group"]["size"] = 3
        self._assert_refuses([entry])

    def test_boolean_size_refuses(self):
        entry = boundary(0, 0)
        entry["group"]["size"] = True
        self._assert_refuses([entry])

    def test_permuted_my_group_rank_refuses(self):
        entry = boundary(0, 0)  # rank 0 of members (0, 1): position is 0
        entry["group"]["my_group_rank"] = 1
        self._assert_refuses([entry])

    def test_out_of_range_my_group_rank_refuses(self):
        entry = boundary(0, 0)
        entry["group"]["my_group_rank"] = 5
        self._assert_refuses([entry])

    def test_selective_my_group_rank_strip_refuses(self):
        # The writer mints None only when dist.get_group_rank RAISES -- a
        # group-level fact -- so mixed presence within one core+group is
        # tamper. This is exactly the strip that used to delete a
        # value_divergence finding from a slice-witnessed join.
        stripped = boundary(0, 0)
        stripped["group"]["my_group_rank"] = None
        self._assert_refuses([stripped, boundary(0, 1)])

    def test_uniform_absence_still_parses(self):
        # get_group_rank genuinely raising on this rank's runtime is honest:
        # every boundary of the group records None uniformly.
        cores = []
        for seq in (0, 1):
            entry = boundary(0, seq)
            entry["group"]["my_group_rank"] = None
            cores.append(entry)
        extract_rank_evidence(
            trace_for_boundaries(cores, seeded_ledger()),
            "honest-uniform-absence",
        )

    def test_engine_belt_refuses_out_of_range_group_rank_typed(self):
        """Direct engine evidence gets a typed refusal, never an IndexError."""

        def role(name: str, index: int) -> dict:
            return {
                "role": name,
                "index": index,
                "shape": [2],
                "logical_shape": None,
                "placements": None,
            }

        root = boundary(
            0,
            0,
            kind="gather",
            reduce_op=None,
            roles=[role("contribution", 0), role("destination", 0), role("destination", 1)],
            witness_policy="digest",
            contribution_digests=[_hex64("a")],
            destination_digests=[_hex64("a"), _hex64("b")],
        )
        leaf = boundary(
            1,
            0,
            kind="gather",
            reduce_op=None,
            roles=[role("contribution", 0)],
            witness_policy="digest",
            contribution_digests=[_hex64("b")],
        )
        leaf["group"]["my_group_rank"] = 7  # bypasses parse: direct evidence
        with pytest.raises(MergeInputError) as excinfo:
            derive_merge({0: evidence(0, [root]), 1: evidence(1, [leaf])})
        assert excinfo.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value


class TestGatherDivergenceSurvivesGroupRankTamper:
    """Findings-never-deleted for the verified my_group_rank strip escape."""

    def _role(self, name: str, index: int) -> dict:
        return {
            "role": name,
            "index": index,
            "shape": [2],
            "logical_shape": None,
            "placements": None,
        }

    def _root(self) -> dict:
        return boundary(
            0,
            0,
            kind="gather",
            reduce_op=None,
            roles=[
                self._role("contribution", 0),
                self._role("destination", 0),
                self._role("destination", 1),
            ],
            witness_policy="digest",
            contribution_digests=[_hex64("a")],
            # Slice 1 disagrees with the leaf's contribution: value_divergence.
            destination_digests=[_hex64("a"), _hex64("f")],
        )

    def _leaf(self) -> dict:
        return boundary(
            1,
            0,
            kind="gather",
            reduce_op=None,
            roles=[self._role("contribution", 0)],
            witness_policy="digest",
            contribution_digests=[_hex64("b")],
        )

    def _extract(self, rank: int, boundaries: list[dict]):
        return extract_rank_evidence(
            trace_for_boundaries(boundaries, seeded_ledger()),
            f"gather-tamper[{rank}]",
        )

    def test_honest_mismatch_yields_value_divergence(self):
        d = derive_merge({0: self._extract(0, [self._root()]), 1: self._extract(1, [self._leaf()])})
        assert d.joins[0].consistency is BoundaryConsistency.MISMATCHED
        assert any(f.kind == "value_divergence" for f in d.findings)

    def test_selective_strip_refuses_at_parse(self):
        # A leaf core holding another boundary of the SAME group with a
        # recorded group rank: stripping only the gather join's my_group_rank
        # is mixed presence and refuses -- the finding can no longer be
        # deleted by the one-field strip that used to work.
        stripped = self._leaf()
        stripped["group"]["my_group_rank"] = None
        with pytest.raises(MergeInputError) as excinfo:
            self._extract(1, [stripped, boundary(1, 1)])
        assert excinfo.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value

    def test_group_wide_strip_retreats_to_no_claim_never_attested(self):
        # Stripping the WHOLE group's my_group_rank on a single-boundary core
        # is byte-identical to an honest get_group_rank failure, so it parses
        # (uniform absence). The witness derivation then retreats to
        # NOT_PRESENT/unwitnessed -- a retreat to NO claim, consistent with
        # the demote-only witness model (identical to stripping the digests
        # themselves) -- and can never IMPROVE to any attested flavor.
        stripped = self._leaf()
        stripped["group"]["my_group_rank"] = None
        d = derive_merge({0: self._extract(0, [self._root()]), 1: self._extract(1, [stripped])})
        assert d.joins[0].consistency is BoundaryConsistency.NOT_PRESENT
        assert d.stored_value_status is MergeValueStatus.UNWITNESSED


class TestC10dGroupSeqLatch:
    """R18-3: per-core c10d_group_seq presence follows the recorder's latch.

    Fail-before: only value+disclosure COEXISTENCE was refused, so nulling a
    single recorded value (or deleting the key) silently removed a
    ``correlation_delta_mismatch`` finding -- the only in-band detector in
    the base-misalignment neighborhood. Legal per-core shapes are exactly the
    recorder's: all values; all null (capability absent); or a prefix of
    values, ONE boundary disclosing ``c10d_group_seq_read_failed``, then an
    all-null suffix.
    """

    def _assert_refuses(self, boundaries: list[dict]) -> None:
        with pytest.raises(MergeInputError) as excinfo:
            extract_rank_evidence(
                trace_for_boundaries(boundaries, seeded_ledger()),
                "seq-latch-tamper",
            )
        assert excinfo.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value

    def test_deleted_key_refuses(self):
        entry = boundary(0, 0)
        del entry["c10d_group_seq"]
        self._assert_refuses([entry])

    def test_undisclosed_drop_after_values_refuses(self):
        self._assert_refuses(
            [boundary(0, 0, c10d_group_seq=10), boundary(0, 1, c10d_group_seq=None)]
        )

    def test_value_after_null_refuses(self):
        self._assert_refuses(
            [boundary(0, 0, c10d_group_seq=None), boundary(0, 1, c10d_group_seq=11)]
        )

    def test_value_after_disclosed_drop_refuses(self):
        dropped = boundary(0, 1, c10d_group_seq=None)
        dropped["disclosures"] = ["c10d_group_seq_read_failed"]
        self._assert_refuses(
            [boundary(0, 0, c10d_group_seq=10), dropped, boundary(0, 2, c10d_group_seq=12)]
        )

    def test_repeated_disclosure_after_latch_refuses(self):
        first = boundary(0, 0, c10d_group_seq=None)
        first["disclosures"] = ["c10d_group_seq_read_failed"]
        second = boundary(0, 1, c10d_group_seq=None)
        second["disclosures"] = ["c10d_group_seq_read_failed"]
        self._assert_refuses([first, second])

    def test_honest_latch_shape_parses(self):
        dropped = boundary(0, 1, c10d_group_seq=None)
        dropped["disclosures"] = ["c10d_group_seq_read_failed"]
        extract_rank_evidence(
            trace_for_boundaries(
                [
                    boundary(0, 0, c10d_group_seq=10),
                    dropped,
                    boundary(0, 2, c10d_group_seq=None),
                ],
                seeded_ledger(),
            ),
            "honest-latch",
        )

    def test_selective_null_cannot_delete_the_delta_mismatch_finding(self):
        # The verified escape, cheap-tamper arm: the delta-disagreement
        # fixture (10,11 vs 20,25) conflicts; nulling ONE contradicting value
        # now refuses at parse instead of silently un-finding the conflict.
        with pytest.raises(MergeInputError) as excinfo:
            extract_rank_evidence(
                trace_for_boundaries(
                    [
                        boundary(1, 0, c10d_group_seq=20),
                        boundary(1, 1, c10d_group_seq=None),
                    ],
                    seeded_ledger(),
                ),
                "selective-null",
            )
        assert excinfo.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value

    def test_all_null_rewrite_is_the_documented_reauthoring_residual(self):
        # Nulling EVERY value on EVERY core is byte-identical to an honest
        # capability-absent capture (the probe returns null with no
        # disclosure from the first boundary on), so it parses and the
        # cross-check honestly reports nothing: "absence of the probe never
        # demotes anything" (contract section 4), and
        # test_c10d_group_seq_absent_never_demotes pins the honest side.
        # This is the coherent-reauthoring boundary -- a weaker program's
        # honest capture -- documented here as a scope statement, not an
        # open residual; the latch rules above force a forger all the way to
        # it instead of the one-field null that used to suffice.
        def cores(seqs0, seqs1):
            return {
                0: extract_rank_evidence(
                    trace_for_boundaries(
                        [boundary(0, i, c10d_group_seq=s) for i, s in enumerate(seqs0)],
                        seeded_ledger(),
                    ),
                    "all-null[0]",
                ),
                1: extract_rank_evidence(
                    trace_for_boundaries(
                        [boundary(1, i, c10d_group_seq=s) for i, s in enumerate(seqs1)],
                        seeded_ledger(),
                    ),
                    "all-null[1]",
                ),
            }

        honest = derive_merge(cores([10, 11], [20, 25]))
        assert honest.stored_alignment is MergeAlignment.CONFLICTED
        rewritten = derive_merge(cores([None, None], [None, None]))
        assert rewritten.stored_alignment is MergeAlignment.ALIGNED
        assert not any(f.kind == "correlation_delta_mismatch" for f in rewritten.findings)


class TestUniformRoleShapeRewrite:
    """R18-1(iii) fallback: asymmetric shape tamper is caught for every
    symmetric kind; the uniform rewrite is the documented residual.

    The full per-core cross-check (role shapes against the member's own op
    records) is not implementable at the parse chokepoint: parse never
    dereferences the member trace (synthetic evidence carriers, payload-free
    and analysis-only loads have no uniformly readable op table). The engine
    belt below extends shape agreement to the remaining symmetric kinds as
    distinct-shape-SET agreement (per-rank role COUNTS legitimately differ:
    root list vs leaf tensor), so an asymmetric rewrite conflicts; a rewrite
    applied identically on EVERY core remains coherent reauthoring -- the
    documented out-of-scope boundary.
    """

    def _all_gather(self, rank: int, shape: list[int]) -> dict:
        def role(name: str, index: int) -> dict:
            return {
                "role": name,
                "index": index,
                "shape": list(shape),
                "logical_shape": None,
                "placements": None,
            }

        return boundary(
            rank,
            0,
            kind="all_gather",
            reduce_op=None,
            roles=[role("contribution", 0), role("destination", 0), role("destination", 1)],
        )

    def test_asymmetric_shape_rewrite_conflicts(self):
        d = derive_merge(
            {
                0: evidence(0, [self._all_gather(0, [2])]),
                1: evidence(1, [self._all_gather(1, [9, 9])]),
            }
        )
        assert d.stored_alignment is MergeAlignment.CONFLICTED
        assert any(
            f.kind == "relation_violation" and "distinct role shapes" in f.detail
            for f in d.findings
        )

    def test_uniform_shape_rewrite_is_the_documented_residual(self):
        d = derive_merge(
            {
                0: evidence(0, [self._all_gather(0, [9, 9])]),
                1: evidence(1, [self._all_gather(1, [9, 9])]),
            }
        )
        assert d.stored_alignment is MergeAlignment.ALIGNED


class TestMemberOutcomeGate:
    """R06c: member capture outcomes gate the merge at input resolution.

    Fail-before: ``merge_ranks`` never consulted member outcomes, so a HALTED
    (or synthetic FAILED) member core merged into aligned/attested_complete
    with zero findings and no disclosure anywhere on the merged surface.
    """

    def _core(self, rank: int, status=None, boundaries=None):
        from torchlens.capture.outcome import CaptureOutcome

        trace = trace_for_boundaries(
            boundaries if boundaries is not None else [boundary(rank, 0)],
            seeded_ledger(),
        )
        if status is not None:
            trace._capture_outcome = CaptureOutcome(status=status)
        return trace

    def _assert_refuses(self, cores) -> None:
        from torchlens.merged import merge_ranks

        with pytest.raises(MergeInputError) as excinfo:
            merge_ranks(cores)
        assert excinfo.value.fields["code"] == MergedErrorCode.MERGE_INPUT_INVALID.value
        assert excinfo.value.fields["reason"] == "member_outcome_not_mergeable"

    def test_failed_member_refuses_typed(self):
        from torchlens.capture.outcome import CaptureStatus

        self._assert_refuses([self._core(0, CaptureStatus.FAILED), self._core(1)])

    def test_aborted_nonfinite_member_refuses_typed(self):
        from torchlens.capture.outcome import CaptureStatus

        self._assert_refuses([self._core(0), self._core(1, CaptureStatus.ABORTED_NONFINITE)])

    def test_unknown_member_refuses_typed(self):
        from torchlens.capture.outcome import CaptureStatus

        self._assert_refuses([self._core(0, CaptureStatus.UNKNOWN), self._core(1)])

    def test_merge_report_shares_the_gate(self):
        from torchlens.capture.outcome import CaptureStatus
        from torchlens.merged import merge_report

        with pytest.raises(MergeInputError) as excinfo:
            merge_report([self._core(0, CaptureStatus.FAILED), self._core(1)])
        assert excinfo.value.fields["reason"] == "member_outcome_not_mergeable"

    def _attested_pair(self, status0, status1):
        def core(rank: int, status):
            return self._core(
                rank,
                status,
                [
                    boundary(
                        rank,
                        0,
                        witness_policy="digest",
                        contribution_digests=[_hex64("c")],
                        destination_digests=[_hex64("a")],
                    )
                ],
            )

        return [core(0, status0), core(1, status1)]

    def test_halted_member_merges_with_disclosure_never_silent_attested_complete(self):
        from torchlens.capture.outcome import CaptureStatus
        from torchlens.merged import merge_ranks

        merged = merge_ranks(self._attested_pair(CaptureStatus.HALTED, CaptureStatus.COMPLETE))
        assert merged.value_status is MergeValueStatus.ATTESTED_COMPLETE
        assert merged.member_outcomes == {0: "halted", 1: "complete"}
        summary = merged.summary()
        assert "halted" in summary and "attested_complete" in summary
        # The disclosure precedes the witness-coverage claim.
        assert summary.index("halted") < summary.index("attested_complete")

    def test_unattested_member_merges_with_disclosure(self):
        from torchlens.capture.outcome import CaptureStatus
        from torchlens.merged import merge_ranks

        merged = merge_ranks(self._attested_pair(CaptureStatus.COMPLETE, CaptureStatus.UNATTESTED))
        assert merged.member_outcomes[1] == "unattested"
        assert "unattested" in merged.summary()

    def test_disclosure_survives_release(self):
        from torchlens.capture.outcome import CaptureStatus
        from torchlens.merged import merge_ranks

        merged = merge_ranks(self._attested_pair(CaptureStatus.HALTED, CaptureStatus.COMPLETE))
        merged.release()
        assert merged.member_outcomes == {0: "halted", 1: "complete"}
        assert "halted" in merged.summary()

    def test_members_without_outcome_sidecar_make_no_claim(self):
        from torchlens.merged import merge_ranks

        merged = merge_ranks([self._core(0), self._core(1)])
        assert merged.member_outcomes == {}
        assert "member capture outcomes" not in merged.summary()

    def test_complete_members_carry_no_disclosure_line(self):
        from torchlens.capture.outcome import CaptureStatus
        from torchlens.merged import merge_ranks

        merged = merge_ranks(self._attested_pair(CaptureStatus.COMPLETE, CaptureStatus.COMPLETE))
        assert merged.member_outcomes == {0: "complete", 1: "complete"}
        assert "member capture outcomes" not in merged.summary()

    def test_path_supplied_member_gets_the_same_gate(self, monkeypatch, tmp_path):
        """Path inputs load through resolve_rank_inputs' one chokepoint."""

        import torchlens._io.bundle as bundle_module
        from torchlens.capture.outcome import CaptureStatus
        from torchlens.merged import merge_ranks

        failed = self._core(0, CaptureStatus.FAILED)
        monkeypatch.setattr(bundle_module, "load", lambda path: failed)
        fake = tmp_path / "rank0.tlspec"
        with pytest.raises(MergeInputError) as excinfo:
            merge_ranks([fake])
        assert excinfo.value.fields["reason"] == "member_outcome_not_mergeable"
        assert excinfo.value.fields["source"] == str(fake)

    def test_path_supplied_halted_member_is_disclosed(self, monkeypatch, tmp_path):
        import torchlens._io.bundle as bundle_module
        from torchlens.capture.outcome import CaptureStatus
        from torchlens.merged import merge_ranks

        cores = {
            str(tmp_path / "rank0.tlspec"): self._attested_pair(
                CaptureStatus.HALTED, CaptureStatus.COMPLETE
            )[0]
        }
        monkeypatch.setattr(bundle_module, "load", lambda path: cores[str(path)])
        merged = merge_ranks(
            [tmp_path / "rank0.tlspec", self._attested_pair(None, CaptureStatus.COMPLETE)[1]]
        )
        assert merged.member_outcomes[0] == "halted"
        assert "halted" in merged.summary()


class TestHaltedMemberArtifactRoundTrip:
    """A merged artifact over a genuinely-persisted HALTED member core loads
    clean and keeps the disclosure -- the same derivation reruns verbatim at
    load, so the outcome gate/disclosure never perturbs descriptor equality."""

    def _real_core(self, rank: int, halted: bool):
        import torch
        from torch import nn

        import torchlens as tl

        torch.manual_seed(0)
        log = tl.trace(nn.Linear(4, 4), torch.randn(2, 4))
        if halted:
            from torchlens.capture.outcome import CaptureOutcome, CaptureStatus

            # Persist a coherent HALTED attestation: the structural halted
            # marker plus the settled stamp, exactly what a halt= capture
            # leaves behind (analysis-level member saves accept halted cores).
            log.halted = True
            log._capture_outcome = CaptureOutcome(
                status=CaptureStatus.HALTED, reason="synthetic halt for merge disclosure"
            )
        entry = boundary(
            rank,
            0,
            witness_policy="digest",
            contribution_digests=[_hex64("c")],
            destination_digests=[_hex64("a")],
        )
        entry["op_labels_raw"] = ["allreduce_1_raw"]
        log.annotations["distributed"] = {
            "boundaries": [entry],
            "group_lifecycle_ledger": seeded_ledger().to_payload(),
            "install_epoch": "seeded",
        }
        return log

    def test_halted_member_round_trip_keeps_disclosure(self, tmp_path):
        from torchlens.merged import merge_ranks
        from torchlens.merged._artifact import load_merged

        merged = merge_ranks([self._real_core(0, halted=True), self._real_core(1, halted=False)])
        assert merged.member_outcomes == {0: "halted", 1: "complete"}
        art = tmp_path / "halted_member.tlspec"
        merged.save(art)
        loaded = load_merged(art)
        assert loaded.load_degradations == ()
        assert loaded.member_outcomes == {0: "halted", 1: "complete"}
        assert "halted" in loaded.summary()
        assert loaded.value_status is merged.value_status


class TestModuleLevelSaveRefusesTyped:
    """tl.save(merged, ...) is a contract-promised typed refusal, not an AttributeError."""

    def test_tl_save_merged_trace_refuses_typed(self, tmp_path):
        # The module-level bundle save used to reach the runnable poison gate
        # and die as a bare AttributeError ('MergedTrace' has no '_runnable');
        # the contract (2.5) promises every merged export surface refuses typed.
        import torchlens as tl
        from torchlens.merged._errors import MergedSurfaceUnsupportedError
        from torchlens.merged._presenter import MergedTrace, _RankHandle

        derivation = derive_merge(
            {0: evidence(0, [boundary(0, 0)]), 1: evidence(1, [boundary(1, 0)])}
        )
        merged = MergedTrace(
            derivation, {0: _RankHandle(0, trace=None), 1: _RankHandle(1, trace=None)}
        )
        with pytest.raises(MergedSurfaceUnsupportedError) as excinfo:
            tl.save(merged, tmp_path / "merged_refused.tlspec")
        assert excinfo.value.fields["code"] == "merged_surface_unsupported"
        assert not (tmp_path / "merged_refused.tlspec").exists()
