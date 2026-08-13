"""C1 merge engine: gloo sims (single-process world + multi-rank spawn).

End-to-end pins for ``tl.merge_ranks`` / ``tl.merge_report`` / the
``merged-directory`` artifact over REAL rank cores: live merge, artifact
round-trip with full load rederivation, the tamper matrix, determinism,
loaded-vs-live parity, multi-rank joins, subset-merge presence gaps, and the
live asymmetric-arming conflict refusal (seed-discharge negative, v5 1.3).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil

import pytest
import torch
from torch import nn

pytestmark = pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="torch.distributed gloo unavailable",
)

import torchlens as tl  # noqa: E402
from torchlens.distributed import _lifecycle as lifecycle  # noqa: E402
from torchlens.merged import (  # noqa: E402
    MergeConflictError,
    MergedArtifactError,
    MergedSurfaceUnsupportedError,
    MergeInputError,
)
from torchlens.merged._artifact import canonical_json_bytes  # noqa: E402


@pytest.fixture()
def gloo_world(tmp_path):
    """Single-process gloo world, armed state guaranteed clean around a test."""

    import torch.distributed as dist

    lifecycle.disarm()
    if dist.is_initialized():
        dist.destroy_process_group()
    store = dist.FileStore(str(tmp_path / "store"), 1)
    dist.init_process_group("gloo", store=store, rank=0, world_size=1)
    try:
        yield dist
    finally:
        lifecycle.disarm()
        if dist.is_initialized():
            dist.destroy_process_group()


class HandRolledTP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        hidden = self.fc(x)
        torch.distributed.all_reduce(hidden)
        return torch.relu(hidden)


def _capture(witness: str = "digest"):
    lifecycle.arm()
    return tl.trace(
        HandRolledTP(),
        torch.randn(2, 4),
        capture=tl.options.CaptureOptions(distributed_witness=witness),
    )


class TestLiveMerge:
    def test_live_merge_aligned_and_attested(self, gloo_world):
        merged = tl.merge_ranks([_capture()])
        assert merged.alignment.value == "aligned"
        assert merged.stored_alignment is merged.alignment
        assert merged.value_status.value == "attested_complete"
        assert len(merged.joins) == 1
        assert merged.rank_ids == (0,)
        assert "witness coverage" in merged.summary()

    def test_merge_report_matches_merge(self, gloo_world):
        log = _capture()
        report = tl.merge_report([log])
        assert report.alignment.value == "aligned"
        assert report.n_joins == 1
        assert report.to_markdown().startswith("# Cross-rank merge report")

    def test_join_ops_resolve_to_boundary_nodes(self, gloo_world):
        merged = tl.merge_ranks([_capture()])
        ops = merged.join_ops(merged.joins[0])
        assert len(ops[0]) == 1 and ops[0][0].type == "allreduce"

    def test_rank_qualified_sugar_and_super_op(self, gloo_world):
        merged = tl.merge_ranks([_capture()])
        label = merged.join_ops(merged.joins[0])[0][0].label
        assert merged[f"r0/{label}"].label == label
        assert set(merged.super_op(label)) == {0}

    def test_selector_getitem_refuses_typed(self, gloo_world):
        merged = tl.merge_ranks([_capture()])
        with pytest.raises(MergedSurfaceUnsupportedError) as excinfo:
            merged[tl.func("relu")]
        assert excinfo.value.fields["code"] == "merged_selector_unsupported"

    def test_run_and_validate_refuse_typed(self, gloo_world):
        merged = tl.merge_ranks([_capture()])
        with pytest.raises(MergedSurfaceUnsupportedError) as excinfo:
            merged.run(torch.randn(2, 4))
        assert excinfo.value.fields["code"] == "merge_run_unsupported"
        with pytest.raises(MergedSurfaceUnsupportedError) as excinfo:
            merged.validate()
        assert excinfo.value.fields["code"] == "merged_surface_unsupported"

    def test_non_distributed_trace_refuses_typed(self, gloo_world):
        lifecycle.arm()
        plain = tl.trace(nn.Sequential(nn.Linear(4, 4)), torch.randn(2, 4))
        with pytest.raises(MergeInputError) as excinfo:
            tl.merge_ranks([plain])
        assert excinfo.value.fields["code"] == "merge_input_invalid"

    def test_duplicate_rank_refuses_typed(self, gloo_world):
        log = _capture()
        with pytest.raises(MergeInputError) as excinfo:
            tl.merge_ranks([log, log])
        assert excinfo.value.fields["code"] == "merge_input_invalid"

    def test_happens_before_orders_sequential_joins(self, gloo_world):
        class TwoCollectives(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 4)

            def forward(self, x):
                hidden = self.fc(x)
                torch.distributed.all_reduce(hidden)
                out = torch.relu(hidden)
                torch.distributed.all_reduce(out)
                return out

        lifecycle.arm()
        log = tl.trace(TwoCollectives(), torch.randn(2, 4))
        merged = tl.merge_ranks([log])
        first, second = (join.key for join in merged.joins)
        assert merged.happens_before(first, second) is True
        assert merged.happens_before(second, first) is False


class TestArtifact:
    def test_round_trip_rederives_and_exposes_ranks(self, gloo_world, tmp_path):
        merged = tl.merge_ranks([_capture()])
        art = tmp_path / "merged.tlspec"
        merged.save(art)
        loaded = tl.load(art)
        assert loaded.alignment.value == "aligned"
        assert loaded.value_status.value == "attested_complete"
        assert len(loaded.joins) == 1
        assert loaded.load_degradations == ()
        # Rank cores are ordinary bundles inside members/, fully loadable.
        assert loaded.ranks[0]["linear_1_1"] is not None

    def test_descriptor_is_deterministic_over_identical_members(self, gloo_world, tmp_path):
        merged = tl.merge_ranks([_capture()])
        art = tmp_path / "merged.tlspec"
        merged.save(art)
        member = art / "members" / "rank_0000.tlspec"
        left = tl.merge_ranks([str(member)])
        right = tl.merge_ranks([str(member)])
        assert canonical_json_bytes(left._derivation.to_payload()) == canonical_json_bytes(
            right._derivation.to_payload()
        )

    def test_loaded_vs_live_parity(self, gloo_world, tmp_path):
        merged = tl.merge_ranks([_capture()])
        art = tmp_path / "merged.tlspec"
        merged.save(art)
        member = art / "members" / "rank_0000.tlspec"
        from_path = tl.merge_ranks([str(member)])
        from_loaded = tl.merge_ranks([tl.load(member)])
        assert canonical_json_bytes(from_path._derivation.to_payload()) == canonical_json_bytes(
            from_loaded._derivation.to_payload()
        )
        # The live derivation equals the loaded one too (P1: rank cores are
        # single truth; save/load changes nothing the merge reads).
        assert canonical_json_bytes(merged._derivation.to_payload()) == canonical_json_bytes(
            from_path._derivation.to_payload()
        )

    def test_resave_of_loaded_artifact_copies_members(self, gloo_world, tmp_path):
        merged = tl.merge_ranks([_capture()])
        first = tmp_path / "first.tlspec"
        merged.save(first)
        loaded = tl.load(first)
        second = tmp_path / "second.tlspec"
        loaded.save(second)
        reloaded = tl.load(second)
        assert reloaded.alignment.value == "aligned"


class TestTamperMatrix:
    """The descriptor is a CACHE; edits refuse typed, never degrade to gaps."""

    def _saved(self, tmp_path):
        merged = tl.merge_ranks([_capture()])
        art = tmp_path / "merged.tlspec"
        merged.save(art)
        return art

    def test_descriptor_edit_without_manifest_is_checksum_tamper(self, gloo_world, tmp_path):
        art = self._saved(tmp_path)
        descriptor = art / "merge" / "descriptor.json"
        data = json.loads(descriptor.read_text())
        data["derivation"]["joins"] = []
        descriptor.write_text(json.dumps(data))
        with pytest.raises(MergedArtifactError) as excinfo:
            tl.load(art)
        assert excinfo.value.fields["code"] == "merged_descriptor_tamper"

    def test_coherent_descriptor_rewrite_fails_rederivation(self, gloo_world, tmp_path):
        # Delete the join from the descriptor AND fix the root checksum: the
        # rank cores still contain the boundary records, so the independent
        # rederivation disagrees with the cache -> typed tamper refusal.
        art = self._saved(tmp_path)
        descriptor = art / "merge" / "descriptor.json"
        data = json.loads(descriptor.read_text())
        data["derivation"]["joins"] = []
        data["derivation"]["stored_value_status"] = "unwitnessed"
        tampered = canonical_json_bytes(data)
        descriptor.write_bytes(tampered)
        manifest_path = art / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["descriptor_sha256"] = hashlib.sha256(tampered).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        with pytest.raises(MergedArtifactError) as excinfo:
            tl.load(art)
        assert excinfo.value.fields["code"] == "merged_descriptor_tamper"

    def test_member_byte_edit_fails_tree_hash(self, gloo_world, tmp_path):
        art = self._saved(tmp_path)
        member_manifest = art / "members" / "rank_0000.tlspec" / "manifest.json"
        member_manifest.write_text(member_manifest.read_text() + " ")
        with pytest.raises(MergedArtifactError) as excinfo:
            tl.load(art)
        assert excinfo.value.fields["code"] == "merged_descriptor_tamper"

    def test_member_deletion_refuses_never_a_gap(self, gloo_world, tmp_path):
        art = self._saved(tmp_path)
        shutil.rmtree(art / "members" / "rank_0000.tlspec")
        with pytest.raises(MergedArtifactError) as excinfo:
            tl.load(art)
        assert excinfo.value.fields["code"] == "merged_descriptor_tamper"

    def test_foreign_bundle_format_refuses_closed_vocabulary(self, gloo_world, tmp_path):
        art = self._saved(tmp_path)
        manifest_path = art / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["bundle_format"] = "merged-directory-v2"
        manifest_path.write_text(json.dumps(manifest))
        with pytest.raises(Exception):
            tl.load(art)


# ---------------------------------------------------------------------------
# Multi-rank spawn sims
# ---------------------------------------------------------------------------


def _tp_worker(rank: int, world_size: int, init_file: str, out_dir: str) -> None:
    import torch
    import torch.distributed as dist

    import torchlens as tl
    from torchlens.distributed import arm

    store = dist.FileStore(init_file, world_size)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=world_size)
    arm()
    torch.manual_seed(1234)

    class TP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x):
            hidden = self.fc(x)
            dist.all_reduce(hidden)
            gathered = [torch.empty_like(hidden) for _ in range(world_size)]
            dist.all_gather(gathered, hidden)
            return torch.relu(gathered[rank])

    model = TP()
    torch.manual_seed(77)
    x = torch.randn(2, 4)
    log = tl.trace(model, x, capture=tl.options.CaptureOptions(distributed_witness="digest"))
    tl.save(log, os.path.join(out_dir, f"rank{rank}.tlspec"))
    dist.destroy_process_group()


def _asymmetric_worker(
    rank: int, world_size: int, init_file: str, init_file_2: str, out_dir: str
) -> None:
    """Sol's seed-discharge-negative shape, live: rank 0 arms before any
    group and witnesses a destroy/re-init cycle; rank 1 arms only after the
    re-init, so its restricted seed denotes generation 1 while claiming
    ordinal 0."""

    import torch
    import torch.distributed as dist

    import torchlens as tl
    from torchlens.distributed import arm

    if rank == 0:
        arm()  # complete witness: armed before ANY group
    store = dist.FileStore(init_file, world_size)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=world_size)
    dist.barrier()
    dist.destroy_process_group()

    store2 = dist.FileStore(init_file_2, world_size)
    dist.init_process_group("gloo", store=store2, rank=rank, world_size=world_size)
    if rank != 0:
        arm()  # late arming: seeds the RECREATED world as ordinal 0

    torch.manual_seed(1234)
    model = HandRolledTP()
    torch.manual_seed(77)
    log = tl.trace(model, torch.randn(2, 4))
    tl.save(log, os.path.join(out_dir, f"rank{rank}.tlspec"))
    dist.destroy_process_group()


@pytest.mark.slow
class TestSpawnSims:
    def _spawn(self, worker, world_size, tmp_path, extra_args=()):
        import torch.multiprocessing as mp

        lifecycle.disarm()
        if torch.distributed.is_initialized():
            pytest.skip("a process group is already initialized in this process")
        init_file = str(tmp_path / "init_store")
        mp.spawn(
            worker,
            args=(world_size, init_file, *extra_args, str(tmp_path)),
            nprocs=world_size,
            join=True,
        )
        return [str(tmp_path / f"rank{rank}.tlspec") for rank in range(world_size)]

    def test_two_rank_merge_aligned_attested_and_round_trips(self, tmp_path):
        paths = self._spawn(_tp_worker, 2, tmp_path)
        merged = tl.merge_ranks(paths)
        assert merged.alignment.value == "aligned"
        assert merged.value_status.value == "attested_complete"
        assert [join.kind for join in merged.joins] == ["all_reduce", "all_gather"]
        assert all(join.presence == (0, 1) for join in merged.joins)
        assert merged.gaps == ()

        # Loaded-vs-path parity.
        from_loaded = tl.merge_ranks([tl.load(path) for path in paths])
        assert canonical_json_bytes(from_loaded._derivation.to_payload()) == canonical_json_bytes(
            merged._derivation.to_payload()
        )

        # Artifact round trip with full rederivation.
        art = tmp_path / "merged.tlspec"
        merged.save(art)
        loaded = tl.load(art)
        assert loaded.alignment.value == "aligned"
        assert len(loaded.joins) == 2
        # Rank cores are ordinary loadable bundles; join ops resolve on them.
        resolved = loaded.join_ops(loaded.joins[0])
        assert resolved[0] and resolved[1]
        assert all(ops[0].type == "allreduce" for ops in resolved.values())
        frame = loaded.to_pandas()
        assert list(frame.index.names) == ["rank", "rank_local_index"]
        assert len(frame) == 4  # 2 joins x 2 ranks

    def test_three_rank_merge_and_subset_gap(self, tmp_path):
        paths = self._spawn(_tp_worker, 3, tmp_path)
        merged = tl.merge_ranks(paths)
        assert merged.alignment.value == "aligned"
        assert all(join.presence == (0, 1, 2) for join in merged.joins)

        # Subset merge: the recorded 3-rank membership makes the missing
        # rank a presence gap on every join (rank cores are gap authority).
        subset = tl.merge_ranks(paths[:2])
        assert subset.alignment.value == "partial"
        assert all(gap.ranks == (2,) for gap in subset.gaps)
        assert len(subset.gaps) == len(subset.joins)
        report = tl.merge_report(paths[:2])
        assert report.alignment.value == "partial"

    def test_asymmetric_arming_refuses_structurally_live(self, tmp_path):
        paths = self._spawn(_asymmetric_worker, 2, tmp_path, extra_args=(str(tmp_path / "init2"),))
        report = tl.merge_report(paths)
        assert report.alignment.value == "conflicted"
        kinds = [finding.kind for finding in report.findings]
        assert kinds == ["group_lifetime_evidence_conflict"]
        assert report.gaps == ()  # structural refusal, never presence gaps
        with pytest.raises(MergeConflictError) as excinfo:
            tl.merge_ranks(paths)
        assert excinfo.value.fields["code"] == "group_lifetime_evidence_conflict"
