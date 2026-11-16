"""Public API surface tests for the Native Split Runtime."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
from dag_helpers import call_ids, close_call_ids

import torchlens as tl
from torchlens.split import (
    BoundarySchema,
    GraphPartition,
    SplitFeatures,
    SplitGraphIR,
    SplitModelProfile,
    SplitRequest,
    SplitRuntime,
    SplitVerificationStatus,
    checkpoint_cache_path,
    prepare,
    prepare_graph,
    register_model_profile,
)
from torchlens.split.errors import (
    SplitBoundaryError,
    SplitError,
    SplitRequestError,
    SplitUnsupportedError,
)
from torchlens.split.profiles import model_cache_dir, profile_cache_dir


def test_split_exports_are_public() -> None:
    """Expose the supported construction API while keeping pipeline helpers internal."""

    assert tl.BoundarySchema is BoundarySchema
    assert tl.SplitFeatures is SplitFeatures
    assert tl.SplitGraphIR is SplitGraphIR
    assert tl.SplitRequest is SplitRequest
    assert tl.SplitRuntime is SplitRuntime
    assert tl.prepare is prepare
    assert GraphPartition.__name__ == "GraphPartition"
    assert SplitVerificationStatus.EXACT.value == "exact"
    for name in (
        "BoundarySchema",
        "SplitFeatures",
        "SplitGraphIR",
        "SplitRequest",
        "SplitRuntime",
        "prepare",
    ):
        assert name in tl.__all__
        assert name in tl.split.__all__
    for name in ("CapabilityStatus", "ReplayOp", "ReplayProgram", "SplitCapabilityReport"):
        assert name in tl.split.__all__
    for name in (
        "capture_model",
        "capture_canonical_model",
        "normalize_to_split_ir",
        "lower_split_program",
        "execute_split_runtime",
        "analyze_split_capabilities",
    ):
        assert name not in tl.split.__all__
        assert not hasattr(tl.split, name)


def test_split_request_validation() -> None:
    """Native partitions and declared batch axes reject invalid requests."""

    request = SplitRequest(GraphPartition(frozenset({"relu_1:1"})))
    assert request.boundary.startswith("partition:")
    assert request.features.batch_axes is None
    assert request.trainable is False
    assert request.placement.is_explicit is False
    with pytest.raises(TypeError):
        SplitRequest("invalid")
    with pytest.raises(ValueError):
        SplitFeatures(batch_axes={"bad-pointer": 0})
    with pytest.raises(TypeError):
        SplitFeatures(batch_axes={"/args/0": "0"})  # type: ignore[dict-item]
    with pytest.raises(ValueError):
        SplitRequest(GraphPartition(frozenset()), validation="unknown")  # type: ignore[arg-type]


def test_split_named_state_exchange_refuses_unsupported_backend() -> None:
    """Backends without named state exchange refuse instead of exporting stale model state."""

    import torch

    graph = prepare_graph(
        torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU()), torch.ones(2, 4)
    )
    dag = graph.compute_dag()
    partition = dag.to_partition(
        close_call_ids(dag, call_ids(graph, modules=frozenset({"1"}), ops=frozenset({"1"})))
    )
    runtime = graph.materialize(graph.analyze(partition))
    for method, args in ((runtime.state_dict, ()), (runtime.load_state_dict, ({},))):
        with pytest.raises(SplitUnsupportedError) as exc:
            method(*args)
        assert exc.value.context.reason == "named_state_exchange_unsupported"


def test_prepare_normalizes_graph_and_exposes_boundary_abi() -> None:
    """Preparation exposes typed replay contracts without deprecated capture warnings."""

    import torch

    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2)
    ).eval()
    inputs = torch.ones(2, 4)
    features = SplitFeatures(boundary_cache=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        graph = prepare_graph(model, inputs, features=features)
        dag = graph.compute_dag()
        partition = dag.to_partition(
            close_call_ids(dag, call_ids(graph, modules=frozenset({"1"}), ops=frozenset({"relu"})))
        )
        runtime = graph.materialize(graph.analyze(partition))
        replay_x = torch.randn(4, 4)
        replayed = runtime.replay(replay_x)
        torch.testing.assert_close(replayed, model(replay_x))

    assert runtime.request.features == features
    assert runtime.graph_ir is not None
    assert runtime.graph_ir.backend == "torch"
    assert runtime.graph_ir.ops
    assert runtime.boundary_schema
    assert runtime.capability_report is not None
    assert runtime.capability_report.preflight_ok is True
    assert runtime.capability_report.replay.supported is True
    assert runtime.capability_report.boundary_cache.supported is True
    assert runtime.prefix_program is not None
    assert runtime.suffix_program is not None
    assert runtime.prefix_program.segment == "prefix"
    assert runtime.suffix_program.segment == "suffix"
    assert runtime.capability_report.verification == SplitVerificationStatus.EXACT
    report = runtime.explain_capabilities()
    assert report["backend_capabilities"]["replay"] is True
    assert "dynamic_batch" not in report["features"]
    assert "backend_handle" not in repr(runtime.graph_ir.as_dict())
    assert tuple(replayed.shape) == (4, 2)


@pytest.mark.parametrize(
    ("error_type", "code"),
    [
        (SplitError, "split_error"),
        (SplitRequestError, "split_request_error"),
        (SplitBoundaryError, "split_boundary_error"),
        (SplitUnsupportedError, "split_unsupported"),
    ],
)
def test_split_error_codes_are_stable(error_type: type[SplitError], code: str) -> None:
    """The split preview exposes distinct typed error codes to callers."""

    assert error_type("split failure").code == code


def test_model_profile_cache_is_user_scoped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Profiles identify pinned metadata while checkpoints stay outside the repo."""

    monkeypatch.setenv("TORCHLENS_MODEL_CACHE", str(tmp_path / "models"))
    profile = SplitModelProfile(
        id="contract.synthetic.torch",
        backend="torch",
        family="mlp",
        checkpoint_source="synthetic",
        checkpoint_revision="2026-01-01",
        checkpoint_sha256="0" * 64,
    )
    register_model_profile(profile, replace=True)

    assert model_cache_dir() == tmp_path / "models"
    assert checkpoint_cache_path(profile, "weights.safetensors") == (
        tmp_path / "models" / profile.id / "weights.safetensors"
    )


def test_profile_cache_dir_rejects_missing_profile() -> None:
    """A missing runtime profile is rejected before building a cache path."""

    with pytest.raises(ValueError, match="requires a model profile"):
        profile_cache_dir(None)
