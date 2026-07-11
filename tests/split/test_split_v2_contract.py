"""Contract tests for the backend-neutral split v2 surface."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.split import (
    SplitFeatures,
    SplitModelProfile,
    SplitRequest,
    SplitVerificationStatus,
    after,
    checkpoint_cache_path,
    register_model_profile,
)
from torchlens.split.profiles import model_cache_dir


def test_v2_prepare_normalizes_graph_and_exposes_boundary_abi() -> None:
    """The typed request produces IR and a structured capability report."""

    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    inputs = torch.randn(2, 4)
    request = SplitRequest(
        point=after("relu"),
        backend="torch",
        features=SplitFeatures(dynamic_batch=(1, 5), boundary_cache=True),
        validation="strict",
    )

    runtime = tl.split.prepare(model, inputs, request)

    assert runtime.request is request
    assert runtime.graph_ir is not None
    assert runtime.graph_ir.backend == "torch"
    assert runtime.graph_ir.ops
    assert runtime.boundary_schema
    assert runtime.capability_report is not None
    assert runtime.capability_report.verification == SplitVerificationStatus.EXACT
    report = runtime.explain_capabilities()
    assert report["backend_capabilities"]["replay"] is True
    assert report["features"]["dynamic_batch"] == (1, 5)
    assert "backend_handle" not in repr(runtime.graph_ir.as_dict())

    replayed = runtime.replay(torch.randn(4, 4))
    assert tuple(replayed.shape) == (4, 2)


def test_model_profile_cache_is_user_scoped(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
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


def test_jax_custom_jvp_is_captured_as_region() -> None:
    """Custom JVP forward calls use region replay metadata, not a false exact op."""

    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from torchlens.backends.jax.jaxpr import (
        JaxRegionCapture,
        derive_closed_jaxpr,
        flatten_dynamic_args,
        interpret_closed_jaxpr_with_inlining,
    )

    @jax.custom_jvp
    def custom(value):
        return value * 2

    @custom.defjvp
    def custom_jvp(primals, tangents):
        (value,) = primals
        (tangent,) = tangents
        return custom(value), tangent * 2

    value = jnp.ones((2,))
    closed = derive_closed_jaxpr(custom, (value,))
    flat, _ = flatten_dynamic_args((value,))
    result = interpret_closed_jaxpr_with_inlining(closed, flat)

    regions = [capture for capture in result.captures if isinstance(capture, JaxRegionCapture)]
    assert regions
    assert regions[0].primitive == "custom_jvp_call"
    assert regions[0].unverified_reason == "custom_jvp_call_forward_region"
