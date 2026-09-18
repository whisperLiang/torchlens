"""Boundary caches authenticate exact bytes before restoring native tensors."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pytest
import torch

from torchlens.split.boundary import ReplayBoundary
from torchlens.split.cache import load_boundary, save_boundary
from torchlens.split.errors import SplitBoundaryError
from torchlens.split.ir import BoundarySchema
from torchlens.split.shape import SymbolicShape

pytestmark = pytest.mark.smoke


@pytest.fixture(autouse=True)
def private_signing_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the signing key outside the artifact and isolate it between tests."""

    monkeypatch.setenv("TORCHLENS_CACHE_DIR", str(tmp_path / "private-key-root"))


@pytest.fixture
def cached_boundary(tmp_path: Path) -> tuple[Path, ReplayBoundary]:
    """Write a minimal boundary using the public cache functions."""

    value = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    boundary = ReplayBoundary(
        backend="torch",
        tensors={"relu:1": value},
        spec={
            "relu:1": BoundarySchema(
                value_id="relu:1",
                container_path=(),
                role="activation",
                shape=SymbolicShape(("B", 3)),
                dtype="torch.float32",
                backend="torch",
            )
        },
        metadata={"split_id": "cut", "runtime_batch_size": 2, "shape_program_hash": "shape"},
    )
    directory = tmp_path / "boundary"
    save_boundary(boundary, directory)
    return directory, boundary


class _WriteMarker:
    """A pickle whose execution would leave an unmistakable local marker."""

    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self) -> Any:
        return eval, (f"__import__('pathlib').Path({str(self.marker)!r}).write_text('executed')",)


@pytest.mark.parametrize("keep_header", [False, True])
def test_boundary_cache_refuses_unsigned_or_tampered_pickle(
    cached_boundary: tuple[Path, ReplayBoundary], keep_header: bool
) -> None:
    """Neither a bare gadget nor a gadget retaining a valid old tag executes."""

    from torchlens.user_funcs import _CAPTURE_CACHE_HEADER_BYTES

    directory, _boundary = cached_boundary
    path = directory / "payload.pkl"
    marker = directory / "executed"
    header = path.read_bytes()[:_CAPTURE_CACHE_HEADER_BYTES] if keep_header else b""
    path.write_bytes(header + pickle.dumps(_WriteMarker(marker)))
    with (
        pytest.warns(UserWarning, match="NOT unpickled"),
        pytest.raises(SplitBoundaryError, match="authentication failed"),
    ):
        load_boundary(directory)
    assert not marker.exists()


def test_boundary_cache_key_is_not_supplied_by_the_artifact(
    cached_boundary: tuple[Path, ReplayBoundary], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Copying a key next to the payload does not authorize it under another root."""

    directory, _boundary = cached_boundary
    key = tmp_path / "private-key-root" / "capture" / ".capture_cache_secret"
    (directory / ".capture_cache_secret").write_bytes(key.read_bytes())
    monkeypatch.setenv("TORCHLENS_CACHE_DIR", str(tmp_path / "different-private-root"))
    with (
        pytest.warns(UserWarning, match="NOT unpickled"),
        pytest.raises(SplitBoundaryError, match="authentication failed"),
    ):
        load_boundary(directory)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("backend", "jax"),
        ("split_id", "other"),
        ("graph_shape_hash", "other"),
        ("runtime_batch_size", 99),
        ("shape_program_hash", "other"),
        ("tensor_ids", []),
        ("boundary_spec", {}),
    ],
)
def test_boundary_cache_manifest_is_bound_to_the_payload(
    cached_boundary: tuple[Path, ReplayBoundary], field: str, replacement: Any
) -> None:
    """Editing the unsigned manifest cannot redirect or relabel a valid payload."""

    directory, _boundary = cached_boundary
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest[field] = replacement
    path.write_text(json.dumps(manifest))
    with pytest.raises(SplitBoundaryError, match="manifest does not match"):
        load_boundary(directory)


@pytest.mark.parametrize("manifest", ["[" * 250 + "0" + "]" * 250, "[", "null"])
def test_invalid_manifest_refuses_before_payload_loading(
    cached_boundary: tuple[Path, ReplayBoundary], manifest: str
) -> None:
    """Deep, malformed, or non-object manifests cannot reach pickle loading."""

    directory, _boundary = cached_boundary
    (directory / "manifest.json").write_text(manifest)
    (directory / "payload.pkl").unlink()
    with pytest.raises((json.JSONDecodeError, SplitBoundaryError)):
        load_boundary(directory)


def test_legacy_unsigned_cache_has_no_load_fallback(
    cached_boundary: tuple[Path, ReplayBoundary],
) -> None:
    """A pre-hardening cache requires regeneration, never a bare-pickle retry."""

    directory, boundary = cached_boundary
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["payload_format"] = "pickle"
    path.write_text(json.dumps(manifest))
    (directory / "payload.pkl").write_bytes(pickle.dumps(boundary))
    with pytest.raises(SplitBoundaryError, match="regenerate"):
        load_boundary(directory)


def test_authenticated_boundary_preserves_values_and_metadata(
    cached_boundary: tuple[Path, ReplayBoundary],
) -> None:
    """Authentication does not change the native payload or boundary ABI."""

    directory, boundary = cached_boundary
    loaded = load_boundary(directory)
    assert loaded.spec == boundary.spec
    assert loaded.metadata == boundary.metadata
    torch.testing.assert_close(loaded.tensors["relu:1"], boundary.tensors["relu:1"])
