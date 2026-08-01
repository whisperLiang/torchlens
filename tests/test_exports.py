"""Tests for Phase 10 export surfaces."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

pd = pytest.importorskip("pandas")

import torchlens as tl  # noqa: E402


EXPORT_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "exports"


def _assert_model_explorer_structure(payload: dict[str, Any]) -> None:
    """Validate required Model Explorer keys and graph referential integrity.

    Parameters
    ----------
    payload:
        Parsed Model Explorer artifact.
    """

    assert payload["schema"] == "torchlens.model_explorer.v1"
    assert isinstance(payload["disclaimer"], str)
    assert isinstance(payload["graphs"], list) and payload["graphs"]
    for graph in payload["graphs"]:
        assert isinstance(graph["id"], str)
        assert isinstance(graph["nodes"], list) and graph["nodes"]
        node_ids = [node["id"] for node in graph["nodes"]]
        assert all(isinstance(node_id, str) for node_id in node_ids)
        assert len(node_ids) == len(set(node_ids))
        for node in graph["nodes"]:
            assert isinstance(node["label"], str)
            assert isinstance(node["namespace"], str)
            assert isinstance(node["attrs"], list)
            assert all(
                isinstance(attr, dict)
                and isinstance(attr.get("key"), str)
                and isinstance(attr.get("value"), str)
                for attr in node["attrs"]
            )
            assert isinstance(node["incomingEdges"], list)
            assert all(
                set(edge) == {"sourceNodeId"}
                and isinstance(edge["sourceNodeId"], str)
                and edge["sourceNodeId"] in node_ids
                for edge in node["incomingEdges"]
            )


def _assert_netron_structure(payload: dict[str, Any]) -> None:
    """Validate required Netron-shaped keys and graph referential integrity.

    Parameters
    ----------
    payload:
        Parsed Netron-shaped artifact.
    """

    assert payload["ir_version"] == "torchlens-lossy-onnx-shaped-v1"
    assert payload["producer_name"] == "torchlens"
    assert payload["runnable"] is False
    assert isinstance(payload["disclaimer"], str)
    graph = payload["graph"]
    assert isinstance(graph["name"], str)
    assert isinstance(graph["node"], list) and graph["node"]
    node_names = [node["name"] for node in graph["node"]]
    outputs = [output for node in graph["node"] for output in node["output"]]
    assert len(node_names) == len(set(node_names))
    assert len(outputs) == len(set(outputs))
    for node in graph["node"]:
        assert isinstance(node["op_type"], str)
        assert isinstance(node["input"], list)
        assert all(isinstance(input_id, str) and input_id in outputs for input_id in node["input"])
        assert isinstance(node["output"], list) and node["output"]
        assert isinstance(node["attribute"], list)
        assert all(
            isinstance(attribute, dict)
            and isinstance(attribute.get("name"), str)
            and isinstance(attribute.get("value"), list)
            for attribute in node["attribute"]
        )


def _assert_or_regenerate_export_golden(name: str, payload: dict[str, Any]) -> None:
    """Compare an export payload with its golden, with explicit opt-in regeneration.

    Set ``TORCHLENS_REGEN_EXPORT_GOLDENS=1`` and run the export test to regenerate
    fixtures after an intentional contract change.

    Parameters
    ----------
    name:
        Golden fixture filename.
    payload:
        Normalized parsed export payload.
    """

    fixture_path = EXPORT_FIXTURE_DIR / name
    if os.environ.get("TORCHLENS_REGEN_EXPORT_GOLDENS") == "1":
        fixture_path.parent.mkdir(parents=True, exist_ok=True)
        fixture_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    expected = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert payload == expected


def _normalize_export_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize process-global trace identifiers in an export payload.

    Parameters
    ----------
    payload:
        Parsed export payload.

    Returns
    -------
    dict[str, Any]
        A detached payload with its generated graph identifier normalized.
    """

    normalized = json.loads(json.dumps(payload))
    if "graphs" in normalized:
        normalized["graphs"][0]["id"] = "<trace-id>"
    else:
        normalized["graph"]["name"] = "<trace-id>"
    return normalized


def test_hash_namespace_is_available_through_all_public_import_patterns() -> None:
    """The structural-hash namespace should preserve lazy facade import patterns."""

    import torchlens.hash as hash_module
    from torchlens import hash as imported_namespace
    from torchlens.hash import model as model_hash

    assert tl.hash is hash_module
    assert imported_namespace is hash_module
    assert model_hash is hash_module.model
    assert tl.assert_unchanged is hash_module.assert_unchanged


class _Tracker:
    """Small object with tracker-like custom_methods for export tests."""

    def __init__(self) -> None:
        """Initialize recorded calls."""

        self.metrics: list[tuple[str, int]] = []

    def log_metric(self, name: str, value: int) -> None:
        """Record an MLflow-like metric call.

        Parameters
        ----------
        name:
            Metric name.
        value:
            Metric value.
        """

        self.metrics.append((name, value))

    def track(self, value: int, name: str) -> None:
        """Record an Aim-like track call.

        Parameters
        ----------
        value:
            Metric value.
        name:
            Metric name.
        """

        self.metrics.append((name, value))


class _FakeHubApi:
    """Small Hugging Face API double."""

    def __init__(self) -> None:
        """Initialize recorded calls."""

        self.created: list[dict[str, Any]] = []
        self.uploaded: list[dict[str, Any]] = []
        self.uploaded_bytes: list[bytes] = []

    def create_repo(self, **kwargs: Any) -> None:
        """Record repository creation.

        Parameters
        ----------
        **kwargs:
            Repository creation arguments.
        """

        self.created.append(kwargs)

    def upload_file(self, **kwargs: Any) -> str:
        """Record file upload.

        Parameters
        ----------
        **kwargs:
            Upload arguments.

        Returns
        -------
        str
            Fake upload URL.
        """

        self.uploaded.append(kwargs)
        # Read the file's bytes immediately: the caller's temp directory is
        # cleaned up as soon as this call returns, so content must be
        # captured now rather than by re-reading the path later.
        self.uploaded_bytes.append(Path(kwargs["path_or_fileobj"]).read_bytes())
        return "https://huggingface.co/example/repo/blob/main/torchlens_artifact.pkl"


@pytest.fixture
def export_log() -> Any:
    """Build a small Trace for export tests.

    Returns
    -------
    Any
        Logged model.
    """

    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    return tl.trace(model, torch.randn(2, 3), capture=tl.options.CaptureOptions())


def test_trace_timeline_exports_are_parseable(export_log: Any, tmp_path: Path) -> None:
    """Trace/timeline exports should write viewer-conformant payloads."""

    chrome_path = tl.export.chrome_trace(export_log, tmp_path / "trace.json")
    chrome_payload = json.loads(chrome_path.read_text(encoding="utf-8"))
    assert "traceEvents" in chrome_payload
    assert any(event.get("ph") == "X" for event in chrome_payload["traceEvents"])

    speedscope_path = tl.export.speedscope(export_log, tmp_path / "profile.json")
    speedscope_payload = json.loads(speedscope_path.read_text(encoding="utf-8"))
    assert speedscope_payload["$schema"].endswith("file-format-schema.json")
    assert speedscope_payload["profiles"][0]["type"] == "evented"

    flamegraph_path = tl.export.flamegraph(export_log, tmp_path / "profile.folded")
    assert ";" in flamegraph_path.read_text(encoding="utf-8")

    memory_path = tl.export.memory_timeline(export_log, tmp_path / "memory.json")
    memory_payload = json.loads(memory_path.read_text(encoding="utf-8"))
    assert memory_payload["scope"] == "tensor"
    assert "not an allocator trace" in memory_payload["disclaimer"]


def test_xarray_export_has_neuroidassembly_shape(export_log: Any) -> None:
    """xarray export should expose presentation and neuroid dimensions."""
    pytest.importorskip("xarray")

    assembly = tl.export.xarray(export_log)

    assert assembly.dims == ("presentation", "neuroid")
    assert "layer" in assembly.coords
    assert assembly.attrs["assembly"] == "NeuroidAssembly"
    assert assembly.sizes["presentation"] == 2
    assert assembly.sizes["neuroid"] > 0


def test_xarray_export_names_mismatched_presentation_layer() -> None:
    """Mismatched presentation counts should identify the offending layer."""
    pytest.importorskip("xarray")

    fake_log = SimpleNamespace(
        layer_list=[
            SimpleNamespace(layer_label="first", out=torch.randn(2, 3)),
            SimpleNamespace(layer_label="bad_layer", out=torch.randn(1, 3)),
        ]
    )

    with pytest.raises(ValueError, match="bad_layer.*1.*expected 2"):
        tl.export.xarray(fake_log)


def test_tracker_exports_accept_existing_objects(export_log: Any, tmp_path: Path) -> None:
    """Tracker helpers should work with caller-owned writer/run objects."""

    pytest.importorskip("tensorboard")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=tmp_path / "tb")
    returned_writer = tl.export.tensorboard(export_log, writer, step=3, prefix="tl")
    writer.close()

    assert returned_writer is writer
    accumulator = EventAccumulator(str(tmp_path / "tb"))
    accumulator.Reload()
    assert "tl/num_layers" in accumulator.Tags()["scalars"]

    tracker = _Tracker()
    assert tl.export.mlflow(export_log, client=tracker, prefix="tl")["num_layers"] > 0
    assert tracker.metrics

    aim_tracker = _Tracker()
    assert tl.export.aim(export_log, run=aim_tracker, prefix="tl")["num_layers"] > 0
    assert aim_tracker.metrics

    pytest.importorskip("wandb")
    wandb_result = tl.export.wandb(export_log)
    assert "table" in wandb_result


def test_tracker_exports_reject_paths_with_clear_type_errors(
    export_log: Any, tmp_path: Path
) -> None:
    """Tracker helpers need live tracker objects, not filesystem paths."""

    with pytest.raises(TypeError, match="tensorboard expects an existing tracker object"):
        tl.export.tensorboard(export_log, str(tmp_path / "tb"))
    with pytest.raises(TypeError, match="mlflow expects an existing tracker object"):
        tl.export.mlflow(export_log, client=tmp_path / "mlruns")
    with pytest.raises(TypeError, match="aim expects an existing tracker object"):
        tl.export.aim(export_log, run=tmp_path / "aim")


def test_emit_nvtx_capture_option_does_not_change_capture() -> None:
    """emit_nvtx should be accepted and should not alter normal logging output."""

    log = tl.trace(
        nn.Linear(2, 2),
        torch.randn(1, 2),
        capture=tl.options.CaptureOptions(emit_nvtx=True),
    )

    assert log.emit_nvtx is True
    assert len(log.layer_list) > 0


def test_tabular_exports_round_trip(export_log: Any, tmp_path: Path) -> None:
    """Canonical tabular exports should round-trip."""

    expected = export_log.to_pandas()
    assert "func_config" in expected.columns
    assert "conditional_then_children" in expected.columns

    csv_path = tl.export.csv(export_log, tmp_path / "model.csv")
    csv_df = pd.read_csv(csv_path)
    assert list(csv_df.columns) == list(expected.columns)
    assert len(csv_df) == len(expected)

    json_path = tl.export.json(export_log, tmp_path / "model.json")
    json_df = pd.read_json(json_path, orient="records")
    assert list(json_df.columns) == list(expected.columns)
    assert len(json_df) == len(expected)

    parquet_path = tmp_path / "model.parquet"
    if importlib.util.find_spec("pyarrow") is None:
        with pytest.raises(ImportError, match=r"torchlens\[tabular\]"):
            tl.export.parquet(export_log, parquet_path)
    else:
        tl.export.parquet(export_log, parquet_path)
        parquet_df = pd.read_parquet(parquet_path)
        assert list(parquet_df.columns) == list(expected.columns)
        assert len(parquet_df) == len(expected)


def test_static_graph_adapters_and_hub_dry_run(export_log: Any, tmp_path: Path) -> None:
    """Static graph adapters and Hub publisher should write planned payloads."""

    explorer_path = tl.export.model_explorer(export_log, tmp_path / "explorer.json")
    explorer_payload = json.loads(explorer_path.read_text(encoding="utf-8"))
    _assert_model_explorer_structure(explorer_payload)
    assert (
        "acceptance by any particular external Model Explorer release is not guaranteed"
        in (explorer_payload["disclaimer"])
    )
    _assert_or_regenerate_export_golden(
        "model_explorer.json", _normalize_export_payload(explorer_payload)
    )

    netron_path = tl.export.netron(export_log, tmp_path / "netron.json")
    netron_payload = json.loads(netron_path.read_text(encoding="utf-8"))
    _assert_netron_structure(netron_payload)
    assert "not a real ONNX model" in netron_payload["disclaimer"]
    assert "acceptance by Netron is not guaranteed" in netron_payload["disclaimer"]
    _assert_or_regenerate_export_golden("netron.json", _normalize_export_payload(netron_payload))

    result = tl.bridge.huggingface.push_to_hub(
        export_log,
        "example/repo",
        dry_run=True,
    )
    assert result["repo_id"] == "example/repo"
    assert result["dry_run"] is True

    api = _FakeHubApi()
    uploaded = tl.bridge.huggingface.push_to_hub(export_log, "example/repo", api=api)
    assert uploaded["upload_result"].startswith("https://huggingface.co/")
    assert api.created
    assert api.uploaded


def test_recurrent_static_graph_exports_use_unique_pass_qualified_ids(tmp_path: Path) -> None:
    """Recurrent exports preserve every execution pass and every connecting edge."""

    class _LoopModel(nn.Module):
        """Three-step recurrent linear/ReLU chain."""

        def __init__(self) -> None:
            """Initialize the reused linear layer."""

            super().__init__()
            self.linear = nn.Linear(2, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Apply the same two operations for three iterations."""

            for _ in range(3):
                x = torch.relu(self.linear(x))
            return x

    log = tl.trace(_LoopModel(), torch.randn(1, 2))
    explorer_path = tl.export.model_explorer(log, tmp_path / "recurrent-explorer.json")
    netron_path = tl.export.netron(log, tmp_path / "recurrent-netron.json")

    explorer_graph = json.loads(explorer_path.read_text(encoding="utf-8"))["graphs"][0]
    explorer_ids = [node["id"] for node in explorer_graph["nodes"]]
    assert len(explorer_ids) == len(set(explorer_ids)) == 8
    assert sum(len(node["incomingEdges"]) for node in explorer_graph["nodes"]) == 7

    _assert_model_explorer_structure(json.loads(explorer_path.read_text(encoding="utf-8")))

    netron_payload = json.loads(netron_path.read_text(encoding="utf-8"))
    _assert_netron_structure(netron_payload)
    netron_nodes = netron_payload["graph"]["node"]
    netron_outputs = {output for node in netron_nodes for output in node["output"]}
    assert len(netron_nodes) == len(netron_outputs) == 8
    assert all(input_id in netron_outputs for node in netron_nodes for input_id in node["input"])


def test_model_explorer_package_accepts_graph_schema(export_log: Any, tmp_path: Path) -> None:
    """The optional Model Explorer graph dataclass loader should accept the artifact."""

    model_explorer = pytest.importorskip("model_explorer")
    dacite = pytest.importorskip("dacite")
    explorer_path = tl.export.model_explorer(export_log, tmp_path / "explorer.json")
    payload = json.loads(explorer_path.read_text(encoding="utf-8"))

    parsed = [
        dacite.from_dict(data_class=model_explorer.graph_builder.Graph, data=graph)
        for graph in payload["graphs"]
    ]

    assert parsed and parsed[0].nodes


def test_hub_push_uploads_real_bundle_not_metadata_stub(export_log: Any) -> None:
    """push_to_hub must upload the real scrubbed artifact, never a JSON stub.

    ``push_to_hub`` previously fell back to a ~240-byte JSON manifest for
    backward-eligible captures while still reporting ``dry_run: False`` success
    (the raw Trace was then unpicklable; it is now picklable via GradFn weakref
    serialization, so the old naive-pickle-fails precondition no longer holds).
    This asserts the uploaded payload is the real, larger, non-JSON
    portable-bundle archive.
    """

    api = _FakeHubApi()
    uploaded = tl.bridge.huggingface.push_to_hub(export_log, "example/repo", api=api)
    assert uploaded["dry_run"] is False
    assert api.uploaded, "expected an upload_file call"

    payload = api.uploaded_bytes[-1]

    # A ~240-byte JSON manifest stub was the old broken fallback. The real
    # artifact must be large and must not be a bare JSON stub.
    assert len(payload) > 1000
    assert uploaded["size_bytes"] > 1000
    assert not payload.lstrip().startswith(b"{"), "expected a real artifact, not a JSON stub"

    # The Trace is picklable (GradFn weakref serialization), so push_to_hub
    # uploads the real pickled artifact (pickle protocol opcode 0x80), never a
    # JSON stub and not the bundle-scrub fallback.
    assert payload[:1] == b"\x80", "expected a real pickle artifact, not a JSON stub"


def test_depyf_bridge_fails_soft_when_extra_missing() -> None:
    """depyf bridge should explain the missing optional dependency."""

    if importlib.util.find_spec("depyf") is not None:
        pytest.skip("Installed depyf API varies; smoke coverage is in the extras matrix.")
    with pytest.raises(ImportError, match=r"torchlens\[depyf\]"):
        tl.bridge.depyf.dump(nn.Linear(1, 1), torch.randn(1, 1))
