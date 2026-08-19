"""Tests for the MCP bridge (``torchlens.bridge.mcp``)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.bridge import mcp as tlmcp


@pytest.fixture()
def saved_trace_path(tmp_path: Path) -> Path:
    """Save a small trace artifact and return its path.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.

    Returns
    -------
    Path
        Path of the saved ``.tlspec`` artifact.
    """

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)).eval()
    log = tl.trace(model, torch.randn(2, 4), save=tl.func("relu"))
    path = tmp_path / "trace.tlspec"
    tl.save(log, str(path))
    return path


def test_tool_specs_are_wellformed_json_schemas() -> None:
    """Every declared tool has a name, description, and object input schema."""

    names = [spec["name"] for spec in tlmcp.TOOL_SPECS]
    assert len(names) == len(set(names))
    for spec in tlmcp.TOOL_SPECS:
        assert spec["name"].startswith("torchlens_")
        assert spec["description"]
        schema = spec["input_schema"]
        assert schema["type"] == "object"
        json.dumps(schema)


def test_call_tool_refusals_teach_the_fix() -> None:
    """Unknown tools and missing paths refuse with the remedy named."""

    with pytest.raises(ValueError, match="Known tools"):
        tlmcp.call_tool("torchlens_nope")
    with pytest.raises(ValueError, match="'path' string"):
        tlmcp.call_tool("torchlens_agent_dump", {})
    with pytest.raises(ValueError, match="tl.save"):
        tlmcp.call_tool("torchlens_agent_dump", {"path": "/nonexistent/file.tlspec"})


def test_doctor_and_api_map_tools_are_json_safe() -> None:
    """Environment and API-map tools return JSON-serializable payloads."""

    doctor = tlmcp.call_tool("torchlens_doctor")
    assert doctor["checks"]
    assert {"name", "status", "detail"} <= set(doctor["checks"][0])
    json.dumps(doctor)

    api_map = tlmcp.call_tool("torchlens_api_map")
    assert api_map["schema"] == "torchlens.api_map.v1"
    names = {row["name"] for row in api_map["names"]}
    assert names == set(tl.__all__)
    assert "tl.report" in api_map["submodules_not_in_all"]
    json.dumps(api_map)


def test_artifact_tools_drive_the_same_public_surface(saved_trace_path: Path) -> None:
    """Overview, dump, and explain tools mirror the in-process spellings."""

    overview = tlmcp.call_tool("torchlens_load_overview", {"path": str(saved_trace_path)})
    assert overview["capture"]["capture_status"] == "complete"
    assert overview["counts"]["operations"] == 3
    assert "Sequential" in overview["summary"]

    dump = tlmcp.call_tool("torchlens_agent_dump", {"path": str(saved_trace_path), "max_ops": 2})
    assert dump["schema"] == "torchlens.agent_trace.v1"
    assert dump["truncation"]["ops_omitted"] == 3
    json.dumps(dump)

    report = tlmcp.call_tool(
        "torchlens_explain", {"path": str(saved_trace_path), "max_tokens": 100}
    )["report"]
    assert "Capture status" in report
    assert "Truncation" in report


def test_server_adapter_serves_the_declared_tools(saved_trace_path: Path) -> None:
    """The mcp>=2.0 server exposes exactly TOOL_SPECS and dispatches calls."""

    pytest.importorskip("mcp")
    pytest.importorskip("mcp.server")

    server = tlmcp._build_server()

    async def _drive() -> None:
        """List tools and run one artifact call through the live server."""

        tools = await server.list_tools()
        assert [tool.name for tool in tools] == [spec["name"] for spec in tlmcp.TOOL_SPECS]
        result = await server.call_tool(
            "torchlens_agent_dump", {"path": str(saved_trace_path), "max_ops": 1}
        )
        assert result.is_error is False
        payload = result.structured_content
        assert payload["schema"] == "torchlens.agent_trace.v1"
        assert payload["truncation"]["ops_omitted"] == 4

    asyncio.run(_drive())


def test_main_without_mcp_teaches_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing mcp package refuses with the extra named."""

    import builtins

    real_import = builtins.__import__

    def _blocked(name: str, *args: object, **kwargs: object) -> object:
        """Simulate an environment without the mcp package."""

        if name == "mcp.server" or name.startswith("mcp"):
            raise ImportError("No module named 'mcp'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    with pytest.raises(ImportError, match=r"torchlens\[mcp\]"):
        tlmcp.main()
