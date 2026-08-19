"""Model Context Protocol (MCP) bridge: serve TorchLens over stdio.

DOCUMENTED-UNSTABLE surface (naming ratification pending). The server exposes
read-only tools over SAVED ``.tlspec`` artifacts and the runtime environment,
wrapping the SAME public surface a human drives (``tl.load``,
``Trace.summary``, ``Trace.to_agent_json``, ``tl.report.explain``,
``tl.utils.doctor``) -- never a parallel API. No tool executes user code, and
no tool mutates anything: live capture stays a Python-process concern.

Run it as ``python -m torchlens.bridge.mcp`` (stdio transport; requires the
``mcp`` extra: ``pip install torchlens[mcp]``). The pure tool layer
(``TOOL_SPECS`` / ``call_tool``) has no ``mcp`` dependency so hosts and tests
can drive it directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

#: JSON Schema tool declarations served verbatim over MCP ``list_tools``.
TOOL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "torchlens_doctor",
        "description": (
            "Run the TorchLens environment health check (PyTorch/CUDA/"
            "Graphviz/extras/capability flags). Call this first when captures "
            "misbehave; each failing row names what is missing."
        ),
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "torchlens_api_map",
        "description": (
            "Machine-readable index of the public torchlens surface: every "
            "name in torchlens.__all__ with its kind and first docstring "
            "line. Use it to discover the exact spelling to write in Python."
        ),
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "torchlens_load_overview",
        "description": (
            "Load a saved .tlspec Trace artifact (analysis-only, no code "
            "execution) and return its text summary plus capture-honesty "
            "facts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to a .tlspec artifact."}
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "torchlens_agent_dump",
        "description": (
            "Return the torchlens.agent_trace.v1 machine-readable dump of a "
            "saved .tlspec Trace: capture facts, counts, pass-qualified op "
            "rows with graph edges, module hierarchy, and a navigation guide."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to a .tlspec artifact."},
                "max_ops": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Optional cap on op rows; omission is disclosed.",
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "torchlens_explain",
        "description": (
            "Plain-language report over a saved .tlspec Trace, optionally "
            "budgeted: max_tokens drops whole sections low-value-first and "
            "discloses every drop; capture-status honesty facts never drop."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to a .tlspec artifact."},
                "max_tokens": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Optional token budget (~4 chars/token).",
                },
                "audience": {
                    "type": "string",
                    "enum": ["researcher", "practitioner", "auto"],
                    "description": "Report style; defaults to 'auto'.",
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
)

#: Small mtime-keyed cache so repeated tool calls do not reload the artifact.
_TRACE_CACHE: dict[str, tuple[float, Any]] = {}
_TRACE_CACHE_MAX = 4


def _load_trace(path_arg: str) -> Any:
    """Load a saved Trace artifact for read-only inspection, with caching.

    Parameters
    ----------
    path_arg:
        Filesystem path to a ``.tlspec`` artifact.

    Returns
    -------
    Any
        Loaded ``Trace``.

    Raises
    ------
    ValueError
        If the path does not exist or the artifact is not a single Trace
        (bundles and intervention specs are out of the v1 tool contract).
    """

    import torchlens as tl

    path = Path(path_arg).expanduser()
    if not path.exists():
        raise ValueError(
            f"No file at {str(path)!r}. Pass the path of an artifact saved "
            "with tl.save(trace, path)."
        )
    key = str(path.resolve())
    mtime = path.stat().st_mtime
    cached = _TRACE_CACHE.get(key)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    loaded = tl.load(path)
    if not isinstance(loaded, tl.Trace):
        raise ValueError(
            f"{str(path)!r} loaded as {type(loaded).__name__}, not a Trace. "
            "The v1 MCP tools cover single-Trace artifacts; load bundles or "
            "intervention specs in Python via tl.load(...)."
        )
    if len(_TRACE_CACHE) >= _TRACE_CACHE_MAX:
        _TRACE_CACHE.pop(next(iter(_TRACE_CACHE)))
    _TRACE_CACHE[key] = (mtime, loaded)
    return loaded


def _tool_doctor() -> dict[str, Any]:
    """Run the environment health check.

    Returns
    -------
    dict[str, Any]
        Doctor rows as ``{"checks": [{"name", "status", "detail"}, ...]}``.
    """

    from ..utils import doctor

    report = doctor()
    return {
        "checks": [
            {"name": check.name, "status": check.status, "detail": check.detail}
            for check in report.checks
        ]
    }


def _api_map_entry(module: Any, name: str) -> dict[str, Any]:
    """Describe one public name for the API map.

    Parameters
    ----------
    module:
        Module owning the name (``torchlens``).
    name:
        Public attribute name.

    Returns
    -------
    dict[str, Any]
        ``{"name", "kind", "summary"}`` row.
    """

    value = getattr(module, name, None)
    if isinstance(value, type):
        kind = "class"
    elif callable(value):
        kind = "function"
    else:
        kind = type(value).__name__
    doc = (getattr(value, "__doc__", None) or "").strip()
    summary = doc.splitlines()[0] if doc else ""
    return {"name": name, "kind": kind, "summary": summary}


def _tool_api_map() -> dict[str, Any]:
    """Build the machine-readable public-surface index.

    Returns
    -------
    dict[str, Any]
        Every ``torchlens.__all__`` name with kind and first docstring line,
        plus the deliberately-unlisted submodules an agent should know about.
    """

    import torchlens as tl

    return {
        "schema": "torchlens.api_map.v1",
        "names": [_api_map_entry(tl, name) for name in sorted(tl.__all__)],
        "submodules_not_in_all": {
            "tl.report": "explain(), TraceProfile/build_profile, log_value",
            "tl.compat": "compat.report(model, x): capture-compatibility findings",
            "tl.debug": "power-user diagnostics (bisect_nan, hot_path, ...)",
            "tl.receptive_field": "lazy influence-geometry submodule",
            "tl.bridge": "optional external-tool adapters (captum, shap, mcp, ...)",
        },
        "docs": "docs/for-ai-agents.md is the agent-facing map of this surface.",
    }


def _tool_load_overview(path: str) -> dict[str, Any]:
    """Summarize a saved Trace artifact.

    Parameters
    ----------
    path:
        Filesystem path to a ``.tlspec`` artifact.

    Returns
    -------
    dict[str, Any]
        Text summary plus the capture block of the agent dump.
    """

    trace = _load_trace(path)
    dump = trace.to_agent_json(max_ops=1)
    return {
        "summary": trace.summary(),
        "capture": dump["capture"],
        "counts": dump["counts"],
    }


def _tool_agent_dump(path: str, max_ops: int | None = None) -> dict[str, Any]:
    """Return the agent dump of a saved Trace artifact.

    Parameters
    ----------
    path:
        Filesystem path to a ``.tlspec`` artifact.
    max_ops:
        Optional cap on emitted op rows.

    Returns
    -------
    dict[str, Any]
        ``torchlens.agent_trace.v1`` dump.
    """

    result = _load_trace(path).to_agent_json(max_ops=max_ops)
    return dict(result)


def _tool_explain(
    path: str,
    max_tokens: int | None = None,
    audience: str = "auto",
) -> dict[str, Any]:
    """Return the plain-language report of a saved Trace artifact.

    Parameters
    ----------
    path:
        Filesystem path to a ``.tlspec`` artifact.
    max_tokens:
        Optional token budget forwarded to ``tl.report.explain``.
    audience:
        Report style forwarded to ``tl.report.explain``.

    Returns
    -------
    dict[str, Any]
        ``{"report": <text>}``.
    """

    from ..report import explain

    trace = _load_trace(path)
    report = explain(trace, audience=audience, max_tokens=max_tokens)  # type: ignore[arg-type]
    return {"report": report}


def call_tool(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """Dispatch one MCP tool call to its handler.

    Parameters
    ----------
    name:
        Tool name from :data:`TOOL_SPECS`.
    arguments:
        JSON arguments matching the tool's ``input_schema``.

    Returns
    -------
    dict[str, Any]
        JSON-serializable tool result.

    Raises
    ------
    ValueError
        If the tool name is unknown or the arguments are invalid; the message
        names the valid tools or the fix.
    """

    args = dict(arguments or {})
    if name == "torchlens_doctor":
        return _tool_doctor()
    if name == "torchlens_api_map":
        return _tool_api_map()
    if name == "torchlens_load_overview":
        return _tool_load_overview(_required_path(args))
    if name == "torchlens_agent_dump":
        return _tool_agent_dump(_required_path(args), max_ops=args.get("max_ops"))
    if name == "torchlens_explain":
        return _tool_explain(
            _required_path(args),
            max_tokens=args.get("max_tokens"),
            audience=args.get("audience", "auto"),
        )
    known = ", ".join(spec["name"] for spec in TOOL_SPECS)
    raise ValueError(f"Unknown tool {name!r}. Known tools: {known}.")


def _required_path(args: dict[str, Any]) -> str:
    """Extract the required ``path`` argument.

    Parameters
    ----------
    args:
        Tool arguments.

    Returns
    -------
    str
        The ``path`` value.

    Raises
    ------
    ValueError
        If ``path`` is missing or not a string.
    """

    path = args.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError("This tool requires a 'path' string naming a .tlspec artifact.")
    return path


def _spec_description(name: str) -> str:
    """Return the declared description for one tool.

    Parameters
    ----------
    name:
        Tool name from :data:`TOOL_SPECS`.

    Returns
    -------
    str
        Declared tool description.
    """

    return next(str(spec["description"]) for spec in TOOL_SPECS if spec["name"] == name)


def _build_server() -> Any:
    """Build the wired MCP server serving :data:`TOOL_SPECS` over ``call_tool``.

    Returns
    -------
    Any
        ``mcp.server.MCPServer`` instance (mcp>=2.0 high-level API).

    Raises
    ------
    ImportError
        If the ``mcp`` package (>=2.0) is unavailable.
    """

    from mcp.server import MCPServer

    server = MCPServer(
        name="torchlens",
        instructions=(
            "Read-only TorchLens tools over saved .tlspec artifacts and the "
            "runtime environment. Live capture stays in Python: write "
            "tl.trace(...) there and tl.save(...) the result for these tools."
        ),
    )

    @server.tool(name="torchlens_doctor", description=_spec_description("torchlens_doctor"))
    def _doctor() -> dict[str, Any]:
        """Run the TorchLens environment health check."""

        return call_tool("torchlens_doctor")

    @server.tool(name="torchlens_api_map", description=_spec_description("torchlens_api_map"))
    def _api_map() -> dict[str, Any]:
        """Index the public torchlens surface."""

        return call_tool("torchlens_api_map")

    @server.tool(
        name="torchlens_load_overview",
        description=_spec_description("torchlens_load_overview"),
    )
    def _load_overview(path: str) -> dict[str, Any]:
        """Summarize a saved .tlspec Trace artifact."""

        return call_tool("torchlens_load_overview", {"path": path})

    @server.tool(
        name="torchlens_agent_dump",
        description=_spec_description("torchlens_agent_dump"),
    )
    def _agent_dump(path: str, max_ops: int | None = None) -> dict[str, Any]:
        """Dump a saved .tlspec Trace in torchlens.agent_trace.v1 form."""

        arguments: dict[str, Any] = {"path": path}
        if max_ops is not None:
            arguments["max_ops"] = max_ops
        return call_tool("torchlens_agent_dump", arguments)

    @server.tool(name="torchlens_explain", description=_spec_description("torchlens_explain"))
    def _explain(
        path: str,
        max_tokens: int | None = None,
        audience: str = "auto",
    ) -> dict[str, Any]:
        """Report on a saved .tlspec Trace in plain language."""

        arguments: dict[str, Any] = {"path": path, "audience": audience}
        if max_tokens is not None:
            arguments["max_tokens"] = max_tokens
        return call_tool("torchlens_explain", arguments)

    return server


async def _serve_stdio() -> None:
    """Run the MCP stdio server until the host disconnects.

    Raises
    ------
    ImportError
        If the ``mcp`` package is unavailable.
    """

    await _build_server().run_stdio_async()


def main() -> None:
    """Entry point for ``python -m torchlens.bridge.mcp``.

    Raises
    ------
    ImportError
        If the ``mcp`` package is unavailable; the message names the extra.
    """

    try:
        from mcp.server import MCPServer  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The MCP bridge requires the `mcp` extra (mcp>=2.0): install "
            "torchlens[mcp] (or `pip install 'mcp>=2.0'`)."
        ) from exc
    import asyncio

    asyncio.run(_serve_stdio())


if __name__ == "__main__":
    main()
