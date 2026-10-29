"""Keep optional backend scenarios syntactically runnable without their native runtimes."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path


def test_embedded_backend_scenarios_compile() -> None:
    """Migrated optional scenarios must remain executable Python when dependencies arrive."""
    checked = 0
    for path in Path(__file__).parent.glob("test_*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in {"_run_backend_subprocess", "run_paddle_subprocess"}
                and node.args
                and isinstance(source := node.args[0], ast.Constant)
                and isinstance(source.value, str)
            ):
                compile(textwrap.dedent(source.value), str(path), "exec")
                checked += 1
    assert checked > 0
