"""Fail before rendering tests when Graphviz binaries or layout plugins are broken."""

from __future__ import annotations

import shutil
import subprocess


def main() -> None:
    """Verify binary provenance, both neato entry points, and pinned rank rendering."""

    for binary in ("dot", "neato"):
        print(f"{binary}: {shutil.which(binary)}", flush=True)
        subprocess.run([binary, "-V"], check=True, timeout=30)
    source = 'digraph { a [pos="0,0!"]; b [pos="72,72!"]; a -> b; }'
    for command in (
        ["dot", "-Tsvg"],
        ["dot", "-Kneato", "-Tsvg"],
        ["neato", "-Tsvg"],
        ["neato", "-n", "-Tsvg"],
    ):
        result = subprocess.run(
            command, input=source, text=True, capture_output=True, check=False, timeout=30
        )
        if result.returncode or "<svg" not in result.stdout:
            raise RuntimeError(f"{' '.join(command)} failed:\n{result.stderr}")
        print(f"{' '.join(command)}: OK", flush=True)


if __name__ == "__main__":
    main()
