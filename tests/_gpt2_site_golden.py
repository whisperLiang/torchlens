"""Identify the upstream reference for the GPT-2 structural census."""

from importlib.metadata import version
from pathlib import Path

from packaging.requirements import Requirement


def is_reference_transformers() -> bool:
    """Return whether the installed Transformers matches the CI golden constraint."""

    path = Path(__file__).resolve().parents[1] / ".github/constraints/transformers-golden.txt"
    requirements = (
        Requirement(line)
        for line in path.read_text().splitlines()
        if line and not line.startswith("#")
    )
    requirement = next(item for item in requirements if item.name == "transformers")
    return requirement.specifier.contains(version("transformers"))
