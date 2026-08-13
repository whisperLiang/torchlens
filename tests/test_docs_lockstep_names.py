"""Lock public documentation names, paths, links, and dated tier claims to code."""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import torchlens as tl

PUBLIC_SURFACE_SIZE = 96
PUBLIC_SURFACE_DOCS = (
    "CLAUDE.md",
    "torchlens/AGENTS.md",
    "torchlens/CLAUDE.md",
    "docs/for-ai-agents.md",
    "docs/migration/v2.0_api_changes.md",
)
CANONICAL_SYMBOLS = {
    "torchlens.backends.torch.wrappers": ("unwrap_torch",),
    "torchlens": ("Bundle", "merge_ranks", "merge_report"),
    "torchlens.distributed": ("arm",),
}
MARKDOWN_LINK_RE = re.compile(r"\[[^]]*\]\((?P<target>[^)]+)\)")


def _repo_root() -> Path:
    """Return the repository root.

    Returns
    -------
    Path
        Absolute repository root.
    """

    return Path(__file__).resolve().parents[1]


def _public_markdown_files() -> tuple[Path, ...]:
    """Return public Markdown files covered by link lockstep.

    Returns
    -------
    tuple[Path, ...]
        README, root guides, and every documentation page.
    """

    root = _repo_root()
    return (
        root / "README.md",
        root / "CLAUDE.md",
        root / "AGENTS.md",
        *sorted((root / "docs").rglob("*.md")),
    )


def test_public_surface_count_claims_match_runtime() -> None:
    """Keep every numeric ``__all__`` claim synchronized with the runtime."""

    root = _repo_root()
    assert len(tl.__all__) == PUBLIC_SURFACE_SIZE
    count_claim = re.compile(
        r"(?:(?P<before>\d+)[ -]?(?:name|top-level)|"
        r"(?:has|exposes|reserves(?: exactly)?)[^\d\n]*(?P<after>\d+))"
    )
    for relative_path in PUBLIC_SURFACE_DOCS:
        text = (root / relative_path).read_text(encoding="utf-8")
        claims = [
            int(match.group("before") or match.group("after"))
            for line in text.splitlines()
            if "__all__" in line
            for match in count_claim.finditer(line)
        ]
        assert claims, f"Expected an __all__ count claim in {relative_path}"
        assert set(claims) == {PUBLIC_SURFACE_SIZE}, (relative_path, claims)


def test_canonical_documented_symbols_resolve() -> None:
    """Resolve public symbol paths whose drift previously broke the guides."""

    for module_name, attributes in CANONICAL_SYMBOLS.items():
        module = importlib.import_module(module_name)
        for attribute in attributes:
            assert hasattr(module, attribute), f"{module_name}.{attribute} does not resolve"


def test_agent_docs_use_current_internal_paths() -> None:
    """Pin corrected backward, schema, utility, and Bundle documentation paths."""

    root = _repo_root()
    expected_paths = (
        "torchlens/backends/torch/backward.py",
        "torchlens/_source_links.py",
        "torchlens/schemas/tlspec_manifest_v1.json",
        "torchlens/schemas/tlspec_manifest_v2.json",
    )
    for relative_path in expected_paths:
        assert (root / relative_path).is_file(), relative_path

    validation_docs = "\n".join(
        (root / path).read_text(encoding="utf-8")
        for path in ("torchlens/validation/AGENTS.md", "torchlens/validation/CLAUDE.md")
    )
    assert "tlspec_manifest_v{schema_version}.json" in validation_docs
    assert "capture/backward.py" not in validation_docs

    intervention_doc = (root / "torchlens/intervention/CLAUDE.md").read_text(encoding="utf-8")
    assert "tl.Bundle" in intervention_doc
    assert "torchlens.bundle.Bundle" not in intervention_doc


def test_dated_test_tier_claim_is_present_and_selector_is_additive() -> None:
    """Require a dated tier record and preserve default rare-test exclusion."""

    root = _repo_root()
    guide = (root / "CLAUDE.md").read_text(encoding="utf-8")
    assert "measured 2026-08-13" in guide
    assert "~3.2k tests" in guide
    assert "1194s (~20 min)" in guide
    assert 'pytest tests/ -m "not rare and not slow"' in guide

    test_guide = (root / "tests/AGENTS.md").read_text(encoding="utf-8")
    assert 'pytest tests/ -m "not rare and not slow"' in test_guide


def test_public_relative_markdown_links_resolve() -> None:
    """Ensure every local Markdown link names an existing repository path."""

    missing: list[tuple[str, str]] = []
    root = _repo_root()
    for page in _public_markdown_files():
        for match in MARKDOWN_LINK_RE.finditer(page.read_text(encoding="utf-8")):
            target = match.group("target").split("#", maxsplit=1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            if not (page.parent / target).resolve().exists():
                missing.append((str(page.relative_to(root)), target))
    assert not missing
