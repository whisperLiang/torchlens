"""Lock public documentation names, paths, links, and dated tier claims to code."""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

import torchlens as tl

pytestmark = pytest.mark.smoke

PUBLIC_SURFACE_SIZE = 97
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


#: Structured smoke-count claim in CLAUDE.md's tier section. The lockstep test
#: below requires the claim to PARSE (dated, with raw collect-only numbers);
#: the drift tripwire compares the parsed numbers against a live collection so
#: the gate detects staleness instead of freezing it (the pre-r3 version
#: hard-asserted the literal count, ENFORCING the stale doc; R41/R81/R88).
TIER_CLAIM_RE = re.compile(
    r"~\d+(?:\.\d+)?k tests \((?P<smoke>\d[\d,]*)/(?P<total>\d[\d,]*) "
    r"collect-only, measured (?P<date>20\d{2}-\d{2}-\d{2})\)"
)


def test_dated_test_tier_claim_is_present_and_selector_is_additive() -> None:
    """Require a parseable dated tier record and default rare-test exclusion."""

    root = _repo_root()
    guide = (root / "CLAUDE.md").read_text(encoding="utf-8")
    assert TIER_CLAIM_RE.search(guide), (
        "CLAUDE.md's Testing Tiers section lost its structured smoke-count "
        "claim ('~Nk tests (S/T collect-only, measured YYYY-MM-DD)'); the "
        "drift tripwire needs it parseable"
    )
    assert re.search(r"measured 20\d{2}-\d{2}-\d{2}.{0,200}took \d+s \(~\d+ min\)", guide, re.S), (
        "CLAUDE.md lost its dated smoke wall-clock measurement record"
    )
    assert 'pytest tests/ -m "not rare and not slow"' in guide

    test_guide = (root / "tests/AGENTS.md").read_text(encoding="utf-8")
    assert 'pytest tests/ -m "not rare and not slow"' in test_guide


#: Structured duration-budget claim, required VERBATIM-parseable in both agent
#: docs. The r3 re-tier shipped a 5s load-scaled budget while CLAUDE.md said
#: "15s" three lines from its own "<5s" partition table and tests/AGENTS.md
#: repeated the 15s — no gate parsed either sentence (R41-1/R81-2/R88-3,
#: cross-filed by all three labs). The numbers are compared against the
#: shipped conftest constants below, so budget and doc can only move together.
BUDGET_CLAIM_RE = re.compile(
    r"budget (?P<smoke>\d+(?:\.\d+)?)s and heavy (?P<heavy>\d+(?:\.\d+)?)s "
    r"\(load-scaled 1x-4x, charged\s+on min\(wall, cpu\)\)"
)


def test_documented_duration_budgets_match_the_shipped_constants() -> None:
    """CLAUDE.md and tests/AGENTS.md budget sentences track conftest reality."""

    root = _repo_root()
    conftest_text = (root / "tests" / "conftest.py").read_text(encoding="utf-8")
    shipped = {
        name: float(match)
        for name, pattern in (
            ("smoke", r"^SMOKE_DURATION_BUDGET_SECONDS\s*=\s*([\d.]+)"),
            ("heavy", r"^HEAVY_DURATION_BUDGET_SECONDS\s*=\s*([\d.]+)"),
        )
        for match in re.findall(pattern, conftest_text, flags=re.MULTILINE)
    }
    assert set(shipped) == {"smoke", "heavy"}, "conftest budget constants moved or renamed"
    for doc in ("CLAUDE.md", "tests/AGENTS.md"):
        text = (root / doc).read_text(encoding="utf-8")
        claim = BUDGET_CLAIM_RE.search(text)
        assert claim is not None, (
            f"{doc} lost its structured duration-budget sentence ('budget Ns and "
            "heavy Ms (load-scaled 1x-4x, charged on min(wall, cpu))'); the doc "
            "and the shipped budget must move together (R41-1)"
        )
        for tier in ("smoke", "heavy"):
            assert float(claim.group(tier)) == shipped[tier], (
                f"{doc} documents a {tier} budget of {claim.group(tier)}s but "
                f"tests/conftest.py ships {shipped[tier]}s — update both together"
            )


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
