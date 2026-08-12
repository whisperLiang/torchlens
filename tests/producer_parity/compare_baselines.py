"""Temporal-baseline comparison (P2..P5 control; Opus v4 C3 lineage).

Runs every parity scenario at the CURRENT tree and Check-A-compares each
snapshot against the frozen pre-migration baseline archive. Any uniform
dual-leg regression the leg-vs-leg parity harness cannot see is red here.

    PYTHONPATH=.:tests python -m producer_parity.compare_baselines <baseline_dir>
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


def _normalize_provenance(node) -> None:
    """Presence-mark provenance keys in place on an artifact tree."""

    if isinstance(node, dict):
        for key, value in node.items():
            if key == "git_commit_hash" and isinstance(value, str):
                node[key] = {"__volatile__": "str", "present": True}
            else:
                _normalize_provenance(value)
    elif isinstance(node, list):
        for item in node:
            _normalize_provenance(item)


def main() -> None:
    baseline_dir = Path(sys.argv[1])
    from producer_parity._comparator import check_a, check_b
    from producer_parity._models import SCENARIOS
    from producer_parity._snapshot import run_scenario
    from producer_parity._worker import load_snapshot

    manifest = json.loads((baseline_dir / "MANIFEST.json").read_text())
    failures = 0
    for scenario in SCENARIOS:
        baseline_path = baseline_dir / f"{scenario.name}.json"
        if not baseline_path.exists():
            print(f"SKIP {scenario.name}: no baseline")
            continue
        baseline = load_snapshot(json.loads(baseline_path.read_text()))
        with tempfile.TemporaryDirectory() as tmp:
            run = run_scenario(scenario, Path(tmp))
        # older baseline archives predate the git_commit_hash canonicalizer;
        # normalize BOTH sides so provenance never masquerades as a regression
        _normalize_provenance(baseline.artifact)
        _normalize_provenance(run.snapshot.artifact)
        diffs = check_a(baseline, run.snapshot) + check_b(run)
        if diffs:
            failures += 1
            print(f"RED  {scenario.name}: {len(diffs)} diffs")
            for diff in diffs[:8]:
                print(f"     [{diff.check}/{diff.kind}] {diff.layer}:{diff.anchor}:{diff.path} {diff.detail[:160]}")
        else:
            print(f"OK   {scenario.name}")
    print(f"baseline tree: {manifest['git_sha']}; failures: {failures}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
