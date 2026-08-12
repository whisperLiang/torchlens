"""Temporal baseline capture (P0 -> P5 control, Opus v4 note lineage C3).

Runs every parity scenario at the CURRENT tree and archives the canonical
three-layer snapshots (plus each run's Check B verdict). P5 diffs the
post-migration LEGACY leg against these frozen baselines, so a regression
introduced uniformly on both legs cannot self-certify through leg-vs-leg
parity alone. Run from the repo root at the pre-migration base:

    PYTHONPATH=.:tests python -m producer_parity.capture_baselines <out_dir>
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)

    from producer_parity._models import SCENARIOS
    from producer_parity._snapshot import run_scenario
    from producer_parity._worker import snapshot_payload

    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    manifest: dict = {"git_sha": sha, "scenarios": []}
    for scenario in SCENARIOS:
        with tempfile.TemporaryDirectory() as tmp:
            run = run_scenario(scenario, Path(tmp))
            payload = snapshot_payload(run)
        if payload["check_b_diffs"]:
            raise SystemExit(
                f"baseline capture is not clean: {scenario.name}: {payload['check_b_diffs'][:3]}"
            )
        (out_dir / f"{scenario.name}.json").write_text(json.dumps(payload))
        manifest["scenarios"].append(scenario.name)
        print(f"archived {scenario.name}")
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print(f"baselines archived at {out_dir} (tree {sha})")


if __name__ == "__main__":
    main()
