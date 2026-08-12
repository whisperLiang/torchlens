"""Cross-process control worker: run one scenario, emit its snapshot as JSON.

Invoked in a fresh interpreter so every process-scoped identity token
(object ids, barcodes, grad_fn addresses) genuinely differs from the parent
run — the real bijective-relabel case Check A must accept. Check B runs
in-process here (it needs the live model) and its verdict rides the payload.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import tempfile
from pathlib import Path


def snapshot_payload(run: object) -> dict:
    """Serialize a RunResult snapshot + Check B verdict to JSON-able form."""

    from ._comparator import check_b

    snapshot = run.snapshot  # type: ignore[attr-defined]
    return {
        "scenario": snapshot.scenario,
        "journal": snapshot.journal,
        "store": snapshot.store,
        "artifact": snapshot.artifact,
        "token_sites": [dataclasses.asdict(site) for site in snapshot.token_sites],
        "coherence": [list(row) for row in snapshot.coherence],
        "presence_only": [list(row) for row in snapshot.presence_only],
        "check_b_diffs": [dataclasses.asdict(diff) for diff in check_b(run)],
    }


def load_snapshot(payload: dict) -> object:
    """Rebuild a Snapshot from a worker payload."""

    from ._snapshot import Snapshot, TokenSite

    snapshot = Snapshot(scenario=payload["scenario"])
    snapshot.journal = payload["journal"]
    snapshot.store = payload["store"]
    snapshot.artifact = payload["artifact"]
    snapshot.token_sites = [TokenSite(**site) for site in payload["token_sites"]]
    snapshot.coherence = [tuple(row) for row in payload["coherence"]]
    snapshot.presence_only = [tuple(row) for row in payload["presence_only"]]
    return snapshot


def main() -> None:
    """Run ``<scenario>`` and write the snapshot payload to ``<out_path>``."""

    scenario_name, out_path = sys.argv[1], sys.argv[2]
    from ._models import scenario_by_name
    from ._snapshot import run_scenario

    with tempfile.TemporaryDirectory() as tmp:
        run = run_scenario(scenario_by_name(scenario_name), Path(tmp))
        payload = snapshot_payload(run)
    Path(out_path).write_text(json.dumps(payload))


if __name__ == "__main__":
    main()
