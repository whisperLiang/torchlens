"""Subprocess entry point for isolated surface-oracle snapshot generation.

The surface goldens freeze CONSTRUCTION-TIME behavior too: a model built
after torch has been wrapped can bake live wrappers into its state (the
SF-53 class — ctor-read ``F.<op>`` defaults, string-activation resolution),
which forks the snapshot depending on what ran earlier in the pytest
session. This worker runs in a fresh interpreter and constructs EVERY
requested model before the first capture, so no ctor ever sees wrapped
torch — for the regen path and the enforce path alike (b10 R78-1, the
capture-oracle isolation pattern batched to one process for the whole axis
family).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Print a JSON mapping of axis -> canonical surface dump on stdout.

    Parameters
    ----------
    argv:
        Optional explicit argument sequence (one or more model axes).

    Returns
    -------
    int
        Process exit status.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_axes", nargs="+")
    args = parser.parse_args(argv)

    from surface_oracle._snapshot import canonical_dump
    from surface_oracle._stages import build_stage_snapshots, prebuild_model_cases

    prebuilt = prebuild_model_cases(tuple(args.model_axes))
    dumps = {
        axis: canonical_dump(build_stage_snapshots(axis, prebuilt=prebuilt[axis]))
        for axis in args.model_axes
    }
    print(json.dumps(dumps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
