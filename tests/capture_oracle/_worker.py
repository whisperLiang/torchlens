"""Subprocess entry point for isolated capture-oracle characterization."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from ._characterize import characterize_case


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the worker command line.

    Parameters
    ----------
    argv:
        Optional explicit argument sequence.

    Returns
    -------
    argparse.Namespace
        Parsed case identifier.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Generate one JSON characterization on standard output.

    Parameters
    ----------
    argv:
        Optional explicit argument sequence.

    Returns
    -------
    int
        Process exit status.
    """

    args = _parse_args(argv)

    # Pin the execution environment the seeds alone do not cover: thread
    # count and kernel selection both steer float bytes, and the goldens
    # compare raw sha256 chunks (b10 R78-7). Pinning lives HERE so the regen
    # path and the enforce path run under identical settings by construction.
    import torch

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)

    print(json.dumps(characterize_case(args.case), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
