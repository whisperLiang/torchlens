"""Subprocess entry point for isolated capture-oracle characterization."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence


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

    # These must precede torch (including _characterize's transitive import):
    # MKL and ATen choose CPU-specific kernels at initialization. A seeded,
    # single-threaded convolution still differed byte-for-byte between the
    # recording host's AVX512 kernel and the nightly runner's CPU kernel.
    # Override inherited settings so regeneration and verification use the
    # same portable CPU paths even on differently configured developer hosts.
    os.environ["MKL_CBWR"] = "COMPATIBLE"
    os.environ["ATEN_CPU_CAPABILITY"] = "default"

    import torch

    from ._characterize import characterize_case

    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    print(json.dumps(characterize_case(args.case), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
