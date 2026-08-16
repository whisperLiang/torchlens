"""Fail a CI leg whose pytest run executed fewer tests than promised.

The per-backend preview legs guard optional-dependency suites whose tests
``importorskip`` their framework. A missing or import-broken framework then
skips every test and pytest exits 0, so the leg stays green while covering
nothing. This check reads the run's junit XML and fails when the number of
EXECUTED tests (collected minus skipped) is below the leg's declared floor.

Usage::

    python scripts/check_ci_executed_tests.py <junit.xml> <min_executed>
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ElementTree
from pathlib import Path


def count_executed_tests(junit_path: Path) -> tuple[int, int]:
    """Return executed and skipped test counts from a junit XML report.

    Parameters
    ----------
    junit_path:
        Path to the pytest ``--junitxml`` output.

    Returns
    -------
    tuple[int, int]
        ``(executed, skipped)`` where executed is collected minus skipped.
    """

    root = ElementTree.parse(junit_path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    total = sum(int(suite.get("tests", 0)) for suite in suites)
    skipped = sum(int(suite.get("skipped", 0)) for suite in suites)
    return total - skipped, skipped


def main(argv: list[str]) -> int:
    """Run the executed-test floor check.

    Parameters
    ----------
    argv:
        ``[junit_xml_path, min_executed]`` with an optional trailing
        ``max_skipped_fraction`` (e.g. ``0.15``). The fraction bound scales
        WITH the selection (r7 R41: a hand floor weakens automatically as
        the suite grows — at a 5,026-test tier a 1,900 floor tolerated
        losing 61% silently), so a mass-importorskip cascade trips even
        when the absolute floor is still met.

    Returns
    -------
    int
        Process exit code: 0 when the floor is met, 1 otherwise.
    """

    if len(argv) not in (2, 3):
        print(
            "usage: check_ci_executed_tests.py <junit.xml> <min_executed> [max_skipped_fraction]",
            file=sys.stderr,
        )
        return 2
    junit_path = Path(argv[0])
    floor = int(argv[1])
    max_skipped_fraction = float(argv[2]) if len(argv) == 3 else None
    if not junit_path.exists():
        print(f"executed-test check FAILED: {junit_path} does not exist.", file=sys.stderr)
        return 1
    executed, skipped = count_executed_tests(junit_path)
    if executed < floor:
        print(
            f"executed-test check FAILED: {executed} executed (< floor {floor}), "
            f"{skipped} skipped. A fully-skipped suite means the backend runtime "
            "is missing or import-broken; this leg is covering nothing.",
            file=sys.stderr,
        )
        return 1
    total = executed + skipped
    if max_skipped_fraction is not None and total and (skipped / total) > max_skipped_fraction:
        print(
            f"executed-test check FAILED: {skipped}/{total} skipped "
            f"({skipped / total:.1%} > bound {max_skipped_fraction:.0%}). The "
            "selection collected but a skip cascade is hollowing it out.",
            file=sys.stderr,
        )
        return 1
    print(f"executed-test check passed: {executed} executed, {skipped} skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
