#!/usr/bin/env python
"""Normalize sdist tarballs in place so equal inputs give equal bytes.

``python -m build`` under ``SOURCE_DATE_EPOCH`` produces a deterministic wheel,
but the sdist stayed non-reproducible for two root-caused reasons (grind r4,
R86-1): the gzip ENVELOPE stores the compression wall-clock time in its header
(4 bytes that differ on every rebuild), and the tar members carry the build
user's uid/gid/uname/gname (which differ across machines, e.g. a CI runner vs
a maintainer checkout reproducing a release). Member mtimes are pinned to
SOURCE_DATE_EPOCH as a belt for the same cross-machine reason.

Usage (exactly how the release build_command and the nightly double-build gate
invoke it)::

    SOURCE_DATE_EPOCH="$(git log -1 --pretty=%ct)" \
        python scripts/normalize_sdist.py dist/*.tar.gz

The script is deterministic and idempotent: PAX output format, uid/gid 0,
empty user/group names, member mtime = SOURCE_DATE_EPOCH, gzip header mtime 0
with no embedded filename, compression level 9. It refuses to run without
SOURCE_DATE_EPOCH rather than silently minting a non-reproducible artifact.
"""

from __future__ import annotations

import gzip
import io
import os
import sys
import tarfile


def normalize_sdist(path: str, epoch: int) -> None:
    """Rewrite one ``.tar.gz`` sdist with deterministic tar and gzip metadata.

    Parameters
    ----------
    path:
        Path of the sdist tarball to rewrite in place.
    epoch:
        SOURCE_DATE_EPOCH value stamped onto every tar member.
    """

    with gzip.open(path, "rb") as compressed:
        raw_tar = compressed.read()

    normalized_tar = io.BytesIO()
    with (
        tarfile.open(fileobj=io.BytesIO(raw_tar)) as source,
        tarfile.open(fileobj=normalized_tar, mode="w", format=tarfile.PAX_FORMAT) as target,
    ):
        for member in source.getmembers():
            member.uid = 0
            member.gid = 0
            member.uname = ""
            member.gname = ""
            member.mtime = epoch
            member.pax_headers = {}
            payload = source.extractfile(member) if member.isreg() else None
            target.addfile(member, payload)

    # filename="" keeps the original-name field out of the gzip header;
    # mtime=0 zeroes the header timestamp (the 4 bytes that differed on
    # every rebuild); a fixed compression level keeps the deflate stream
    # itself stable.
    with (
        open(path, "wb") as output,
        gzip.GzipFile(
            filename="", mode="wb", fileobj=output, mtime=0, compresslevel=9
        ) as recompressed,
    ):
        recompressed.write(normalized_tar.getvalue())


def main(argv: list[str]) -> int:
    """Normalize every sdist named on the command line.

    Parameters
    ----------
    argv:
        Paths of ``.tar.gz`` sdists to normalize.

    Returns
    -------
    int
        Process exit code.
    """

    if not argv:
        print("usage: normalize_sdist.py DIST.tar.gz [...]", file=sys.stderr)
        return 2
    epoch_text = os.environ.get("SOURCE_DATE_EPOCH")
    if not epoch_text or not epoch_text.isdigit():
        print(
            "normalize_sdist.py: SOURCE_DATE_EPOCH must be set to an integer "
            "timestamp (refusing to mint a non-reproducible sdist)",
            file=sys.stderr,
        )
        return 2
    epoch = int(epoch_text)
    for path in argv:
        normalize_sdist(path, epoch)
        print(f"normalized {path} (epoch {epoch})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
