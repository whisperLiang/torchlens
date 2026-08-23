#!/usr/bin/env python
"""Normalize sdists AND wheels in place so equal inputs give equal bytes.

``python -m build`` under ``SOURCE_DATE_EPOCH`` produces a wheel whose
timestamps are deterministic, but two machine-dependent residuals remained
(grind r4 R86-1; grind r5 b10 R84-1, MEASURED): the sdist's gzip ENVELOPE
stores the compression wall-clock time, its tar members carry the build
user's uid/gid/uname/gname, and BOTH artifacts carry source-file MODES
(``chmod 644`` vs ``664`` under umask 022 vs 002 — a GitHub runner vs a
group-writable maintainer checkout), in the tar member headers and the
wheel zip entries' ``external_attr`` respectively. Modes are normalized to
0o644 (0o755 when any execute bit is set) so a checkout's umask can never
change published bytes.

Usage (exactly how the release build_command and the nightly double-build gate
invoke it)::

    SOURCE_DATE_EPOCH="$(git log -1 --pretty=%ct)" \
        python scripts/normalize_sdist.py dist/*.tar.gz dist/*.whl

The script is deterministic and idempotent: PAX output format, uid/gid 0,
empty user/group names, member mtime = SOURCE_DATE_EPOCH, normalized member
modes, gzip header mtime 0 with no embedded filename, compression level 9
for both containers. It refuses to run without SOURCE_DATE_EPOCH rather
than silently minting a non-reproducible artifact.
"""

from __future__ import annotations

import gzip
import io
import os
import sys
import tarfile
import zipfile


def _normalized_mode(mode: int) -> int:
    """Map any source-file mode to the canonical 644/755 pair."""

    return 0o755 if mode & 0o111 else 0o644


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
            # The build user's umask leaks into member modes (644 vs 664)
            # and forked sdist bytes across machines (grind r5, b10 R84-1).
            member.mode = _normalized_mode(member.mode)
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


def normalize_wheel(path: str) -> None:
    """Rewrite one ``.whl`` with normalized zip entry modes.

    The wheel's timestamps are already deterministic under
    ``SOURCE_DATE_EPOCH``, but data/license members inherit their source
    files' modes in ``external_attr`` (measured: 7 members differing 0o644
    vs 0o664 across umasks, CRCs identical — grind r5, b10 R84-1). Entry
    order, names, timestamps, and contents are preserved; only the unix
    mode bits are canonicalized, so RECORD hashes stay valid.

    Parameters
    ----------
    path:
        Path of the wheel to rewrite in place.
    """

    normalized = io.BytesIO()
    with (
        zipfile.ZipFile(path) as source,
        zipfile.ZipFile(normalized, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as target,
    ):
        for info in source.infolist():
            data = source.read(info.filename)
            clone = zipfile.ZipInfo(info.filename, date_time=info.date_time)
            clone.compress_type = zipfile.ZIP_DEFLATED
            clone.create_system = 3  # unix, so the mode bits are authoritative
            mode = (info.external_attr >> 16) & 0o7777
            # Carry the original file-TYPE bits through and rewrite only the
            # permission bits (r7 R84: stamping S_IFREG unconditionally would
            # republish an explicit ``dir/`` member as a zero-length regular
            # file; latent on setuptools-83 wheels, wrong for the general
            # normalizer this is written as). Untyped entries get S_IFREG.
            type_bits = (info.external_attr >> 16) & 0o170000 or 0o100000
            clone.external_attr = (type_bits | _normalized_mode(mode)) << 16
            target.writestr(clone, data)
    with open(path, "wb") as output:
        output.write(normalized.getvalue())


def main(argv: list[str]) -> int:
    """Normalize every sdist / wheel named on the command line.

    Parameters
    ----------
    argv:
        Paths of ``.tar.gz`` sdists and ``.whl`` wheels to normalize.

    Returns
    -------
    int
        Process exit code.
    """

    if not argv:
        print("usage: normalize_sdist.py DIST.tar.gz DIST.whl [...]", file=sys.stderr)
        return 2
    epoch_text = os.environ.get("SOURCE_DATE_EPOCH")
    if not epoch_text or not epoch_text.isdigit():
        print(
            "normalize_sdist.py: SOURCE_DATE_EPOCH must be set to an integer "
            "timestamp (refusing to mint a non-reproducible artifact)",
            file=sys.stderr,
        )
        return 2
    epoch = int(epoch_text)
    for path in argv:
        if path.endswith(".whl"):
            normalize_wheel(path)
        else:
            normalize_sdist(path, epoch)
        print(f"normalized {path} (epoch {epoch})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
