"""Bounded JSON reads for attacker-controlled ``.tlspec`` manifest boundaries.

A hostile ``.tlspec`` can carry a deeply-nested ``manifest.json`` whose stdlib
``json.load`` blows the C recursion stack -- an uncaught ``RecursionError`` that
escapes ``tl.load(path)`` before any descriptor-parse graceful-degradation net
runs (round 54 ``free_2``). This module owns the ONE bounded reader every manifest
/ metadata / format-detection JSON boundary routes through:

1. a byte ceiling (rejects an absurd manifest before it is read into memory), and
2. a single-pass, string-aware bracket-depth *prescan* that fails BEFORE the
   recursive ``json.loads`` ever runs.

Both violations surface as ``json.JSONDecodeError`` so every existing
``except (OSError, json.JSONDecodeError)`` handler at the call sites catches them
unchanged and applies its existing disposition (``None`` / typed
``TorchLensIOError``) -- an over-nested artifact degrades typed, never crashes.
``RecursionError`` is also caught as a belt in case the ambient recursion limit is
below the depth ceiling.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import IO, Any

# Manifests for large models carry many tensor entries but are shallow (nesting is
# a handful of levels); the deepest legitimate structure is a nested literal-list
# constant, still far below this. 200 sits comfortably above any real manifest and
# comfortably below the ~496-frame depth at which stdlib ``json.load`` overflows
# the C stack at the default recursion limit, so the prescan always fires first.
_MAX_JSON_DEPTH = 200

# 512 MiB: no legitimate ``manifest.json`` approaches this; it bounds the prescan's
# worst-case cost and rejects an absurd artifact before it is read into memory.
_MAX_JSON_BYTES = 512 * 1024 * 1024

_OPEN_BRACKETS = frozenset("[{")
_CLOSE_BRACKETS = frozenset("]}")

# The prescan's state machine only ever transitions on a quote, a backslash, or a
# bracket; every other character is inert. Both patterns below let the C regex
# engine skip the inert bulk (a multi-MB manifest is >90% inert) instead of paying
# one Python loop iteration per character.
_QUOTE_OR_BRACKET = re.compile(r'["\[\]{}]')
_ESCAPE_RELEVANT = re.compile(r'["\\\[\]{}]')

# Reduce/scan in bounded slices so a nesting bomb is still refused after roughly
# one chunk rather than after a full-payload reduction (the original per-character
# loop bailed within ~``max_depth`` characters).
_PRESCAN_CHUNK_CHARS = 1 << 20


def _refuse(detail: str, text: str) -> json.JSONDecodeError:
    """Build a ``JSONDecodeError`` so existing JSON-boundary handlers catch it."""

    return json.JSONDecodeError(detail, text if text else " ", 0)


def _fd_size(handle: IO[Any]) -> int | None:
    """Return the size of an open handle's fd, or ``None`` if it cannot be stat'd.

    ``fstat`` on the fd we are about to read (rather than a separate ``path.stat``)
    keeps the size measurement bound to the exact bytes the read will consume.
    """

    try:
        return os.fstat(handle.fileno()).st_size
    except (OSError, ValueError, AttributeError):  # non-seekable / pipe / no fileno
        return None


def _bounded_read_bytes(handle: IO[bytes], max_bytes: int) -> bytes:
    """Read up to ``max_bytes`` bytes, allocating the FILE size, not the ceiling.

    ``handle.read(max_bytes + 1)`` pre-allocates a ``max_bytes + 1`` buffer whatever
    the real file size is, so a tiny manifest under the 512-MiB ceiling still
    transiently requested ~512 MiB (R33-1). We stat the open fd first: a file already
    over the ceiling is refused before any allocation, and a file within it reads
    exactly its own size plus one sentinel byte (to still catch a file that grew after
    the stat). When the size cannot be determined (a pipe/non-seekable handle) we fall
    back to the ceiling read, since there is nothing else to bound it by.
    """

    size = _fd_size(handle)
    if size is not None and size > max_bytes:
        raise _refuse(f"manifest JSON exceeds the maximum size of {max_bytes} bytes", "")
    read_count = (size + 1) if size is not None else (max_bytes + 1)
    data = handle.read(read_count)
    if len(data) > max_bytes:  # grew after the stat, or unbounded fallback
        raise _refuse(f"manifest JSON exceeds the maximum size of {max_bytes} bytes", "")
    return data


def _prescan_depth(text: str, *, max_depth: int) -> None:
    """Refuse over-nested JSON via a single-pass, string-aware bracket-depth scan.

    Zero recursion: a running bracket depth that never enters ``json``'s recursive
    decoder. Quote/escape aware so brackets inside string literals do not count.
    The scan bails as soon as the running depth exceeds the ceiling, so a nesting
    bomb whose brackets are front-loaded is refused within the FIRST chunk rather
    than after scanning the whole payload. Note the bail is not per-character:
    ``findall`` first materializes the matches for the current
    ``_PRESCAN_CHUNK_CHARS`` slice, so up to one chunk's quotes/brackets are
    collected before the depth check can fire.

    The scan is the same state machine as a naive per-character loop, but the
    inert characters are skipped by the regex engine. When the payload carries no
    backslash at all the escape branch is unreachable, so each chunk is reduced to
    just its quotes/brackets before the Python loop runs; otherwise the scan walks
    only the escape-relevant character positions (a backslash's effect on the very
    next character is recovered from the match offsets).
    """

    depth = 0
    in_string = False
    if "\\" not in text:
        for start in range(0, len(text) or 1, _PRESCAN_CHUNK_CHARS):
            for char in _QUOTE_OR_BRACKET.findall(text[start : start + _PRESCAN_CHUNK_CHARS]):
                if in_string:
                    if char == '"':
                        in_string = False
                    continue
                if char == '"':
                    in_string = True
                elif char in _OPEN_BRACKETS:
                    depth += 1
                    if depth > max_depth:
                        raise _refuse(
                            f"manifest JSON nesting exceeds the maximum depth of {max_depth}",
                            text,
                        )
                elif depth > 0:
                    depth -= 1
        return

    # ``escaped_at`` is the absolute offset of the character a backslash escapes.
    # Reaching an escape-relevant character at that exact offset consumes the
    # escape; reaching a later one means the escaped character was inert and the
    # escape has already been consumed. Either way the flag clears, which is
    # exactly what a per-character loop does.
    escaped_at = -1
    for match in _ESCAPE_RELEVANT.finditer(text):
        char = match[0]
        if in_string:
            position = match.start()
            if position == escaped_at:
                escaped_at = -1
                continue
            escaped_at = -1
            if char == "\\":
                escaped_at = position + 1
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in _OPEN_BRACKETS:
            depth += 1
            if depth > max_depth:
                raise _refuse(
                    f"manifest JSON nesting exceeds the maximum depth of {max_depth}", text
                )
        elif char in _CLOSE_BRACKETS:
            if depth > 0:
                depth -= 1


def loads_bounded(
    text: str,
    *,
    max_depth: int = _MAX_JSON_DEPTH,
    max_bytes: int = _MAX_JSON_BYTES,
) -> Any:
    """Parse a JSON string after a byte-size ceiling and a depth prescan.

    Raises ``json.JSONDecodeError`` on an over-size or over-nested payload (so the
    existing boundary handlers catch it), and re-raises a ``RecursionError`` from
    the stdlib decoder as a ``json.JSONDecodeError`` belt.
    """

    # ``len(text.encode("utf-8"))`` allocates a full extra copy of the payload just
    # to measure it -- ~1.5x the 512-MiB ceiling before json even parses (B8-14).
    # UTF-8 uses >= 1 byte per code point, so ``len(text) <= byte length`` always:
    # a character count over the ceiling is a byte count over the ceiling (refuse
    # without encoding), and only when the character count is within the ceiling can
    # multibyte inflation push the byte count over, so the encode runs only there.
    if len(text) > max_bytes or len(text.encode("utf-8")) > max_bytes:
        raise _refuse(f"manifest JSON exceeds the maximum size of {max_bytes} bytes", text)
    _prescan_depth(text, max_depth=max_depth)

    def _reject_constant(name: str) -> Any:
        # The writers use ``allow_nan=False``, so NaN/Infinity in an artifact
        # is a forgery -- and one that loads fine but makes every re-save
        # raise (a stillborn artifact). Refuse at parse instead.
        raise _refuse(f"manifest JSON contains the non-finite constant {name}", text)

    try:
        return json.loads(text, parse_constant=_reject_constant)
    except RecursionError as exc:  # pragma: no cover - prescan normally fires first
        raise _refuse(
            "manifest JSON nesting exceeded the interpreter recursion limit", text
        ) from exc


def load_bounded(
    handle: IO[str],
    *,
    max_depth: int = _MAX_JSON_DEPTH,
    max_bytes: int = _MAX_JSON_BYTES,
) -> Any:
    """Read a JSON object from a text handle with a bounded size and nesting depth.

    Reads at most ``max_bytes + 1`` bytes so an oversize file is rejected without
    being fully loaded into memory, then delegates to :func:`loads_bounded`.
    """

    raw_handle = getattr(handle, "buffer", None)
    if raw_handle is not None:
        raw = _bounded_read_bytes(raw_handle, max_bytes)
        encoding = getattr(handle, "encoding", None) or "utf-8"
        text = raw.decode(encoding)
    else:
        # Text handle with no binary buffer: read one char past the fd size (or the
        # ceiling if it cannot be stat'd) rather than the whole ceiling.
        size = _fd_size(handle)
        if size is not None and size > max_bytes:
            raise _refuse(f"manifest JSON exceeds the maximum size of {max_bytes} bytes", "")
        text = handle.read((size + 1) if size is not None else (max_bytes + 1))
        if len(text.encode("utf-8")) > max_bytes:
            raise _refuse(f"manifest JSON exceeds the maximum size of {max_bytes} bytes", text)
    return loads_bounded(text, max_depth=max_depth, max_bytes=max_bytes)


def read_bounded(
    path: Path,
    *,
    encoding: str = "utf-8",
    max_depth: int = _MAX_JSON_DEPTH,
    max_bytes: int = _MAX_JSON_BYTES,
) -> Any:
    """Read and parse a JSON file without ever allocating the whole file first.

    ``Path.read_text()`` materializes the ENTIRE attacker-controlled file before
    any ceiling can be applied, so a multi-GiB ``manifest.json`` is an allocation
    DoS even though :func:`loads_bounded` would have refused it a microsecond
    later. This helper opens the file and reads at most ``max_bytes + 1`` bytes, so
    the ceiling is enforced BEFORE the allocation.

    Parameters
    ----------
    path:
        JSON file to read.
    encoding:
        Text encoding of the file.
    max_depth:
        Maximum permitted bracket nesting depth.
    max_bytes:
        Maximum permitted payload size.

    Returns
    -------
    Any
        The decoded JSON value.

    Raises
    ------
    json.JSONDecodeError
        On an over-size or over-nested payload, so the existing JSON-boundary
        handlers at each call site catch it unchanged.
    """

    with path.open("rb") as handle:
        data = _bounded_read_bytes(handle, max_bytes)
    return loads_bounded(data.decode(encoding), max_depth=max_depth, max_bytes=max_bytes)


def read_bytes_bounded(path: Path, *, max_bytes: int = _MAX_JSON_BYTES) -> bytes:
    """Read a JSON file's RAW bytes under the same ceiling as :func:`read_bounded`.

    For the boundaries that must hash the exact on-disk bytes (the merged-artifact
    descriptor checksum) before parsing them. ``Path.read_bytes()`` would allocate
    the whole attacker-sized file first; this reads at most ``max_bytes + 1`` and
    refuses over-size payloads through the same ``JSONDecodeError`` channel.

    Parameters
    ----------
    path:
        JSON file to read.
    max_bytes:
        Maximum permitted payload size.

    Returns
    -------
    bytes
        The file's raw bytes.

    Raises
    ------
    json.JSONDecodeError
        When the payload exceeds ``max_bytes``.
    """

    with path.open("rb") as handle:
        return _bounded_read_bytes(handle, max_bytes)
