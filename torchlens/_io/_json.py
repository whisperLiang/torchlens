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
import re
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


def _prescan_depth(text: str, *, max_depth: int) -> None:
    """Refuse over-nested JSON via a single-pass, string-aware bracket-depth scan.

    Zero recursion: a running bracket depth that never enters ``json``'s recursive
    decoder. Quote/escape aware so brackets inside string literals do not count.
    Bails as soon as the depth ceiling is exceeded, so a nesting bomb is refused
    after ~``max_depth`` characters rather than scanning the whole payload.

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

    if len(text) > max_bytes:
        raise _refuse(f"manifest JSON exceeds the maximum size of {max_bytes} bytes", text)
    _prescan_depth(text, max_depth=max_depth)
    try:
        return json.loads(text)
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

    text = handle.read(max_bytes + 1)
    return loads_bounded(text, max_depth=max_depth, max_bytes=max_bytes)
