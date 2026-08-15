"""Hash-seed-independent metadata pickling for portable bundles (B3R4-R21-2).

Split out of ``scrub.py`` under the R43 file-size ratchet: this is the
byte-determinism seam for ``metadata.pkl``, consumed by the bundle and
streaming writers, independent of the scrub walk itself.
"""

import pickle
from typing import Any

__all__ = ["dump_canonical_metadata"]


class _CanonicalMetadataPickler(pickle._Pickler):
    """Pickler that emits exact set/frozenset members in sorted order (B3R4-R21-2).

    A ``frozenset`` pickles in its hash-table iteration order, which for str
    elements is salted by ``PYTHONHASHSEED`` -- so two processes capturing the
    identical program emitted byte-different ``metadata.pkl`` for identical
    logical content (the M6 relation views are exact frozensets of labels).
    Rewriting the reduction as ``cls(sorted_members)`` makes the persisted
    bytes hash-seed independent; ``builtins.set``/``builtins.frozenset`` are
    already on the safe-unpickler's explicit-globals allowlist, so loads
    admit the REDUCE spelling. Exact types only: subclasses keep their own
    reduce protocol.

    Deliberately the pure-Python ``pickle._Pickler``: the C pickler hardcodes
    exact set/frozenset saves and consults neither ``reducer_override`` nor
    ``dispatch_table`` for them (probed on 3.10). Metadata is the small
    sidecar of a bundle (tensor payloads ride safetensors blobs), so the
    slower Python walk is a save-time-only, determinism-buying cost.
    """

    def reducer_override(self, obj: Any) -> Any:
        """Return the sorted-members reduction for exact set/frozenset values."""

        cls = type(obj)
        if cls is frozenset or cls is set:
            try:
                members = sorted(obj)
            except TypeError:
                # Heterogeneous members: any deterministic total order works
                # for byte stability; type-name-then-repr is stable for the
                # pure-data values the scrub admits.
                members = sorted(obj, key=lambda member: (type(member).__qualname__, repr(member)))
            return (cls, (members,))
        return NotImplemented


def dump_canonical_metadata(state: Any, handle: Any) -> None:
    """Pickle scrubbed trace state with hash-seed-independent container bytes."""

    _CanonicalMetadataPickler(handle, protocol=pickle.HIGHEST_PROTOCOL).dump(state)
