"""The two-check parity discharge (design-of-record section 6.2).

Check A — cross-leg structural comparison. Non-token cells compare exactly;
volatile-token buckets compare structurally: presence/type/count per anchored
site, plus the alias partition of anchored sites compared exactly as sets of
site-sets. Bijective token relabelings are green BY CONSTRUCTION (the ids are
arbitrary); co-location-breaking permutations of multi-site tokens are red.
Payload-digest ties ride the exact non-token cell comparison (payload digests
are ordinary cells, so a digest moving between anchors is an ordinary diff).

Check B — within-leg referent attestation. Each attested bucket's stored
tokens are compared against ground truth recorded by the P0 shims (or, for
the coherence bucket, at snapshot time) WITHIN the same leg. A producer that
writes op B's referent token into op A's row fails here even when the swap is
structurally invisible to Check A (the singleton-token case that made the v3
one-comparator design undecidable).

Check A's partition soundness rests on a token-retention premise (Opus v4
note N4): id-derived tokens must not be recycled within one run's comparison
window. ``test_attestation`` asserts the strong-reference premise for the
grad_fn bucket.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from ._snapshot import Snapshot, TokenSite


# Buckets with a sound token-retention premise (strong refs / model-owned
# objects for the whole run): the alias-partition check applies. Buckets NOT
# here (backend_handle_id and its id-space siblings) are presence + Check B
# only, because their referents are transient and ids get recycled (N4).
PARTITION_BUCKETS: frozenset[str] = frozenset(
    {"grad_fn_object_id", "tl_barcode", "tl_barcode_derived"}
)


@dataclass(frozen=True)
class Diff:
    """One parity finding."""

    check: str  # "A" | "B"
    kind: str
    layer: str
    anchor: str
    path: str
    detail: str


def _cell_diffs(kind_prefix: str, left: dict, right: dict, layer: str) -> list[Diff]:
    diffs: list[Diff] = []
    for anchor in sorted(set(left) | set(right)):
        if anchor not in left or anchor not in right:
            diffs.append(
                Diff("A", f"{kind_prefix}_anchor_presence", layer, anchor, "", "anchor missing in one leg")
            )
            continue
        row_left, row_right = left[anchor], right[anchor]
        for path in sorted(set(row_left) | set(row_right)):
            value_left = row_left.get(path, "__ABSENT__")
            value_right = row_right.get(path, "__ABSENT__")
            if value_left != value_right:
                diffs.append(
                    Diff(
                        "A",
                        f"{kind_prefix}_cell",
                        layer,
                        anchor,
                        path,
                        f"{_short(value_left)} != {_short(value_right)}",
                    )
                )
    return diffs


def _short(value: Any) -> str:
    text = str(value)
    return text if len(text) <= 120 else text[:117] + "..."


def _bucket_sites(snapshot: Snapshot) -> dict[str, list[TokenSite]]:
    grouped: dict[str, list[TokenSite]] = defaultdict(list)
    for site in snapshot.token_sites:
        grouped[site.bucket].append(site)
    return grouped


def _site_key(site: TokenSite) -> tuple[str, str, str]:
    return (site.layer, site.anchor, site.path)


def check_a(left: Snapshot, right: Snapshot) -> list[Diff]:
    """Cross-leg structural comparison over all three layers."""

    diffs: list[Diff] = []
    diffs.extend(_cell_diffs("journal", left.journal, right.journal, "journal"))
    diffs.extend(_cell_diffs("store", left.store, right.store, "store"))
    diffs.extend(
        _cell_diffs("artifact", {"<artifact>": left.artifact}, {"<artifact>": right.artifact}, "artifact")
    )

    left_buckets = _bucket_sites(left)
    right_buckets = _bucket_sites(right)
    for bucket in sorted(set(left_buckets) | set(right_buckets)):
        sites_left = left_buckets.get(bucket, [])
        sites_right = right_buckets.get(bucket, [])
        keys_left = {_site_key(site) for site in sites_left}
        keys_right = {_site_key(site) for site in sites_right}
        for missing in sorted(keys_left ^ keys_right):
            diffs.append(
                Diff(
                    "A",
                    "token_site_presence",
                    missing[0],
                    missing[1],
                    missing[2],
                    f"bucket {bucket}: site present in only one leg",
                )
            )
        if bucket not in PARTITION_BUCKETS:
            # Buckets whose token-retention premise fails (backend_handle_id:
            # transient tensors are freed mid-run, so CPython id reuse merges
            # partition groups coincidentally — Opus v4 note N4) compare by
            # presence only here; their swap detection lives in Check B.
            continue
        shared = keys_left & keys_right
        partition_left = _partition({s for s in sites_left if _site_key(s) in shared})
        partition_right = _partition({s for s in sites_right if _site_key(s) in shared})
        if partition_left != partition_right:
            only_left = partition_left - partition_right
            only_right = partition_right - partition_left
            diffs.append(
                Diff(
                    "A",
                    "token_partition",
                    "*",
                    "*",
                    bucket,
                    f"alias partition differs: {sorted(map(sorted, only_left))!r} vs "
                    f"{sorted(map(sorted, only_right))!r}",
                )
            )
    return diffs


def _partition(sites: set[TokenSite]) -> frozenset[frozenset[tuple[str, str, str]]]:
    by_token: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    for site in sites:
        by_token[site.token].add(_site_key(site))
    return frozenset(frozenset(group) for group in by_token.values())


def check_b(run: Any) -> list[Diff]:
    """Within-leg referent attestation over the three layers (Opus note N2).

    ``run`` is a ``RunResult`` (snapshot + attestation log + live model).
    """

    diffs: list[Diff] = []
    snapshot: Snapshot = run.snapshot
    attestation = run.attestation

    # ---- grad_fn_object_id: stored token == id(selected grad_fn) at source.
    for site in snapshot.token_sites:
        if site.bucket != "grad_fn_object_id":
            continue
        expected = attestation.grad_fn_by_anchor.get(site.anchor)
        if expected is None:
            # Anchor never passed the semantic-selection boundary (source
            # events, synthetic outputs): presence-only for this bucket.
            continue
        if site.token != str(expected):
            diffs.append(
                Diff(
                    "B",
                    "grad_fn_binding",
                    site.layer,
                    site.anchor,
                    site.path,
                    f"stored {site.token} != attested {expected}",
                )
            )

    # ---- tl_barcode: per-anchor BINDING where the record row names the param
    # address (journal params[i]), membership provenance elsewhere (disclosed).
    expected_by_address = _live_param_barcodes(run.model)
    known_barcodes = (
        set(attestation.minted_barcodes)
        | set(attestation.barcode_by_param_id.values())
        | set(expected_by_address.values())
    )
    import re as _re

    for site in snapshot.token_sites:
        if site.bucket != "tl_barcode":
            continue
        bound = False
        match = _re.fullmatch(r"params\[(\d+)\]\.barcode", site.path)
        if match is not None and site.layer == "journal":
            row = snapshot.journal.get(site.anchor, {})
            params_row = row.get("params") or []
            index = int(match.group(1))
            if index < len(params_row) and isinstance(params_row[index], dict):
                address = params_row[index].get("address")
                expected = expected_by_address.get(address) if address else None
                if expected is not None:
                    bound = True
                    if site.token != expected:
                        diffs.append(
                            Diff(
                                "B",
                                "barcode_binding",
                                site.layer,
                                site.anchor,
                                site.path,
                                f"stored {site.token} != live param barcode {expected} "
                                f"for address {address!r}",
                            )
                        )
        if not bound and site.token not in known_barcodes:
            diffs.append(
                Diff(
                    "B",
                    "barcode_provenance",
                    site.layer,
                    site.anchor,
                    site.path,
                    "stored barcode neither minted this run nor attached to a live param",
                )
            )

    # ---- tl_barcode_derived (equivalence_class): recompute from the record's
    # own attested param barcodes; only parameterized ops carry the derivation.
    for site in snapshot.token_sites:
        if site.bucket != "tl_barcode_derived" or site.layer != "journal":
            continue
        row = snapshot.journal.get(site.anchor, {})
        params_row = row.get("params") or []
        barcodes: list[str] = []
        for entry in snapshot.token_sites:
            if (
                entry.bucket == "tl_barcode"
                and entry.layer == "journal"
                and entry.anchor == site.anchor
                and entry.path.startswith("params[")
            ):
                barcodes.append(entry.token)
        if not barcodes or not params_row:
            continue
        layer_type = row.get("layer_type")
        expected_prefix = f"{layer_type}_{'_'.join(sorted(barcodes))}"
        if not site.token.startswith(expected_prefix):
            diffs.append(
                Diff(
                    "B",
                    "equivalence_class_derivation",
                    site.layer,
                    site.anchor,
                    site.path,
                    f"stored {site.token!r} does not derive from attested barcodes "
                    f"(expected prefix {expected_prefix!r})",
                )
            )

    # ---- backend_handle_id: stored token == id(live output tensor) recorded
    # by the shim at the commit boundary. Only the primary-output path is
    # attested; transformed-output and dedup-annotation sites lack a live root
    # and stay in the presence-only residual (disclosed, Opus note N1).
    for site in snapshot.token_sites:
        if site.bucket != "backend_handle_id":
            continue
        if site.layer != "journal" or not site.path.endswith("output.tensor.backend_handle_id"):
            continue
        expected_handle = attestation.handle_by_anchor.get(site.anchor)
        if expected_handle is None:
            snapshot.presence_only.append((site.layer, site.anchor, site.path))
            continue
        if site.token != str(expected_handle):
            diffs.append(
                Diff(
                    "B",
                    "handle_binding",
                    site.layer,
                    site.anchor,
                    site.path,
                    f"stored {site.token} != attested live-tensor id {expected_handle}",
                )
            )

    # ---- duplicate anchors would make attestation ambiguous: refuse loudly.
    for anchor in attestation.duplicate_anchors:
        diffs.append(
            Diff("B", "ambiguous_anchor", "journal", anchor, "", "anchor observed twice at source")
        )
    return diffs


def _live_param_barcodes(model: Any) -> dict[str, str]:
    """Read barcodes off the LIVE parameters (independent of any record)."""

    from torchlens.backends.torch._tl import get_param_meta

    result: dict[str, str] = {}
    if model is None:
        return result
    for name, param in model.named_parameters():
        meta = get_param_meta(param)
        barcode = getattr(meta, "param_barcode", None) if meta is not None else None
        if barcode is not None:
            result[name] = barcode
    return result


def compare_runs(left: Any, right: Any) -> list[Diff]:
    """Full two-check discharge for a pair of runs (A cross-leg, B per leg)."""

    diffs = check_a(left.snapshot, right.snapshot)
    diffs.extend(check_b(left))
    diffs.extend(check_b(right))
    return diffs
