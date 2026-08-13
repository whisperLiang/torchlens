"""``merged-directory`` artifact: canonical descriptor, tree hashes, load rederivation.

Layout (design-merge-ranks-c v5, 4.1)::

    <root>/
      manifest.json            # root integrity record (merged-directory)
      merge/descriptor.json    # canonical-JSON derivation CACHE
      members/rank_NNNN.tlspec # ordinary rank-core bundles, byte-identical
                               # to standalone saves (P1)

The descriptor is a CACHE; the rank cores are the authority (4.3). Every load
verifies integrity (descriptor bytes vs root-manifest checksum, member tree
hashes), then INDEPENDENTLY rederives the merge from the rank cores' boundary
records and requires EXACT equality with the cache -- any inequality is a
typed tamper refusal, never a gap. Presence expectations derive from the group
memberships recorded INSIDE the surviving rank cores, never from the editable
descriptor input list (which can only widen them).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from .._io import _json
from ._engine import derive_merge
from ._enums import (
    MERGED_BUNDLE_FORMAT,
    MERGED_DESCRIPTOR_KIND,
    MERGED_DESCRIPTOR_SCHEMA_VERSION,
    MERGED_TLSPEC_VERSION,
    MergedErrorCode,
)
from ._errors import MergedArtifactError
from ._evidence import extract_rank_evidence
from ._presenter import MergedTrace, _RankHandle

__all__ = ["canonical_json_bytes", "load_merged", "save_merged", "tree_hash"]

_TREE_HASH_CHUNK_BYTES = 1 << 20

CANONICAL_ENCODING = "torchlens-canonical-json-v1"
"""UTF-8, sorted keys, no NaN/Infinity, LF, no insignificant whitespace."""


def canonical_json_bytes(obj: Any) -> bytes:
    """Serialize ``obj`` under the canonical encoding declared above."""

    text = json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return (text + "\n").encode("utf-8")


def tree_hash(root: Path) -> str:
    """Canonical tree hash of a rank-core directory (4.1).

    Sorted POSIX-relative paths; per entry ``path \\0 size \\0 sha256(bytes)``;
    the tree hash is the SHA-256 of the concatenated entries. Symlinks are
    REJECTED at hash time, matching the loader guards.
    """

    entries: list[bytes] = []
    for candidate in sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix()):
        if candidate.is_symlink():
            raise MergedArtifactError(
                f"Symlink {candidate} inside a rank core; merged artifacts "
                "reject symlinks at hash time.",
                code=MergedErrorCode.MERGED_SCHEMA_INVALID,
            )
        if not candidate.is_file():
            continue
        # Chunked: a rank core's safetensors blobs are legitimately multi-GiB, so
        # ``read_bytes()`` materialized the whole file just to hash it.
        size = 0
        digest = hashlib.sha256()
        with candidate.open("rb") as handle:
            while True:
                chunk = handle.read(_TREE_HASH_CHUNK_BYTES)
                if not chunk:
                    break
                size += len(chunk)
                digest.update(chunk)
        entry = (
            candidate.relative_to(root).as_posix().encode("utf-8")
            + b"\0"
            + str(size).encode("ascii")
            + b"\0"
            + digest.hexdigest().encode("ascii")
        )
        entries.append(entry)
    return hashlib.sha256(b"\n".join(entries)).hexdigest()


def _member_dirname(rank: int) -> str:
    """Canonical per-member directory name for one rank."""

    return f"rank_{rank:04d}.tlspec"


def _tamper(detail: str, **payload: Any) -> MergedArtifactError:
    """Build the typed tamper refusal (integrity failure, never a presence gap)."""

    return MergedArtifactError(
        f"Merged artifact integrity failure: {detail}",
        code=MergedErrorCode.MERGED_DESCRIPTOR_TAMPER,
        **payload,
    )


def _schema_refusal(detail: str, **payload: Any) -> MergedArtifactError:
    """Build the typed merged-artifact schema refusal."""

    return MergedArtifactError(
        f"Merged artifact schema refusal: {detail}",
        code=MergedErrorCode.MERGED_SCHEMA_INVALID,
        **payload,
    )


def save_merged(merged: MergedTrace, path: str | Path, *, overwrite: bool = False) -> None:
    """Write a ``merged-directory`` artifact.

    Parameters
    ----------
    merged:
        The merged trace to persist.
    path:
        Output directory path.
    overwrite:
        Whether an existing directory may be replaced.

    Raises
    ------
    MergedArtifactError
        If the target exists (without ``overwrite``) or a member cannot be
        persisted.
    """

    root = Path(path)
    if root.exists() and not overwrite:
        raise MergedArtifactError(
            f"{root} already exists; pass overwrite=True to replace it.",
            code=MergedErrorCode.MERGE_INPUT_INVALID,
        )
    staging_root = root.parent / f"{root.name}.tmp.{uuid.uuid4().hex}"
    backup_root: Path | None = None
    try:
        members_dir = staging_root / "members"
        members_dir.mkdir(parents=True)
        (staging_root / "merge").mkdir()

        members_payload: list[dict[str, Any]] = []
        for rank in merged.rank_ids:
            handle = merged._handles[rank]
            member_path = members_dir / _member_dirname(rank)
            if handle.path is not None:
                source = Path(handle.path)
                if source.is_dir():
                    # Copy is the only link mode: member bytes stay byte-identical
                    # to the standalone rank-core save (P1).
                    shutil.copytree(source, member_path)
                else:
                    raise MergedArtifactError(
                        f"Rank {rank} core path {source} is not a bundle directory.",
                        code=MergedErrorCode.MERGE_INPUT_INVALID,
                    )
            else:
                from .._io.bundle import save as save_bundle

                save_bundle(handle.trace, member_path)
            members_payload.append(
                {
                    "rank": rank,
                    "path": f"members/{_member_dirname(rank)}",
                    "tree_sha256": tree_hash(member_path),
                }
            )

        descriptor = {
            "descriptor_kind": MERGED_DESCRIPTOR_KIND,
            "schema_version": MERGED_DESCRIPTOR_SCHEMA_VERSION,
            "encoding": CANONICAL_ENCODING,
            "members": members_payload,
            "derivation": merged._derivation.to_payload(),
        }
        descriptor_bytes = canonical_json_bytes(descriptor)
        (staging_root / "merge" / "descriptor.json").write_bytes(descriptor_bytes)

        import platform as platform_module
        from datetime import datetime, timezone

        import torch

        from .. import __version__ as torchlens_version

        manifest = {
            "tlspec_version": MERGED_TLSPEC_VERSION,
            "bundle_format": MERGED_BUNDLE_FORMAT,
            "descriptor_sha256": hashlib.sha256(descriptor_bytes).hexdigest(),
            "members": {str(entry["rank"]): entry["tree_sha256"] for entry in members_payload},
            "torchlens_version": str(torchlens_version),
            "torch_version": str(torch.__version__),
            "python_version": platform_module.python_version(),
            "platform": platform_module.platform(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (staging_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        if root.exists():
            backup_root = root.parent / f"{root.name}.bak.{uuid.uuid4().hex}"
            root.rename(backup_root)
        staging_root.rename(root)
        if backup_root is not None:
            shutil.rmtree(backup_root, ignore_errors=True)
    except BaseException:
        if staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)
        if backup_root is not None and not root.exists() and backup_root.exists():
            try:
                backup_root.rename(root)
            except OSError:
                pass
        raise


def _resolve_member_path(root: Path, relative: str) -> Path:
    """Resolve a descriptor member path under the root, rejecting escapes."""

    candidate_path = Path(relative)
    if candidate_path.is_absolute() or ".." in candidate_path.parts:
        raise _schema_refusal(f"member path {relative!r} escapes the artifact root")
    resolved = (root / candidate_path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise _schema_refusal(f"member path {relative!r} escapes the artifact root") from exc
    if resolved == root.resolve():
        raise _schema_refusal(f"member path {relative!r} is self-referential")
    return resolved


def load_merged(path: str | Path) -> MergedTrace:
    """Load and REDERIVE a ``merged-directory`` artifact (4.3).

    Parameters
    ----------
    path:
        Artifact root directory.

    Returns
    -------
    MergedTrace
        The merged presenter, with ``load_degradations`` recording any rank
        core that no longer parses on this runtime (capping the effective
        alignment at ``partial``).

    Raises
    ------
    MergedArtifactError
        Typed schema refusal (closed vocabularies, unknown versions) or
        typed tamper refusal (checksum mismatch, tree-hash mismatch, or any
        inequality between the descriptor cache and the independent
        rederivation from the rank cores). Tamper is never a gap.
    """

    root = Path(path)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise _schema_refusal(f"{root} has no manifest.json")
    try:
        # Bounded PATH read: ``read_text`` allocated the whole attacker-sized file
        # before the ceiling applied, so the byte limit was advisory only here.
        manifest = _json.read_bounded(manifest_path)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _schema_refusal(f"root manifest does not parse ({exc})") from exc
    if not isinstance(manifest, dict):
        raise _schema_refusal("root manifest is not a JSON object")
    if manifest.get("bundle_format") != MERGED_BUNDLE_FORMAT:
        raise _schema_refusal(
            f"bundle_format {manifest.get('bundle_format')!r} is not {MERGED_BUNDLE_FORMAT!r}"
        )
    if manifest.get("tlspec_version") != MERGED_TLSPEC_VERSION:
        raise _schema_refusal(
            f"merged root tlspec_version {manifest.get('tlspec_version')!r} is "
            f"not supported by this runtime (expected {MERGED_TLSPEC_VERSION})"
        )

    descriptor_path = root / "merge" / "descriptor.json"
    if not descriptor_path.is_file():
        raise _tamper("merge/descriptor.json is missing")
    # The descriptor's EXACT on-disk bytes are the checksum subject, so they must be
    # read raw -- but under the same ceiling, not via an unbounded ``read_bytes``.
    try:
        descriptor_bytes = _json.read_bytes_bounded(descriptor_path)
    except json.JSONDecodeError as exc:
        raise _schema_refusal(f"descriptor does not parse ({exc})") from exc
    recorded_sha = manifest.get("descriptor_sha256")
    if hashlib.sha256(descriptor_bytes).hexdigest() != recorded_sha:
        raise _tamper("descriptor bytes do not match the root-manifest checksum")
    try:
        descriptor = _json.loads_bounded(descriptor_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _schema_refusal(f"descriptor does not parse ({exc})") from exc
    if not isinstance(descriptor, dict):
        raise _schema_refusal("descriptor is not a JSON object")
    if descriptor.get("descriptor_kind") != MERGED_DESCRIPTOR_KIND:
        raise _schema_refusal(
            f"descriptor_kind {descriptor.get('descriptor_kind')!r} is not "
            f"{MERGED_DESCRIPTOR_KIND!r}"
        )
    if descriptor.get("schema_version") != MERGED_DESCRIPTOR_SCHEMA_VERSION:
        raise _schema_refusal(
            f"descriptor schema_version {descriptor.get('schema_version')!r} is "
            f"not supported (expected {MERGED_DESCRIPTOR_SCHEMA_VERSION})"
        )
    members = descriptor.get("members")
    if not isinstance(members, list) or not members:
        raise _schema_refusal("descriptor members table is absent or empty")

    manifest_members = manifest.get("members")
    if not isinstance(manifest_members, dict):
        raise _schema_refusal("root manifest members table is absent")

    # 1. Integrity: every member's canonical tree hash must match BOTH records.
    member_paths: dict[int, Path] = {}
    for entry in members:
        rank = int(entry["rank"])
        member_path = _resolve_member_path(root, str(entry["path"]))
        if not member_path.is_dir():
            raise _tamper(f"rank {rank} core {entry['path']!r} is missing")
        recorded = str(entry["tree_sha256"])
        manifest_recorded = manifest_members.get(str(rank))
        if manifest_recorded != recorded:
            raise _tamper(f"rank {rank} tree hash disagrees between descriptor and manifest")
        actual = tree_hash(member_path)
        if actual != recorded:
            raise _tamper(f"rank {rank} core bytes do not match the recorded tree hash")
        member_paths[rank] = member_path

    # 2. Independent rederivation from the parsable rank cores.
    from .._io.bundle import load as load_bundle

    evidence = {}
    handles: dict[int, _RankHandle] = {}
    load_degradations: list[str] = []
    for rank, member_path in sorted(member_paths.items()):
        try:
            trace = load_bundle(member_path)
            evidence[rank] = extract_rank_evidence(trace, str(member_path))
        except Exception as exc:
            load_degradations.append(f"rank {rank} core no longer parses on this runtime: {exc}")
            continue
        handles[rank] = _RankHandle(rank, trace=trace, path=str(member_path))

    if not evidence:
        raise _schema_refusal(
            "no rank core of this artifact parses on this runtime; nothing can "
            "be rederived. Load degradations: " + "; ".join(load_degradations)
        )

    cached = descriptor.get("derivation")
    if not isinstance(cached, dict):
        raise _schema_refusal("descriptor derivation cache is absent")
    expected = cached.get("expected_ranks")
    rederived = derive_merge(evidence, expected)

    if not load_degradations:
        # 3. EXACT equality with the cache is required (any inequality = tamper).
        if canonical_json_bytes(rederived.to_payload()) != canonical_json_bytes(cached):
            raise _tamper(
                "the merge rederived from the rank cores does not equal the "
                "descriptor cache; the descriptor was edited or the cores were "
                "coherently rewritten"
            )
        derivation = rederived
    else:
        # Degraded environment: rederivation covers the parsable subset; the
        # STORED verdicts stay visible while the effective alignment is capped
        # at partial by the degradation ledger (3.2).
        from ._enums import MergeAlignment, MergeValueStatus

        derivation = replace(
            rederived,
            stored_alignment=MergeAlignment(cached["stored_alignment"]),
            stored_value_status=MergeValueStatus(cached["stored_value_status"]),
        )

    merged = MergedTrace(derivation, handles, tuple(load_degradations))
    merged._source_path = str(root)
    return merged
