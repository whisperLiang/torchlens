"""r58 A3e/A3g: hostile-input parity for the merged + distributed load surfaces.

Two gaps this closes.

A3e -- the distributed group-lifecycle ledger was NOT fail-closed at its parse
boundary. ``GroupLifecycleEvent.from_payload`` assigned its closed-vocabulary fields
(``kind`` / ``ordinal_source`` / ``install_epoch``) straight from the payload, and
``lineage_vectors()`` then SILENTLY DROPPED an event whose kind it did not recognize --
so a forged sidecar could erase a generation from the lineage evidence the pre-join
membership audit reads, or promote a ``seeded`` rank to the complete witness it never
was. ``GroupLifecycleLedger.from_payload`` also bypassed ``append()``, so duplicate or
decreasing event indices loaded happily.

A3g -- the RCE corpus (13 ``test_r*_rce.py`` files at the time) had ZERO coverage of the
merged-directory / ``load_merged`` / ledger surface the sprint had just added. The
artifact-boundary invariant is that a hostile artifact refuses TYPED and never crashes,
never executes, and never degrades a tamper into a presence gap.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from torchlens.distributed._ledger import (
    GroupLifecycleEvent,
    GroupLifecycleLedger,
    membership_digest_for_ranks,
)
from torchlens.merged._artifact import load_merged
from torchlens.merged._enums import (
    MERGED_BUNDLE_FORMAT,
    MERGED_DESCRIPTOR_KIND,
    MERGED_DESCRIPTOR_SCHEMA_VERSION,
    MERGED_TLSPEC_VERSION,
    MergedErrorCode,
)
from torchlens.merged._errors import MergedArtifactError, MergeInputError

pytestmark = pytest.mark.smoke

_DIGEST = membership_digest_for_ranks([0, 1])


def _event_payload(**overrides: Any) -> dict[str, Any]:
    """Return a valid group-lifecycle event payload with optional overrides."""

    payload = GroupLifecycleEvent(
        event_index=0,
        kind="create",
        membership_digest=_DIGEST,
        ordinal=0,
        ordinal_source="wrapped",
        install_epoch="armed_before_any_group",
        local_creation_index=0,
        group_name="group:0",
        name_scheme="pg.group_name",
    ).to_payload()
    payload.update(overrides)
    return payload


# --------------------------------------------------------------------------- #
# A3e: the ledger parse boundary is fail-closed                                #
# --------------------------------------------------------------------------- #


def test_valid_ledger_payload_round_trips() -> None:
    """The tightened parser still accepts what ``to_payload`` emits."""

    ledger = GroupLifecycleLedger()
    ledger.append(GroupLifecycleEvent.from_payload(_event_payload()))
    ledger.append(GroupLifecycleEvent.from_payload(_event_payload(event_index=1, kind="destroy")))
    rebuilt = GroupLifecycleLedger.from_payload(ledger.to_payload())
    assert rebuilt.events == ledger.events
    assert rebuilt.lineage_vectors()[_DIGEST].entries[0].destroyed is True


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"kind": "frobnicate"}, id="kind-outside-vocabulary"),
        pytest.param({"install_epoch": "armed_before_anything"}, id="epoch-outside-vocabulary"),
        pytest.param({"ordinal_source": "conjured"}, id="ordinal-source-outside-vocabulary"),
        pytest.param({"membership_digest": "not-a-digest"}, id="digest-not-sha256"),
        pytest.param({"membership_digest": 7}, id="digest-not-a-string"),
        pytest.param({"event_index": -1}, id="negative-event-index"),
        pytest.param({"event_index": "0"}, id="event-index-not-an-int"),
        pytest.param({"ordinal": True}, id="ordinal-is-a-bool"),
        pytest.param({"group_name": 3}, id="diagnostic-not-a-string"),
        pytest.param({"smuggled": "field"}, id="unknown-key"),
    ],
)
def test_forged_event_field_refuses(overrides: dict[str, Any]) -> None:
    """Every closed-vocabulary / typed field refuses at the parse boundary.

    Fail-before: an unknown ``kind`` parsed fine and was then silently omitted from
    ``lineage_vectors()``, i.e. the forged sidecar ERASED evidence rather than being
    rejected.
    """

    with pytest.raises((ValueError, TypeError)):
        GroupLifecycleEvent.from_payload(_event_payload(**overrides))


def test_missing_required_field_refuses() -> None:
    """A truncated payload refuses rather than defaulting a verdict-steering field."""

    payload = _event_payload()
    del payload["install_epoch"]
    with pytest.raises(ValueError, match="missing keys"):
        GroupLifecycleEvent.from_payload(payload)


def test_non_monotone_ledger_payload_refuses() -> None:
    """A loaded ledger obeys the SAME monotone contract as a live one.

    Fail-before: ``from_payload`` handed the list straight to ``__init__``, bypassing
    ``append()``, so duplicate or decreasing indices could reorder or mask evidence.
    """

    with pytest.raises(ValueError, match="strictly increasing"):
        GroupLifecycleLedger.from_payload([_event_payload(), _event_payload()])
    with pytest.raises(ValueError, match="strictly increasing"):
        GroupLifecycleLedger.from_payload([_event_payload(event_index=5), _event_payload()])


def test_non_list_ledger_payload_refuses() -> None:
    """A payload of the wrong shape refuses typed instead of iterating a mapping."""

    with pytest.raises(TypeError, match="must be a list"):
        GroupLifecycleLedger.from_payload({"event_index": 0})  # type: ignore[arg-type]


def test_lineage_derivation_never_silently_drops_a_kind() -> None:
    """The derivation raises on an unrecognized kind rather than omitting it.

    Constructed directly (bypassing the parse boundary) to prove the belt is inside the
    derivation too, so a future kind cannot be added and silently ignored.
    """

    rogue = GroupLifecycleEvent(
        event_index=0,
        kind="frobnicate",  # type: ignore[arg-type]
        membership_digest=_DIGEST,
        ordinal=0,
        ordinal_source="wrapped",
        install_epoch="seeded",
    )
    ledger = GroupLifecycleLedger([rogue])
    with pytest.raises(ValueError, match="outside the closed"):
        ledger.lineage_vectors()


def test_ledger_epoch_disagreement_refuses_at_evidence_extraction() -> None:
    """A ledger whose events disagree with the record epoch refuses typed.

    ``lineage_vectors()`` stamps each vector's epoch from the EVENTS while the audit
    also reads the record-level epoch, so a disagreement is a sidecar trying to promote
    a ``seeded`` rank to a complete witness.
    """

    from torchlens.merged._evidence import extract_rank_evidence

    boundary = {
        "schema": "collective_boundary_v1",
        "kind": "all_reduce",
        "correlation": {
            "membership_digest": _DIGEST,
            "lifetime_ordinal": 0,
            "channel": "default",
            "seq": 0,
        },
        "group": {"global_ranks": [0, 1], "my_global_rank": 0},
        "events": {"completion_binding": "issue_sync"},
        "witness": {"policy_resolved": "digest"},
        "op_labels_raw": ["all_reduce_1_1"],
    }

    class _FakeTrace:
        annotations = {
            "distributed": {
                "boundaries": [boundary],
                "install_epoch": "seeded",
                "group_lifecycle_ledger": [_event_payload(install_epoch="armed_before_any_group")],
            }
        }

    with pytest.raises(MergeInputError) as caught:
        extract_rank_evidence(_FakeTrace(), "forged")
    assert caught.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value


# --------------------------------------------------------------------------- #
# A3g: the merged-directory load surface refuses hostile artifacts typed        #
# --------------------------------------------------------------------------- #


def _merged_artifact(root: Path, descriptor: dict[str, Any], **manifest: Any) -> Path:
    """Write a syntactically well-formed merged-directory artifact."""

    root.mkdir(parents=True, exist_ok=True)
    (root / "merge").mkdir(exist_ok=True)
    descriptor_bytes = json.dumps(descriptor).encode("utf-8")
    (root / "merge" / "descriptor.json").write_bytes(descriptor_bytes)
    payload = {
        "bundle_format": MERGED_BUNDLE_FORMAT,
        "tlspec_version": MERGED_TLSPEC_VERSION,
        "descriptor_sha256": hashlib.sha256(descriptor_bytes).hexdigest(),
        "members": {"0": "a" * 64},
    }
    payload.update(manifest)
    (root / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    return root


def _descriptor(**overrides: Any) -> dict[str, Any]:
    """Return a well-formed merged descriptor with optional overrides."""

    payload: dict[str, Any] = {
        "descriptor_kind": MERGED_DESCRIPTOR_KIND,
        "schema_version": MERGED_DESCRIPTOR_SCHEMA_VERSION,
        "members": [{"rank": 0, "path": "rank0", "tree_sha256": "a" * 64}],
    }
    payload.update(overrides)
    return payload


def test_deeply_nested_merged_manifest_refuses_typed(tmp_path: Path) -> None:
    """A depth-5000 root manifest is a typed schema refusal, never a RecursionError."""

    root = tmp_path / "deep.merged"
    root.mkdir()
    (root / "manifest.json").write_text('{"d": ' + "[" * 5000 + "0" + "]" * 5000 + "}")
    with pytest.raises(MergedArtifactError) as caught:
        load_merged(root)
    assert caught.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value


def test_deeply_nested_merged_descriptor_refuses_typed(tmp_path: Path) -> None:
    """An over-nested descriptor refuses typed AFTER its checksum verifies."""

    root = tmp_path / "deepdesc.merged"
    root.mkdir()
    (root / "merge").mkdir()
    descriptor_bytes = ("[" * 5000 + "0" + "]" * 5000).encode("utf-8")
    (root / "merge" / "descriptor.json").write_bytes(descriptor_bytes)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "bundle_format": MERGED_BUNDLE_FORMAT,
                "tlspec_version": MERGED_TLSPEC_VERSION,
                "descriptor_sha256": hashlib.sha256(descriptor_bytes).hexdigest(),
                "members": {"0": "a" * 64},
            }
        )
    )
    with pytest.raises(MergedArtifactError) as caught:
        load_merged(root)
    assert caught.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value


def test_descriptor_checksum_tamper_is_a_typed_tamper_not_a_gap(tmp_path: Path) -> None:
    """Rewritten descriptor bytes refuse as TAMPER; tamper is never a presence gap."""

    root = _merged_artifact(tmp_path / "tampered.merged", _descriptor())
    (root / "merge" / "descriptor.json").write_bytes(b'{"descriptor_kind": "swapped"}')
    with pytest.raises(MergedArtifactError) as caught:
        load_merged(root)
    assert caught.value.fields["code"] == MergedErrorCode.MERGED_DESCRIPTOR_TAMPER.value


@pytest.mark.parametrize(
    "member_path",
    [
        pytest.param("../escape", id="parent-traversal"),
        pytest.param("/etc", id="absolute-path"),
        pytest.param(".", id="self-referential"),
    ],
)
def test_member_path_escape_refuses_typed(tmp_path: Path, member_path: str) -> None:
    """A descriptor member path outside the artifact root refuses typed."""

    root = _merged_artifact(
        tmp_path / f"escape_{abs(hash(member_path))}.merged",
        _descriptor(members=[{"rank": 0, "path": member_path, "tree_sha256": "a" * 64}]),
    )
    with pytest.raises(MergedArtifactError) as caught:
        load_merged(root)
    assert caught.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value


@pytest.mark.parametrize(
    "manifest_override",
    [
        pytest.param({"bundle_format": "trace-bundle"}, id="wrong-bundle-format"),
        pytest.param({"tlspec_version": 999}, id="unsupported-tlspec-version"),
        pytest.param({"members": []}, id="members-table-not-a-mapping"),
    ],
)
def test_merged_manifest_vocabulary_violations_refuse_typed(
    tmp_path: Path, manifest_override: dict[str, Any]
) -> None:
    """Closed-vocabulary violations in the root manifest refuse typed."""

    root = _merged_artifact(
        tmp_path / f"vocab_{next(iter(manifest_override))}.merged",
        _descriptor(),
        **manifest_override,
    )
    with pytest.raises(MergedArtifactError) as caught:
        load_merged(root)
    assert caught.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value


def test_missing_merged_members_refuse_typed(tmp_path: Path) -> None:
    """A descriptor naming a rank core that is absent refuses as tamper."""

    root = _merged_artifact(tmp_path / "absent.merged", _descriptor())
    with pytest.raises(MergedArtifactError) as caught:
        load_merged(root)
    assert caught.value.fields["code"] == MergedErrorCode.MERGED_DESCRIPTOR_TAMPER.value
