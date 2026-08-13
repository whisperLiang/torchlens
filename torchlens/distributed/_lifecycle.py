"""Arming, group-lifecycle wraps, restricted seeding, and seq counters.

``torchlens.distributed.arm()`` is the explicit process-start opt-in for
distributed capture. Arming:

1. derives and verifies the five-namespace collective recognizer (fail-closed
   arm-time census, :mod:`._recognizer`);
2. installs the group-lifecycle wraps IMMEDIATELY (no first-capture laziness):
   ``init_process_group`` / ``new_group`` / ``split_group`` assign lifetime
   ordinals at creation; ``destroy_process_group`` (and ``_abort_process_group``
   where present) are wrapped observationally;
3. stamps the install-epoch record -- ``armed_before_any_group`` when no
   process group has ever been observed in this process, else ``seeded``.

Group identity is ``group_uid = (membership_digest, lifetime_ordinal)``
(exactly two fields). Ordinals are TorchLens-owned, monotone, ever-created:
unobserved DESTRUCTION is harmless (a dead ordinal is simply retired), and
every rule here exists to make unobserved CREATION either impossible or
fail-closed. Registry seeding of groups that predate arming is RESTRICTED to
the provably unambiguous case: only ordinal 0, only when exactly one
same-membership group is alive and no churn of that membership was ever
observed; anything else refuses typed (``ambiguous_group_lifetime``).

SPMD programs that first-capture symmetrically may be armed lazily by capture
entry (:func:`maybe_auto_arm`); ``arm()`` before any group creation is the
REQUIRED spelling for MPMD programs and the only way a rank can be a COMPLETE
WITNESS in the merge-time pre-join lineage audit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
from typing import Any
import warnings

import torch

from ..errors._base import CompatibilityError
from ._ledger import (
    GroupLifecycleEvent,
    GroupLifecycleLedger,
    InstallEpoch,
    membership_digest_for_ranks,
)
from ._recognizer import (
    CollectiveRecognizer,
    UncapturedCollectiveOpError,
    derive_collective_recognizer,
)

__all__ = [
    "AMBIGUOUS_GROUP_LIFETIME",
    "AmbiguousGroupLifetimeError",
    "ArmingRecord",
    "GroupIdentity",
    "arm",
    "armed_state",
    "disarm",
    "is_armed",
    "maybe_auto_arm",
]

AMBIGUOUS_GROUP_LIFETIME = "ambiguous_group_lifetime"
"""Finding kind for refused local seeding of a group with ambiguous lifetime."""


class AmbiguousGroupLifetimeError(CompatibilityError, RuntimeError):
    """Raised when a pre-arming group's lifetime ordinal cannot be proven.

    Structured context on ``fields``: ``kind`` (always
    ``"ambiguous_group_lifetime"``), ``membership_digest``, and ``reason``.
    """


@dataclass(frozen=True)
class GroupIdentity:
    """Resolved lifetime identity of one process group on this rank."""

    membership_digest: str
    lifetime_ordinal: int
    ordinal_source: str
    global_ranks: tuple[int, ...]
    backend: str | None

    @property
    def group_uid(self) -> tuple[str, int]:
        """The exactly-two-field correlation group identity."""

        return (self.membership_digest, self.lifetime_ordinal)


@dataclass(frozen=True)
class ArmingRecord:
    """What ``arm()`` established, stamped on every boundary record."""

    install_epoch: InstallEpoch
    recognizer_snapshot: str
    source: str


@dataclass
class _ArmedState:
    """Process-global armed distributed-capture state."""

    arming: ArmingRecord
    recognizer: CollectiveRecognizer
    ledger: GroupLifecycleLedger = field(default_factory=GroupLifecycleLedger)
    identities: dict[int, GroupIdentity] = field(default_factory=dict)
    seq_counters: dict[tuple[str, int, str], int] = field(default_factory=dict)
    originals: dict[tuple[Any, str], Any] = field(default_factory=dict)


_LOCK = threading.Lock()
_STATE: _ArmedState | None = None
_AUTO_ARM_WARNED = False


def is_armed() -> bool:
    """Whether the distributed opt-in is currently armed in this process."""

    return _STATE is not None


def armed_state() -> _ArmedState | None:
    """Return the live armed state, or ``None`` when unarmed."""

    return _STATE


def _dist() -> Any:
    """Return ``torch.distributed``, refusing typed when unavailable."""

    if not torch.distributed.is_available():
        raise CompatibilityError(
            "torchlens.distributed.arm() requires a torch build with "
            "distributed support (torch.distributed.is_available() is False).",
            kind="distributed_unavailable",
        )
    return torch.distributed


def _any_group_history() -> bool:
    """Whether any process group has been observed alive in this process."""

    dist = torch.distributed
    if not dist.is_available():
        return False
    try:
        if dist.is_initialized():
            return True
    except Exception:
        return False
    world = getattr(getattr(dist, "distributed_c10d", None), "_world", None)
    pg_map = getattr(world, "pg_map", None)
    return bool(pg_map)


def _group_global_ranks(group: Any) -> tuple[int, ...]:
    """Return a process group's member ranks in global numbering."""

    dist = torch.distributed
    return tuple(int(rank) for rank in dist.get_process_group_ranks(group))


def _group_backend_name(group: Any) -> str | None:
    """Best-effort backend name for a group; diagnostics only."""

    try:
        return str(torch.distributed.get_backend(group))
    except Exception:
        return None


def _group_display_name(group: Any) -> tuple[str | None, str | None]:
    """Return (group_name, name_scheme) diagnostics; never identity."""

    try:
        name = getattr(group, "group_name", None)
        if name is not None:
            return str(name), "pg.group_name"
    except Exception:
        pass
    return None, None


def _record_created_group(state: _ArmedState, group: Any) -> None:
    """Assign a wrapped-creation ordinal to a newly created group."""

    if group is None or not _is_member_group(group):
        return
    try:
        global_ranks = _group_global_ranks(group)
    except Exception:
        return
    digest = membership_digest_for_ranks(global_ranks)
    ordinal = state.ledger.next_ordinal(digest)
    name, scheme = _group_display_name(group)
    state.ledger.append(
        GroupLifecycleEvent(
            event_index=state.ledger.next_event_index(),
            kind="create",
            membership_digest=digest,
            ordinal=ordinal,
            ordinal_source="wrapped",
            install_epoch=state.arming.install_epoch,
            local_creation_index=state.ledger.creation_count(),
            group_name=name,
            name_scheme=scheme,
        )
    )
    state.identities[id(group)] = GroupIdentity(
        membership_digest=digest,
        lifetime_ordinal=ordinal,
        ordinal_source="wrapped",
        global_ranks=global_ranks,
        backend=_group_backend_name(group),
    )


def _is_member_group(group: Any) -> bool:
    """Whether ``group`` is a real ProcessGroup this rank belongs to."""

    dist = torch.distributed
    non_member = getattr(getattr(dist, "GroupMember", None), "NON_GROUP_MEMBER", object())
    return group is not None and group is not non_member


def _record_destroyed_group(state: _ArmedState, group: Any) -> None:
    """Observationally mark a group (or, for None, all groups) destroyed."""

    dist = torch.distributed
    targets: list[GroupIdentity] = []
    if group is None:
        # destroy_process_group(None) tears down the world and every subgroup.
        targets = list(state.identities.values())
        state.identities.clear()
    else:
        identity = state.identities.pop(id(group), None)
        if identity is None and _is_member_group(group):
            # A destroy of a group we never identified: try to at least record
            # the membership churn so restricted seeding refuses it later.
            try:
                ranks = _group_global_ranks(group)
            except Exception:
                return
            identity = GroupIdentity(
                membership_digest=membership_digest_for_ranks(ranks),
                lifetime_ordinal=state.ledger.next_ordinal(
                    membership_digest_for_ranks(ranks)
                ),
                ordinal_source="wrapped",
                global_ranks=ranks,
                backend=None,
            )
        if identity is not None:
            targets = [identity]
    for identity in targets:
        state.ledger.append(
            GroupLifecycleEvent(
                event_index=state.ledger.next_event_index(),
                kind="destroy",
                membership_digest=identity.membership_digest,
                ordinal=identity.lifetime_ordinal,
                ordinal_source=identity.ordinal_source,  # type: ignore[arg-type]
                install_epoch=state.arming.install_epoch,
            )
        )
    if group is None:
        return
    _ = dist  # narrow: dist retained for parity with the None branch above


_CREATE_WRAP_TARGETS = ("init_process_group", "new_group", "split_group")
_DESTROY_WRAP_TARGETS = ("destroy_process_group", "_abort_process_group")


def _patch_modules() -> list[Any]:
    """Modules whose lifecycle-function attributes are patched at arm time."""

    dist = torch.distributed
    modules = [dist]
    c10d = getattr(dist, "distributed_c10d", None)
    if c10d is not None:
        modules.append(c10d)
    device_mesh_mod = getattr(dist, "device_mesh", None)
    if device_mesh_mod is not None:
        modules.append(device_mesh_mod)
    return modules


def _install_lifecycle_wraps(state: _ArmedState) -> None:
    """Install creation/destroy wraps on every module holding a reference."""

    def make_create_wrap(original: Any, returns_group: bool) -> Any:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            group = result if returns_group else None
            if not returns_group:
                # init_process_group returns None; the world group is the
                # default group after it completes.
                try:
                    group = torch.distributed.group.WORLD
                except Exception:
                    group = None
            with _LOCK:
                if _STATE is state and id(group) not in state.identities:
                    _record_created_group(state, group)
            return result

        wrapped.__wrapped__ = original  # type: ignore[attr-defined]
        wrapped.__name__ = getattr(original, "__name__", "wrapped")
        return wrapped

    def make_destroy_wrap(original: Any) -> Any:
        def wrapped(group: Any = None, *args: Any, **kwargs: Any) -> Any:
            with _LOCK:
                if _STATE is state:
                    resolved = group
                    if resolved is not None and not _is_member_group(resolved):
                        resolved = None
                    _record_destroyed_group(state, resolved)
            return original(group, *args, **kwargs)

        wrapped.__wrapped__ = original  # type: ignore[attr-defined]
        wrapped.__name__ = getattr(original, "__name__", "wrapped")
        return wrapped

    for module in _patch_modules():
        for name in _CREATE_WRAP_TARGETS:
            original = getattr(module, name, None)
            if original is None or (module, name) in state.originals:
                continue
            state.originals[(module, name)] = original
            returns_group = name != "init_process_group"
            setattr(module, name, make_create_wrap(original, returns_group))
        for name in _DESTROY_WRAP_TARGETS:
            original = getattr(module, name, None)
            if original is None or (module, name) in state.originals:
                continue
            state.originals[(module, name)] = original
            setattr(module, name, make_destroy_wrap(original))


def arm() -> ArmingRecord:
    """Arm distributed capture for this process.

    Returns
    -------
    ArmingRecord
        The install-epoch record and recognizer snapshot established. Calling
        ``arm()`` again is idempotent and returns the original record.

    Raises
    ------
    UncapturedCollectiveOpError
        When the arm-time recognizer derivation refuses this torch runtime.
    CompatibilityError
        When torch has no distributed support.
    """

    return _arm(source="explicit")


def _arm(source: str) -> ArmingRecord:
    global _STATE
    with _LOCK:
        if _STATE is not None:
            return _STATE.arming
        _dist()
        recognizer = derive_collective_recognizer()
        epoch: InstallEpoch = (
            "seeded" if _any_group_history() else "armed_before_any_group"
        )
        arming = ArmingRecord(
            install_epoch=epoch,
            recognizer_snapshot=recognizer.snapshot_name,
            source=source,
        )
        state = _ArmedState(arming=arming, recognizer=recognizer)
        _install_lifecycle_wraps(state)
        _STATE = state
        return arming


def maybe_auto_arm() -> ArmingRecord | None:
    """Lazily arm at capture entry when torch.distributed is initialized.

    Returns
    -------
    ArmingRecord | None
        The arming record when armed (newly or already), ``None`` when
        distributed is not in play or lazy arming was refused.

    Notes
    -----
    Explicit ``arm()`` raises on recognizer refusal; the lazy path degrades to
    unarmed capture with a one-time warning instead, because refusing every
    capture in a process that merely initialized a process group would break
    previously-working dense captures that issue no collectives at all.
    Unarmed capture records no collective boundaries -- the pre-tier-(b)
    status quo -- and the warning names the typed finding.
    """

    global _AUTO_ARM_WARNED
    if _STATE is not None:
        return _STATE.arming
    try:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return None
    except Exception:
        return None
    try:
        return _arm(source="auto")
    except UncapturedCollectiveOpError as error:
        if not _AUTO_ARM_WARNED:
            _AUTO_ARM_WARNED = True
            warnings.warn(
                "torchlens could not arm distributed collective capture on this "
                "torch runtime (uncaptured_collective_op); collective boundary "
                f"nodes will NOT be recorded. {error}",
                stacklevel=3,
            )
        return None


def disarm() -> None:
    """Remove the lifecycle wraps and drop all armed state.

    Primarily for tests; ordinary programs stay armed for process lifetime.
    """

    global _STATE, _AUTO_ARM_WARNED
    with _LOCK:
        state = _STATE
        if state is None:
            return
        for (module, name), original in state.originals.items():
            try:
                setattr(module, name, original)
            except Exception:
                pass
        _STATE = None
        _AUTO_ARM_WARNED = False


def resolve_group_identity(group: Any) -> GroupIdentity:
    """Resolve a process group's lifetime identity, seeding if provably safe.

    Parameters
    ----------
    group:
        A live ProcessGroup (or ``None`` for the default/world group).

    Returns
    -------
    GroupIdentity
        The two-field ``group_uid`` identity plus diagnostics.

    Raises
    ------
    AmbiguousGroupLifetimeError
        When the group predates arming and restricted seeding cannot prove
        its lifetime ordinal (two or more same-membership groups alive, or
        observed churn of that membership).
    RuntimeError
        When called while unarmed.
    """

    state = _STATE
    if state is None:
        raise RuntimeError(
            "resolve_group_identity() requires torchlens.distributed to be armed"
        )
    dist = torch.distributed
    if group is None:
        group = dist.group.WORLD
    with _LOCK:
        identity = state.identities.get(id(group))
        if identity is not None:
            return identity
        return _seed_group_locked(state, group)


def _alive_same_membership_count(digest: str) -> int:
    """Count live registry groups whose membership digest equals ``digest``."""

    dist = torch.distributed
    world = getattr(getattr(dist, "distributed_c10d", None), "_world", None)
    pg_map = getattr(world, "pg_map", None)
    if not pg_map:
        return 0
    count = 0
    for candidate in list(pg_map):
        try:
            ranks = _group_global_ranks(candidate)
        except Exception:
            continue
        if membership_digest_for_ranks(ranks) == digest:
            count += 1
    return count


def _seed_group_locked(state: _ArmedState, group: Any) -> GroupIdentity:
    """Restricted registry seeding: only ordinal 0, only provably unambiguous."""

    global_ranks = _group_global_ranks(group)
    digest = membership_digest_for_ranks(global_ranks)

    def refuse(reason: str) -> AmbiguousGroupLifetimeError:
        return AmbiguousGroupLifetimeError(
            "torchlens cannot assign a provable lifetime ordinal to a process "
            f"group created before arming: {reason}. Call "
            "torchlens.distributed.arm() at process start, before any process "
            "group is created.",
            kind=AMBIGUOUS_GROUP_LIFETIME,
            membership_digest=digest,
            reason=reason,
        )

    churn = [
        event
        for event in state.ledger.events
        if event.membership_digest == digest
    ]
    if churn:
        raise refuse(
            "lifecycle churn of this membership was already observed "
            f"({len(churn)} ledger event(s)), so an unwrapped group of the same "
            "membership cannot be generation 0"
        )
    alive = _alive_same_membership_count(digest)
    if alive > 1:
        raise refuse(
            f"{alive} live groups share this membership; the seed cannot prove "
            "which generation this group is"
        )
    name, scheme = _group_display_name(group)
    state.ledger.append(
        GroupLifecycleEvent(
            event_index=state.ledger.next_event_index(),
            kind="seed",
            membership_digest=digest,
            ordinal=0,
            ordinal_source="seeded",
            install_epoch=state.arming.install_epoch,
            local_creation_index=state.ledger.creation_count(),
            group_name=name,
            name_scheme=scheme,
        )
    )
    identity = GroupIdentity(
        membership_digest=digest,
        lifetime_ordinal=0,
        ordinal_source="seeded",
        global_ranks=global_ranks,
        backend=_group_backend_name(group),
    )
    state.identities[id(group)] = identity
    return identity


def next_seq(identity: GroupIdentity, channel: str) -> int:
    """Tick and return the issue-time seq counter for ``(group_uid, channel)``.

    Seq counters key on the FULL two-field ``group_uid``: a recreated
    communicator is a new uid and its counters start fresh at 0.
    """

    state = _STATE
    if state is None:
        raise RuntimeError("next_seq() requires torchlens.distributed to be armed")
    key = (identity.membership_digest, identity.lifetime_ordinal, channel)
    with _LOCK:
        value = state.seq_counters.get(key, 0)
        state.seq_counters[key] = value + 1
        return value
