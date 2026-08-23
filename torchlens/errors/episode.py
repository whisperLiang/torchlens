"""Episode-capture and bundle-relation exception families (L2 core).

Stable refusal codes ride ``exc.fields["code"]``; public code branches on the
code value or the structured report, never on exception text. The exhaustive
episode code vocabulary is maintained in
``docs/reference/error_refusal_contract.md``.

Every spelling is DOCUMENTED-UNSTABLE pending the rolling naming session;
the codes themselves are the stable branch surface.
"""

from __future__ import annotations

from enum import Enum

from ._base import ConfigurationError, TorchLensError, ValidationError

__all__ = [
    "BundleRelationError",
    "EpisodeCaptureError",
    "EpisodeDeclarationError",
    "EpisodeErrorCode",
    "EpisodeLedgerError",
]


class EpisodeErrorCode(str, Enum):
    """Frozen episode/bundle-relation refusal codes (S6/S7 contracts)."""

    #: Ledger geometry violates the monotone prefix law, a coherence arm, or
    #: the value-mode token presence rule; the outcome derivation degrades
    #: fail-closed to UNKNOWN.
    EPISODE_LEDGER_INCOHERENT = "episode_ledger_incoherent"

    #: Declared episode-carried state without snapshot/restore support inside
    #: the declared checkpoint scope (E-A4/E-B4); refuses at DECLARATION time,
    #: before execution.
    EPISODE_STATE_UNSNAPSHOTABLE = "episode_state_unsnapshotable"

    #: An episode ledger is present on a capture that carries no episode
    #: declaration (S2 combination table: ILLEGAL, refuse at load).
    EPISODE_LEDGER_WITHOUT_DECLARATION = "episode_ledger_without_declaration"

    #: A structure-only episode ledger carries token payloads (S7 presence
    #: rule, structure-only direction).
    EPISODE_LEDGER_PAYLOAD_IN_STRUCTURE_ONLY = "episode_ledger_payload_in_structure_only"

    #: The episode declaration itself is unusable (stepped module not a
    #: submodule of the episode root, token axis unreadable, forced-token
    #: declaration malformed, or a structure-only episode was declared —
    #: TYPED REFUSE per the S2 combination table this sprint).
    EPISODE_DECLARATION_INVALID = "episode_declaration_invalid"

    #: A bundle relation row names a member absent from the Bundle (S6 R1:
    #: no dangling edges, ever).
    BUNDLE_RELATION_MEMBER_MISSING = "bundle_relation_member_missing"

    #: A bundle mutator would orphan relation rows (S6 R5: cascade explicitly
    #: or refuse typed; silent orphaning is forbidden).
    BUNDLE_MEMBER_HAS_RELATIONS = "bundle_member_has_relations"

    #: A relation row is outside the closed S6 schema (unknown kind, wrong
    #: row shape for the kind, or undeclared param keys — S6 R2).
    BUNDLE_RELATION_SCHEMA_INVALID = "bundle_relation_schema_invalid"


class EpisodeCaptureError(TorchLensError):
    """Base class for episode-capture (``capture_kind=episode``) failures."""


class EpisodeDeclarationError(EpisodeCaptureError, ConfigurationError, ValueError):
    """Episode declaration refused at entry, before execution.

    ``fields["code"]`` carries ``episode_state_unsnapshotable`` or
    ``episode_declaration_invalid``.
    """


class EpisodeLedgerError(EpisodeCaptureError, ValidationError, ValueError):
    """Episode ledger validation refusal.

    ``fields["code"]`` carries ``episode_ledger_incoherent``,
    ``episode_ledger_without_declaration``, or
    ``episode_ledger_payload_in_structure_only``.
    """


class BundleRelationError(ValidationError, ValueError):
    """Bundle member-relation table refusal (S6).

    ``fields["code"]`` carries ``bundle_relation_member_missing``,
    ``bundle_member_has_relations``, or ``bundle_relation_schema_invalid``.
    """
