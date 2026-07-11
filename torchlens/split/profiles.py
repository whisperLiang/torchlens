"""Pinned real-model profile registry for split validation."""

from __future__ import annotations

import os
from pathlib import Path
from threading import RLock
from typing import Iterable

from .ir import SplitModelProfile

_PROFILE_LOCK = RLock()
_PROFILES: dict[str, SplitModelProfile] = {}


def register_model_profile(profile: SplitModelProfile, *, replace: bool = False) -> None:
    """Register one real-model profile.

    Profiles contain loader metadata only.  Checkpoints remain in the user
    cache and are never copied into the source tree.
    """

    with _PROFILE_LOCK:
        if profile.id in _PROFILES and not replace:
            raise ValueError(f"model profile {profile.id!r} is already registered")
        _PROFILES[profile.id] = profile


def get_model_profile(profile_id: str) -> SplitModelProfile:
    """Return a registered model profile or raise a clear lookup error."""

    with _PROFILE_LOCK:
        try:
            return _PROFILES[profile_id]
        except KeyError as exc:
            known = ", ".join(sorted(_PROFILES)) or "<none>"
            raise KeyError(f"unknown split model profile {profile_id!r}; known: {known}") from exc


def resolve_model_profile(
    profile: str | SplitModelProfile | None,
) -> SplitModelProfile | None:
    """Resolve a profile ID or return an already materialized profile."""

    if profile is None or isinstance(profile, SplitModelProfile):
        return profile
    return get_model_profile(profile)


def iter_model_profiles() -> Iterable[SplitModelProfile]:
    """Iterate over a snapshot of registered profiles."""

    with _PROFILE_LOCK:
        return tuple(_PROFILES.values())


def model_cache_dir() -> Path:
    """Return the user-controlled real-model cache directory."""

    configured = os.environ.get("TORCHLENS_MODEL_CACHE")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "torchlens" / "models"


def profile_cache_dir(profile: str | SplitModelProfile) -> Path:
    """Return a deterministic cache directory for one profile."""

    resolved = resolve_model_profile(profile)
    if resolved is None:  # pragma: no cover - defensive typing guard.
        raise ValueError("profile_cache_dir requires a model profile")
    return model_cache_dir() / resolved.id


def checkpoint_cache_path(profile: str | SplitModelProfile, filename: str) -> Path:
    """Return a profile-scoped checkpoint path without creating directories."""

    if Path(filename).name != filename:
        raise ValueError("checkpoint filename must not contain path components")
    return profile_cache_dir(profile) / filename


__all__ = [
    "checkpoint_cache_path",
    "get_model_profile",
    "iter_model_profiles",
    "model_cache_dir",
    "profile_cache_dir",
    "register_model_profile",
    "resolve_model_profile",
]
