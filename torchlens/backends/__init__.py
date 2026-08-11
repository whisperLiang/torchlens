"""Backend Protocol and public registry exports for TorchLens adapters."""

from __future__ import annotations

from ._protocol import CaptureBackend
from .registry import (
    GATED_CAPABILITY_FLAGS,
    BackendAmbiguityError,
    BackendCapabilities,
    BackendCapabilityConformanceError,
    BackendMismatchError,
    BackendName,
    BackendPayloadUnsupportedError,
    BackendRegistryError,
    BackendRuntimeCompatibilityError,
    BackendSpec,
    BackendUnsupportedError,
    SerializationPolicy,
    UnknownBackendError,
    get_backend_spec,
    registered_backend_specs,
    register_backend_spec,
    require_capability_implementation,
    resolve_backend_spec,
    unregister_backend_spec,
)

# Import for backend registration side effects.
from . import default_specs as _default_specs  # noqa: F401

__all__ = [
    "GATED_CAPABILITY_FLAGS",
    "BackendAmbiguityError",
    "BackendCapabilities",
    "BackendCapabilityConformanceError",
    "BackendMismatchError",
    "BackendName",
    "BackendPayloadUnsupportedError",
    "BackendRegistryError",
    "BackendRuntimeCompatibilityError",
    "BackendSpec",
    "BackendUnsupportedError",
    "CaptureBackend",
    "SerializationPolicy",
    "UnknownBackendError",
    "get_backend_spec",
    "registered_backend_specs",
    "register_backend_spec",
    "require_capability_implementation",
    "resolve_backend_spec",
    "unregister_backend_spec",
]
