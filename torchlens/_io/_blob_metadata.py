"""Portable tensor-blob kind and owner-label metadata."""

from __future__ import annotations

from typing import Any

from . import TorchLensIOError


def _blob_kind_for_field(owner: Any, field_name: str) -> str:
    """Map an object field name to the portable manifest tensor kind."""

    kind = {
        "out": "out",
        "transformed_out": "transformed_out",
        "grad": "grad",
        "transformed_grad": "transformed_grad",
        "saved_args": "captured_arg",
        "saved_kwargs": "captured_arg",
        "out_versions_by_child": "child_version",
        "func_rng_states": "rng_state",
        "forward_args": "module_arg",
        "forward_kwargs": "module_arg",
        "func_config": "func_config",
        "custom_attributes": "module_meta",
        "grad_inputs": "grad_fn_grad",
        "grad_outputs": "grad_fn_grad",
        "_buffer_initial_values": "buffer_initial_value",
        "_annotation_blobs": "annotation_blob",
        "orphan_records": "orphan_payload",
        # L6 stage 3: occurrence-granular substituted-value payloads.
        "edge_substitutions": "edge_substitution",
    }.get(field_name)
    if kind is not None:
        return kind
    if field_name in {"_args", "_kwargs", "payload"} and type(owner).__name__ in {
        "ModuleInputSnapshot",
        "TensorInputObservation",
    }:
        return "pre_hook_input"
    raise TorchLensIOError(f"No blob kind mapping defined for {type(owner).__name__}.{field_name}.")


def _blob_label_for_owner(owner: Any) -> str:
    """Return the human-readable label stored alongside a blob spec."""

    if hasattr(owner, "label") and getattr(owner, "label") is not None:
        return str(getattr(owner, "label"))
    if hasattr(owner, "call_label") and getattr(owner, "call_label") is not None:
        return str(getattr(owner, "call_label"))
    if hasattr(owner, "address") and getattr(owner, "address") is not None:
        return str(getattr(owner, "address"))
    return type(owner).__name__
