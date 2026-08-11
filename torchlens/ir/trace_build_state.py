"""Mutable trace-build state for capture and postprocessing."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Any

from .container import ContainerSpec
from .container_registry import ContainerRegistry

LEGACY_TRACE_BUILD_STATE_KEYS = frozenset(
    {
        "_raw_layer_dict",
        "_raw_layer_labels_list",
        "_layer_counter",
        "_raw_layer_type_counter",
        "_current_func_barcode",
        "_mod_entered",
        "_mod_exited",
        "_mod_call_index",
        "_mod_call_labels",
        "_module_build_data",
        "_module_metadata",
        "_module_forward_args",
        "_module_containment_engine",
        "_exhaustive_module_stack",
        "_grad_fn_strong_refs",
        "_in_exhaustive_pass",
        "_input_tensor_addresses",
    }
)
"""Legacy flat Trace scratch keys accepted only for drop-on-restore compatibility."""


@dataclass(slots=True)
class TraceBuildState:
    """Transient capture/postprocess state discarded before returning a Trace."""

    raw_layer_dict: dict[str, Any] = field(default_factory=OrderedDict)
    raw_layer_labels_list: list[str] = field(default_factory=list)
    mod_entered: dict[int, list[str]] = field(default_factory=dict)
    mod_exited: dict[int, list[str]] = field(default_factory=dict)
    mod_call_index: dict[int, int] = field(default_factory=dict)
    mod_call_labels: dict[int, list[tuple[str, int]]] = field(default_factory=dict)
    exhaustive_module_stack: list[Any] = field(default_factory=list)
    module_build_data: dict[str, Any] = field(default_factory=dict)
    module_metadata: dict[Any, Any] = field(default_factory=dict)
    module_forward_args: dict[Any, Any] = field(default_factory=dict)
    module_containment_engine: str = "hook_stack"
    current_func_barcode: Any = None
    grad_fn_strong_refs: list[Any] = field(default_factory=list)
    in_exhaustive_pass: bool = True
    layer_counter: int = 0
    raw_layer_type_counter: dict[str, int] = field(default_factory=lambda: defaultdict(lambda: 0))
    output_container_specs_by_raw_label: dict[str, ContainerSpec] = field(default_factory=dict)
    output_container_specs: tuple[ContainerSpec, ...] = ()
    container_registry: ContainerRegistry = field(default_factory=ContainerRegistry)
    input_tensor_addresses: list[str] = field(default_factory=list)
