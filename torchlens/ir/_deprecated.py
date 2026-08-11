"""Inert compatibility shims for event types deleted from the capture IR.

The ``ModuleEvent`` and ``BufferEvent`` lanes were removed in the backend
migration (their production writers/consumers were verified dead at the base
revision), but both names were listed in ``torchlens.ir.__all__``. These
definitions keep the import path alive for external code; nothing in
TorchLens emits or consumes them anymore. Access through ``torchlens.ir``
raises a :class:`DeprecationWarning`.
"""

from __future__ import annotations

from dataclasses import dataclass

from .events import ModuleFrame


@dataclass(frozen=True, slots=True)
class BufferEvent:
    """DEPRECATED inert shim: captured module buffer metadata event.

    TorchLens no longer emits this event kind; buffer capture flows through
    :class:`~torchlens.ir.events.BufferWriteEvent`. The class exists only so
    ``from torchlens.ir import BufferEvent`` keeps importing.
    """

    address: str
    name: str
    module_address: str
    buffer_pass: int
    parent_label_raw: str | None
    shape: tuple[int, ...] | None
    dtype: str | None
    memory: int | None
    module_stack: tuple[ModuleFrame, ...]


@dataclass(frozen=True, slots=True)
class ModuleEvent:
    """DEPRECATED inert shim: captured module-call event.

    TorchLens no longer emits this event kind; module containment flows
    through :class:`~torchlens.ir.events.ModuleEnterEvent` and
    :class:`~torchlens.ir.events.ModuleExitEvent`. The class exists only so
    ``from torchlens.ir import ModuleEvent`` keeps importing.
    """

    address: str
    all_addresses: tuple[str, ...]
    call_index: int
    call_label: str
    layers_raw: tuple[str, ...]
    input_layers_raw: tuple[str, ...]
    output_layers_raw: tuple[str, ...]
    forward_args_summary: object
    forward_kwargs_summary: object
    forward_args: object | None
    forward_kwargs: object | None
    call_parent: str | None
    call_children: tuple[str, ...]
