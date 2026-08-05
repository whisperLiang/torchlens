"""Compatibility re-exports for the historical ``torchlens.ir.buffer`` path.

``CaptureEvents`` and its helpers now live in
:mod:`torchlens.ir.capture_events`, and every in-tree caller imports them from
there (including :mod:`torchlens.ir`, which does not route through this module).
No module in the tree imports ``torchlens.ir.buffer``, so this shim exists solely
to preserve the historical import path for any out-of-tree consumer. Two of its
re-exports (``live_record_for_label`` and ``register_live_event``) are themselves
retired stubs. This is a de-bloat deletion candidate, but because the module is a
public import path, removing it is an owner-reserved public-surface change.
"""

from __future__ import annotations

from .capture_events import (
    CaptureEvents,
    LiveOpRecord,
    live_record_for_label,
    register_live_event,
    replace_op_event,
)

__all__ = [
    "CaptureEvents",
    "LiveOpRecord",
    "live_record_for_label",
    "register_live_event",
    "replace_op_event",
]
