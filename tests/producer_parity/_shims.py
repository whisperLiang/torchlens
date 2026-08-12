"""Check B attestation shims (test-only, observe-and-delegate).

The shims record within-leg ground truth for the volatile-token buckets at
their semantic sources (design-of-record section 6.2, Sol v4 note 1, Opus v4
notes N1/N3):

* ``grad_fn_object_id`` — wraps ``ops._log_output_tensor_info`` and observes
  the tensor's SELECTED autograd node (``tl_user_grad_fn`` when present, else
  ``t.grad_fn``) BEFORE delegation, i.e. before the producer deletes the
  ``tl_user_grad_fn`` marker. Rooted in the live tensor, never the draft.
* ``tl_barcode`` — wraps ``_tl.set_param_meta`` (binding: live param object ->
  barcode at attachment time) and ``hashing.make_random_barcode`` (the minted
  set), covering both param-registration mint sites.
* ``backend_handle_id`` — no shim: the bucket is coherence-attested against the
  record's own carried payload object (see ``_fields.VOLATILE_BUCKETS``).

Attach points (Opus v4 note N3): the legacy exhaustive commit pipeline is
reached through ``_log_output_tensor_info`` (called for every per-output
commit feeding the three ``_make_layer_log_entry`` sites); the sparse leg
hard-sets ``grad_fn_handle=None`` and carries no attestable token for these
buckets, which is disclosed rather than papered over. The shims never mutate
an argument and never change control flow — the shim-perturbation control
(legacy-vs-legacy with shims armed must stay empty) proves observation is
side-effect-free.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator


@dataclass
class AttestationLog:
    """Within-leg ground truth recorded by the shims for one capture run."""

    # anchor (label_raw) -> id() of the selected grad_fn at semantic source.
    grad_fn_by_anchor: dict[str, int | None] = field(default_factory=dict)
    # anchor (label_raw) -> id() of the LIVE output tensor at the commit
    # boundary (the referent backend_handle_id names; the retained payload may
    # be a copy under save_mode="copy", so the live object is the only root).
    handle_by_anchor: dict[str, int] = field(default_factory=dict)
    # id(param object) -> barcode attached at set_param_meta time.
    barcode_by_param_id: dict[int, str] = field(default_factory=dict)
    # every barcode minted through make_random_barcode this run.
    minted_barcodes: list[str] = field(default_factory=list)
    # anchors observed more than once (would make attestation ambiguous).
    duplicate_anchors: list[str] = field(default_factory=list)


@contextlib.contextmanager
def attestation_shims() -> Iterator[AttestationLog]:
    """Install the Check B shims around one capture; restore on exit."""

    import torchlens.backends.torch._tl as tl_meta
    import torchlens.backends.torch.ops as ops_module
    import torchlens.utils.hashing as hashing_module

    log = AttestationLog()

    original_log_output = ops_module._log_output_tensor_info

    def observing_log_output(
        self: Any,
        t: Any,
        i: int,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        parent_param_ops: dict[str, int],
        fields_dict: dict[str, Any],
        autograd_saved_stats: tuple[int | None, int | None],
    ) -> None:
        """Observe the selected grad_fn pre-deletion, then delegate."""

        selected = getattr(t, "tl_user_grad_fn", None)
        if selected is None:
            selected = t.grad_fn
        observed_id = None if selected is None else id(selected)
        original_log_output(
            self, t, i, args, kwargs, parent_param_ops, fields_dict, autograd_saved_stats
        )
        anchor = fields_dict.get("_label_raw")
        if isinstance(anchor, str):
            if anchor in log.grad_fn_by_anchor:
                log.duplicate_anchors.append(anchor)
            log.grad_fn_by_anchor[anchor] = observed_id
            log.handle_by_anchor[anchor] = id(t)

    # Lookback retention legitimately REBINDS backend_handle_id post-commit
    # (the retained, possibly detached payload replaces the live output —
    # Sol v4 note 1's second case). Attest the token at its rebind site.
    original_replace_retained = ops_module._replace_event_with_retained_payload

    def observing_replace_retained(trace: Any, raw_label: str, payload: Any) -> None:
        if payload.raw_out is not None:
            log.handle_by_anchor[raw_label] = id(payload.raw_out)
        original_replace_retained(trace, raw_label, payload)

    original_set_param_meta = tl_meta.set_param_meta

    def observing_set_param_meta(
        p: Any, *, barcode: str, address: str, requires_grad_before: bool
    ) -> None:
        """Record the (live param -> barcode) binding, then delegate."""

        log.barcode_by_param_id[id(p)] = barcode
        original_set_param_meta(
            p, barcode=barcode, address=address, requires_grad_before=requires_grad_before
        )

    original_make_barcode = hashing_module.make_random_barcode

    def observing_make_barcode(barcode_len: int = 8) -> str:
        """Record every minted barcode, then return it unchanged."""

        minted = original_make_barcode(barcode_len)
        log.minted_barcodes.append(minted)
        return minted

    restores: list[Callable[[], None]] = []
    ops_module._log_output_tensor_info = observing_log_output
    restores.append(lambda: setattr(ops_module, "_log_output_tensor_info", original_log_output))
    ops_module._replace_event_with_retained_payload = observing_replace_retained
    restores.append(
        lambda: setattr(
            ops_module, "_replace_event_with_retained_payload", original_replace_retained
        )
    )
    tl_meta.set_param_meta = observing_set_param_meta
    restores.append(lambda: setattr(tl_meta, "set_param_meta", original_set_param_meta))
    # model_prep and tensor_tracking import set_param_meta by name; patch those
    # references too so both mint sites are observed.
    import torchlens.backends.torch.model_prep as model_prep_module
    import torchlens.backends.torch.tensor_tracking as tracking_module

    original_prep_ref = model_prep_module.set_param_meta
    original_track_ref = tracking_module.set_param_meta
    model_prep_module.set_param_meta = observing_set_param_meta
    tracking_module.set_param_meta = observing_set_param_meta
    restores.append(lambda: setattr(model_prep_module, "set_param_meta", original_prep_ref))
    restores.append(lambda: setattr(tracking_module, "set_param_meta", original_track_ref))

    import torchlens.backends.torch.wrappers as wrappers_module

    original_prep_mint = model_prep_module.make_random_barcode
    original_track_mint = tracking_module.make_random_barcode
    original_wrap_mint = wrappers_module.make_random_barcode
    hashing_module.make_random_barcode = observing_make_barcode
    model_prep_module.make_random_barcode = observing_make_barcode
    tracking_module.make_random_barcode = observing_make_barcode
    wrappers_module.make_random_barcode = observing_make_barcode
    restores.append(lambda: setattr(hashing_module, "make_random_barcode", original_make_barcode))
    restores.append(lambda: setattr(model_prep_module, "make_random_barcode", original_prep_mint))
    restores.append(lambda: setattr(tracking_module, "make_random_barcode", original_track_mint))
    restores.append(lambda: setattr(wrappers_module, "make_random_barcode", original_wrap_mint))

    try:
        yield log
    finally:
        for restore in reversed(restores):
            restore()
