"""SAE splice-experiment helper (SAE-library-agnostic).

Splicing swaps a sparse autoencoder's reconstruction back into the model at
the site the SAE was trained on and replays the downstream computation, so
the causal footprint of the SAE's features can be measured against the clean
run. This module composes existing TorchLens primitives (fork, replay,
``tl.splice_module``) into the one-call experiment; it never adds capture
machinery of its own.

Every spelling in this module is DOCUMENTED-UNSTABLE: whether the splice
experiment earns a stable public name is a UI-sprint question, and the
surface here is deliberately minimal until that ruling.

The SAE object is duck-typed: anything exposing ``encode(x) -> latents`` and
``decode(latents) -> reconstruction`` works (an SAE Lens ``SAE``, or any
hand-built module). No third-party SAE package is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from ._utils import resolve_one_site

if TYPE_CHECKING:
    from collections.abc import Callable


def _require_codec(sae: Any) -> None:
    """Refuse SAE objects without an encode/decode pair.

    Parameters
    ----------
    sae:
        Candidate SAE object.

    Raises
    ------
    TypeError
        If ``sae`` does not expose callable ``encode`` and ``decode``.
    """

    if not callable(getattr(sae, "encode", None)) or not callable(getattr(sae, "decode", None)):
        raise TypeError(
            "SAE splice requires an SAE exposing encode(...) and decode(...); "
            f"got {type(sae).__name__!r}. Any object with that pair works -- "
            "an SAE Lens SAE, or a hand-built module."
        )


class _SAEReconstruction(nn.Module):
    """Shape-preserving encode/decode splice around a duck-typed SAE.

    Parameters
    ----------
    sae:
        Object exposing ``encode`` and ``decode``.
    latents_edit:
        Optional transform applied to the encoded latents before decoding.
    """

    def __init__(
        self, sae: Any, latents_edit: Callable[[torch.Tensor], torch.Tensor] | None
    ) -> None:
        """Store the SAE and optional latent transform."""

        super().__init__()
        self._sae = sae
        self._latents_edit = latents_edit

    def forward(self, out: torch.Tensor) -> torch.Tensor:
        """Return the (optionally latent-edited) SAE reconstruction of ``out``.

        Parameters
        ----------
        out:
            Site activation being replaced.

        Returns
        -------
        torch.Tensor
            Reconstruction cast back to the site's dtype and device.
        """

        return _reconstruct(self._sae, out, self._latents_edit)


def _reconstruct(
    sae: Any,
    out: torch.Tensor,
    latents_edit: Callable[[torch.Tensor], torch.Tensor] | None,
) -> torch.Tensor:
    """Return the (optionally latent-edited) SAE reconstruction of ``out``.

    Parameters
    ----------
    sae:
        Duck-typed SAE.
    out:
        Site activation being reconstructed.
    latents_edit:
        Optional transform applied to the encoded latents before decoding.

    Returns
    -------
    torch.Tensor
        Reconstruction cast to the site's dtype and device.

    Raises
    ------
    TypeError
        If ``decode`` returns a non-tensor.
    """

    latents = sae.encode(out)
    if latents_edit is not None:
        latents = latents_edit(latents)
    reconstruction = sae.decode(latents)
    if not isinstance(reconstruction, torch.Tensor):
        raise TypeError(
            f"SAE decode(...) must return a torch.Tensor; got {type(reconstruction).__name__!r}"
        )
    return reconstruction.to(device=out.device, dtype=out.dtype)


@dataclass(frozen=True)
class SpliceResult:
    """Settled record of one SAE splice experiment.

    Parameters
    ----------
    site:
        Resolved pass-qualified layer label the reconstruction replaced.
    clean:
        Clean capture the experiment forked from.
    spliced:
        Replayed fork carrying the spliced computation.
    clean_site_out:
        Clean activation at the splice site.
    reconstruction:
        SAE reconstruction that replaced it (after any latent edit).
    reconstruction_mse:
        Mean squared error between reconstruction and clean activation.
    fraction_variance_explained:
        ``1 - FVU`` of the reconstruction against the clean activation.
    clean_outputs:
        Clean model-output tensors in output order.
    spliced_outputs:
        Spliced model-output tensors in the same order.
    output_delta_l2:
        L2 norm of the concatenated output difference.
    output_delta_max:
        Largest absolute per-element output difference.
    """

    site: str
    clean: Any
    spliced: Any
    clean_site_out: torch.Tensor
    reconstruction: torch.Tensor
    reconstruction_mse: float
    fraction_variance_explained: float
    clean_outputs: tuple[torch.Tensor, ...]
    spliced_outputs: tuple[torch.Tensor, ...]
    output_delta_l2: float
    output_delta_max: float


def _tensor_outputs(log: Any) -> tuple[torch.Tensor, ...]:
    """Return every saved tensor model output of a trace in output order.

    Parameters
    ----------
    log:
        TorchLens ``Trace``.

    Returns
    -------
    tuple[torch.Tensor, ...]
        Saved output tensors.

    Raises
    ------
    ValueError
        If no model output carries a saved tensor payload.
    """

    outputs = tuple(
        op.out for op in log.output_ops if isinstance(getattr(op, "out", None), torch.Tensor)
    )
    if not outputs:
        raise ValueError(
            "SAE splice needs saved tensor model outputs to compare runs; "
            "this capture retained none. Capture with the default exhaustive "
            "save (tl.trace(model, x, capture=tl.options.CaptureOptions("
            "intervention_ready=True)))."
        )
    return outputs


def splice(
    model_or_log: Any,
    inputs: Any = None,
    *,
    site: Any,
    sae: Any,
    latents_edit: Callable[[torch.Tensor], torch.Tensor] | None = None,
    fork_name: str = "sae_splice",
) -> SpliceResult:
    """Run the standard SAE splice experiment at one site.

    Captures (or reuses) a clean intervention-ready trace, forks it, replaces
    the site activation with the SAE's reconstruction via
    ``tl.splice_module``, replays the downstream computation, and reports
    reconstruction fidelity plus output-level causal effect.

    Parameters
    ----------
    model_or_log:
        Live ``nn.Module`` (captured here with ``intervention_ready=True``)
        or an already-captured intervention-ready ``Trace``.
    inputs:
        Model inputs, required when ``model_or_log`` is a module and refused
        when it is a trace.
    site:
        Layer label, selector, or layer object naming the splice site.
    sae:
        Duck-typed SAE exposing ``encode`` and ``decode``.
    latents_edit:
        Optional transform applied to the encoded latents before decoding --
        the causal-testing knob (ablate or steer individual features).
    fork_name:
        Name for the spliced fork.

    Returns
    -------
    SpliceResult
        Frozen experiment record; ``result.spliced`` is the replayed fork.

    Raises
    ------
    TypeError
        If ``sae`` lacks an encode/decode pair, or ``inputs`` disagrees with
        the ``model_or_log`` kind.
    """

    _require_codec(sae)
    if isinstance(model_or_log, nn.Module):
        if inputs is None:
            raise TypeError("SAE splice from a live model requires `inputs`.")
        import torchlens as tl

        clean = tl.trace(
            model_or_log,
            inputs,
            capture=tl.options.CaptureOptions(intervention_ready=True),
        )
    else:
        if inputs is not None:
            raise TypeError(
                "SAE splice from an existing Trace takes no `inputs`; the "
                "fork replays the captured inputs."
            )
        clean = model_or_log

    from torchlens.intervention import label as label_selector, splice_module

    site_label = str(resolve_one_site(clean, site).label)
    clean_site_out = clean[site_label].out
    with torch.no_grad():
        reconstruction = _reconstruct(sae, clean_site_out, latents_edit)
    clean_detached = clean_site_out.detach()
    residual = (reconstruction.detach() - clean_detached).float()
    centered = (clean_detached - clean_detached.float().mean()).float()
    total_variance = float(centered.pow(2).sum())
    reconstruction_mse = float(residual.pow(2).mean())
    fraction_variance_explained = (
        1.0 - float(residual.pow(2).sum()) / total_variance if total_variance > 0.0 else 0.0
    )

    spliced = clean.fork(fork_name)
    spliced.attach_hooks(
        label_selector(site_label),
        splice_module(_SAEReconstruction(sae, latents_edit), input="out"),
    )
    spliced.push()

    clean_outputs = _tensor_outputs(clean)
    spliced_outputs = _tensor_outputs(spliced)
    deltas = [
        (spliced_out.detach().float() - clean_out.detach().float()).flatten()
        for clean_out, spliced_out in zip(clean_outputs, spliced_outputs)
    ]
    flat_delta = torch.cat(deltas) if deltas else torch.zeros(0)
    return SpliceResult(
        site=site_label,
        clean=clean,
        spliced=spliced,
        clean_site_out=clean_site_out,
        reconstruction=reconstruction,
        reconstruction_mse=reconstruction_mse,
        fraction_variance_explained=fraction_variance_explained,
        clean_outputs=clean_outputs,
        spliced_outputs=spliced_outputs,
        output_delta_l2=float(flat_delta.norm()),
        output_delta_max=float(flat_delta.abs().max()) if flat_delta.numel() else 0.0,
    )


__all__ = ["SpliceResult", "splice"]
