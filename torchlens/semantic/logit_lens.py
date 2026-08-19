"""Logit-lens appliance over semantic facets.

``logit_lens`` pushes each transformer block's residual-stream state through
the model's OWN final normalization + unembedding, showing what the model
"would have said" at every depth. Every architecture-specific fact (which
child is the unembedding, which norm feeds it, the norm kind/parameters)
comes from the ``language_model_head`` facet recipe -- this module is generic
math over that facet vocabulary plus the existing block ``resid_*`` facets,
so new architectures extend coverage by registering recipes, never by
editing this appliance.

Every spelling here is DOCUMENTED-UNSTABLE pending naming-session
ratification.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F

from .facets import Facet, MissingFacet

__all__ = ["LogitLensEntry", "LogitLensError", "LogitLensResult", "logit_lens"]

LensFunc = Callable[[torch.Tensor], torch.Tensor]


class LogitLensError(RuntimeError):
    """Raised when a logit lens cannot be built, validated, or applied."""


@dataclass(frozen=True)
class LogitLensEntry:
    """Per-layer logit-lens projection.

    Parameters
    ----------
    address:
        Module address of the transformer block the hidden state came from.
    layer_index:
        Zero-based position of the block among the projected layers.
    facet:
        Hidden-state facet name that was projected (e.g. ``"resid_post"``).
    logits:
        Projected logits with the hidden state's leading dimensions preserved.
    """

    address: str
    layer_index: int
    facet: str
    logits: torch.Tensor

    def top_k(self, k: int = 5, *, position: int = -1, batch_index: int = 0) -> tuple[Any, Any]:
        """Return top-k probabilities and token ids at one sequence position.

        Parameters
        ----------
        k:
            Number of top tokens.
        position:
            Sequence position (last dimension before vocab), when present.
        batch_index:
            Batch element to inspect.

        Returns
        -------
        tuple[Any, Any]
            ``(probabilities, token_ids)`` tensors of length ``k``.
        """

        vector = _position_vector(self.logits, position=position, batch_index=batch_index)
        probabilities = torch.softmax(vector.float(), dim=-1)
        values, indices = probabilities.topk(min(k, probabilities.shape[-1]))
        return values, indices


@dataclass(frozen=True)
class LogitLensResult:
    """Result of a :func:`logit_lens` sweep across transformer blocks.

    Parameters
    ----------
    entries:
        Per-layer projections in block execution order.
    facet:
        Hidden-state facet name that was projected.
    lens_source:
        ``"model_head"`` when the lens was reconstructed from the
        ``language_model_head`` facets, ``"user"`` when supplied via ``lens=``.
    validated:
        Whether the reconstructed lens was numerically validated against the
        captured last-block ``resid_post`` -> logits pair. Always ``False``
        for user lenses (TorchLens has no oracle for them).
    final_logits:
        The model's captured real output logits when readable, else ``None``.
    """

    entries: tuple[LogitLensEntry, ...]
    facet: str
    lens_source: str
    validated: bool
    final_logits: torch.Tensor | None

    def stacked(self) -> torch.Tensor:
        """Return all per-layer logits stacked on a new leading layer axis.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(n_layers, *logits_shape)``.
        """

        shapes = {tuple(entry.logits.shape) for entry in self.entries}
        if len(shapes) > 1:
            raise LogitLensError(
                f"Cannot stack per-layer logits with differing shapes {sorted(shapes)!r}; "
                "read result.entries[i].logits individually."
            )
        return torch.stack([entry.logits for entry in self.entries])

    def top_tokens(
        self,
        k: int = 5,
        *,
        position: int = -1,
        batch_index: int = 0,
        tokenizer: Any | None = None,
    ) -> list[tuple[str, list[tuple[Any, float]]]]:
        """Return per-layer top-k tokens at one sequence position.

        Parameters
        ----------
        k:
            Number of top tokens per layer.
        position:
            Sequence position to inspect.
        batch_index:
            Batch element to inspect.
        tokenizer:
            Optional tokenizer with ``convert_ids_to_tokens`` or ``decode``;
            when omitted, raw token ids are returned.

        Returns
        -------
        list[tuple[str, list[tuple[Any, float]]]]
            One ``(address, [(token, probability), ...])`` row per layer.
        """

        rows: list[tuple[str, list[tuple[Any, float]]]] = []
        for entry in self.entries:
            values, indices = entry.top_k(k, position=position, batch_index=batch_index)
            ids = [int(index) for index in indices]
            tokens: Sequence[Any] = ids if tokenizer is None else _decode_ids(tokenizer, ids)
            rows.append((entry.address, list(zip(tokens, [float(value) for value in values]))))
        return rows

    def summary(
        self,
        *,
        position: int = -1,
        batch_index: int = 0,
        tokenizer: Any | None = None,
    ) -> str:
        """Return a human-readable per-layer top-1 table.

        Parameters
        ----------
        position:
            Sequence position to inspect.
        batch_index:
            Batch element to inspect.
        tokenizer:
            Optional tokenizer used to decode token ids.

        Returns
        -------
        str
            One line per layer with the top token and its probability.
        """

        lines = [f"logit lens ({self.facet}, lens={self.lens_source})"]
        for entry in self.entries:
            values, indices = entry.top_k(1, position=position, batch_index=batch_index)
            token: Any = int(indices[0])
            if tokenizer is not None:
                token = _decode_ids(tokenizer, [int(indices[0])])[0]
            lines.append(
                f"  [{entry.layer_index:>3}] {entry.address}: {token!r} ({float(values[0]):.4f})"
            )
        return "\n".join(lines)


def logit_lens(
    trace: Any,
    *,
    facet: str = "resid_post",
    layers: Sequence[str] | None = None,
    lens: LensFunc | Mapping[str, LensFunc] | None = None,
    validate: bool = True,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> LogitLensResult:
    """Project per-block hidden states through the model's output head.

    Parameters
    ----------
    trace:
        Completed TorchLens trace of a language model captured with the
        relevant activations saved (the default exhaustive save suffices).
    facet:
        Hidden-state facet to project (``"resid_post"``, ``"resid_pre"``, or
        ``"resid_mid"``).
    layers:
        Explicit module addresses to project; defaults to every module
        exposing ``facet``, in trace order.
    lens:
        Optional user lens: one callable ``hidden -> logits`` applied to every
        layer, or a mapping from module address to per-layer callables (e.g. a
        tuned lens). When omitted, the lens is reconstructed from the
        ``language_model_head`` facets and numerically validated.
    validate:
        Whether to validate the reconstructed default lens by re-projecting
        the last block's captured ``resid_post`` and comparing against the
        captured logits. Fail-closed: validation that cannot run (missing
        anchors) refuses rather than silently trusting the reconstruction.
        Ignored for user lenses.
    rtol:
        Relative tolerance for lens validation.
    atol:
        Absolute tolerance for lens validation.

    Returns
    -------
    LogitLensResult
        Per-layer projections in block order.
    """

    addresses = _resolve_layer_addresses(trace, facet=facet, layers=layers)
    head_view = _find_head_view(trace)
    final_logits = _read_final_logits(head_view)
    if lens is not None:
        lens_by_address = _user_lens_by_address(lens, addresses)
        lens_source, validated = "user", False
    else:
        if head_view is None:
            raise LogitLensError(
                "No module in this trace exposes the 'unembed_weight' facet, so the "
                "model's own lens cannot be reconstructed. Register a facet recipe "
                "producing the language_model_head facet names for this architecture "
                "(tl.facets.register), or pass lens= explicitly."
            )
        default_lens = _build_default_lens(head_view)
        if validate:
            _validate_lens(trace, head_view, default_lens, rtol=rtol, atol=atol)
        lens_by_address = dict.fromkeys(addresses, default_lens)
        lens_source, validated = "model_head", validate
    entries: list[LogitLensEntry] = []
    with torch.no_grad():
        for layer_index, address in enumerate(addresses):
            hidden = _facet_tensor(trace.modules[address].facets[facet], name=facet)
            entries.append(
                LogitLensEntry(
                    address=address,
                    layer_index=layer_index,
                    facet=facet,
                    logits=lens_by_address[address](hidden.detach()),
                )
            )
    return LogitLensResult(
        entries=tuple(entries),
        facet=facet,
        lens_source=lens_source,
        validated=validated,
        final_logits=final_logits,
    )


def _resolve_layer_addresses(trace: Any, *, facet: str, layers: Sequence[str] | None) -> list[str]:
    """Return the block addresses to project, validating availability."""

    available = [
        str(module.address)
        for module in trace.modules
        if getattr(module, "address", None) != "self" and module.facets.has(facet)
    ]
    if layers is None:
        if not available:
            raise LogitLensError(
                f"No modules in this trace expose facet {facet!r}. Logit lens needs "
                "transformer blocks with residual-stream facets; capture with the "
                "default exhaustive save, or register a block facet recipe for this "
                "architecture."
            )
        return available
    missing = [address for address in layers if address not in available]
    if missing:
        raise LogitLensError(
            f"Modules {tuple(missing)!r} do not expose facet {facet!r} in this trace. "
            f"Available: {tuple(available)!r}."
        )
    return [str(address) for address in layers]


def _find_head_view(trace: Any) -> Any | None:
    """Return the first module facet view exposing the unembedding facets."""

    for module in trace.modules:
        view = module.facets
        if view.has("unembed_weight"):
            return view
    return None


def _read_final_logits(head_view: Any | None) -> torch.Tensor | None:
    """Return the captured real output logits when readable."""

    if head_view is None:
        return None
    value = head_view.get("logits")
    if value is None or isinstance(value, MissingFacet):
        return None
    try:
        return _facet_tensor(value, name="logits").detach()
    except LogitLensError:
        return None


def _user_lens_by_address(
    lens: LensFunc | Mapping[str, LensFunc], addresses: Sequence[str]
) -> dict[str, LensFunc]:
    """Return a per-address lens mapping from the user ``lens=`` argument."""

    if callable(lens):
        return dict.fromkeys(addresses, lens)
    if isinstance(lens, Mapping):
        missing = [address for address in addresses if address not in lens]
        if missing:
            raise LogitLensError(
                f"lens= mapping is missing entries for modules {tuple(missing)!r}; "
                "provide one callable per projected layer or a single callable."
            )
        return {address: lens[address] for address in addresses}
    raise LogitLensError(
        "lens= must be a callable hidden -> logits or a mapping from module "
        f"address to such callables, got {type(lens).__name__}."
    )


def _build_default_lens(head_view: Any) -> LensFunc:
    """Reconstruct the model's own final-norm + unembed lens from facets."""

    kind = head_view.get("final_norm_kind")
    if not isinstance(kind, str):
        raise LogitLensError(
            "The final normalization feeding the unembedding head was not "
            "identified (facet 'final_norm_kind' is unavailable), so the model's "
            "own lens cannot be reconstructed. Pass lens= explicitly, or register "
            "a facet recipe classifying this model's final norm."
        )
    weight = _facet_tensor(head_view["unembed_weight"], name="unembed_weight").detach()
    gamma = _facet_tensor(head_view["final_norm_gamma"], name="final_norm_gamma").detach()
    beta_value = head_view.get("final_norm_beta")
    beta = (
        None
        if beta_value is None or isinstance(beta_value, MissingFacet)
        else _facet_tensor(beta_value, name="final_norm_beta").detach()
    )
    bias_value = head_view.get("unembed_bias")
    bias = (
        None
        if bias_value is None or isinstance(bias_value, MissingFacet)
        else _facet_tensor(bias_value, name="unembed_bias").detach()
    )
    eps_value = head_view.get("final_norm_eps")
    eps = float(eps_value) if isinstance(eps_value, (int, float)) else 1e-5

    def _apply(hidden: torch.Tensor) -> torch.Tensor:
        """Apply the reconstructed final norm + unembedding to a hidden state."""

        if kind == "layer_norm":
            normalized = F.layer_norm(hidden, gamma.shape, gamma, beta, eps)
        elif kind == "rms_norm":
            normalized = (
                hidden * torch.rsqrt(hidden.pow(2).mean(dim=-1, keepdim=True) + eps) * gamma
            )
        else:
            raise LogitLensError(f"Unknown final_norm_kind {kind!r}; pass lens= explicitly.")
        logits = normalized @ weight.transpose(-2, -1)
        if bias is not None:
            logits = logits + bias
        return logits

    return _apply


def _validate_lens(trace: Any, head_view: Any, lens: LensFunc, *, rtol: float, atol: float) -> None:
    """Validate a reconstructed lens against captured last-block -> logits data.

    Applying the reconstructed final norm + unembedding to the LAST block's
    captured ``resid_post`` must reproduce the model's captured real logits:
    that is exactly the path the real forward took, so the check validates the
    full norm + unembed reconstruction end-to-end. This is a tripwire, not a
    formality: an unconventional head (extra projection, nonstandard norm
    scaling such as a ``(1 + weight)`` RMSNorm, dropout before the head) makes
    the reconstruction WRONG, and a wrong lens is a confidently mislabelled
    reading of someone's model. Fail-closed: when the anchors needed to
    validate were not captured, this refuses rather than silently trusting the
    reconstruction (pass ``validate=False`` to opt out explicitly).
    """

    captured = head_view.get("logits")
    if captured is None:
        raise LogitLensError(
            "Cannot validate the reconstructed lens: the model's captured head "
            "logits were not saved in this trace. Re-capture with the default "
            "exhaustive save (or a save= predicate including the head output), or "
            "pass validate=False to trust the reconstruction without a numeric check."
        )
    resid_addresses = [
        str(module.address)
        for module in trace.modules
        if getattr(module, "address", None) != "self" and module.facets.has("resid_post")
    ]
    if not resid_addresses:
        raise LogitLensError(
            "Cannot validate the reconstructed lens: no module exposes a captured "
            "'resid_post' facet to replay through it. Re-capture with the default "
            "exhaustive save, or pass validate=False to trust the reconstruction "
            "without a numeric check."
        )
    reference_hidden = _facet_tensor(
        trace.modules[resid_addresses[-1]].facets["resid_post"], name="resid_post"
    ).detach()
    reference_logits = _facet_tensor(captured, name="logits").detach()
    with torch.no_grad():
        reconstructed = lens(reference_hidden)
    if reconstructed.shape != reference_logits.shape or not torch.allclose(
        reconstructed, reference_logits, rtol=rtol, atol=atol
    ):
        raise LogitLensError(
            "Reconstructed lens failed validation: applying the reconstructed "
            "final norm + unembedding to the last block's captured resid_post does "
            f"not reproduce the captured logits (rtol={rtol}, atol={atol}). The "
            "head does something the reconstruction does not capture (nonstandard "
            "norm scaling, an extra projection, or dropout between the last block "
            "and the head). Pass lens= explicitly or register a corrected facet "
            "recipe."
        )


def _position_vector(logits: torch.Tensor, *, position: int, batch_index: int) -> torch.Tensor:
    """Return the vocab vector at one batch element / sequence position."""

    if logits.ndim >= 3:
        return logits[batch_index, ..., position, :]
    if logits.ndim == 2:
        return logits[batch_index]
    return logits


def _decode_ids(tokenizer: Any, ids: Sequence[int]) -> list[Any]:
    """Decode token ids with whichever tokenizer surface is available."""

    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    if callable(convert):
        return list(convert(list(ids)))
    decode = getattr(tokenizer, "decode", None)
    if callable(decode):
        return [decode([token_id]) for token_id in ids]
    return list(ids)


def _facet_tensor(value: Any, *, name: str) -> torch.Tensor:
    """Return a tensor from a facet-like value or refuse with a clear message."""

    if isinstance(value, Facet):
        value = value.value
    if isinstance(value, MissingFacet):
        raise LogitLensError(f"Facet {name!r} is unavailable: {value.reason}")
    if not isinstance(value, torch.Tensor):
        raise LogitLensError(
            f"Facet {name!r} did not produce a tensor (got {type(value).__name__})."
        )
    return value
