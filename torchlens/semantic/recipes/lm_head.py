"""Built-in semantic recipes for language-model unembedding heads.

The ``language_model_head`` recipe anchors every architecture-specific fact a
logit-lens style projection needs -- which child module is the unembedding
head, which normalization module feeds it, and that norm's kind/parameters --
so downstream appliances (``torchlens.semantic.logit_lens``) stay generic math
over these facets. New architectures extend coverage by registering another
recipe that produces the same facet names; the appliance never changes.
"""

from __future__ import annotations

from typing import Any

from ..facets import AbsenceReason, register
from ._helpers import (
    add_if_present,
    child_module,
    child_output_spec,
    config_value,
    module_input_op_spec,
    parameter_spec,
    structural,
)

_LM_HEAD_FACETS = (
    "logits",
    "unembed_weight",
    "unembed_bias",
    "final_norm_kind",
    "final_norm_eps",
    "final_norm_gamma",
    "final_norm_beta",
    "final_norm_input",
)

#: Conventional direct-child names for the unembedding projection. A model
#: whose head is named differently is covered by registering a user recipe
#: producing the same facet names, never by widening this list speculatively.
_HEAD_CHILD_NAMES = ("lm_head", "embed_out", "output_projection")

_LAYER_NORM_CLASS_NAMES = frozenset({"LayerNorm"})


def _norm_kind_for_class(class_name: str) -> str | None:
    """Classify a module class name into a closed normalization-kind vocabulary.

    The ``*RMSNorm`` suffix match is deliberately broad (``LlamaRMSNorm``,
    ``Qwen2RMSNorm``, ...) and is safe ONLY because every consumer of
    ``final_norm_kind`` must numerically validate its reconstruction against
    the captured norm-input -> logits pair before trusting it (the
    ``logit_lens`` appliance refuses on mismatch). A nonstandard family member
    (e.g. a ``(1 + weight)``-scaled RMSNorm) therefore fails loudly downstream
    instead of being silently mislabelled here.

    Parameters
    ----------
    class_name:
        Module class name.

    Returns
    -------
    str | None
        ``"layer_norm"``, ``"rms_norm"``, or ``None`` when unclassified.
    """

    if class_name in _LAYER_NORM_CLASS_NAMES:
        return "layer_norm"
    if class_name == "RMSNorm" or class_name.endswith("RMSNorm"):
        return "rms_norm"
    return None


def _head_child_name(module: Any) -> str | None:
    """Return the conventional unembedding-head child name when present.

    Parameters
    ----------
    module:
        Candidate module record.

    Returns
    -------
    str | None
        Matched local child name, or ``None`` when no head child exists.
    """

    for child in getattr(module, "address_children", ()) or ():
        local = str(child).rsplit(".", maxsplit=1)[-1]
        if local in _HEAD_CHILD_NAMES:
            return local
    return None


def _has_lm_head(module: Any) -> bool:
    """Return whether a module record has a conventional unembedding child."""

    return _head_child_name(module) is not None


def _strip_pass(label: str) -> str:
    """Return a pass-qualified module label with its ``:<pass>`` suffix removed."""

    base, sep, tail = label.rpartition(":")
    if sep and tail.isdigit():
        return base
    return label


def _final_norm_record(head: Any) -> Any | None:
    """Return the norm module record whose output feeds the head's input.

    The head's captured input op carries its containing-module stack; walking
    it innermost-first finds the normalization module that produced the value
    the unembedding consumed. This is a structural derivation from the traced
    dataflow, never a name search over the module tree.

    Parameters
    ----------
    head:
        Unembedding-head module record.

    Returns
    -------
    Any | None
        Norm module record, or ``None`` when no classified norm feeds the head.
    """

    trace = getattr(head, "trace", None)
    if trace is None:
        return None
    try:
        call = head._single_call_or_error()
        input_ops = list(getattr(call, "input_ops", ()) or ())
        if not input_ops:
            return None
        op = trace.ops[input_ops[0]]
    except (AttributeError, KeyError, IndexError, RuntimeError, ValueError):
        return None
    for entry in reversed(tuple(getattr(op, "modules", ()) or ())):
        address = _strip_pass(str(entry))
        try:
            record = trace.modules[address]
        except (KeyError, ValueError):
            continue
        if _norm_kind_for_class(str(getattr(record, "class_name", ""))) is not None:
            return record
    return None


@register(predicate=_has_lm_head, target_scope="module", facets=_LM_HEAD_FACETS)
def language_model_head(module: Any) -> dict[str, Any]:
    """Return unembedding-head facets for language-model modules.

    Parameters
    ----------
    module:
        TorchLens module record with a conventional unembedding child.

    Returns
    -------
    dict[str, Any]
        Head and final-norm facets anchored to captured records.
    """

    result: dict[str, Any] = {}
    head_name = _head_child_name(module)
    head = child_module(module, head_name) if head_name is not None else None
    if head_name is None or head is None:
        absent = structural(f"unembedding child {head_name!r} record is unavailable")
        for name in _LM_HEAD_FACETS:
            add_if_present(result, name, absent)
        return result
    add_if_present(result, "logits", child_output_spec(module, head_name, "language_model_head"))
    add_if_present(result, "unembed_weight", parameter_spec(head, "weight", "language_model_head"))
    add_if_present(result, "unembed_bias", parameter_spec(head, "bias", "language_model_head"))
    norm = _final_norm_record(head)
    if norm is None:
        absent = structural("no classified normalization module feeds the unembedding head input")
        for name in _LM_HEAD_FACETS[3:]:
            add_if_present(result, name, absent)
        return result
    kind = _norm_kind_for_class(str(getattr(norm, "class_name", "")))
    add_if_present(result, "final_norm_kind", kind)
    add_if_present(result, "final_norm_input", module_input_op_spec(norm, "language_model_head"))
    add_if_present(
        result, "final_norm_gamma", parameter_spec(norm, "weight", "language_model_head")
    )
    if kind == "rms_norm":
        add_if_present(result, "final_norm_beta", structural("RMSNorm has no beta/bias parameter"))
    else:
        add_if_present(
            result, "final_norm_beta", parameter_spec(norm, "bias", "language_model_head")
        )
    eps = config_value(norm, "eps", "variance_epsilon")
    if isinstance(eps, AbsenceReason):
        eps = structural("normalization epsilon metadata is absent")
    add_if_present(result, "final_norm_eps", eps)
    return result
