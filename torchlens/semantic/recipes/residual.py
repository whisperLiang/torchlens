"""Built-in semantic recipes for transformer residual stream facets."""

from __future__ import annotations

from typing import Any

from ..facets import AbsenceReason, FacetSpec, register
from ._helpers import (
    add_if_present,
    module_input_op_spec,
    module_output_spec,
    needs_capture,
    op_output_readable,
)

_RESIDUAL_FACETS = ("resid_pre", "resid_mid", "resid_post")
_ATTENTION_CHILD_NAMES = ("attn", "attention", "self_attn", "self_attention")
_MLP_CHILD_NAMES = ("mlp", "feed_forward", "ffn", "intermediate", "output")


def _is_transformer_block(module: Any) -> bool:
    """Return whether a module record is a genuine transformer block.

    A block qualifies only on STRUCTURAL evidence: it must contain both an
    attention child and an MLP/feed-forward child. A class-name marker alone
    (``*Block*`` / ``*Layer*``) is neither necessary nor sufficient -- plain
    non-transformer modules routinely carry those names (a scaling ``*Layer*``,
    a conv ``*Block*``), so matching on the name fabricated resid_pre/mid/post
    facets on modules that have no residual stream at all. Requiring the real
    attention + MLP substructure keeps the recipe honest.

    Parameters
    ----------
    module:
        Candidate module record.

    Returns
    -------
    bool
        Whether the recipe should attempt residual stream facets.
    """

    children = set(getattr(module, "address_children", ()) or ())
    local_children = {str(child).rsplit(".", maxsplit=1)[-1] for child in children}
    has_attention = any(name in local_children for name in _ATTENTION_CHILD_NAMES)
    has_mlp = any(name in local_children for name in _MLP_CHILD_NAMES)
    return has_attention and has_mlp


@register(predicate=_is_transformer_block, target_scope="module", facets=_RESIDUAL_FACETS)
def transformer_residuals(module: Any) -> dict[str, Any]:
    """Return residual stream facets for transformer-like block modules.

    Parameters
    ----------
    module:
        TorchLens module record.

    Returns
    -------
    dict[str, Any]
        Residual facets anchored to captured ops where available.
    """

    result: dict[str, Any] = {}
    add_if_present(result, "resid_pre", module_input_op_spec(module, "transformer_residuals"))
    add_if_present(result, "resid_mid", _resid_mid_spec(module))
    add_if_present(result, "resid_post", module_output_spec(module, "transformer_residuals"))
    return result


def _resid_mid_spec(module: Any) -> FacetSpec | AbsenceReason | None:
    """Return a spec for the post-attention residual add inside a block.

    The residual midpoint is only well defined when an add op genuinely consumes
    the block's attention output. The previous "first add op" fallback anchored
    resid_mid to an arbitrary add when no attention-consuming add existed; on
    single-add blocks that add is also the block output, so resid_mid collapsed
    onto resid_post (a degenerate, meaningless midpoint). We now return absence
    (``None``) rather than fabricate a midpoint that is not the post-attention
    residual.

    Parameters
    ----------
    module:
        TorchLens module record.

    Returns
    -------
    FacetSpec | AbsenceReason | None
        Op-anchored spec for the real post-attention add, a needs-capture
        reason when that add was not saved, or ``None`` when no such add exists.
    """

    trace = getattr(module, "trace", None)
    if trace is None:
        return None
    attention_outputs = {_base_label(label) for label in _attention_output_labels(module)}
    if not attention_outputs:
        return None
    for label in _module_op_labels(module):
        try:
            op = trace.ops[label]
        except (KeyError, TypeError):
            continue
        if str(getattr(op, "func_name", "")) not in {"add", "__add__", "add_"}:
            continue
        parents = {_base_label(parent) for parent in (getattr(op, "parents", ()) or ())}
        if not parents.intersection(attention_outputs):
            continue
        if not op_output_readable(op):
            return needs_capture(
                f"residual midpoint op {getattr(op, 'label', '<unknown>')!r} was not saved",
                f"save=... including {getattr(op, 'label', 'the residual midpoint')!r}",
            )
        return FacetSpec.from_home(op, home_kind="op", recipe_id="transformer_residuals")
    return None


def _base_label(label: Any) -> str:
    """Return an op label with any trailing ``:<pass>`` recurrence suffix removed.

    Module ``output_ops`` are pass-qualified (``"linear_1_1:1"``) while an op's
    ``parents`` are bare (``"linear_1_1"``). Comparing them raw never matched, so
    the post-attention add was previously only found via the removed first-add
    fallback. Normalizing both sides to the base label makes the attention-output
    parent match actually fire.
    """

    text = str(label)
    base, sep, tail = text.rpartition(":")
    if sep and tail.isdigit():
        return base
    return text


def _module_op_labels(module: Any) -> list[str]:
    """Return op labels contained by a module record."""

    try:
        return list(module._op_labels())
    except (AttributeError, TypeError, ValueError):
        return list(getattr(module, "output_ops", ()) or ())


def _attention_output_labels(module: Any) -> set[str]:
    """Return output op labels for direct attention children."""

    trace = getattr(module, "trace", None)
    if trace is None:
        return set()
    labels: set[str] = set()
    for child_address in getattr(module, "address_children", ()) or ():
        local_name = str(child_address).rsplit(".", maxsplit=1)[-1]
        if local_name not in _ATTENTION_CHILD_NAMES:
            continue
        try:
            child = trace.modules[child_address]
        except (KeyError, ValueError):
            continue
        labels.update(str(label) for label in getattr(child, "output_ops", ()) or ())
    return labels
