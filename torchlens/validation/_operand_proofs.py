"""Bounded saved-operand proofs used by perturbation validation.

Registry ownership stays in exemptions; imports of its shared primitives are
local so loading either module cannot create an initialization cycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from ..data_classes.op import Op
    from ..data_classes.trace import Trace
    from .exemptions import PosthocPerturbDecision


def _setitem_index_has_single_valid_position(layer: Op, parents: list[str]) -> bool:
    """Prove an integer index selects an axis with no alternate valid position."""

    from .exemptions import _INDEX_DOMAIN_INT_DTYPES, _perturbed_parent_arg_positions

    args: tuple[Any, ...] = tuple(layer.saved_args or ())
    positions = _perturbed_parent_arg_positions(layer, parents)
    if len(args) < 2 or len(positions) != 1 or not isinstance(args[0], torch.Tensor):
        return False
    position = next(iter(positions))
    axis = 0
    index = args[1]
    if isinstance(position, tuple) and len(position) == 2 and position[0] == 1:
        axis = position[1]
        if not isinstance(axis, int) or not isinstance(index, tuple):
            return False
        if not 0 <= axis < len(index) or any(
            not isinstance(item, (torch.Tensor, slice, int))
            or (isinstance(item, torch.Tensor) and item.dtype not in _INDEX_DOMAIN_INT_DTYPES)
            for item in index[:axis]
        ):
            return False
        index = index[axis]
    elif position != 1:
        return False
    return (
        axis < args[0].ndim
        and args[0].shape[axis] == 1
        and isinstance(index, torch.Tensor)
        and index.dtype in _INDEX_DOMAIN_INT_DTYPES
        and index.numel() > 0
        and bool(((index == 0) | (index == -1)).all())
    )


def _check_scatter_reduce_exempt(self: Trace, layer: Op, parents: list[str]) -> bool:
    """Prove a fully covered reduction destination is excluded by include_self."""

    from .exemptions import _perturbed_parent_arg_positions, _scatter_index_fully_overwrites_dim

    del self
    args: tuple[Any, ...] = tuple(layer.saved_args or ())
    kwargs = layer.saved_kwargs or {}
    if (
        _perturbed_parent_arg_positions(layer, parents) != {0}
        or kwargs.get("include_self", True) is not False
        or len(args) < 3
    ):
        return False
    destination, dim, index = args[:3]
    if (
        not isinstance(destination, torch.Tensor)
        or not isinstance(dim, int)
        or not -destination.ndim <= dim < destination.ndim
    ):
        return False
    dim %= destination.ndim
    return isinstance(index, torch.Tensor) and _scatter_index_fully_overwrites_dim(
        destination, dim, index
    )


def _posthoc_absorbing_operand_decision(
    layer: Op, parents: list[str], args: tuple[Any, ...]
) -> PosthocPerturbDecision:
    """Prove operand irrelevance from a saved absorbing operand or total mask."""

    from .exemptions import (
        PosthocPerturbDecision,
        _is_all_zero_value,
        _perturbed_parent_arg_positions,
    )

    positions = _perturbed_parent_arg_positions(layer, parents)
    output = layer.out
    if (
        layer.func_name in {"__or__", "__ior__", "logical_or", "logical_or_", "bitwise_or"}
        and len(args) == 2
        and positions in ({0}, {1})
    ):
        other = args[1 if positions == {0} else 0]
        if isinstance(other, torch.Tensor) and other.dtype == torch.bool and bool(other.all()):
            return PosthocPerturbDecision(
                True,
                "boolean_true_annihilator",
                "the independent all-true boolean operand absorbs OR",
            )
    if (
        layer.func_name == "scaled_dot_product_attention"
        and len(args) >= 3
        and positions
        and positions.issubset({0, 1})
        and (
            _attention_values_absorb_scores(layer, args)
            or (positions == {0} and _is_all_zero_value(args[1]))
            or (positions == {1} and _is_all_zero_value(args[0]))
        )
        and isinstance(output, torch.Tensor)
        and bool(torch.isfinite(output).all())
    ):
        return PosthocPerturbDecision(
            True,
            "attention_value_absorption",
            "an independent zero score operand, or zero/constant attention values, absorbs the parent",
        )
    if (
        layer.func_name in {"masked_scatter", "masked_scatter_"}
        and len(args) >= 3
        and positions == {0}
        and isinstance(args[1], torch.Tensor)
        and args[1].dtype == torch.bool
        and isinstance(output, torch.Tensor)
        and bool(torch.broadcast_to(args[1], output.shape).all())
    ):
        return PosthocPerturbDecision(
            True,
            "full_destination_overwrite",
            "the broadcast mask replaces every destination value",
        )
    return PosthocPerturbDecision(False, "not_absorbing_operand")


def _attention_values_absorb_scores(layer: Op, args: tuple[Any, ...]) -> bool:
    """Prove zero V, or identical V at every unmasked key with no dropout."""
    from .exemptions import _is_all_zero_value

    value = args[2]
    if _is_all_zero_value(value):
        return True
    kwargs = layer.saved_kwargs or {}
    mask = kwargs.get("attn_mask", args[3] if len(args) > 3 else None)
    dropout = kwargs.get("dropout_p", args[4] if len(args) > 4 else 0.0)
    return (
        mask is None
        and dropout == 0
        and isinstance(value, torch.Tensor)
        and value.ndim >= 2
        and value.shape[-2] > 0
        and bool((value == value[..., :1, :]).all())
    )


def _singleton_softmax_decision(
    layer: Op, parents: list[str], args: tuple[Any, ...]
) -> PosthocPerturbDecision:
    """Prove softmax over one finite scalar is the constant one."""
    from .exemptions import PosthocPerturbDecision, _perturbed_parent_arg_positions

    dim = layer.saved_kwargs.get("dim", args[1] if len(args) > 1 else None)
    if (
        layer.func_name in {"softmax", "log_softmax"}
        and args
        and _perturbed_parent_arg_positions(layer, parents) == {0}
        and isinstance(args[0], torch.Tensor)
        and isinstance(dim, int)
        and -args[0].ndim <= dim < args[0].ndim
        and args[0].shape[dim] == 1
        and bool(torch.isfinite(args[0]).all())
        and isinstance(layer.out, torch.Tensor)
        and bool(torch.isfinite(layer.out).all())
    ):
        return PosthocPerturbDecision(
            True, "singleton_normalization", "the normalization axis has one finite element"
        )
    return PosthocPerturbDecision(False, "not_singleton_normalization")


def _einsum_zero_annihilator_decision(
    layer: Op, layers_to_perturb: list[str], args: tuple[Any, ...]
) -> PosthocPerturbDecision:
    """Prove an einsum input irrelevant when a different operand is all zero.

    Parameters
    ----------
    layer, layers_to_perturb, args:
        Captured contraction, the perturbed parent, and its saved arguments.

    Returns
    -------
    PosthocPerturbDecision
        A zero-product proof for equation-form calls, otherwise no exemption.
    """

    from .exemptions import PosthocPerturbDecision, _is_all_zero_value

    not_proved = PosthocPerturbDecision(False, "not_einsum_zero_annihilator")
    if len(layers_to_perturb) != 1:
        return not_proved
    if not _is_all_zero_value(layer.out):
        return not_proved
    operands = _einsum_operand_slots(args)
    parent_map = layer.parent_arg_positions.get("args", {})
    perturbed = layers_to_perturb[0]
    if not any(parent_map.get(slot) == perturbed for slot, _ in operands):
        return not_proved
    if any(
        parent_map.get(slot) != perturbed and _is_all_zero_value(value) for slot, value in operands
    ):
        return PosthocPerturbDecision(
            True,
            "multiplicative_zero_annihilator",
            "einsum contraction contains a different, non-perturbed all-zero operand",
        )
    return not_proved


def _einsum_operand_slots(
    args: tuple[Any, ...],
) -> list[tuple[int | tuple[int, int], torch.Tensor]]:
    """Parse equation-form einsum operands and their concrete argument slots.

    Parameters
    ----------
    args:
        Captured arguments, using either packed or variadic operands.

    Returns
    -------
    list of tuple
        Tensor operands with their slots, or no entries for unsupported forms.
    """
    if len(args) < 2 or not isinstance(args[0], str):
        return []
    packed = len(args) == 2 and isinstance(args[1], (list, tuple))
    operands = args[1] if packed else args[1:]
    result: list[tuple[int | tuple[int, int], torch.Tensor]] = []
    for index, value in enumerate(operands):
        if not isinstance(value, torch.Tensor):
            return []
        result.append(((1, index) if packed else index + 1, value))
    return result


def _conv_zero_operand_decision(
    layer: Op,
    layers_to_perturb: list[str],
    args: tuple[Any, ...],
) -> PosthocPerturbDecision:
    """Return a proof for convolution by a different all-zero operand.

    Parameters
    ----------
    layer:
        Convolution op whose unchanged perturbed replay is being classified.
    layers_to_perturb:
        Parent labels selected for perturbation.
    args:
        Saved positional arguments for ``layer``.

    Returns
    -------
    PosthocPerturbDecision
        Exempt decision for zero kernels annihilating the input, or zero input
        annihilating the kernel. The affine bias remains value-sensitive.
    """

    from .exemptions import (
        PosthocPerturbDecision,
        _is_all_zero_value,
        _perturbed_parent_arg_positions,
    )

    if len(layers_to_perturb) != 1:
        return PosthocPerturbDecision(False, "not_conv_zero_weight_input")
    positions = _perturbed_parent_arg_positions(layer, layers_to_perturb)
    other_position = 1 if positions == {0} else 0 if positions == {1} else None
    if other_position is not None and _is_all_zero_value(args[other_position]):
        return PosthocPerturbDecision(
            True,
            "multiplicative_zero_annihilator",
            "convolution operand is annihilated by the other all-zero saved operand",
        )
    return PosthocPerturbDecision(False, "not_conv_zero_weight_input")


def _linear_zero_weight_input_decision(
    layer: Op,
    layers_to_perturb: list[str],
    args: tuple[Any, ...],
) -> PosthocPerturbDecision:
    """Return a proof decision for a linear input under all-zero weights.

    Parameters
    ----------
    layer:
        Linear op whose unchanged perturbed replay is being classified.
    layers_to_perturb:
        Parent labels selected for perturbation.
    args:
        Saved positional arguments for ``layer``.

    Returns
    -------
    PosthocPerturbDecision
        Exempt decision when the perturbed parent is the input and the saved
        weight matrix is all zero.
    """

    from .exemptions import (
        PosthocPerturbDecision,
        _is_all_zero_value,
    )

    if len(layers_to_perturb) != 1:
        return PosthocPerturbDecision(False, "not_linear_zero_weight_input")
    arg_positions = layer.parent_arg_positions.get("args", {})
    perturbed_label = layers_to_perturb[0]
    if arg_positions.get(0) != perturbed_label:
        return PosthocPerturbDecision(False, "not_linear_zero_weight_input")
    if _is_all_zero_value(args[1]):
        return PosthocPerturbDecision(
            True,
            "multiplicative_zero_annihilator",
            "linear input is annihilated by an all-zero saved weight matrix",
        )
    return PosthocPerturbDecision(False, "not_linear_zero_weight_input")
