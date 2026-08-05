"""Regression tests for the unattributed-tensor-argument witness false positive.

Context
-------
The postprocess "no graph/source provenance" witness
(``torchlens.postprocess._warn_unattributed_tensor_args``, fed by
``torchlens.backends.torch.ops._unattributed_tensor_arg_positions``) exists to
catch capture gaps: tensors that flow into an op with no traced provenance. Its
branch (2) additionally flags a *provenanced* tensor whose producer label is not
one of the op's recorded parent edges.

That branch (2) mis-fired on dynamic-shape families: a scalar tensor that is
fully traced (``has_known_provenance``) but is consumed as a **size/shape**
argument (a ``torch.zeros`` size dim, a ``view``/``reshape``/``as_strided`` size
arg, a ``new_zeros`` size element) is deliberately excluded from the op's
recorded parents, so branch (2) wrongly promoted it to an error. This broke
``test_packed_sequence`` (LSTM internal ``h0``/``c0`` allocation) and
``test_longformer`` (windowed-attention dynamic dims).

The fix narrows branch (2) so it suppresses a provenanced tensor ONLY at a
schema-confirmed non-operand (size/shape) position. The authority is the ATen
**schema** (ground truth), not the local ``FUNC_ARG_SPECS`` -- so a dropped edge
at a genuine Tensor-operand position (including one caused by an under-specified
spec) still fires. An UN-provenanced tensor at any position is caught earlier by
branch (1) and is unaffected by the narrowing, so no capture gap can be masked.
"""

import warnings

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.backends.torch.ops import _arg_position_is_tensor_operand

_PROVENANCE_MATCH = "no graph/source provenance"

# Foreign tensors constructed OUTSIDE any capture session carry no TorchLens
# provenance at all -- the "real leak" the witness must never miss.
_FOREIGN_SIZE_A = torch.tensor(4)
_FOREIGN_SIZE_B = torch.tensor(6)
_FOREIGN_OPERAND = torch.ones(24)


class _ProvenancedDynamicSize(nn.Module):
    """Feed a fully-traced scalar tensor as a ``reshape`` size argument."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape ``x`` using a size derived from a traced op.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            ``x`` reshaped with a dynamically computed final dimension.
        """

        cols = torch.tensor([x.shape[1] * x.shape[2]])  # traced producer
        return x.reshape(x.shape[0], cols[0])  # cols[0] is provenanced, at arg2 (size)


class _ForeignSize(nn.Module):
    """Feed UN-provenanced foreign scalar tensors as ``reshape`` size args."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape a flattened view with foreign size tensors.

        Parameters
        ----------
        x:
            Input tensor with 24 elements.

        Returns
        -------
        torch.Tensor
            ``x`` reshaped to ``(4, 6)`` via foreign size tensors.
        """

        return x.flatten().reshape(_FOREIGN_SIZE_A, _FOREIGN_SIZE_B)


class _ForeignOperand(nn.Module):
    """Add an UN-provenanced foreign tensor at a genuine data-operand position."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add a foreign tensor to a flattened view.

        Parameters
        ----------
        x:
            Input tensor with 24 elements.

        Returns
        -------
        torch.Tensor
            The elementwise sum with the foreign operand.
        """

        return x.flatten() + _FOREIGN_OPERAND  # foreign at add arg1 (data operand)


def _provenance_warnings(model: nn.Module, x: torch.Tensor) -> list[str]:
    """Return every provenance-witness warning message raised while tracing.

    Parameters
    ----------
    model:
        Module to trace.
    x:
        Model input.

    Returns
    -------
    list[str]
        Messages of captured provenance-witness ``UserWarning`` instances.
    """

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tl.trace(model, x)
    return [str(w.message) for w in caught if _PROVENANCE_MATCH in str(w.message)]


def test_provenanced_dynamic_size_arg_no_false_positive() -> None:
    """A fully-traced scalar used as a size argument must NOT be flagged."""

    assert _provenance_warnings(_ProvenancedDynamicSize().eval(), torch.randn(2, 3, 4)) == []


def test_foreign_unprovenanced_size_arg_still_flags() -> None:
    """An un-provenanced foreign tensor at a size position MUST still fire.

    Guards the no-false-negative direction: the narrowing gates on
    ``has_known_provenance`` (branch 1), so a real leak at a size position is
    never masked.
    """

    with pytest.warns(UserWarning, match=_PROVENANCE_MATCH):
        tl.trace(_ForeignSize().eval(), torch.randn(2, 3, 4))


def test_foreign_unprovenanced_operand_arg_still_flags() -> None:
    """An un-provenanced foreign tensor at a data-operand position MUST fire."""

    with pytest.warns(UserWarning, match=_PROVENANCE_MATCH):
        tl.trace(_ForeignOperand().eval(), torch.randn(2, 3, 4))


def test_arg_position_operand_classifier_contract() -> None:
    """The schema-authoritative operand classifier separates the two classes.

    This is the exact new condition gating branch (2). Data-operand slots must
    classify as operand (``True`` -> keep flagging); schema-confirmed size/shape
    slots must classify as non-operand (``False`` -> suppress). ``polygamma``
    arg1 is the incomplete-spec guard: the ATen schema types it ``Tensor`` even
    though a corrupted ``FUNC_ARG_SPECS`` could drop it, so the witness still
    fires there. Uncertain authority (variadic ops, no ATen schema) fails safe.
    """

    # Data-operand positions -> keep flagging.
    assert _arg_position_is_tensor_operand("reshape", "arg0") is True
    assert _arg_position_is_tensor_operand("view", "arg0") is True
    assert _arg_position_is_tensor_operand("new_zeros", "arg0") is True
    assert _arg_position_is_tensor_operand("as_strided", "arg0") is True
    assert _arg_position_is_tensor_operand("add", "arg1") is True
    assert _arg_position_is_tensor_operand("cat", "arg0.1") is True
    # Incomplete-spec guard: schema types polygamma self (arg1) as Tensor.
    assert _arg_position_is_tensor_operand("polygamma", "arg1") is True

    # Schema-confirmed size/shape positions -> suppress benign false positive.
    assert _arg_position_is_tensor_operand("reshape", "arg2") is False
    assert _arg_position_is_tensor_operand("view", "arg2") is False
    assert _arg_position_is_tensor_operand("zeros", "arg0") is False
    assert _arg_position_is_tensor_operand("zeros", "arg1") is False
    assert _arg_position_is_tensor_operand("new_zeros", "arg1.1") is False
    assert _arg_position_is_tensor_operand("as_strided", "kw:size.1") is False

    # Uncertain authority -> fail safe (keep flagging).
    assert _arg_position_is_tensor_operand("__getitem__", "arg1") is True
    assert _arg_position_is_tensor_operand("einsum", "arg2") is True
