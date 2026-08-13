"""Order-aligned activation stacking for completed traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from .trace import Trace

    _TraceMixinBase = Trace
else:
    _TraceMixinBase = object


@dataclass(frozen=True)
class StackedActivations:
    """Saved activation rows with index-aligned operation provenance.

    Attributes
    ----------
    tensor:
        Tensor with shape ``(n_matches, *activation_shape)``.
    labels:
        Layer labels aligned one-to-one with the tensor's leading dimension.
    """

    tensor: torch.Tensor
    labels: tuple[str, ...]

    def __len__(self) -> int:
        """Return the number of stacked activation rows.

        Returns
        -------
        int
            Number of matched operations.
        """

        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor]:
        """Return one label and its aligned activation row.

        Parameters
        ----------
        index:
            Leading-dimension row index.

        Returns
        -------
        tuple[str, torch.Tensor]
            Layer label and activation tensor row.
        """

        return self.labels[index], self.tensor[index]

    def __repr__(self) -> str:
        """Return a compact plain-text representation.

        Returns
        -------
        str
            Result type, row count, and tensor shape.
        """

        return f"StackedActivations(n={len(self)}, shape={tuple(self.tensor.shape)})"


class TraceStackMixin(_TraceMixinBase):
    """Mixin providing order-aligned saved activation stacks."""

    def stack(self: Trace, selector: Any) -> StackedActivations:
        """Stack selector-matched saved outputs in recorded execution order.

        Selector evaluation reuses :meth:`Trace.find_sites`. Matching operations
        are then sorted explicitly by their recorded ``ordinal_index``; selector
        iteration order is never treated as authoritative. The resulting rows can
        feed ``torchlens.stats.cka``, across-layer PCA, or ``torchlens.repgeom.rdm``.

        Parameters
        ----------
        selector:
            TorchLens selector accepted by :meth:`Trace.find_sites`.

        Returns
        -------
        StackedActivations
            Tensor rows and their aligned layer labels.

        Raises
        ------
        ValueError
            If no operations match, execution ordinals are invalid, a payload was
            not saved, a primary output is not one tensor, or shapes differ.
        """

        max_fanout = max(1, len(self.layer_list))
        sites = tuple(self.find_sites(selector, max_fanout=max_fanout))
        if not sites:
            raise ValueError(f"trace.stack selector {selector!r} matched 0 sites.")

        ordered_pairs: list[tuple[int, Any]] = []
        for site in sites:
            ordinal = getattr(site, "ordinal_index", None)
            if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
                raise ValueError("trace.stack requires matched operations with recorded ordinals.")
            ordered_pairs.append((ordinal, site))
        if len({ordinal for ordinal, _ in ordered_pairs}) != len(ordered_pairs):
            raise ValueError("trace.stack found duplicate recorded execution ordinals.")
        ordered_sites = [site for _, site in sorted(ordered_pairs, key=lambda pair: pair[0])]

        tensors: list[torch.Tensor] = []
        labels: list[str] = []
        expected_shape: tuple[int, ...] | None = None
        expected_label: str | None = None
        for site in ordered_sites:
            label = str(getattr(site, "layer_label", getattr(site, "label", "<unknown>")))
            if not bool(getattr(site, "has_saved_activation", False)):
                raise ValueError(f"{label!r} has no saved activation payload.")
            output = getattr(site, "out", None)
            if not isinstance(output, torch.Tensor):
                raise ValueError(
                    f"trace.stack requires a single tensor primary saved out; {label!r} "
                    f"has {type(output).__name__}."
                )
            shape = tuple(output.shape)
            if expected_shape is None:
                expected_shape = shape
                expected_label = label
            elif shape != expected_shape:
                raise ValueError(
                    "trace.stack refuses shape mismatch between "
                    f"{expected_label!r} {expected_shape} and {label!r} {shape}."
                )
            tensors.append(output)
            labels.append(label)

        return StackedActivations(torch.stack(tensors, dim=0), tuple(labels))


__all__ = ["StackedActivations", "TraceStackMixin"]
