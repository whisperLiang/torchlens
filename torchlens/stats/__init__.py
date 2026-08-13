"""Streaming statistics for out aggregation."""

from __future__ import annotations

import gc
import heapq
import math
import random
import warnings
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch
from torch import nn

from ..intervention.errors import MultiMatchWarning


class StreamingStat(Protocol):
    """Protocol implemented by all streaming statistic accumulators."""

    name: str | None

    def update(self, value: Any) -> None:
        """Update the accumulator with one batch value."""

    def result(self) -> Any:
        """Return the finalized statistic value."""


def _as_float_tensor(value: Any) -> torch.Tensor:
    """Return ``value`` as a detached CPU float tensor.

    Parameters
    ----------
    value:
        Tensor-like value.

    Returns
    -------
    torch.Tensor
        Flattened CPU float tensor.
    """

    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    return torch.as_tensor(value, dtype=torch.float64).reshape(-1)


class Mean:
    """Running mean accumulator."""

    def __init__(self, name: str | None = None) -> None:
        """Initialize the accumulator.

        Parameters
        ----------
        name:
            Optional metric name.
        """

        self.name = name
        self._count = 0
        self._mean: torch.Tensor | None = None

    def update(self, value: Any) -> None:
        """Update the running mean.

        Parameters
        ----------
        value:
            Tensor-like batch value.
        """

        tensor = _as_float_tensor(value)
        if tensor.numel() == 0:
            return
        batch_mean = tensor.mean()
        batch_count = int(tensor.numel())
        if self._mean is None:
            self._mean = batch_mean
            self._count = batch_count
            return
        total = self._count + batch_count
        self._mean = self._mean + (batch_mean - self._mean) * (batch_count / total)
        self._count = total

    def result(self) -> float:
        """Return the finalized mean.

        Returns
        -------
        float
            Running mean, or NaN when no values were seen.
        """

        if self._mean is None:
            return math.nan
        return float(self._mean.item())


class Norm:
    """Running mean of per-update tensor norms.

    Each ``update()`` reduces the provided batch to one scalar norm before the
    running mean is updated. This is intentionally different from a global norm
    over all elements seen across all updates, so regrouping the same values
    into different update batches can change the reported result.
    """

    def __init__(self, p: float = 2.0, name: str | None = None) -> None:
        """Initialize the norm accumulator.

        Parameters
        ----------
        p:
            Norm order passed to ``torch.linalg.vector_norm``.
        name:
            Optional metric name.
        """

        self.name = name
        self.p = float(p)
        self._mean = Mean()

    def update(self, value: Any) -> None:
        """Update the running norm mean."""

        tensor = _as_float_tensor(value)
        if tensor.numel() == 0:
            return
        self._mean.update(torch.linalg.vector_norm(tensor, ord=self.p))

    def result(self) -> float:
        """Return the finalized mean norm."""

        return self._mean.result()


class Quantile:
    """Reservoir-sampling running quantile estimator."""

    def __init__(
        self,
        quantiles: Iterable[float] = (0.5, 0.95, 0.99),
        name: str | None = None,
        reservoir_size: int = 8192,
    ) -> None:
        """Initialize the estimator.

        Parameters
        ----------
        quantiles:
            Quantiles in ``[0, 1]`` to estimate.
        name:
            Optional metric name.
        reservoir_size:
            Maximum sampled values retained in memory.
        """

        self.name = name
        self.quantiles = tuple(float(q) for q in quantiles)
        self.reservoir_size = int(reservoir_size)
        self._seen = 0
        self._reservoir: list[float] = []

    def update(self, value: Any) -> None:
        """Update the reservoir.

        Parameters
        ----------
        value:
            Tensor-like batch value.
        """

        for item in _as_float_tensor(value).tolist():
            self._seen += 1
            if len(self._reservoir) < self.reservoir_size:
                self._reservoir.append(float(item))
                continue
            replacement = random.randint(0, self._seen - 1)
            if replacement < self.reservoir_size:
                self._reservoir[replacement] = float(item)

    def result(self) -> dict[float, float]:
        """Return finalized quantile estimates.

        Returns
        -------
        dict[float, float]
            Mapping from requested quantile to estimated value.
        """

        if not self._reservoir:
            return dict.fromkeys(self.quantiles, math.nan)
        tensor = torch.tensor(self._reservoir, dtype=torch.float64)
        return {q: float(torch.quantile(tensor, q).item()) for q in self.quantiles}


class TopK:
    """Streaming top-k value tracker."""

    def __init__(self, k: int = 10, name: str | None = None) -> None:
        """Initialize the tracker.

        Parameters
        ----------
        k:
            Number of largest scalar values to retain.
        name:
            Optional metric name.
        """

        self.name = name
        self.k = int(k)
        self._heap: list[float] = []

    def update(self, value: Any) -> None:
        """Update the tracked top-k values.

        Parameters
        ----------
        value:
            Tensor-like batch value.
        """

        if self.k <= 0:
            return
        for item in _as_float_tensor(value).tolist():
            scalar = float(item)
            if len(self._heap) < self.k:
                heapq.heappush(self._heap, scalar)
            elif scalar > self._heap[0]:
                heapq.heapreplace(self._heap, scalar)

    def result(self) -> list[float]:
        """Return top values in descending order.

        Returns
        -------
        list[float]
            Retained top-k values.
        """

        return sorted(self._heap, reverse=True)


class Covariance:
    """Running covariance matrix accumulator."""

    def __init__(self, name: str | None = None) -> None:
        """Initialize the accumulator.

        Parameters
        ----------
        name:
            Optional metric name.
        """

        self.name = name
        self._count = 0
        self._mean: torch.Tensor | None = None
        self._m2: torch.Tensor | None = None

    def update(self, value: Any) -> None:
        """Update covariance from one batch.

        Parameters
        ----------
        value:
            Tensor-like batch. One-dimensional inputs are treated as one row.
        """

        tensor = torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float64)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.reshape(tensor.shape[0], -1)
        if self._mean is not None and tensor.shape[1] != self._mean.numel():
            raise ValueError("Covariance feature dimensions cannot change across updates.")
        for row in tensor:
            self._count += 1
            if self._mean is None:
                self._mean = torch.zeros_like(row)
                self._m2 = torch.zeros((row.numel(), row.numel()), dtype=torch.float64)
            assert self._m2 is not None
            delta = row - self._mean
            self._mean = self._mean + delta / self._count
            self._m2 = self._m2 + torch.outer(delta, row - self._mean)

    def result(self) -> torch.Tensor:
        """Return the finalized covariance matrix.

        Returns
        -------
        torch.Tensor
            Covariance matrix.
        """

        if self._m2 is None:
            return torch.empty((0, 0), dtype=torch.float64)
        if self._count < 2:
            return torch.zeros_like(self._m2)
        return self._m2 / (self._count - 1)


def _as_feature_matrix(value: Any) -> torch.Tensor:
    """Return ``value`` as a detached CPU float64 feature matrix.

    Parameters
    ----------
    value:
        Tensor-like batch. One-dimensional inputs are treated as one row.

    Returns
    -------
    torch.Tensor
        A two-dimensional ``(n_rows, n_features)`` tensor.
    """

    tensor = torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float64)
    if tensor.ndim == 0:
        raise ValueError("A feature batch must have at least one dimension.")
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor.reshape(tensor.shape[0], -1)


class CrossCovariance:
    """Running cross-covariance matrix accumulator.

    The accumulator retains only feature-sized running means and the cross
    second moment. Inputs are converted to detached CPU ``float64`` tensors.
    """

    def __init__(self, name: str | None = None) -> None:
        """Initialize the accumulator.

        Parameters
        ----------
        name:
            Optional metric name.
        """

        self.name = name
        self._count = 0
        self._mean_a: torch.Tensor | None = None
        self._mean_b: torch.Tensor | None = None
        self._m2: torch.Tensor | None = None

    def update(self, a: Any, b: Any) -> None:
        """Update cross-covariance from one paired batch.

        Parameters
        ----------
        a:
            First tensor-like batch, with rows as observations.
        b:
            Second tensor-like batch, with rows as observations.

        Raises
        ------
        ValueError
            If the batches have different row counts or a feature dimension
            changes across updates.
        """

        matrix_a = _as_feature_matrix(a)
        matrix_b = _as_feature_matrix(b)
        if matrix_a.shape[0] != matrix_b.shape[0]:
            raise ValueError(
                "CrossCovariance requires matched row counts; "
                f"got {matrix_a.shape[0]} and {matrix_b.shape[0]}."
            )
        if self._mean_a is not None and matrix_a.shape[1] != self._mean_a.numel():
            raise ValueError("CrossCovariance feature dimensions cannot change across updates.")
        if self._mean_b is not None and matrix_b.shape[1] != self._mean_b.numel():
            raise ValueError("CrossCovariance feature dimensions cannot change across updates.")
        for row_a, row_b in zip(matrix_a, matrix_b, strict=True):
            self._count += 1
            if self._mean_a is None or self._mean_b is None:
                self._mean_a = torch.zeros_like(row_a)
                self._mean_b = torch.zeros_like(row_b)
                self._m2 = torch.zeros((row_a.numel(), row_b.numel()), dtype=torch.float64)
            assert self._m2 is not None
            delta_a = row_a - self._mean_a
            delta_b = row_b - self._mean_b
            self._mean_a = self._mean_a + delta_a / self._count
            self._mean_b = self._mean_b + delta_b / self._count
            self._m2 = self._m2 + torch.outer(delta_a, row_b - self._mean_b)

    def result(self) -> torch.Tensor:
        """Return the finalized sample cross-covariance matrix.

        Returns
        -------
        torch.Tensor
            Cross-covariance with shape ``(d_a, d_b)``. Fewer than two rows
            produce a zero matrix of the established feature shape.
        """

        if self._m2 is None:
            return torch.empty((0, 0), dtype=torch.float64)
        if self._count < 2:
            return torch.zeros_like(self._m2)
        return self._m2 / (self._count - 1)


class CKA:
    r"""Streaming linear centered kernel alignment accumulator.

    Linear CKA is

    .. math::

        \operatorname{CKA}(A, B) =
        \frac{\lVert C_{AB}\rVert_F^2}
        {\lVert C_{AA}\rVert_F\,\lVert C_{BB}\rVert_F}.

    Only feature-sized covariance terms are retained. If either input has
    zero variance, ``result()`` returns NaN because the alignment denominator
    is zero. This follows the linear CKA formulation of Kornblith et al. (2019).
    """

    def __init__(self, name: str | None = None) -> None:
        """Initialize the accumulator.

        Parameters
        ----------
        name:
            Optional metric name.
        """

        self.name = name
        self._cross = CrossCovariance()
        self._covariance_a = Covariance()
        self._covariance_b = Covariance()

    def update(self, a: Any, b: Any) -> None:
        """Update linear CKA from one paired batch.

        Parameters
        ----------
        a:
            First tensor-like batch, with rows as observations.
        b:
            Second tensor-like batch, with rows as observations.
        """

        matrix_a = _as_feature_matrix(a)
        matrix_b = _as_feature_matrix(b)
        self._cross.update(matrix_a, matrix_b)
        self._covariance_a.update(matrix_a)
        self._covariance_b.update(matrix_b)

    def result(self) -> float:
        """Return the finalized linear CKA value.

        Returns
        -------
        float
            Linear CKA, or NaN when either representation has zero variance.
        """

        cross = self._cross.result()
        covariance_a = self._covariance_a.result()
        covariance_b = self._covariance_b.result()
        numerator = torch.linalg.matrix_norm(cross, ord="fro").square()
        denominator = torch.linalg.matrix_norm(covariance_a, ord="fro") * torch.linalg.matrix_norm(
            covariance_b, ord="fro"
        )
        if denominator.item() == 0.0:
            return math.nan
        return float((numerator / denominator).item())


def cka(a: Any, b: Any) -> float:
    r"""Compute one-shot linear centered kernel alignment.

    Linear CKA is

    .. math::

        \operatorname{CKA}(A, B) =
        \frac{\lVert C_{AB}\rVert_F^2}
        {\lVert C_{AA}\rVert_F\,\lVert C_{BB}\rVert_F}.

    This is the linear CKA measure described by Kornblith et al. (2019).
    Inputs are treated as ``(n_observations, n_features)`` matrices and all
    computation uses CPU ``float64``. A zero-variance input produces NaN.

    Parameters
    ----------
    a:
        First tensor-like representation.
    b:
        Second tensor-like representation with the same row count.

    Returns
    -------
    float
        Linear CKA value, or NaN for a degenerate zero-variance input.
    """

    matrix_a = _as_feature_matrix(a)
    matrix_b = _as_feature_matrix(b)
    if matrix_a.shape[0] != matrix_b.shape[0]:
        raise ValueError(
            f"CKA requires matched row counts; got {matrix_a.shape[0]} and {matrix_b.shape[0]}."
        )

    centered_a = matrix_a - matrix_a.mean(dim=0, keepdim=True)
    centered_b = matrix_b - matrix_b.mean(dim=0, keepdim=True)
    gram_a = centered_a @ centered_a.T
    gram_b = centered_b @ centered_b.T
    denominator = torch.linalg.matrix_norm(gram_a, ord="fro") * torch.linalg.matrix_norm(
        gram_b, ord="fro"
    )
    if denominator.item() == 0.0:
        return math.nan
    numerator = torch.sum(gram_a * gram_b)
    return float((numerator / denominator).item())


class PCA:
    """Simple incremental PCA backed by running covariance."""

    def __init__(self, n_components: int, name: str | None = None) -> None:
        """Initialize the estimator.

        Parameters
        ----------
        n_components:
            Number of principal components to return.
        name:
            Optional metric name.
        """

        self.name = name
        self.n_components = int(n_components)
        self._covariance = Covariance(name=name)

    def update(self, value: Any) -> None:
        """Update the PCA estimator.

        Parameters
        ----------
        value:
            Tensor-like batch.
        """

        self._covariance.update(value)

    def result(self) -> dict[str, torch.Tensor]:
        """Return components and explained variances.

        Returns
        -------
        dict[str, torch.Tensor]
            ``components`` and ``explained_variance`` tensors.
        """

        cov = self._covariance.result()
        if cov.numel() == 0:
            return {
                "components": torch.empty((0, 0), dtype=torch.float64),
                "explained_variance": torch.empty((0,), dtype=torch.float64),
            }
        values, vectors = torch.linalg.eigh(cov)
        order = torch.argsort(values, descending=True)[: self.n_components]
        return {"components": vectors[:, order].T, "explained_variance": values[order]}


class Aggregator:
    """Combine multiple streaming accumulators in one update pass."""

    def __init__(self, *stats: StreamingStat, name: str | None = None) -> None:
        """Initialize the combined aggregator.

        Parameters
        ----------
        *stats:
            Streaming statistic instances.
        name:
            Optional metric name.
        """

        self.name = name
        self.stats = tuple(stats)

    def update(self, value: Any) -> None:
        """Update each child statistic.

        Parameters
        ----------
        value:
            Tensor-like batch value.
        """

        for stat in self.stats:
            stat.update(value)

    def result(self) -> dict[str, Any]:
        """Return each child statistic result.

        Returns
        -------
        dict[str, Any]
            Mapping from child names/classes to finalized results.
        """

        results: dict[str, Any] = {}
        for index, stat in enumerate(self.stats):
            key = stat.name or type(stat).__name__
            if key in results:
                key = f"{key}_{index}"
            results[key] = stat.result()
        return results


def _resolve_metric_out(log: Any, metric_name: str) -> tuple[Any, Any]:
    """Resolve one metric input value and its matched site from a Trace.

    Parameters
    ----------
    log:
        Captured model log.
    metric_name:
        Layer selector or ``"output"``.

    Returns
    -------
    tuple[Any, Any]
        ``(value, site)`` where ``site`` is the matched layer object whose
        ``out`` supplied the value.
    """

    if metric_name == "output" and log.output_layers:
        site = log[log.output_layers[-1]]
        return site.out, site
    try:
        site = log[metric_name]
        return site.out, site
    except Exception:
        matches = _matching_layers(
            log,
            metric_name,
            require_grad=False,
        )
        if not matches:
            raise KeyError(f"No saved out matched metric {metric_name!r}.")
        _warn_on_ambiguous_metric_match(metric_name, matches)
        return matches[0].out, matches[0]


def _metric_value_from_log(log: Any, metric_name: str) -> Any:
    """Resolve one metric input value from a Trace.

    Parameters
    ----------
    log:
        Captured model log.
    metric_name:
        Layer selector or ``"output"``.

    Returns
    -------
    Any
        Tensor-like value for the metric.
    """

    value, _site = _resolve_metric_out(log, metric_name)
    return value


def _metric_grad_from_log(log: Any, metric_name: str) -> Any:
    """Resolve one gradient metric input value from a Trace."""

    if metric_name == "output":
        raise KeyError(f"No saved grad matched metric {metric_name!r}.")
    try:
        value = log[metric_name].grad
    except Exception:
        matches = _matching_layers(
            log,
            metric_name,
            require_grad=True,
        )
        if not matches:
            raise KeyError(f"No saved grad matched metric {metric_name!r}.")
        _warn_on_ambiguous_metric_match(metric_name, matches)
        value = matches[0].grad
    if value is None:
        raise KeyError(f"No saved grad matched metric {metric_name!r}.")
    return value


def _matching_layers(log: Any, metric_name: str, *, require_grad: bool) -> list[Any]:
    """Return saved layers whose labels contain ``metric_name``.

    Parameters
    ----------
    log:
        Captured model trace.
    metric_name:
        User-provided selector substring.
    require_grad:
        Whether to require gradient availability instead of saved activations.

    Returns
    -------
    list[Any]
        Matching saved layers in trace order.
    """

    attribute = "has_grad" if require_grad else "has_saved_activation"
    return [
        layer
        for layer in log.layer_list
        if metric_name in str(layer.layer_label) and bool(getattr(layer, attribute, False))
    ]


def _warn_on_ambiguous_metric_match(metric_name: str, matches: list[Any]) -> None:
    """Warn when a metric substring selector matches multiple saved sites.

    Parameters
    ----------
    metric_name:
        User-provided selector substring.
    matches:
        Matching saved sites in trace order.

    Returns
    -------
    None
        Emits a warning when multiple sites match.
    """

    if len(matches) < 2:
        return
    first_label = str(getattr(matches[0], "layer_label", metric_name))
    warnings.warn(
        (
            f"metric selector {metric_name!r} matched {len(matches)} sites; "
            f"using the first saved site {first_label!r}."
        ),
        MultiMatchWarning,
        stacklevel=3,
    )


def _split_batch_for_loss(batch: Any) -> tuple[Any, tuple[Any, ...]]:
    """Return model input and extra loss arguments from a dataloader batch."""

    if isinstance(batch, tuple) and len(batch) >= 2:
        return batch[0], tuple(batch[1:])
    if isinstance(batch, list) and len(batch) >= 2:
        return batch[0], tuple(batch[1:])
    return batch, ()


# Each sparse record batch leaves a cyclic capture-event graph behind, which
# only a cyclic collection can free. With the default gen-0 threshold that
# garbage gets promoted to gen 2 mid-batch and either accumulates for many
# batches or forces expensive full-heap collections. Raising the gen-0
# threshold for the duration of the fast loop keeps each batch's garbage in
# the young generations, where a per-batch ``gc.collect(1)`` frees it without
# scanning the full heap.
_FAST_LOOP_GEN0_THRESHOLD = 100_000

# One fingerprint entry per operation in capture order: the normalized op type
# plus each parent encoded structurally -- ("op", stream position),
# ("input", label), ("buffer", first-reference ordinal), or ("other", label).
# Raw capture labels are deliberately absent: their numbering diverges between
# exhaustive trace and sparse record when buffer-write events consume indexes.
_StreamEntry = tuple[str, tuple[tuple[str, Any], ...]]


@dataclass(frozen=True)
class _CompiledAggregatePlan:
    """Discovery-batch measurement plan for ``aggregate(target='out')``."""

    fingerprint: tuple[_StreamEntry, ...]
    sites: dict[str, int]


def _trace_reference_stream(log: Any) -> tuple[tuple[_StreamEntry, ...], dict[str, int]] | None:
    """Rebuild the raw op-event stream fingerprint from a finalized Trace.

    Returns ``None`` whenever any structural detail cannot be mapped; callers
    treat that as "do not compile a plan" and keep the exact per-batch path.
    """

    try:
        ops = []
        input_labels: set[str] = set()
        buffer_labels: set[str] = set()
        for layer in log.layer_list:
            if layer.is_input:
                input_labels.add(layer.layer_label)
            elif layer.is_buffer:
                buffer_labels.add(layer.layer_label)
            elif not layer.is_output:
                ops.append(layer)
        ops.sort(key=lambda op: op.raw_index)
        final_to_pos = {op.layer_label: pos for pos, op in enumerate(ops)}
        if len(final_to_pos) != len(ops):
            return None
        buffer_ordinals: dict[str, int] = {}
        entries: list[_StreamEntry] = []
        for op in ops:
            parents: list[tuple[str, Any]] = []
            for parent in op.parents:
                if parent in final_to_pos:
                    parents.append(("op", final_to_pos[parent]))
                elif parent in input_labels:
                    parents.append(("input", parent))
                elif parent in buffer_labels:
                    parents.append(
                        ("buffer", buffer_ordinals.setdefault(parent, len(buffer_ordinals)))
                    )
                else:
                    return None
            layer_type = op.layer_type
            if not isinstance(layer_type, str) or not layer_type:
                return None
            entries.append((layer_type, tuple(parents)))
        return tuple(entries), final_to_pos
    except Exception:
        return None


def _site_stream_position(site: Any, final_to_pos: dict[str, int]) -> int | None:
    """Return the op-stream position measured for a resolved metric site."""

    try:
        if getattr(site, "is_output", False):
            parents = list(getattr(site, "parents", ()))
            if len(parents) != 1:
                return None
            return final_to_pos.get(parents[0])
        label = getattr(site, "layer_label", None)
        if label is None:
            return None
        return final_to_pos.get(label)
    except Exception:
        return None


def _compile_aggregate_plan(
    log: Any, resolved_sites: Mapping[str, Any]
) -> _CompiledAggregatePlan | None:
    """Compile the discovery trace into a sparse measurement plan, or refuse."""

    reference = _trace_reference_stream(log)
    if reference is None:
        return None
    fingerprint, final_to_pos = reference
    sites: dict[str, int] = {}
    for metric_name, site in resolved_sites.items():
        position = _site_stream_position(site, final_to_pos)
        if position is None:
            return None
        sites[metric_name] = position
    return _CompiledAggregatePlan(fingerprint=fingerprint, sites=sites)


@dataclass
class _RecordStreamCursor:
    """Per-batch op-event stream state built inside the record predicate."""

    entries: list[_StreamEntry] = field(default_factory=list)
    raw_to_pos: dict[str, int] = field(default_factory=dict)
    event_to_pos: dict[Any, int] = field(default_factory=dict)
    buffer_ordinals: dict[str, int] = field(default_factory=dict)
    last_event: Any = None
    last_decision: bool = False
    broken: bool = False


def _strip_raw_suffix(label: str) -> str:
    """Return ``label`` without the in-flight ``_raw`` capture suffix."""

    return label[:-4] if label.endswith("_raw") else label


def _make_plan_predicate(
    cursor: _RecordStreamCursor, wanted: frozenset[int]
) -> Callable[[Any], bool]:
    """Build the record predicate that fingerprints the stream and saves sites."""

    def _predicate(ctx: Any) -> bool:
        """Fingerprint the record stream and decide whether this site is wanted.

        Stateful in ``cursor``: a repeat call for the same event index under an
        alias label replays the previous decision instead of advancing the stream.
        """

        try:
            if ctx.kind != "op":
                return False
            event_index = ctx.event_index
            if event_index is not None and event_index == cursor.last_event:
                # Compatibility retry for the same event under an alias label.
                return cursor.last_decision
            cursor.last_event = event_index
            position = len(cursor.entries)
            raw_label = ctx.raw_label
            if isinstance(raw_label, str) and raw_label:
                cursor.raw_to_pos[_strip_raw_suffix(raw_label)] = position
            parents: list[tuple[str, Any]] = []
            for parent in ctx.parent_labels:
                name = _strip_raw_suffix(parent)
                parent_pos = cursor.raw_to_pos.get(name)
                if parent_pos is not None:
                    parents.append(("op", parent_pos))
                elif name.startswith("input_"):
                    parents.append(("input", name))
                elif name.startswith("buffer_"):
                    parents.append(
                        (
                            "buffer",
                            cursor.buffer_ordinals.setdefault(name, len(cursor.buffer_ordinals)),
                        )
                    )
                else:
                    parents.append(("other", name))
            layer_type = ctx.layer_type
            cursor.entries.append(
                (layer_type if isinstance(layer_type, str) else "", tuple(parents))
            )
            decision = position in wanted
            if decision:
                cursor.event_to_pos[event_index] = position
            cursor.last_decision = decision
            return decision
        except Exception:
            cursor.broken = True
            cursor.last_decision = False
            return False

    return _predicate


def _run_compiled_aggregate_batch(
    model: nn.Module,
    model_input: Any,
    plan: _CompiledAggregatePlan,
) -> dict[int, torch.Tensor] | None:
    """Measure one batch with the compiled sparse plan.

    Returns the payload per compiled stream position, or ``None`` when the
    batch's op stream does not match the discovery fingerprint (the caller
    then falls back to the exact full-trace path).
    """

    from .. import record

    wanted = frozenset(plan.sites.values())
    cursor = _RecordStreamCursor()
    recording = None
    try:
        recording = record(model, model_input, save=_make_plan_predicate(cursor, wanted))
        if cursor.broken or tuple(cursor.entries) != plan.fingerprint:
            return None
        payloads: dict[int, torch.Tensor] = {}
        for entry in recording:
            position = cursor.event_to_pos.get(entry.ctx.event_index)
            if position is None:
                return None
            payload = entry.ram_payload
            if payload is None:
                return None
            payloads[position] = payload
        if any(position not in payloads for position in wanted):
            return None
        return payloads
    finally:
        del recording, cursor


def _rng_snapshot() -> tuple[Any, Any, Any, Any] | None:
    """Snapshot the global RNG engines a capture can consume.

    Covers Python's ``random`` module, the torch CPU generator, torch CUDA
    generators, and NumPy's legacy global generator. Returns ``None`` when any
    engine cannot be snapshotted; callers then skip the sparse fast path.
    """

    try:
        cuda_states = None
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            cuda_states = [state.clone() for state in torch.cuda.get_rng_state_all()]
        try:
            import numpy as np

            numpy_state = np.random.get_state()
        except Exception:
            numpy_state = None
        return (
            random.getstate(),
            torch.random.get_rng_state().clone(),
            cuda_states,
            numpy_state,
        )
    except Exception:
        return None


def _rng_restore(snapshot: tuple[Any, Any, Any, Any]) -> None:
    """Rewind the global RNG engines to a ``_rng_snapshot()`` state."""

    python_state, torch_state, cuda_states, numpy_state = snapshot
    random.setstate(python_state)
    torch.random.set_rng_state(torch_state)
    if cuda_states is not None:
        for device_index, state in enumerate(cuda_states):
            torch.cuda.set_rng_state(state, device_index)
    if numpy_state is not None:
        import numpy as np

        np.random.set_state(numpy_state)


def _model_state_allows_compiled_plan(model: nn.Module) -> bool:
    """Gate the sparse plan on fully-eval models.

    A fingerprint mismatch re-traces the same batch, so the fast path is only
    compiled when a second forward cannot mutate module state (train-mode
    BatchNorm running stats being the canonical hazard).
    """

    try:
        return not any(module.training for module in model.modules())
    except Exception:
        return False


def aggregate(
    model: nn.Module,
    dataloader: Iterable[Any],
    metrics: Mapping[str, StreamingStat],
    *,
    target: str = "out",
    loss_fn: Callable[..., torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Stream outs through metric accumulators.

    Parameters
    ----------
    model:
        Model to capture.
    dataloader:
        Iterable of model inputs.
    metrics:
        Mapping from layer selector to streaming statistic.
    target:
        ``"out"`` for activation statistics or ``"grad"`` for gradient
        statistics.
    loss_fn:
        Callable used to build a loss from ``(output, *batch_tail)`` when
        ``target="grad"``.

    Returns
    -------
    dict[str, Any]
        Finalized metric results.
    """

    from .. import trace
    from ..options import CaptureOptions

    if target not in {"out", "grad"}:
        raise ValueError("target must be 'out' or 'grad'")
    if target == "grad" and loss_fn is None:
        raise TypeError("aggregate(target='grad') requires loss_fn=")
    grad_loss_fn = loss_fn

    layers = [name for name in metrics if name != "output"]
    capture_layers: str | list[str] = layers if layers else "all"

    if target == "grad":
        for batch in dataloader:
            model_input, loss_args = _split_batch_for_loss(batch)
            log = trace(
                model,
                model_input,
                capture=CaptureOptions(
                    layers_to_save=capture_layers,
                    save_grads=capture_layers,
                ),
            )
            try:
                if grad_loss_fn is None:
                    raise TypeError("aggregate(target='grad') requires loss_fn=")
                loss = grad_loss_fn(_metric_value_from_log(log, "output"), *loss_args)
                log.log_backward(loss)
                for metric_name, stat in metrics.items():
                    stat.update(_metric_grad_from_log(log, metric_name))
            finally:
                log.cleanup()
        return {name: stat.result() for name, stat in metrics.items()}

    # target == "out": trace the structure once, then execute a compiled sparse
    # measurement per batch. Any surprise -- unstable structure, non-eval
    # modules, unmappable sites -- keeps or restores the exact per-batch
    # full-trace path. Trace and sparse record share the per-capture seed draw,
    # so the global RNG streams the model and the stats see stay identical; a
    # fingerprint mismatch rewinds the RNG engines before re-tracing so the
    # fallback consumes the exact stream the full path would have.
    plan: _CompiledAggregatePlan | None = None
    compile_allowed = _model_state_allows_compiled_plan(model)
    saved_gc_thresholds: tuple[int, int, int] | None = None
    try:
        for batch in dataloader:
            model_input, _loss_args = _split_batch_for_loss(batch)
            if plan is not None:
                snapshot = _rng_snapshot()
                payloads = (
                    _run_compiled_aggregate_batch(model, model_input, plan)
                    if snapshot is not None
                    else None
                )
                if payloads is not None:
                    for metric_name, stat in metrics.items():
                        stat.update(payloads[plan.sites[metric_name]])
                    gc.collect(1)
                    continue
                # Structure drifted from the discovery batch (or the RNG
                # engines could not be protected): re-trace this batch exactly
                # and stay on the full path for the rest of the loop.
                plan = None
                compile_allowed = False
                if saved_gc_thresholds is not None:
                    gc.set_threshold(*saved_gc_thresholds)
                    saved_gc_thresholds = None
                if snapshot is not None:
                    _rng_restore(snapshot)
            log = trace(
                model,
                model_input,
                capture=CaptureOptions(layers_to_save=capture_layers),
            )
            try:
                resolved_sites: dict[str, Any] = {}
                for metric_name, stat in metrics.items():
                    value, site = _resolve_metric_out(log, metric_name)
                    resolved_sites[metric_name] = site
                    stat.update(value)
                if compile_allowed:
                    plan = _compile_aggregate_plan(log, resolved_sites)
                    compile_allowed = False
                    if plan is not None:
                        saved_gc_thresholds = gc.get_threshold()
                        gc.set_threshold(
                            max(saved_gc_thresholds[0], _FAST_LOOP_GEN0_THRESHOLD),
                            *saved_gc_thresholds[1:],
                        )
            finally:
                log.cleanup()
    finally:
        if saved_gc_thresholds is not None:
            gc.collect(1)
            gc.set_threshold(*saved_gc_thresholds)
    return {name: stat.result() for name, stat in metrics.items()}


__all__ = [
    "Aggregator",
    "CKA",
    "Covariance",
    "CrossCovariance",
    "Mean",
    "Norm",
    "PCA",
    "Quantile",
    "StreamingStat",
    "TopK",
    "aggregate",
    "cka",
]
