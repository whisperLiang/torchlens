"""Native layer-attribution methods for TorchLens."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch.utils.hooks import RemovableHandle

from torchlens.attribution._core import (
    AttributionError,
    AttributionResult,
    InputKwargs,
    TargetSpec,
    _call_model,
    _interned_path_leaves,
    _make_input_leaves,
    _normalize_model_inputs,
    _PreparedInputs,
    _scalarize_output,
    _target_repr,
    _temporarily_eval,
    _validate_baselines,
    _validate_positive_int,
)

LayerAttributionMethod: TypeAlias = Literal["activation_x_grad", "grad"]


@dataclass
class _LayerCapture:
    """Container for captured layer activations across every hook firing.

    Attributes
    ----------
    activations
        Distinct forward activations emitted by the target layer, one entry per
        distinct output tensor object, in firing order. A module that returns
        the very tensor object it received (``nn.Identity``) contributes one
        entry no matter how often it fires: autograd already accumulates every
        use of that node into one gradient, so a second entry would
        double-count its contribution.
    """

    activations: list[Tensor] = field(default_factory=list)

    def add(self, output: Any) -> None:
        """Record one hook firing.

        Parameters
        ----------
        output
            Value returned by the target layer for this firing.

        Raises
        ------
        AttributionError
            If the layer emitted a non-tensor output.
        """

        if not isinstance(output, Tensor):
            raise AttributionError("target layer must return a tensor activation")
        if not any(output is existing for existing in self.activations):
            self.activations.append(output)


def _layer_gradients(
    scalar: Tensor,
    activations: tuple[Tensor, ...],
    layer: str,
) -> tuple[Tensor, ...]:
    """Compute per-firing gradients of the scalar target for every captured activation.

    Firings that carry gradient tracking but provably do not reach the target
    contribute exact-zero gradients: the target does not depend on them, and
    zero is that statement, not a fallback. If NO firing reaches the target the
    request is a user error and the single-firing error contract is preserved.

    Parameters
    ----------
    scalar
        Scalarized model output.
    activations
        Distinct captured activations in firing order.
    layer
        Layer name used for error reporting.

    Returns
    -------
    tuple[Tensor, ...]
        Gradient of ``scalar`` with respect to each activation, in firing order.

    Raises
    ------
    AttributionError
        If no captured firing is differentiable or none reaches the target.
    """

    differentiable = [activation for activation in activations if activation.requires_grad]
    if not differentiable:
        raise AttributionError(
            f"layer {layer!r} activation is not differentiable with respect to target"
        )
    try:
        raw_gradients = torch.autograd.grad(scalar, differentiable, allow_unused=True)
    except RuntimeError as exc:
        raise AttributionError(
            f"target scalar is not differentiable with respect to layer {layer!r}"
        ) from exc
    if all(gradient is None for gradient in raw_gradients):
        raise AttributionError(
            f"target scalar is not differentiable with respect to layer {layer!r}"
        )
    gradient_by_id = {
        id(activation): gradient
        for activation, gradient in zip(differentiable, raw_gradients, strict=True)
    }
    return tuple(
        gradient
        if (gradient := gradient_by_id.get(id(activation))) is not None
        else torch.zeros_like(activation)
        for activation in activations
    )


def _validate_uniform_firing_shapes(activations: tuple[Tensor, ...], layer: str) -> None:
    """Require matching activation shapes across firings of a reused layer.

    Per-firing contributions are accumulated into one activation-shaped result,
    which is only well-defined when every firing produced the same shape.

    Parameters
    ----------
    activations
        Distinct captured activations in firing order.
    layer
        Layer name used for error reporting.

    Raises
    ------
    AttributionError
        If the layer produced differently shaped activations across firings.
    """

    shapes = {tuple(activation.shape) for activation in activations}
    if len(shapes) > 1:
        raise AttributionError(
            f"layer {layer!r} fired {len(activations)} times with mismatched activation "
            f"shapes {sorted(shapes)}; per-firing accumulation requires matching shapes"
        )


def _validate_matching_firings(
    reference: tuple[Tensor, ...],
    observed: tuple[Tensor, ...],
    layer: str,
) -> None:
    """Require consistent firing count and shapes across input-path points.

    Input-path layer methods pair the ``i``-th firing at one path point with the
    ``i``-th firing at every other; a count or shape change means the model took
    different control flow along the path and the pairing is meaningless.

    Parameters
    ----------
    reference
        Firing activations (or aligned gradients) at the reference path point.
    observed
        Firing activations (or aligned gradients) at another path point.
    layer
        Layer name used for error reporting.

    Raises
    ------
    AttributionError
        If the firing count or any per-firing shape differs.
    """

    if len(observed) != len(reference):
        raise AttributionError(
            f"layer {layer!r} fired {len(observed)} times at one input-path point and "
            f"{len(reference)} times at another; input-path layer attribution requires "
            "consistent control flow along the path"
        )
    for reference_item, observed_item in zip(reference, observed, strict=True):
        if tuple(reference_item.shape) != tuple(observed_item.shape):
            raise AttributionError(
                f"layer {layer!r} produced mismatched activation shapes across input-path "
                "points; input-path layer attribution requires consistent shapes"
            )


def _autograd_leaf_variable_ids(activations: tuple[Tensor, ...]) -> set[int]:
    """Collect ids of every autograd leaf tensor reachable from the activations.

    Parameters
    ----------
    activations
        Captured activations still connected to their autograd graphs.

    Returns
    -------
    set[int]
        ``id()`` values of leaf tensors (``AccumulateGrad`` variables) that are
        graph ancestors of any activation, plus the activations themselves so a
        passthrough layer that returns an input leaf unchanged still reports it.
    """

    found: set[int] = {id(activation) for activation in activations}
    seen: set[int] = set()
    stack: list[Any] = [
        activation.grad_fn for activation in activations if activation.grad_fn is not None
    ]
    while stack:
        node = stack.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        variable = getattr(node, "variable", None)
        if isinstance(variable, Tensor):
            found.add(id(variable))
        stack.extend(next_node for next_node, _ in node.next_functions)
    return found


def _feeding_original_leaves(
    activations: tuple[Tensor, ...],
    input_leaves: tuple[Tensor, ...],
    original_leaves: tuple[Tensor, ...],
) -> tuple[Tensor, ...]:
    """Return the unique original input leaves that feed the captured activations.

    "Feeds" is autograd-graph ancestry of the substituted differentiable clone,
    not traversal order: an input that never reaches the target layer must not
    be treated as its coordinate system.

    Parameters
    ----------
    activations
        Captured activations still connected to their autograd graphs.
    input_leaves
        Substituted differentiable clones, slot-aligned with ``original_leaves``.
    original_leaves
        Original attributed leaves from the user's call.

    Returns
    -------
    tuple[Tensor, ...]
        Unique original leaves whose clones are graph ancestors of the
        activations, in traversal order.
    """

    reachable = _autograd_leaf_variable_ids(activations)
    feeding: list[Tensor] = []
    seen_clone_ids: set[int] = set()
    for clone, original in zip(input_leaves, original_leaves, strict=True):
        if id(clone) in seen_clone_ids:
            continue
        seen_clone_ids.add(id(clone))
        if id(clone) in reachable:
            feeding.append(original)
    return tuple(feeding)


def _format_layer_options(model: Module) -> str:
    """Return a compact list of useful layer-name suggestions.

    Parameters
    ----------
    model
        Model whose named modules should be searched.

    Returns
    -------
    str
        Comma-separated module names suitable for an error message.
    """

    named_modules = dict(model.named_modules())
    conv_like = [
        name
        for name, module in named_modules.items()
        if name and ("conv" in name.lower() or "conv" in type(module).__name__.lower())
    ]
    options = conv_like[:5]
    if not options:
        options = [name for name in named_modules if name][:5]
    if not options:
        return "<no named child modules>"
    return ", ".join(options)


def _resolve_named_layer(model: Module, layer: str) -> Module:
    """Resolve a user-specified module name.

    Parameters
    ----------
    model
        Model containing the target layer.
    layer
        Name from ``model.named_modules()``.

    Returns
    -------
    Module
        Resolved PyTorch module.

    Raises
    ------
    AttributionError
        If ``layer`` is not a named module string in ``model``.
    """

    if not isinstance(layer, str):
        raise AttributionError("layer must be a module name string")

    named_modules = dict(model.named_modules())
    if layer not in named_modules:
        options = _format_layer_options(model)
        raise AttributionError(
            f"layer {layer!r} was not found; available conv-like layers include: {options}"
        )
    return named_modules[layer]


def _capture_layer_activation(
    model: Module,
    inputs: _PreparedInputs,
    target: TargetSpec,
    layer: str,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...], tuple[Tensor, ...]]:
    """Capture every distinct layer firing and its target gradient.

    A layer reused ``N`` times contributes ``N`` distinct activations; keeping
    only one would silently drop the other calls' contributions to the target.

    Parameters
    ----------
    model
        PyTorch module to attribute.
    inputs
        Normalized attribution inputs.
    target
        Integer class index or callable scalarizer.
    layer
        Name of the module whose activation should be captured.

    Returns
    -------
    tuple[tuple[Tensor, ...], tuple[Tensor, ...], tuple[Tensor, ...]]
        Detached per-firing activations, aligned per-firing gradients of the
        scalar target, and the unique original attributed input leaves that
        feed the captured activations through the autograd graph.

    Raises
    ------
    AttributionError
        If the target layer does not emit a differentiable tensor activation.
    """

    target_layer = _resolve_named_layer(model, layer)
    capture = _LayerCapture()
    hook_handles: list[RemovableHandle] = []

    def _forward_hook(_module: Module, _args: tuple[Any, ...], output: Any) -> None:
        """Record the target layer output for each firing of the forward pass."""

        capture.add(output)

    hook_handles.append(target_layer.register_forward_hook(_forward_hook))
    try:
        with _temporarily_eval(model):
            input_leaves = _make_input_leaves(inputs)
            output = _call_model(model, inputs, input_leaves)
            activations = tuple(capture.activations)
            if not activations:
                raise AttributionError(f"layer {layer!r} did not run during the forward pass")
            if not any(activation.requires_grad for activation in activations):
                raise AttributionError(
                    f"layer {layer!r} activation is not differentiable with respect to target"
                )
            feeding_leaves = _feeding_original_leaves(
                activations, input_leaves, inputs.attributed_leaves
            )
            scalar = _scalarize_output(output, target)
            gradients = _layer_gradients(scalar, activations, layer)
    finally:
        for handle in hook_handles:
            handle.remove()

    return (
        tuple(activation.detach() for activation in activations),
        tuple(gradient.detach() for gradient in gradients),
        feeding_leaves,
    )


def _capture_layer_activation_for_leaves(
    model: Module,
    inputs: _PreparedInputs,
    target: TargetSpec,
    layer: str,
    input_leaves: tuple[Tensor, ...],
    *,
    require_gradient: bool,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...] | None]:
    """Capture every distinct layer firing and optionally the target gradients.

    Parameters
    ----------
    model
        PyTorch module to attribute.
    inputs
        Normalized attribution inputs.
    target
        Integer class index or callable scalarizer.
    layer
        Name of the module whose activation should be captured.
    input_leaves
        Differentiable leaves substituted into the model call.
    require_gradient
        Whether to compute ``dTarget / dActivation`` per firing.

    Returns
    -------
    tuple[tuple[Tensor, ...], tuple[Tensor, ...] | None]
        Detached per-firing activations and optional aligned per-firing
        gradients with respect to them.

    Raises
    ------
    AttributionError
        If the target layer does not emit a differentiable tensor activation.
    """

    target_layer = _resolve_named_layer(model, layer)
    capture = _LayerCapture()
    hook_handles: list[RemovableHandle] = []

    def _forward_hook(_module: Module, _args: tuple[Any, ...], output: Any) -> None:
        """Record the target layer output for each firing of the forward pass."""

        capture.add(output)

    hook_handles.append(target_layer.register_forward_hook(_forward_hook))
    try:
        output = _call_model(model, inputs, input_leaves)
        activations = tuple(capture.activations)
        if not activations:
            raise AttributionError(f"layer {layer!r} did not run during the forward pass")
        if not require_gradient:
            return tuple(activation.detach() for activation in activations), None
        if not any(activation.requires_grad for activation in activations):
            raise AttributionError(
                f"layer {layer!r} activation is not differentiable with respect to target"
            )
        scalar = _scalarize_output(output, target)
        gradients = _layer_gradients(scalar, activations, layer)
    finally:
        for handle in hook_handles:
            handle.remove()

    return (
        tuple(activation.detach() for activation in activations),
        tuple(gradient.detach() for gradient in gradients),
    )


def _layer_path_basics(
    model: Module,
    inputs: _PreparedInputs,
    target: TargetSpec,
    layer: str,
    baseline_tensors: tuple[Tensor, ...],
    n_steps: int,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...], list[tuple[Tensor, ...]], tuple[Tensor, ...]]:
    """Capture per-firing endpoint activations and midpoint gradients along an input path.

    Every path point captures ALL distinct firings of the target layer. The
    firing count and per-firing shapes are validated to be consistent across
    path points (pairing firing ``i`` across points is otherwise meaningless)
    and uniform across firings (per-firing contributions are accumulated into
    one activation-shaped result).

    Parameters
    ----------
    model
        PyTorch module to attribute.
    inputs
        Normalized attribution inputs.
    target
        Integer class index or callable scalarizer.
    layer
        Name from ``dict(model.named_modules())`` identifying the target layer.
    baseline_tensors
        Baseline leaves matching attributed inputs.
    n_steps
        Number of midpoint Riemann samples.

    Returns
    -------
    tuple
        Per-firing baseline activations, per-firing input activations,
        per-step tuples of per-firing midpoint gradients, and input deltas.
    """

    deltas = tuple(
        input_leaf.detach() - baseline_tensor
        for input_leaf, baseline_tensor in zip(
            inputs.attributed_leaves, baseline_tensors, strict=True
        )
    )
    with _temporarily_eval(model):
        baseline_activations, _baseline_gradients = _capture_layer_activation_for_leaves(
            model,
            inputs,
            target,
            layer,
            _interned_path_leaves(inputs, baseline_tensors, deltas, 0.0),
            require_gradient=False,
        )
        _validate_uniform_firing_shapes(baseline_activations, layer)
        input_activations, _input_gradients = _capture_layer_activation_for_leaves(
            model,
            inputs,
            target,
            layer,
            _interned_path_leaves(inputs, baseline_tensors, deltas, 1.0),
            require_gradient=False,
        )
        _validate_matching_firings(baseline_activations, input_activations, layer)
        gradients_by_step: list[tuple[Tensor, ...]] = []
        for step in range(n_steps):
            alpha = (step + 0.5) / n_steps
            _activations, gradients = _capture_layer_activation_for_leaves(
                model,
                inputs,
                target,
                layer,
                _interned_path_leaves(inputs, baseline_tensors, deltas, alpha),
                require_gradient=True,
            )
            if gradients is None:
                raise AttributionError("internal error: missing layer path gradient")
            _validate_matching_firings(baseline_activations, gradients, layer)
            gradients_by_step.append(gradients)
    return baseline_activations, input_activations, gradients_by_step, deltas


def _spatial_reference_tensor(
    inputs: _PreparedInputs,
    feeding_leaves: tuple[Tensor, ...],
    layer: str,
) -> Tensor:
    """Return the input tensor whose grid the Grad-CAM should be upsampled onto.

    The CAM lives in the coordinate system of the input that actually feeds the
    target layer. Upsampling onto any other input would place the attribution
    in an unrelated coordinate space, so candidates are restricted to
    dependency-proven feeders and ambiguity between different grids is refused
    rather than resolved by traversal order.

    Parameters
    ----------
    inputs
        Normalized attribution inputs.
    feeding_leaves
        Unique original attributed leaves proven to feed the target layer.
    layer
        Layer name used for error reporting.

    Returns
    -------
    Tensor
        Spatial (``ndim >= 4``) feeding leaf; when several feed the layer they
        must share one spatial grid.

    Raises
    ------
    AttributionError
        If no attributed leaf has spatial dimensions, no spatial leaf feeds the
        layer, or the feeding spatial grids are heterogeneous.
    """

    if not any(input_tensor.ndim >= 4 for input_tensor in inputs.attributed_leaves):
        raise AttributionError("grad_cam requires an input tensor with spatial dimensions")
    candidates = [input_tensor for input_tensor in feeding_leaves if input_tensor.ndim >= 4]
    if not candidates:
        raise AttributionError(
            f"grad_cam found no spatial (ndim >= 4) input tensor feeding layer {layer!r}; "
            "the CAM has no input coordinate system to upsample onto"
        )
    grids = {tuple(candidate.shape[-2:]) for candidate in candidates}
    if len(grids) > 1:
        raise AttributionError(
            f"grad_cam target layer {layer!r} is fed by spatial inputs with different "
            f"grids {sorted(grids)}; the upsampling target is ambiguous"
        )
    return candidates[0]


def _validate_conv_activation(activation: Tensor, layer: str) -> None:
    """Validate that an activation is a 2D convolution-style feature map.

    Parameters
    ----------
    activation
        Captured target-layer activation.
    layer
        Layer name used for error reporting.

    Raises
    ------
    AttributionError
        If ``activation`` is not shaped ``N, C, H, W``.
    """

    if activation.ndim != 4:
        raise AttributionError(
            f"grad_cam requires layer {layer!r} to produce a 4D N,C,H,W feature map; "
            f"got shape {tuple(activation.shape)}"
        )


def grad_cam(
    model: Module,
    inputs: Any,
    input_kwargs: InputKwargs = None,
    *,
    target: TargetSpec,
    layer: str,
    relu: bool = True,
) -> AttributionResult:
    """Compute Grad-CAM for a named convolution-style layer.

    Parameters
    ----------
    model
        PyTorch module to attribute.
    inputs
        Bare tensor for v1 behavior, or tuple/list of positional model arguments.
        Floating-point or complex tensor leaves are threaded through the model call.
    input_kwargs
        Optional keyword arguments for ``model``.
    target
        Integer class index selecting ``output[..., target]`` and summing the
        selected values, or callable ``output -> scalar tensor``.
    layer
        Name from ``dict(model.named_modules())`` identifying the target layer.
        The layer must fire exactly once during the forward pass; Grad-CAM has
        no defined semantics for a reused layer, so multi-fire raises rather
        than silently attributing one arbitrary call.
    relu
        Whether to apply ReLU to the channel-reduced CAM.

    Returns
    -------
    AttributionResult
        Grad-CAM values upsampled to the spatial size of the input that feeds
        the target layer, with shape ``N, 1, Hin, Win``.

    Raises
    ------
    AttributionError
        If the target layer fired more than once, no spatial input feeds it,
        or several feeding spatial inputs have different grids.
    """

    prepared_inputs = _normalize_model_inputs(inputs, input_kwargs)
    activations, gradients, feeding_leaves = _capture_layer_activation(
        model, prepared_inputs, target, layer
    )
    if len(activations) != 1:
        raise AttributionError(
            f"grad_cam requires layer {layer!r} to fire exactly once during the forward "
            f"pass; it fired {len(activations)} times"
        )
    activation, gradient = activations[0], gradients[0]
    _validate_conv_activation(activation, layer)
    spatial_reference = _spatial_reference_tensor(prepared_inputs, feeding_leaves, layer)
    alpha = gradient.mean(dim=(2, 3), keepdim=True)
    cam = (alpha * activation).sum(dim=1, keepdim=True)
    if relu:
        cam = torch.relu(cam)
    upsampled_cam = F.interpolate(
        cam,
        size=spatial_reference.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    return AttributionResult(
        method="grad_cam",
        values=upsampled_cam.detach(),
        target_repr=_target_repr(target),
        extra={"layer": layer, "relu": relu},
    )


def layer_integrated_gradients(
    model: Module,
    inputs: Any,
    input_kwargs: InputKwargs = None,
    *,
    target: TargetSpec,
    layer: str,
    baseline: Any | None = None,
    n_steps: int = 50,
) -> AttributionResult:
    """Compute Layer Integrated Gradients for a named intermediate layer.

    The input path is the straight line from baseline leaves to input leaves.
    The returned values have the same shape as the target layer activation and
    use the Captum-style ``(A(input) - A(baseline)) * mean(dTarget / dA)`` rule.

    Parameters
    ----------
    model
        PyTorch module to attribute.
    inputs
        Bare tensor for v1 behavior, or tuple/list of positional model arguments.
        Floating-point or complex tensor leaves are threaded through the model call.
    input_kwargs
        Optional keyword arguments for ``model``.
    target
        Integer class index selecting ``output[..., target]`` and summing the
        selected values, or callable ``output -> scalar tensor``.
    layer
        Name from ``dict(model.named_modules())`` identifying the target layer.
    baseline
        Optional baseline tree matching attributed input leaves. A bare tensor is
        accepted when there is exactly one attributed leaf.
    n_steps
        Number of midpoint Riemann samples along the straight input path.

    Returns
    -------
    AttributionResult
        Layer attribution values with the same shape as the captured activation.
    """

    _validate_positive_int("n_steps", n_steps)
    prepared_inputs = _normalize_model_inputs(inputs, input_kwargs)
    baseline_tensors = _validate_baselines(prepared_inputs, baseline)
    baseline_activations, input_activations, gradients_by_step, _deltas = _layer_path_basics(
        model,
        prepared_inputs,
        target,
        layer,
        baseline_tensors,
        n_steps,
    )
    # A layer reused N times contributes through every firing; the honest total
    # sums the per-firing (activation delta) x (mean path gradient) terms.
    values = torch.zeros_like(baseline_activations[0])
    for firing_index, (baseline_activation, input_activation) in enumerate(
        zip(baseline_activations, input_activations, strict=True)
    ):
        mean_gradient = torch.stack(
            [step_gradients[firing_index] for step_gradients in gradients_by_step], dim=0
        ).mean(dim=0)
        values = values + (input_activation - baseline_activation) * mean_gradient
    return AttributionResult(
        method="layer_integrated_gradients",
        values=values.detach(),
        target_repr=_target_repr(target),
        extra={
            "layer": layer,
            "n_steps": n_steps,
        },
    )


def layer_conductance(
    model: Module,
    inputs: Any,
    input_kwargs: InputKwargs = None,
    *,
    target: TargetSpec,
    layer: str,
    baseline: Any | None = None,
    n_steps: int = 50,
) -> AttributionResult:
    """Compute Layer Conductance for a named intermediate layer.

    Conductance decomposes input Integrated Gradients onto hidden units by
    integrating ``(dTarget / dA) * (dA / dalpha)`` along the input path. This
    implementation uses midpoint layer gradients and finite activation
    differences for each path interval.

    Parameters
    ----------
    model
        PyTorch module to attribute.
    inputs
        Bare tensor for v1 behavior, or tuple/list of positional model arguments.
        Floating-point or complex tensor leaves are threaded through the model call.
    input_kwargs
        Optional keyword arguments for ``model``.
    target
        Integer class index selecting ``output[..., target]`` and summing the
        selected values, or callable ``output -> scalar tensor``.
    layer
        Name from ``dict(model.named_modules())`` identifying the target layer.
    baseline
        Optional baseline tree matching attributed input leaves. A bare tensor is
        accepted when there is exactly one attributed leaf.
    n_steps
        Number of midpoint Riemann samples along the straight input path.

    Returns
    -------
    AttributionResult
        Layer conductance values with the same shape as the captured activation.
    """

    _validate_positive_int("n_steps", n_steps)
    prepared_inputs = _normalize_model_inputs(inputs, input_kwargs)
    baseline_tensors = _validate_baselines(prepared_inputs, baseline)
    baseline_activations, _input_activations, gradients_by_step, deltas = _layer_path_basics(
        model,
        prepared_inputs,
        target,
        layer,
        baseline_tensors,
        n_steps,
    )
    # A layer reused N times contributes through every firing; the honest total
    # sums the per-firing gradient x (activation interval) terms.
    activations_left = baseline_activations
    conductance = torch.zeros_like(baseline_activations[0])
    with _temporarily_eval(model):
        for step, gradients in enumerate(gradients_by_step):
            alpha_right = (step + 1) / n_steps
            activations_right, _right_gradients = _capture_layer_activation_for_leaves(
                model,
                prepared_inputs,
                target,
                layer,
                _interned_path_leaves(prepared_inputs, baseline_tensors, deltas, alpha_right),
                require_gradient=False,
            )
            _validate_matching_firings(baseline_activations, activations_right, layer)
            for gradient, activation_right, activation_left in zip(
                gradients, activations_right, activations_left, strict=True
            ):
                conductance = conductance + gradient * (activation_right - activation_left)
            activations_left = activations_right
    return AttributionResult(
        method="layer_conductance",
        values=conductance.detach(),
        target_repr=_target_repr(target),
        extra={
            "layer": layer,
            "n_steps": n_steps,
        },
    )


def layer_attribution(
    model: Module,
    inputs: Any,
    input_kwargs: InputKwargs = None,
    *,
    target: TargetSpec,
    layer: str,
    method: LayerAttributionMethod = "activation_x_grad",
) -> AttributionResult:
    """Compute attribution for a named intermediate layer.

    Parameters
    ----------
    model
        PyTorch module to attribute.
    inputs
        Bare tensor for v1 behavior, or tuple/list of positional model arguments.
        Floating-point or complex tensor leaves are threaded through the model call.
    input_kwargs
        Optional keyword arguments for ``model``.
    target
        Integer class index selecting ``output[..., target]`` and summing the
        selected values, or callable ``output -> scalar tensor``.
    layer
        Name from ``dict(model.named_modules())`` identifying the target layer.
    method
        Layer attribution method. ``"activation_x_grad"`` returns
        ``activation * gradient``. ``"grad"`` returns ``abs(gradient)``. A
        layer reused ``N`` times contributes through every firing, so both
        methods sum the per-firing terms.

    Returns
    -------
    AttributionResult
        Layer-attribution values with the same shape as the captured activation.

    Raises
    ------
    AttributionError
        If ``method`` is unsupported, or a reused layer produced mismatched
        activation shapes across firings.
    """

    if method not in ("activation_x_grad", "grad"):
        raise AttributionError("method must be 'activation_x_grad' or 'grad'")

    prepared_inputs = _normalize_model_inputs(inputs, input_kwargs)
    activations, gradients, _feeding_leaves = _capture_layer_activation(
        model, prepared_inputs, target, layer
    )
    _validate_uniform_firing_shapes(activations, layer)
    values = torch.zeros_like(activations[0])
    for activation, gradient in zip(activations, gradients, strict=True):
        if method == "activation_x_grad":
            values = values + activation * gradient
        else:
            values = values + gradient.abs()
    return AttributionResult(
        method=f"layer_{method}",
        values=values.detach(),
        target_repr=_target_repr(target),
        extra={"layer": layer},
    )
