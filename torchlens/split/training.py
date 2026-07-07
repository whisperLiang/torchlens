"""Backend-dispatched split training helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from .boundary import ReplayBoundary
from .errors import SplitErrorContext, SplitUnsupportedError

BoundaryGradients = dict[str, Any]


@dataclass(frozen=True)
class TrainingStepResult:
    """Structured result from a split suffix training step."""

    loss: Any
    boundary_grads: BoundaryGradients
    param_grads: dict[str, Any] | None = None
    optimizer_applied: bool = False

    def as_tuple(self) -> tuple[Any, BoundaryGradients]:
        """Return the legacy public ``(loss, boundary_grads)`` shape."""

        return self.loss, self.boundary_grads


class BackendTrainingEngine(Protocol):
    """Protocol for backend-owned split training engines."""

    name: str

    def train_suffix(
        self,
        runtime: Any,
        boundary: ReplayBoundary,
        targets: Any,
        loss_fn: Callable[[Any, Any], Any] | None = None,
        optimizer: Any | None = None,
    ) -> TrainingStepResult:
        """Differentiate or train a split suffix."""
        ...

    def backward_prefix(
        self,
        runtime: Any,
        boundary: ReplayBoundary,
        boundary_grads: BoundaryGradients,
        optimizer: Any | None = None,
    ) -> Any:
        """Propagate suffix gradients through a graph-connected prefix."""
        ...


def _require_torch(runtime: Any) -> Any:
    """Import torch and verify that ``runtime`` is a Torch split runtime."""

    if runtime.adapter.name != "torch":
        raise SplitUnsupportedError(
            f"backend={runtime.adapter.name!r} does not support split training.",
            context=SplitErrorContext(
                backend=runtime.adapter.name,
                split_point=runtime.split_spec.boundary,
                module_path=None,
                op_type=None,
                layer_label=None,
                reason="unsupported split training",
            ),
        )
    import torch

    return torch


def _context(runtime: Any, reason: str) -> SplitErrorContext:
    """Build a generic split-training error context."""

    return SplitErrorContext(
        backend=runtime.adapter.name,
        split_point=runtime.split_spec.boundary,
        module_path=None,
        op_type=None,
        layer_label=None,
        reason=reason,
    )


def _is_diff_tensor(torch: Any, value: Any) -> bool:
    """Return whether ``value`` can serve as a differentiable boundary root."""

    return isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex())


def _default_loss(torch: Any, output: Any, targets: Any) -> Any:
    """Compute the default split-training loss."""

    if not isinstance(output, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise SplitUnsupportedError(
            "Non-tensor split-training outputs require an explicit loss_fn.",
            context=SplitErrorContext(
                backend="torch",
                split_point="",
                module_path=None,
                op_type=None,
                layer_label=None,
                reason="default loss requires tensor output and target",
            ),
        )
    if targets.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.long):
        if output.ndim >= 2 and targets.ndim == output.ndim - 1:
            return torch.nn.functional.cross_entropy(output, targets)
    return torch.nn.functional.mse_loss(output, targets)


def _trainable_param_handles(runtime: Any, node_ids: frozenset[str]) -> list[Any]:
    """Return unique live trainable parameter handles used by split nodes."""

    handles: list[Any] = []
    seen: set[int] = set()
    for node_id in node_ids:
        node = runtime.trace_graph.node_by_id.get(node_id)
        if node is None:
            continue
        for param in node.param_refs:
            if not getattr(param, "is_trainable", False):
                continue
            handle = getattr(param, "_param_ref", None)
            if handle is None:
                handle = getattr(param, "handle", None)
            if handle is None or id(handle) in seen:
                continue
            seen.add(id(handle))
            handles.append(handle)
    if getattr(runtime.adapter, "name", None) == "tf":
        _extend_tf_resource_param_handles(runtime, node_ids, handles, seen)
    return handles


def _tf_resource_name(handle: Any) -> str | None:
    """Extract TensorFlow eager resource-handle name from its stable repr."""

    text = repr(handle)
    marker = 'ResourceHandle(name="'
    start = text.find(marker)
    if start < 0:
        return None
    start += len(marker)
    end = text.find('"', start)
    if end < 0:
        return None
    return text[start:end]


def _extend_tf_resource_param_handles(
    runtime: Any,
    node_ids: frozenset[str],
    handles: list[Any],
    seen: set[int],
) -> None:
    """Add TensorFlow variables referenced by raw ``ReadVariableOp`` resource handles."""

    variables = tuple(getattr(runtime.model, "trainable_variables", ()) or ())
    variables_by_resource = {
        resource_name: variable
        for variable in variables
        if (resource_name := _tf_resource_name(getattr(variable, "handle", None))) is not None
    }
    if not variables_by_resource:
        return
    for node_id in node_ids:
        node = runtime.trace_graph.node_by_id.get(node_id)
        capture = getattr(node, "target", None)
        if node is None or getattr(capture, "op_type", None) != "ReadVariableOp":
            continue
        for input_record in getattr(capture, "inputs", ()):
            if getattr(input_record, "source_kind", None) != "resource":
                continue
            resource_name = _tf_resource_name(getattr(input_record, "tensor", None))
            if resource_name is None:
                continue
            variable = variables_by_resource.get(resource_name)
            if variable is None or id(variable) in seen:
                continue
            seen.add(id(variable))
            handles.append(variable)


def _train_suffix_torch(
    runtime: Any,
    boundary: ReplayBoundary,
    targets: Any,
    loss_fn: Callable[[Any, Any], Any] | None = None,
    optimizer: Any | None = None,
) -> tuple[Any, BoundaryGradients, bool]:
    """Train a Torch suffix from a boundary and return boundary gradients."""

    torch = _require_torch(runtime)
    runtime.validate_boundary(boundary)
    root_tensors: dict[str, Any] = {}
    replay_tensors: dict[str, Any] = {}
    for key, value in boundary.tensors.items():
        if _is_diff_tensor(torch, value):
            root = value.detach().clone().requires_grad_(True)
            root_tensors[key] = root
            replay_tensors[key] = root
        else:
            replay_tensors[key] = value
    replay_boundary = ReplayBoundary(
        backend=boundary.backend,
        tensors=replay_tensors,
        spec=boundary.spec,
        metadata={**boundary.metadata, "suffix_training_roots": tuple(root_tensors)},
    )
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    output = runtime.run_suffix(replay_boundary)
    loss = (
        loss_fn(output, targets) if loss_fn is not None else _default_loss(torch, output, targets)
    )
    loss.backward()
    gradients: BoundaryGradients = {
        key: root.grad.detach().clone()
        for key, root in root_tensors.items()
        if root.grad is not None
    }
    if optimizer is not None:
        optimizer.step()
    return loss, gradients, optimizer is not None


def _is_diff_tf_tensor(tf: Any, value: Any) -> bool:
    """Return whether ``value`` is differentiable for TensorFlow."""

    if not isinstance(value, (tf.Tensor, tf.Variable)):
        return False
    dtype = getattr(value, "dtype", None)
    return bool(getattr(dtype, "is_floating", False) or getattr(dtype, "is_complex", False))


def _tf_gradient_source(tf: Any, value: Any) -> Any:
    """Return the TensorFlow variable/tensor watched by ``GradientTape``."""

    if isinstance(value, (tf.Tensor, tf.Variable)):
        return value
    keras_value = getattr(value, "value", None)
    if isinstance(keras_value, (tf.Tensor, tf.Variable)):
        return keras_value
    return value


def _default_tf_loss(tf: Any, output: Any, targets: Any) -> Any:
    """Compute a default TensorFlow split-training loss."""

    if not _is_diff_tf_tensor(tf, output):
        raise SplitUnsupportedError(
            "Non-tensor TensorFlow split-training outputs require an explicit loss_fn.",
            context=SplitErrorContext(
                backend="tf",
                split_point="",
                module_path=None,
                op_type=None,
                layer_label=None,
                reason="default loss requires tensor output",
            ),
        )
    if isinstance(targets, tf.Tensor) and getattr(targets.dtype, "is_integer", False):
        if len(output.shape) >= 2 and len(targets.shape) == len(output.shape) - 1:
            return tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    targets,
                    output,
                    from_logits=True,
                )
            )
    return tf.reduce_mean(tf.math.squared_difference(output, targets))


def _train_suffix_tf(
    runtime: Any,
    boundary: ReplayBoundary,
    targets: Any,
    loss_fn: Callable[[Any, Any], Any] | None = None,
    optimizer: Any | None = None,
) -> tuple[Any, BoundaryGradients, bool]:
    """Train a TensorFlow suffix and return boundary gradients."""

    import tensorflow as tf

    runtime.validate_boundary(boundary)
    root_tensors: dict[str, Any] = {}
    replay_tensors: dict[str, Any] = {}
    for key, value in boundary.tensors.items():
        if _is_diff_tf_tensor(tf, value):
            root = tf.identity(value)
            root_tensors[key] = root
            replay_tensors[key] = root
        else:
            replay_tensors[key] = value
    replay_boundary = ReplayBoundary(
        backend=boundary.backend,
        tensors=replay_tensors,
        spec=boundary.spec,
        metadata={**boundary.metadata, "suffix_training_roots": tuple(root_tensors)},
    )
    suffix_vars = _trainable_param_handles(runtime, runtime.plan.suffix_node_ids)
    suffix_sources = [_tf_gradient_source(tf, var) for var in suffix_vars]
    with tf.GradientTape(persistent=True) as tape:
        for root in root_tensors.values():
            tape.watch(root)
        for source in suffix_sources:
            tape.watch(source)
        output = runtime.run_suffix(replay_boundary)
        loss = (
            loss_fn(output, targets)
            if loss_fn is not None
            else _default_tf_loss(tf, output, targets)
        )
    root_values = list(root_tensors.values())
    root_grads = tape.gradient(loss, root_values) if root_values else []
    gradients = {
        key: grad
        for key, grad in zip(root_tensors, root_grads, strict=False)
        if grad is not None
    }
    optimizer_applied = False
    if optimizer is not None and suffix_vars:
        var_grads = tape.gradient(loss, suffix_sources)
        pairs = [(grad, var) for grad, var in zip(var_grads, suffix_vars) if grad is not None]
        if pairs:
            optimizer.apply_gradients(pairs)
            optimizer_applied = True
    return loss, gradients, optimizer_applied


def _is_diff_paddle_tensor(paddle: Any, value: Any) -> bool:
    """Return whether ``value`` is differentiable for Paddle."""

    if not isinstance(value, paddle.Tensor):
        return False
    dtype = str(getattr(value, "dtype", ""))
    return any(token in dtype for token in ("float", "complex"))


def _default_paddle_loss(paddle: Any, output: Any, targets: Any) -> Any:
    """Compute a default Paddle split-training loss."""

    if not isinstance(output, paddle.Tensor):
        raise SplitUnsupportedError(
            "Non-tensor Paddle split-training outputs require an explicit loss_fn.",
            context=SplitErrorContext(
                backend="paddle",
                split_point="",
                module_path=None,
                op_type=None,
                layer_label=None,
                reason="default loss requires tensor output",
            ),
        )
    if isinstance(targets, paddle.Tensor) and "int" in str(getattr(targets, "dtype", "")):
        if len(output.shape) >= 2 and len(targets.shape) == len(output.shape) - 1:
            return paddle.nn.functional.cross_entropy(output, targets)
    return paddle.nn.functional.mse_loss(output, targets)


def _optimizer_clear_grad(optimizer: Any) -> None:
    """Clear gradients on a backend optimizer when it exposes a known method."""

    clear_grad = getattr(optimizer, "clear_grad", None)
    zero_grad = getattr(optimizer, "zero_grad", None)
    if callable(clear_grad):
        clear_grad()
    elif callable(zero_grad):
        zero_grad()


def _train_suffix_paddle(
    runtime: Any,
    boundary: ReplayBoundary,
    targets: Any,
    loss_fn: Callable[[Any, Any], Any] | None = None,
    optimizer: Any | None = None,
) -> tuple[Any, BoundaryGradients, bool]:
    """Train a Paddle suffix and return boundary gradients."""

    import paddle

    runtime.validate_boundary(boundary)
    root_tensors: dict[str, Any] = {}
    replay_tensors: dict[str, Any] = {}
    for key, value in boundary.tensors.items():
        if _is_diff_paddle_tensor(paddle, value):
            root = value.detach()
            root.stop_gradient = False
            root_tensors[key] = root
            replay_tensors[key] = root
        else:
            replay_tensors[key] = value
    replay_boundary = ReplayBoundary(
        backend=boundary.backend,
        tensors=replay_tensors,
        spec=boundary.spec,
        metadata={**boundary.metadata, "suffix_training_roots": tuple(root_tensors)},
    )
    if optimizer is not None:
        _optimizer_clear_grad(optimizer)
    output = runtime.run_suffix(replay_boundary)
    loss = (
        loss_fn(output, targets)
        if loss_fn is not None
        else _default_paddle_loss(paddle, output, targets)
    )
    loss.backward()
    gradients: BoundaryGradients = {}
    for key, root in root_tensors.items():
        grad = getattr(root, "grad", None)
        if grad is not None:
            gradients[key] = paddle.clone(grad)
    if optimizer is not None:
        optimizer.step()
    return loss, gradients, optimizer is not None


def _is_diff_jax_tensor(value: Any) -> bool:
    """Return whether ``value`` is differentiable for JAX."""

    dtype = getattr(value, "dtype", None)
    kind = getattr(dtype, "kind", None)
    return kind in {"f", "c"}


def _default_jax_loss(output: Any, targets: Any) -> Any:
    """Compute a default JAX split-training loss."""

    jnp = __import__("jax.numpy", fromlist=["numpy"])
    target_dtype = getattr(getattr(targets, "dtype", None), "kind", None)
    if target_dtype in {"i", "u"} and len(output.shape) >= 2:
        if len(targets.shape) == len(output.shape) - 1:
            import jax

            log_probs = jax.nn.log_softmax(output, axis=-1)
            gathered = jnp.take_along_axis(log_probs, jnp.expand_dims(targets, -1), axis=-1)
            return -jnp.mean(jnp.squeeze(gathered, axis=-1))
    return jnp.mean((output - targets) ** 2)


def _train_suffix_jax(
    runtime: Any,
    boundary: ReplayBoundary,
    targets: Any,
    loss_fn: Callable[[Any, Any], Any] | None = None,
    optimizer: Any | None = None,
) -> tuple[Any, BoundaryGradients, bool]:
    """Return JAX suffix boundary gradients without mutating parameters."""

    if optimizer is not None:
        raise SplitUnsupportedError(
            "JAX split training returns gradients; apply optimizer updates outside TorchLens.",
            context=_context(runtime, "jax optimizer unsupported"),
        )
    import jax

    runtime.validate_boundary(boundary)
    keys = [key for key, value in boundary.tensors.items() if _is_diff_jax_tensor(value)]
    values = [boundary.tensors[key] for key in keys]

    def suffix_loss(*roots: Any) -> Any:
        tensors = dict(boundary.tensors)
        tensors.update({key: root for key, root in zip(keys, roots, strict=False)})
        replay_boundary = ReplayBoundary(
            backend=boundary.backend,
            tensors=tensors,
            spec=boundary.spec,
            metadata={**boundary.metadata, "suffix_training_roots": tuple(keys)},
        )
        output = runtime.run_suffix(replay_boundary)
        if loss_fn is not None:
            return loss_fn(output, targets)
        return _default_jax_loss(output, targets)

    if not values:
        loss = suffix_loss()
        return loss, {}, False
    loss, grads = jax.value_and_grad(suffix_loss, argnums=tuple(range(len(values))))(*values)
    return (
        loss,
        {key: grad for key, grad in zip(keys, grads, strict=False) if grad is not None},
        False,
    )


def _tinygrad_tensor_type() -> Any:
    """Return the tinygrad Tensor class."""

    from tinygrad import Tensor

    return Tensor


def _is_diff_tinygrad_tensor(value: Any) -> bool:
    """Return whether ``value`` is differentiable for tinygrad."""

    Tensor = _tinygrad_tensor_type()
    if not isinstance(value, Tensor):
        return False
    dtype = str(getattr(value, "dtype", ""))
    return "float" in dtype or "half" in dtype or "bfloat" in dtype


def _default_tinygrad_loss(output: Any, targets: Any) -> Any:
    """Compute a default tinygrad split-training loss."""

    Tensor = _tinygrad_tensor_type()
    if not isinstance(output, Tensor):
        raise SplitUnsupportedError(
            "Non-tensor tinygrad split-training outputs require an explicit loss_fn.",
            context=SplitErrorContext(
                backend="tinygrad",
                split_point="",
                module_path=None,
                op_type=None,
                layer_label=None,
                reason="default loss requires tensor output",
            ),
        )
    target_dtype = str(getattr(targets, "dtype", ""))
    if isinstance(targets, Tensor) and "int" in target_dtype and len(output.shape) >= 2:
        if len(targets.shape) == len(output.shape) - 1:
            return output.sparse_categorical_crossentropy(targets)
    return ((output - targets) ** 2).mean()


def _require_tinygrad_optimizer(runtime: Any, optimizer: Any) -> None:
    """Validate a tinygrad optimizer-like object."""

    if not callable(getattr(optimizer, "zero_grad", None)) or not callable(
        getattr(optimizer, "step", None)
    ):
        raise SplitUnsupportedError(
            "tinygrad split training optimizer must expose zero_grad() and step().",
            context=_context(runtime, "invalid tinygrad optimizer"),
        )


def _tinygrad_optimizer_step(runtime: Any, optimizer: Any | None, *, before: bool = False) -> None:
    """Run a tinygrad optimizer method with Tensor.training temporarily enabled."""

    if optimizer is None:
        return
    _require_tinygrad_optimizer(runtime, optimizer)
    Tensor = _tinygrad_tensor_type()
    previous_training = bool(getattr(Tensor, "training", False))
    try:
        Tensor.training = True
        if before:
            optimizer.zero_grad()
        else:
            optimizer.step()
    finally:
        Tensor.training = previous_training


def _tinygrad_clone_grad(value: Any) -> Any:
    """Return a detached realized copy of a tinygrad gradient tensor."""

    from .adapters.tinygrad import TinygradSplitAdapter

    try:
        return TinygradSplitAdapter().clone(value)
    except (RuntimeError, AssertionError):
        return value


def _train_suffix_tinygrad(
    runtime: Any,
    boundary: ReplayBoundary,
    targets: Any,
    loss_fn: Callable[[Any, Any], Any] | None = None,
    optimizer: Any | None = None,
) -> tuple[Any, BoundaryGradients, bool]:
    """Train a tinygrad suffix and return boundary gradients."""

    runtime.validate_boundary(boundary)
    root_tensors: dict[str, Any] = {}
    replay_tensors: dict[str, Any] = {}
    for key, value in boundary.tensors.items():
        if _is_diff_tinygrad_tensor(value):
            root = value.detach()
            root.requires_grad = True
            root_tensors[key] = root
            replay_tensors[key] = root
        else:
            replay_tensors[key] = value
    replay_boundary = ReplayBoundary(
        backend=boundary.backend,
        tensors=replay_tensors,
        spec=boundary.spec,
        metadata={**boundary.metadata, "suffix_training_roots": tuple(root_tensors)},
    )
    _tinygrad_optimizer_step(runtime, optimizer, before=True)
    output = runtime.run_suffix(replay_boundary)
    loss = (
        loss_fn(output, targets)
        if loss_fn is not None
        else _default_tinygrad_loss(output, targets)
    )
    loss.backward()
    gradients: BoundaryGradients = {}
    for key, root in root_tensors.items():
        grad = getattr(root, "grad", None)
        if grad is not None:
            gradients[key] = _tinygrad_clone_grad(grad)
    _tinygrad_optimizer_step(runtime, optimizer, before=False)
    return loss, gradients, optimizer is not None


class TorchTrainingEngine:
    """Torch split-training engine."""

    name = "torch"

    def train_suffix(
        self,
        runtime: Any,
        boundary: ReplayBoundary,
        targets: Any,
        loss_fn: Callable[[Any, Any], Any] | None = None,
        optimizer: Any | None = None,
    ) -> TrainingStepResult:
        """Train a Torch suffix and return a structured result."""

        loss, grads, optimizer_applied = _train_suffix_torch(
            runtime,
            boundary,
            targets,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )
        return TrainingStepResult(loss, grads, optimizer_applied=optimizer_applied)

    def backward_prefix(
        self,
        runtime: Any,
        boundary: ReplayBoundary,
        boundary_grads: BoundaryGradients,
        optimizer: Any | None = None,
    ) -> Any:
        """Backpropagate Torch boundary gradients through the prefix."""

        return _backward_prefix_torch(runtime, boundary, boundary_grads, optimizer=optimizer)


class TensorFlowTrainingEngine:
    """TensorFlow split-training engine."""

    name = "tf"

    def train_suffix(
        self,
        runtime: Any,
        boundary: ReplayBoundary,
        targets: Any,
        loss_fn: Callable[[Any, Any], Any] | None = None,
        optimizer: Any | None = None,
    ) -> TrainingStepResult:
        """Train a TensorFlow suffix and return a structured result."""

        loss, grads, optimizer_applied = _train_suffix_tf(
            runtime,
            boundary,
            targets,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )
        return TrainingStepResult(loss, grads, optimizer_applied=optimizer_applied)

    def backward_prefix(
        self,
        runtime: Any,
        boundary: ReplayBoundary,
        boundary_grads: BoundaryGradients,
        optimizer: Any | None = None,
    ) -> Any:
        """Backpropagate TensorFlow boundary gradients through the prefix."""

        return _backward_prefix_tf(runtime, boundary, boundary_grads, optimizer=optimizer)


class PaddleTrainingEngine:
    """Paddle split-training engine."""

    name = "paddle"

    def train_suffix(
        self,
        runtime: Any,
        boundary: ReplayBoundary,
        targets: Any,
        loss_fn: Callable[[Any, Any], Any] | None = None,
        optimizer: Any | None = None,
    ) -> TrainingStepResult:
        """Train a Paddle suffix and return a structured result."""

        loss, grads, optimizer_applied = _train_suffix_paddle(
            runtime,
            boundary,
            targets,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )
        return TrainingStepResult(loss, grads, optimizer_applied=optimizer_applied)

    def backward_prefix(
        self,
        runtime: Any,
        boundary: ReplayBoundary,
        boundary_grads: BoundaryGradients,
        optimizer: Any | None = None,
    ) -> Any:
        """Backpropagate Paddle boundary gradients through the prefix."""

        return _backward_prefix_paddle(runtime, boundary, boundary_grads, optimizer=optimizer)


class JaxTrainingEngine:
    """JAX split-training engine."""

    name = "jax"

    def train_suffix(
        self,
        runtime: Any,
        boundary: ReplayBoundary,
        targets: Any,
        loss_fn: Callable[[Any, Any], Any] | None = None,
        optimizer: Any | None = None,
    ) -> TrainingStepResult:
        """Differentiate a JAX suffix and return a structured result."""

        loss, grads, optimizer_applied = _train_suffix_jax(
            runtime,
            boundary,
            targets,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )
        return TrainingStepResult(loss, grads, optimizer_applied=optimizer_applied)

    def backward_prefix(
        self,
        runtime: Any,
        boundary: ReplayBoundary,
        boundary_grads: BoundaryGradients,
        optimizer: Any | None = None,
    ) -> Any:
        """Return JAX prefix gradients via VJP recomputation."""

        return _backward_prefix_jax(runtime, boundary, boundary_grads, optimizer=optimizer)


class TinygradTrainingEngine:
    """tinygrad split-training engine."""

    name = "tinygrad"

    def train_suffix(
        self,
        runtime: Any,
        boundary: ReplayBoundary,
        targets: Any,
        loss_fn: Callable[[Any, Any], Any] | None = None,
        optimizer: Any | None = None,
    ) -> TrainingStepResult:
        """Train a tinygrad suffix and return a structured result."""

        loss, grads, optimizer_applied = _train_suffix_tinygrad(
            runtime,
            boundary,
            targets,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )
        return TrainingStepResult(loss, grads, optimizer_applied=optimizer_applied)

    def backward_prefix(
        self,
        runtime: Any,
        boundary: ReplayBoundary,
        boundary_grads: BoundaryGradients,
        optimizer: Any | None = None,
    ) -> Any:
        """Backpropagate tinygrad boundary gradients through the prefix."""

        return _backward_prefix_tinygrad(runtime, boundary, boundary_grads, optimizer=optimizer)


_TRAINING_ENGINES: dict[str, BackendTrainingEngine] = {
    "torch": TorchTrainingEngine(),
    "tf": TensorFlowTrainingEngine(),
    "tensorflow": TensorFlowTrainingEngine(),
    "paddle": PaddleTrainingEngine(),
    "jax": JaxTrainingEngine(),
    "tinygrad": TinygradTrainingEngine(),
}


def training_engine_for(backend: str) -> BackendTrainingEngine:
    """Return the split training engine for ``backend``."""

    try:
        return _TRAINING_ENGINES[backend]
    except KeyError as exc:
        raise SplitUnsupportedError(
            f"backend={backend!r} does not support split training.",
            context=SplitErrorContext(
                backend=backend,
                split_point="",
                module_path=None,
                op_type=None,
                layer_label=None,
                reason="unsupported split training",
            ),
        ) from exc


def train_suffix_result(
    runtime: Any,
    boundary: ReplayBoundary,
    targets: Any,
    loss_fn: Callable[[Any, Any], Any] | None = None,
    optimizer: Any | None = None,
) -> TrainingStepResult:
    """Train or differentiate a backend split suffix and return structured metadata."""

    return training_engine_for(runtime.adapter.name).train_suffix(
        runtime,
        boundary,
        targets,
        loss_fn=loss_fn,
        optimizer=optimizer,
    )


def train_suffix(
    runtime: Any,
    boundary: ReplayBoundary,
    targets: Any,
    loss_fn: Callable[[Any, Any], Any] | None = None,
    optimizer: Any | None = None,
) -> tuple[Any, BoundaryGradients]:
    """Train or differentiate a backend split suffix."""

    return train_suffix_result(
        runtime,
        boundary,
        targets,
        loss_fn=loss_fn,
        optimizer=optimizer,
    ).as_tuple()


def _backward_prefix_torch(
    runtime: Any,
    boundary: ReplayBoundary,
    boundary_grads: BoundaryGradients,
    optimizer: Any | None = None,
) -> None:
    """Backpropagate suffix boundary gradients through a graph-connected prefix."""

    torch = _require_torch(runtime)
    runtime.validate_boundary(boundary)
    if not boundary.metadata.get("supports_prefix_backward"):
        raise SplitUnsupportedError(
            "backward_prefix requires a boundary from run_training_prefix().",
            context=SplitErrorContext(
                backend="torch",
                split_point=runtime.split_spec.boundary,
                module_path=None,
                op_type=None,
                layer_label=None,
                reason="boundary is not graph-connected",
            ),
        )
    prefix_tensors = boundary.metadata.get("prefix_boundary_tensors", {})
    tensors: list[Any] = []
    grads: list[Any] = []
    for key, grad in boundary_grads.items():
        tensor = prefix_tensors.get(key)
        if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
            tensors.append(tensor)
            grads.append(grad)
    if not tensors:
        return
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    torch.autograd.backward(tensors, grads)
    if optimizer is not None:
        optimizer.step()


def _backward_prefix_tf(
    runtime: Any,
    boundary: ReplayBoundary,
    boundary_grads: BoundaryGradients,
    optimizer: Any | None = None,
) -> dict[str, Any]:
    """Backpropagate TensorFlow suffix gradients through a prefix tape."""

    import tensorflow as tf

    runtime.validate_boundary(boundary)
    if not boundary.metadata.get("supports_prefix_backward"):
        raise SplitUnsupportedError(
            "backward_prefix requires a boundary from run_training_prefix().",
            context=_context(runtime, "boundary is not graph-connected"),
        )
    tape = boundary.metadata.get("tf_tape")
    prefix_tensors = boundary.metadata.get("prefix_boundary_tensors", {})
    targets: list[Any] = []
    output_grads: list[Any] = []
    for key, grad in boundary_grads.items():
        tensor = prefix_tensors.get(key)
        if tensor is not None:
            targets.append(tensor)
            output_grads.append(grad)
    if not targets:
        return {}
    sources = _trainable_param_handles(runtime, runtime.plan.prefix_node_ids)
    gradient_sources = [_tf_gradient_source(tf, source) for source in sources]
    if not gradient_sources:
        return {}
    if tape is None:
        raise SplitUnsupportedError(
            "TensorFlow backward_prefix requires a live GradientTape boundary.",
            context=_context(runtime, "missing tensorflow gradient tape"),
        )
    grads = tape.gradient(targets, gradient_sources, output_gradients=output_grads)
    result = {
        str(getattr(source, "name", index)): grad
        for index, (source, grad) in enumerate(zip(sources, grads, strict=False))
        if grad is not None
    }
    if optimizer is not None:
        pairs = [(grad, source) for source, grad in zip(sources, grads) if grad is not None]
        if pairs:
            optimizer.apply_gradients(pairs)
    return result


def _backward_prefix_paddle(
    runtime: Any,
    boundary: ReplayBoundary,
    boundary_grads: BoundaryGradients,
    optimizer: Any | None = None,
) -> dict[str, Any]:
    """Backpropagate Paddle suffix gradients through a graph-connected prefix."""

    import paddle

    runtime.validate_boundary(boundary)
    if not boundary.metadata.get("supports_prefix_backward"):
        raise SplitUnsupportedError(
            "backward_prefix requires a boundary from run_training_prefix().",
            context=_context(runtime, "boundary is not graph-connected"),
        )
    prefix_tensors = boundary.metadata.get("prefix_boundary_tensors", {})
    tensors: list[Any] = []
    grads: list[Any] = []
    for key, grad in boundary_grads.items():
        tensor = prefix_tensors.get(key)
        if isinstance(tensor, paddle.Tensor):
            tensors.append(tensor)
            grads.append(grad)
    if not tensors:
        return {}
    if optimizer is not None:
        _optimizer_clear_grad(optimizer)
    paddle.autograd.backward(tensors, grad_tensors=grads)
    if optimizer is not None:
        optimizer.step()
    return {key: grad for key, grad in boundary_grads.items() if key in prefix_tensors}


def _backward_prefix_jax(
    runtime: Any,
    boundary: ReplayBoundary,
    boundary_grads: BoundaryGradients,
    optimizer: Any | None = None,
) -> dict[str, Any]:
    """Return JAX prefix input gradients via VJP recomputation."""

    if optimizer is not None:
        raise SplitUnsupportedError(
            "JAX split training returns gradients; apply optimizer updates outside TorchLens.",
            context=_context(runtime, "jax optimizer unsupported"),
        )
    import jax

    runtime.validate_boundary(boundary)
    if not boundary.metadata.get("supports_prefix_backward"):
        raise SplitUnsupportedError(
            "backward_prefix requires a boundary from run_training_prefix().",
            context=_context(runtime, "boundary is not graph-connected"),
        )
    keys = [key for key in boundary.spec if key in boundary_grads]
    if not keys:
        return {}
    inputs = tuple(boundary.metadata.get("prefix_inputs", ()))

    def prefix_outputs(*args: Any) -> tuple[Any, ...]:
        replay_boundary = runtime.run_training_prefix(*args)
        return tuple(replay_boundary.tensors[key] for key in keys)

    _outputs, pullback = jax.vjp(prefix_outputs, *inputs)
    input_grads = pullback(tuple(boundary_grads[key] for key in keys))
    return {"inputs": input_grads}


def _backward_prefix_tinygrad(
    runtime: Any,
    boundary: ReplayBoundary,
    boundary_grads: BoundaryGradients,
    optimizer: Any | None = None,
) -> dict[str, Any]:
    """Backpropagate tinygrad suffix gradients through a graph-connected prefix."""

    runtime.validate_boundary(boundary)
    if not boundary.metadata.get("supports_prefix_backward"):
        raise SplitUnsupportedError(
            "backward_prefix requires a boundary from run_training_prefix().",
            context=_context(runtime, "boundary is not graph-connected"),
        )
    prefix_tensors = boundary.metadata.get("prefix_boundary_tensors", {})
    _tinygrad_optimizer_step(runtime, optimizer, before=True)
    applied: dict[str, Any] = {}
    for key, grad in boundary_grads.items():
        tensor = prefix_tensors.get(key)
        if tensor is None or not _is_diff_tinygrad_tensor(tensor):
            continue
        tensor.backward(gradient=grad)
        applied[key] = grad
    _tinygrad_optimizer_step(runtime, optimizer, before=False)
    return applied


def backward_prefix(
    runtime: Any,
    boundary: ReplayBoundary,
    boundary_grads: BoundaryGradients,
    optimizer: Any | None = None,
) -> Any:
    """Backpropagate suffix boundary gradients through a graph-connected prefix."""

    return training_engine_for(runtime.adapter.name).backward_prefix(
        runtime,
        boundary,
        boundary_grads,
        optimizer=optimizer,
    )


def build_feature_cache(runtime: Any, dataloader: Any, cache_dir: str | Path) -> list[Path]:
    """Build a directory of boundary cache entries from a dataloader."""

    paths: list[Path] = []
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    for index, batch in enumerate(dataloader):
        inputs = batch if isinstance(batch, tuple) else (batch,)
        boundary = runtime.run_prefix(*inputs)
        path = root / f"boundary_{index:06d}"
        runtime.save_boundary(boundary, path)
        paths.append(path)
    return paths


class BoundaryCacheDataset:
    """Minimal dataset that yields cached replay boundaries."""

    def __init__(self, runtime: Any, cache_dir: str | Path) -> None:
        """Create a dataset over boundary cache directories."""

        self.runtime = runtime
        self.paths = sorted(Path(cache_dir).glob("boundary_*"))

    def __len__(self) -> int:
        """Return the number of cached boundaries."""

        return len(self.paths)

    def __getitem__(self, index: int) -> ReplayBoundary:
        """Load one cached boundary."""

        return self.runtime.load_boundary(self.paths[index])


def train_suffix_from_cache(
    runtime: Any,
    dataset: BoundaryCacheDataset,
    targets: Any,
    loss_fn: Callable[[Any, Any], Any] | None = None,
    optimizer: Any | None = None,
) -> list[tuple[Any, BoundaryGradients]]:
    """Train a suffix over cached boundaries."""

    results: list[tuple[Any, BoundaryGradients]] = []
    for index in range(len(dataset)):
        results.append(
            train_suffix(
                runtime,
                dataset[index],
                targets[index] if hasattr(targets, "__getitem__") else targets,
                loss_fn=loss_fn,
                optimizer=optimizer,
            )
        )
    return results


__all__ = [
    "BackendTrainingEngine",
    "BoundaryCacheDataset",
    "BoundaryGradients",
    "JaxTrainingEngine",
    "PaddleTrainingEngine",
    "TensorFlowTrainingEngine",
    "TinygradTrainingEngine",
    "TorchTrainingEngine",
    "TrainingStepResult",
    "backward_prefix",
    "build_feature_cache",
    "train_suffix",
    "train_suffix_result",
    "train_suffix_from_cache",
    "training_engine_for",
]
