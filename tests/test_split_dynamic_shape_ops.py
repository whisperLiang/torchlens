"""Focused dynamic-batch split replay tests for shape-producing ops."""

from __future__ import annotations

import copy
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.split.generated import _dynamic_batch_literal
from torchlens.split.generated import _runtime_batch_from_env
from torchlens.split.trace_graph import TraceGraph, TraceNode


RUNTIME_BATCHES = (1, 2, 4, 8, 32)
TRACE_BATCH = 2


class NestedTensorLike:
    """Minimal object exposing RF-DETR-style tensor and mask leaves."""

    def __init__(self, tensors: torch.Tensor, mask: torch.Tensor) -> None:
        self.tensors = tensors
        self.mask = mask


class NestedTensorPropertyWithCache:
    """Tensor-like input whose semantic tensor is exposed as a property."""

    def __init__(self, tensors: torch.Tensor, cache: torch.Tensor) -> None:
        self.cache = cache
        self._tensors = tensors

    @property
    def tensors(self) -> torch.Tensor:
        """Return the tensor leaf consumed by the model."""

        return self._tensors


class NestedMaskFirstNet(nn.Module):
    """Toy nested-input model that reads mask before image tensors."""

    def __init__(self) -> None:
        super().__init__()
        self.prefix = nn.Conv2d(3, 4, kernel_size=1)
        self.head = nn.Linear(4, 2)

    def forward(self, nested: NestedTensorLike) -> torch.Tensor:
        """Use both nested leaves, with mask access preceding tensor access."""

        mask = nested.mask
        mask_bias = mask.float()[:, :1, :1].view(mask.shape[0], 1, 1, 1)
        x = nested.tensors
        hidden = torch.relu(self.prefix(x + mask_bias))
        return self.head(hidden.mean(dim=(2, 3)))


class NestedTensorPropertyNet(nn.Module):
    """Toy model consuming a tensor-like property input."""

    def forward(self, nested: NestedTensorPropertyWithCache) -> torch.Tensor:
        """Use only the semantic ``tensors`` property."""

        return nested.tensors * 2


class DynamicReshapeViewNet(nn.Module):
    """Toy model covering reshape, view, flatten, and runtime B."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(3 * 4 * 5, 16)
        self.out = nn.Linear(16, 7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run shape-producing tensor views using runtime batch size."""

        batch = x.shape[0]
        y = x.reshape(batch, -1)
        y = self.proj(y)
        y = y.view(batch, -1)
        y = torch.flatten(y, 1)
        return self.out(y)


class DynamicFoldedBatchViewNet(nn.Module):
    """Toy model covering leading reshape/view dimensions derived from B multiples."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fold channels into the leading dimension, then restore runtime B."""

        batch = x.shape[0]
        y = x.reshape(batch, 2, -1)
        y = y.reshape(batch * 2, -1)
        return y.reshape(batch, -1)


class DynamicFactoryNet(nn.Module):
    """Toy model covering batch-dependent factory tensors."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Build factory tensors whose first dimension follows runtime B."""

        batch = x.shape[0]
        idx = torch.arange(batch, device=x.device, dtype=x.dtype).view(batch, 1)
        bias = torch.ones((batch, 1), device=x.device, dtype=x.dtype)
        scratch = torch.zeros((batch, 1), device=x.device, dtype=x.dtype)
        inherited_bias = x.new_ones((batch, 1))
        inherited_scratch = x.new_zeros((batch, 1))
        y = x.flatten(1).mean(dim=1, keepdim=True)
        like_bias = torch.ones_like(y)
        like_scratch = torch.zeros_like(y)
        return y + idx * 0.01 + bias + scratch + inherited_bias + inherited_scratch + like_bias + like_scratch


class DynamicExpandRepeatNet(nn.Module):
    """Toy model covering expand, repeat, and batch-dependent expand shapes."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expand a parameter over runtime B and repeat input-derived features."""

        batch = x.shape[0]
        base = self.weight.view(1, -1).expand(batch, -1)
        repeated = x.mean(dim=(2, 3)).repeat(1, 2)
        return torch.cat([base, repeated], dim=1)


class DynamicRepeatBatchlessNet(nn.Module):
    """Toy model covering repeat(batch, ...) on a batchless tensor."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat a batchless parameter over the runtime batch size."""

        batch = x.shape[0]
        base = self.weight.view(1, -1).repeat(batch, 1)
        return base + x.mean(dim=(2, 3))


class BatchlessRepeatOnlyNet(nn.Module):
    """Toy model whose suffix needs B but whose boundary may be batchless."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat a parameter from ``x.shape[0]`` without tensor-input parents."""

        batch = x.shape[0]
        return self.weight.view(1, -1).repeat(batch, 1)


class LiveParamAndBatchConstantNet(nn.Module):
    """Toy model mixing live parameter-derived no-input nodes and B-shaped constants."""

    def __init__(self) -> None:
        super().__init__()
        self.prefix = nn.Linear(3 * 4 * 5, 4)
        self.query = nn.Parameter(torch.randn(4, 6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input with a live parameter view and add a runtime-B constant."""

        batch = x.shape[0]
        hidden = self.prefix(x.flatten(1))
        query, _unused = self.query.chunk(2, dim=1)
        query = query.transpose(0, 1)[:, :4].transpose(0, 1)
        dynamic_bias = torch.ones((batch, 3), device=x.device, dtype=x.dtype)
        return hidden @ query + dynamic_bias


class DynamicStackCatChunkNet(nn.Module):
    """Toy model covering stack, cat, chunk, and tuple getitem."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Round-trip stacked features through chunk/getitem and cat."""

        a = x.mean(dim=(2, 3))
        b = x.amax(dim=(2, 3))
        c = torch.stack([a, b], dim=1)
        d = c.flatten(1)
        left, right = torch.chunk(d, chunks=2, dim=1)
        return torch.cat([left, right], dim=1)


class DynamicMeshgridAnchorToy(nn.Module):
    """Toy model covering arange/meshgrid anchor-like generation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate a grid on the input device/dtype and combine with x."""

        _batch, _channels, height, width = x.shape
        yy, xx = torch.meshgrid(
            torch.arange(height, device=x.device, dtype=x.dtype),
            torch.arange(width, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        grid = (xx + yy).view(1, 1, height, width)
        y = x + grid
        return y.mean(dim=(1, 2, 3))


class DynamicBatchIndexNet(nn.Module):
    """Toy model covering batch-dependent arange indexing."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Use arange(B) as a batch index into flattened features."""

        batch = x.shape[0]
        flat = x.flatten(1)
        idx = torch.arange(batch, device=x.device)
        col = torch.remainder(idx, flat.shape[1])
        return flat[idx, col].view(batch, 1)


class KwargGatherIndexNet(nn.Module):
    """Toy model whose gather index dependency is passed by keyword."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Gather the highest-scoring row using an index built by repeat."""

        scores = x.mean(dim=2)
        idx = scores.argmax(dim=1, keepdim=True)
        index = idx.unsqueeze(-1).repeat(1, 1, x.shape[2])
        return torch.gather(x, dim=1, index=index).squeeze(1)


@pytest.mark.parametrize(
    ("model_cls", "input_shape_without_batch"),
    (
        (DynamicReshapeViewNet, (3, 4, 5)),
        (DynamicFoldedBatchViewNet, (2, 3, 5)),
        (DynamicFactoryNet, (3, 4, 5)),
        (DynamicExpandRepeatNet, (3, 4, 5)),
        (DynamicRepeatBatchlessNet, (3, 4, 5)),
        (DynamicStackCatChunkNet, (3, 4, 5)),
        (DynamicMeshgridAnchorToy, (3, 4, 5)),
        (DynamicBatchIndexNet, (3, 4, 5)),
    ),
)
def test_dynamic_batch_shape_ops_replay_across_split_points(
    model_cls: type[nn.Module],
    input_shape_without_batch: tuple[int, ...],
) -> None:
    """Trace at B=2 and replay shape-producing ops at multiple runtime batches."""

    assert_dynamic_batch_split_replay(
        model_cls,
        input_shape_without_batch,
        boundaries=("percent:25", "percent:50", "percent:75"),
    )


def test_dynamic_shape_codegen_does_not_hardcode_traced_batch_size() -> None:
    """Batch-sensitive shape literals are rewritten to the runtime batch."""

    reshape_graph = _trace_for(DynamicReshapeViewNet(), (3, 4, 5))
    folded_graph = _trace_for(DynamicFoldedBatchViewNet(), (2, 3, 5))
    factory_graph = _trace_for(DynamicFactoryNet(), (3, 4, 5))
    expand_graph = _trace_for(DynamicExpandRepeatNet(), (3, 4, 5))
    repeat_graph = _trace_for(DynamicRepeatBatchlessNet(), (3, 4, 5))

    cases = (
        (reshape_graph, _first_node(reshape_graph, "reshape"), ("args", 1), TRACE_BATCH, 4),
        (reshape_graph, _first_node(reshape_graph, "view"), ("args", 1), TRACE_BATCH, 4),
        (
            folded_graph,
            _first_node_with_shape(folded_graph, "reshape", (TRACE_BATCH * 2, 15)),
            ("args", 1),
            TRACE_BATCH * 2,
            8,
        ),
        (expand_graph, _first_node(expand_graph, "expand"), ("args", 1), TRACE_BATCH, 4),
        (repeat_graph, _first_node(repeat_graph, "repeat"), ("args", 1), TRACE_BATCH, 4),
        (factory_graph, _first_node(factory_graph, "zeros"), ("args", 0, 0), TRACE_BATCH, 4),
        (factory_graph, _first_node(factory_graph, "ones"), ("args", 0, 0), TRACE_BATCH, 4),
        (factory_graph, _first_node(factory_graph, "new_zeros"), ("args", 1, 0), TRACE_BATCH, 4),
        (factory_graph, _first_node(factory_graph, "new_ones"), ("args", 1, 0), TRACE_BATCH, 4),
        (factory_graph, _first_node(factory_graph, "arange"), ("args", 0), TRACE_BATCH, 4),
    )

    for graph, node, path, literal_value, expected_value in cases:
        env = _static_codegen_env(graph, node)
        assert _dynamic_batch_literal(literal_value, node, graph, env, path) == expected_value

    repeat_existing_batch_node = _first_node(expand_graph, "repeat")
    env = {expand_graph.input_nodes[0]: torch.randn(4, 3, 4, 5)}
    assert _dynamic_batch_literal(1, repeat_existing_batch_node, expand_graph, env, ("args", 1)) is None


def test_nested_tensor_like_inputs_replay_across_runtime_batches() -> None:
    """Nested ``.tensors``/``.mask`` inputs keep leaf matching dynamic across B."""

    torch.manual_seed(0)
    model = NestedMaskFirstNet().eval()
    trace_inputs = _nested_batch(1)
    runtime = tl.prepare_split(
        model,
        trace_inputs,
        tl.SplitSpec(boundary="after:prefix", dynamic_batch=(1, 4)),
    )

    for batch_size in (1, 2, 4):
        inputs = _nested_batch(batch_size)
        with torch.no_grad():
            split_out = runtime.replay(inputs)
            full_out = model(inputs)
        assert_tree_allclose(split_out, full_out)


def test_nested_tensor_like_inputs_split_training_across_runtime_batches() -> None:
    """Split training accepts nested inputs whose runtime batch differs from trace B."""

    torch.manual_seed(0)
    base_model = NestedMaskFirstNet().train()
    full_model = copy.deepcopy(base_model).train()
    split_model = copy.deepcopy(base_model).train()
    inputs = _nested_batch(2)
    targets = torch.randn(2, 2)

    full_optimizer = torch.optim.SGD(full_model.parameters(), lr=0.05)
    full_optimizer.zero_grad(set_to_none=True)
    full_loss = torch.nn.functional.mse_loss(full_model(inputs), targets)
    full_loss.backward()
    full_optimizer.step()

    runtime = tl.prepare_split(
        split_model,
        _nested_batch(1),
        tl.SplitSpec(boundary="after:prefix", dynamic_batch=(1, 4), trainable=True),
    )
    prefix_optimizer = torch.optim.SGD(split_model.prefix.parameters(), lr=0.05)
    suffix_optimizer = torch.optim.SGD(split_model.head.parameters(), lr=0.05)
    boundary = runtime.run_training_prefix(inputs)
    loss, boundary_grads = runtime.train_suffix(
        boundary,
        targets,
        torch.nn.functional.mse_loss,
        suffix_optimizer,
    )
    runtime.backward_prefix(boundary, boundary_grads, prefix_optimizer)

    assert torch.allclose(loss, full_loss.detach(), atol=1e-5, rtol=1e-4)
    for full_param, split_param in zip(
        full_model.parameters(),
        split_model.parameters(),
        strict=True,
    ):
        assert torch.allclose(full_param, split_param, atol=1e-5, rtol=1e-4)


def test_nested_tensor_property_precedes_generic_tensor_attrs() -> None:
    """Semantic ``.tensors`` leaves are matched before unrelated object attrs."""

    model = NestedTensorPropertyNet().eval()
    trace_inputs = NestedTensorPropertyWithCache(
        tensors=torch.arange(3, dtype=torch.float32).view(1, 3),
        cache=torch.full((1, 3), 100.0),
    )
    runtime = tl.prepare_split(
        model,
        trace_inputs,
        tl.SplitSpec(boundary="percent:50", dynamic_batch=(1, 4)),
    )

    inputs = NestedTensorPropertyWithCache(
        tensors=torch.arange(6, dtype=torch.float32).view(2, 3),
        cache=torch.full((2, 3), 200.0),
    )
    with torch.no_grad():
        split_out = runtime.replay(inputs)
        full_out = model(inputs)
    assert_tree_allclose(split_out, full_out)


def test_runtime_batch_inference_prefers_symbolic_batch_tensors() -> None:
    """Suffix envs may contain non-batch tensors before the true batch tensor."""

    graph = _trace_for(DynamicRepeatBatchlessNet(), (3, 4, 5))
    batch_node = next(
        node
        for node in graph.ordered_nodes()
        if node.symbolic_output_shape is not None
        and node.symbolic_output_shape
        and node.symbolic_output_shape[0] == graph.batch_symbol
        and not node.is_input
    )
    non_batch_node = next(
        node
        for node in graph.ordered_nodes()
        if node.output_shape is not None
        and node.output_shape
        and node.symbolic_output_shape is not None
        and node.symbolic_output_shape[0] != graph.batch_symbol
    )

    env = {
        non_batch_node.torchlens_label: torch.randn(2),
        batch_node.torchlens_label: torch.randn(4, *batch_node.output_shape[1:]),
    }

    assert _runtime_batch_from_env(env, graph) == 4


def test_live_param_policy_does_not_capture_batch_dynamic_constants() -> None:
    """Trainable suffix live-param closure leaves unrelated B-shaped constants dynamic."""

    torch.manual_seed(0)
    model = LiveParamAndBatchConstantNet().eval()
    trace_x = torch.randn(TRACE_BATCH, 3, 4, 5)
    runtime = tl.prepare_split(
        model,
        trace_x,
        tl.SplitSpec(boundary="after:prefix", dynamic_batch=(1, 8), trainable=True),
    )

    policy_by_op = {
        (node.op_type, node.torchlens_label): node.replay_source_policy
        for node in runtime.trace_graph.ordered_nodes()
    }
    assert any(op_type == "chunk" and policy == "live_param" for (op_type, _), policy in policy_by_op.items())
    assert any(policy == "live_param_derived" for policy in policy_by_op.values())
    assert any(
        op_type == "ones" and policy == "batch_dynamic_constant"
        for (op_type, _), policy in policy_by_op.items()
    )

    x = torch.randn(4, 3, 4, 5)
    with torch.no_grad():
        split_out = runtime.replay(x)
        full_out = model(x)
    assert_tree_allclose(split_out, full_out)


def test_batchless_repeat_dynamicizes_singleton_trace_batch() -> None:
    """Batchless singleton views still repeat to runtime B after a B=1 trace."""

    assert_dynamic_batch_split_replay(
        DynamicRepeatBatchlessNet,
        (3, 4, 5),
        boundaries=("percent:25", "percent:50", "percent:75"),
        runtime_batches=(1, 2, 4),
        trace_batch=1,
    )


def test_boundary_metadata_carries_batch_for_batchless_suffix_shape_ops() -> None:
    """Suffix shape ops use boundary metadata when boundary tensors are batchless."""

    torch.manual_seed(0)
    model = BatchlessRepeatOnlyNet().eval()
    trace_x = torch.randn(1, 3, 4, 5)
    probe_runtime = tl.prepare_split(
        model,
        trace_x,
        tl.SplitSpec(boundary="percent:50", dynamic_batch=(1, 4)),
    )
    view_label = _first_node(probe_runtime.trace_graph, "view").torchlens_label
    runtime = tl.prepare_split(
        model,
        trace_x,
        tl.SplitSpec(boundary=f"after:{view_label}", dynamic_batch=(1, 4)),
    )

    for batch_size in (2, 4):
        x = torch.randn(batch_size, 3, 4, 5)
        boundary = runtime.run_prefix(x)
        assert boundary.batch_size == batch_size
        assert {tuple(tensor.shape) for tensor in boundary.tensors.values()} == {(1, 3)}
        with torch.no_grad():
            split_out = runtime.run_suffix(boundary)
            full_out = model(x)
        assert_tree_allclose(split_out, full_out)


def _nested_batch(batch_size: int) -> NestedTensorLike:
    """Return a deterministic nested tensor-like input batch."""

    tensors = torch.randn(batch_size, 3, 4, 5)
    mask = torch.zeros(batch_size, 4, 5, dtype=torch.bool)
    if batch_size > 1:
        mask[1::2, 0, 0] = True
    return NestedTensorLike(tensors=tensors, mask=mask)


def _static_codegen_env(graph: TraceGraph, node: TraceNode) -> dict[str, torch.Tensor]:
    """Build enough runtime-shaped env for static dynamic literal checks."""

    env = {graph.input_nodes[0]: torch.randn(4, 3, 4, 5)}
    for parent_label in node.parents:
        parent = graph.get(parent_label)
        if parent.output_shape is None:
            continue
        shape = list(parent.output_shape)
        if (
            parent.symbolic_output_shape is not None
            and parent.symbolic_output_shape
            and parent.symbolic_output_shape[0] == graph.batch_symbol
        ):
            shape[0] = 4
        env[parent.torchlens_label] = torch.randn(*shape)
    return env


def test_frontier_includes_kwarg_parent_refs() -> None:
    """Boundary frontier includes ParentRef dependencies stored in kwargs."""

    torch.manual_seed(0)
    model = KwargGatherIndexNet().eval()
    trace_x = torch.randn(2, 5, 3)
    graph = _trace_for(model, (5, 3))
    repeat_label = _first_node(graph, "repeat").torchlens_label

    runtime = tl.prepare_split(
        model,
        trace_x,
        tl.SplitSpec(boundary=f"after:{repeat_label}", dynamic_batch=(1, 8)),
    )
    assert repeat_label in runtime.plan.boundary_nodes

    for batch_size in (1, 2, 4):
        x = torch.randn(batch_size, 5, 3)
        with torch.no_grad():
            split_out = runtime.replay(x)
            full_out = model(x)
        assert_tree_allclose(split_out, full_out)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("model_cls", (DynamicFactoryNet, DynamicBatchIndexNet))
def test_dynamic_factory_and_index_ops_replay_on_cuda(model_cls: type[nn.Module]) -> None:
    """CUDA trace/runtime keeps factory tensors on the runtime CUDA device."""

    assert_dynamic_batch_split_replay(
        model_cls,
        (3, 4, 5),
        boundaries=("percent:25", "percent:50", "percent:75"),
        device=torch.device("cuda"),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("model_cls", (DynamicFactoryNet, DynamicBatchIndexNet))
def test_dynamic_factory_and_index_cpu_prefix_cuda_suffix(model_cls: type[nn.Module]) -> None:
    """CPU prefix boundaries can feed CUDA suffixes for factory/index toy models."""

    torch.manual_seed(0)
    base_model = model_cls().eval()
    cpu_model = model_cls().eval()
    cuda_model = model_cls().cuda().eval()
    cpu_model.load_state_dict(base_model.state_dict())
    cuda_model.load_state_dict(base_model.state_dict())

    trace_x = torch.randn(TRACE_BATCH, 3, 4, 5)
    spec = tl.SplitSpec(boundary="percent:50", dynamic_batch=(1, max(RUNTIME_BATCHES)))
    cpu_runtime = tl.prepare_split(cpu_model, trace_x, spec)
    cuda_runtime = tl.prepare_split(cuda_model, trace_x.cuda(), spec)

    for batch_size in RUNTIME_BATCHES:
        x_cpu = torch.randn(batch_size, 3, 4, 5)
        x_cuda = x_cpu.cuda()
        with torch.no_grad():
            boundary = cpu_runtime.run_prefix(x_cpu)
            split_out = cuda_runtime.run_suffix(boundary)
            full_out = cuda_model(x_cuda)
        assert_tree_allclose(split_out, full_out)
        assert _tree_device(split_out) == "cuda"


def assert_dynamic_batch_split_replay(
    model_cls_or_model: type[nn.Module] | nn.Module,
    input_shape_without_batch: tuple[int, ...],
    boundaries: tuple[str, ...],
    runtime_batches: tuple[int, ...] = RUNTIME_BATCHES,
    trace_batch: int = TRACE_BATCH,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    device: torch.device | None = None,
) -> None:
    """Assert split replay matches full model for dynamic runtime batches."""

    torch.manual_seed(0)
    model = model_cls_or_model() if isinstance(model_cls_or_model, type) else model_cls_or_model
    if device is not None:
        model = model.to(device)
    model.eval()
    trace_x = torch.randn(trace_batch, *input_shape_without_batch, device=device)

    for boundary in boundaries:
        runtime = tl.prepare_split(
            model,
            trace_x,
            tl.SplitSpec(boundary=boundary, dynamic_batch=(1, max(runtime_batches))),
        )
        for batch_size in runtime_batches:
            x = torch.randn(batch_size, *input_shape_without_batch, device=device)
            with torch.no_grad():
                full_out = model(x)
                split_out = runtime.run_suffix(runtime.run_prefix(x))
            assert_tree_allclose(split_out, full_out, atol=atol, rtol=rtol)


def _trace_for(model: nn.Module, input_shape_without_batch: tuple[int, ...]) -> TraceGraph:
    """Return the split TraceGraph produced by a trace-batch example."""

    runtime = tl.prepare_split(
        model.eval(),
        torch.randn(TRACE_BATCH, *input_shape_without_batch),
        tl.SplitSpec(boundary="percent:50", dynamic_batch=(1, max(RUNTIME_BATCHES))),
    )
    return runtime.trace_graph


def _first_node(graph: TraceGraph, op_type: str) -> TraceNode:
    """Return the first node with the requested op type."""

    for node in graph.ordered_nodes():
        if node.op_type == op_type:
            return node
    raise AssertionError(f"Trace graph did not contain op {op_type!r}.")


def _first_node_with_shape(
    graph: TraceGraph,
    op_type: str,
    output_shape: tuple[int, ...],
) -> TraceNode:
    """Return the first node with the requested op type and traced output shape."""

    for node in graph.ordered_nodes():
        if node.op_type == op_type and node.output_shape == output_shape:
            return node
    raise AssertionError(f"Trace graph did not contain op {op_type!r} with shape {output_shape!r}.")


def assert_tree_allclose(
    left: Any,
    right: Any,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> None:
    """Assert nested outputs match."""

    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        assert torch.allclose(left, right, atol=atol, rtol=rtol)
        return
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)) and len(left) == len(right):
        for left_item, right_item in zip(left, right, strict=True):
            assert_tree_allclose(left_item, right_item, atol=atol, rtol=rtol)
        return
    if isinstance(left, dict) and isinstance(right, dict) and set(left) == set(right):
        for key in left:
            assert_tree_allclose(left[key], right[key], atol=atol, rtol=rtol)
        return
    assert left == right


def _tree_device(value: Any) -> str | None:
    if isinstance(value, torch.Tensor):
        return value.device.type
    if isinstance(value, (tuple, list)):
        devices = {_tree_device(item) for item in value}
        devices.discard(None)
        return devices.pop() if len(devices) == 1 else None
    if isinstance(value, dict):
        devices = {_tree_device(item) for item in value.values()}
        devices.discard(None)
        return devices.pop() if len(devices) == 1 else None
    return None
