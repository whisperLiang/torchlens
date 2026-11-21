"""Exercise each registered TensorFlow native replay contract through its callers."""

from __future__ import annotations

from collections.abc import Callable
from inspect import signature
from typing import Any

import pytest

import torchlens as tl
from torchlens.backends.tf._native_op_specs import _MISSING, _NATIVE_OP_SPECS
from torchlens.backends.tf._raw_op_replay import _pause_tf_replay, _prepare_native_call
from torchlens.backends.tf.validation import validate_tf_trace_detailed
from torchlens.split import prepare_graph

pytestmark = pytest.mark.tf_backend

# Explicit names retain coverage if an entry disappears from the registry.
_UNARY = {
    "Elu": "features",
    "Erf": "x",
    "Erfc": "x",
    "Exp": "x",
    "Identity": "input",
    "Log": "x",
    "Neg": "x",
    "Relu": "features",
    "Relu6": "features",
    "Rsqrt": "x",
    "Shape": "input",
    "Sigmoid": "x",
    "Sin": "x",
    "Softmax": "logits",
    "Sqrt": "x",
    "Squeeze": "input",
    "StopGradient": "input",
    "Tanh": "x",
}
_BINARY = (
    "AddV2",
    "Greater",
    "Less",
    "Maximum",
    "Minimum",
    "Mul",
    "Pow",
    "RealDiv",
    "SquaredDifference",
    "Sub",
)
_REDUCTIONS = ("Mean", "Prod", "Sum")
_SPECIAL = (
    "AvgPool",
    "BatchMatMulV2",
    "BiasAdd",
    "BroadcastTo",
    "Cast",
    "ConcatV2",
    "Conv2D",
    "DepthwiseConv2dNative",
    "Einsum",
    "ExpandDims",
    "Fill",
    "GatherV2",
    "MatMul",
    "MaxPool",
    "Pack",
    "Pad",
    "ReadVariableOp",
    "Reshape",
    "SelectV2",
    "Slice",
    "Split",
    "StridedSlice",
    "Tile",
    "Transpose",
)
_CASES = sorted((*_UNARY, *_BINARY, *_REDUCTIONS, *_SPECIAL))


def test_tf_registry_matches_raw_ops_python_signatures() -> None:
    """Check keyword names, requiredness and native defaults without OpDef at runtime."""

    tf = pytest.importorskip("tensorflow")
    assert set(_CASES) == _NATIVE_OP_SPECS.keys()
    for name, spec in _NATIVE_OP_SPECS.items():
        parameters = signature(getattr(tf.raw_ops, name)).parameters
        assert set(spec.inputs).issubset(parameters), name
        assert all(parameters[key].default is parameters[key].empty for key in spec.inputs)
        for rule in spec.attrs:
            if rule.name not in parameters and rule.default is not _MISSING:
                # The matmul gradient hints are optional across TensorFlow versions.
                assert name in {"MatMul", "BatchMatMulV2"} and rule.name.startswith("grad_")
                continue
            native = parameters[rule.name].default
            if rule.default is _MISSING:
                assert native is parameters[rule.name].empty, (name, rule.name)
            else:
                if isinstance(native, list):
                    native = tuple(native)
                elif isinstance(native, tf.dtypes.DType):
                    native = native.name
                assert rule.default == native, (name, rule.name)


def test_tf_native_replay_pauses_capture_for_its_segment() -> None:
    """Replay can run during a TF callback capture without logging its native calls."""

    tf = pytest.importorskip("tensorflow")
    inputs = tf.constant([[0.25, 0.5]], dtype=tf.float64)
    graph = prepare_graph(lambda x: x * x, inputs, backend="tf")
    runtime = graph.materialize(graph.analyze(graph.compute_dag().to_partition(())))

    def model(x: Any) -> Any:
        intermediate = tf.sin(x)
        runtime.replay(x)
        return tf.tanh(intermediate)

    trace = tl.trace(model, inputs, backend="tf")
    captured = {record.op_type for record in trace._tf_op_captures}
    assert {"Sin", "Tanh"}.issubset(captured)
    assert "Mul" not in captured


def test_tf_validation_detects_a_wrong_native_output_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Detect a native replay with identical values placed on another device."""

    tf = pytest.importorskip("tensorflow")
    if not tf.config.list_logical_devices("GPU"):
        pytest.skip("TensorFlow GPU unavailable")

    with tf.device("/GPU:0"):
        inputs = tf.constant([[-1.0, 2.0]], dtype=tf.float32)
        trace = tl.trace(tf.nn.relu, inputs, backend="tf")
    capture = next(item for item in trace._tf_op_captures if item.op_type == "Relu")
    assert "GPU:0" in capture.output_tensor.device
    assert validate_tf_trace_detailed(trace).failures == ()
    raw_relu = tf.raw_ops.Relu

    def execute_on_cpu(**kwargs: Any) -> Any:
        with tf.device("/CPU:0"):
            return raw_relu(**kwargs)

    monkeypatch.setattr(tf.raw_ops, "Relu", execute_on_cpu)
    result = validate_tf_trace_detailed(trace)
    assert any("Relu:replay_mismatch" in failure for failure in result.failures)


def _pool_or_conv(tf: Any, name: str, raw: Callable[..., Any]) -> tuple[Any, Callable[..., Any]]:
    """Generate spatial inputs with distinct values for the native window families."""

    image = tf.reshape(tf.range(18, dtype=tf.float64) / 9, (1, 3, 3, 2))
    if name in {"AvgPool", "MaxPool"}:
        field = "value" if name == "AvgPool" else "input"
        return image, lambda x: raw(
            **{field: x}, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=b"SAME"
        )
    kernel = tf.reshape(tf.range(1, 9, dtype=tf.float64) / 8, (2, 2, 2, 1))
    kwargs = {"strides": [1, 1, 1, 1], "padding": "VALID", "dilations": [1, 1, 1, 1]}
    return image, lambda x: raw(input=x, filter=kernel, **kwargs)


def _resource_read(tf: Any, raw: Callable[..., Any]) -> Any:
    """Use a real TensorFlow variable and keep the model input in the DAG."""

    class Reader(tf.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = tf.Variable([[0.1, 0.2, 0.3, 0.4]], dtype=tf.float64)

        def __call__(self, x: Any) -> Any:
            return x + raw(resource=self.weight.handle, dtype=self.weight.dtype)

    return Reader()


def _remaining_model(tf: Any, name: str, raw: Callable[..., Any]) -> Callable[..., Any]:
    """Supply nontrivial native operand bindings for the remaining signatures."""

    scalar = tf.constant(1, dtype=tf.int32)
    other = tf.constant([[0.3, 0.6, 0.9, 0.4]], dtype=tf.float64)
    dimension = tf.constant([1, 2, 2], dtype=tf.int32)
    matrix = tf.constant([[0.2], [0.6], [0.7], [0.4]], dtype=tf.float64)
    choices: dict[str, Callable[..., Any]] = {
        "BiasAdd": lambda x: raw(value=x, bias=other[0]),
        "BroadcastTo": lambda x: raw(input=x, shape=(2, 4)),
        "Cast": lambda x: raw(x=x, DstT=tf.float32, Truncate=None),
        "ConcatV2": lambda x: raw(values=(x, other), axis=scalar),
        "Einsum": lambda x: raw(inputs=(x, matrix), equation=b"ij,jk->ik"),
        "ExpandDims": lambda x: raw(input=x, axis=scalar),
        "Fill": lambda x: x + raw(dims=(1, 4), value=0.25),
        "GatherV2": lambda x: raw(params=x, indices=((3, 1),), axis=scalar, batch_dims=1),
        "MatMul": lambda x: raw(a=x, b=tf.transpose(matrix), transpose_b=True),
        "Pack": lambda x: raw(values=(x,), axis=1),
        "Pad": lambda x: raw(input=x, paddings=((0, 0), (1, 0))),
        "Reshape": lambda x: raw(tensor=x, shape=dimension),
        "SelectV2": lambda x: raw(condition=((True, False, True, False),), t=x, e=other),
        "Slice": lambda x: raw(input=x, begin=(0, 1), size=(-1, 2)),
        "Split": lambda x: raw(axis=scalar, value=x, num_split=2),
        "StridedSlice": lambda x: raw(
            input=x, begin=(0, 1), end=(0, 4), strides=(1, 2), begin_mask=1, end_mask=1
        ),
        "Tile": lambda x: raw(input=x, multiples=(1, 2)),
        "Transpose": lambda x: raw(x=x, perm=(1, 0)),
    }
    return choices[name]


def _native_case(tf: Any, name: str) -> tuple[Any, Callable[..., Any]]:
    """Return independent native model inputs and its raw TensorFlow reference."""

    inputs = tf.constant([[0.25, 0.5, 0.75, 1.0]], dtype=tf.float64)
    raw = getattr(tf.raw_ops, name)
    if name == "Shape":
        return inputs, lambda x: raw(input=x, out_type=tf.int64)
    if name in _UNARY:
        return inputs, lambda x: raw(**{_UNARY[name]: x})
    if name in _BINARY:
        other = tf.constant([[0.3, 0.6, 0.4, 0.9]], dtype=tf.float64)
        return inputs, lambda x: raw(x=x, y=other)
    if name in _REDUCTIONS:
        return inputs, lambda x: raw(input=x, axis=(1,), keep_dims=True)
    if name in {"AvgPool", "MaxPool", "Conv2D", "DepthwiseConv2dNative"}:
        return _pool_or_conv(tf, name, raw)
    if name == "ReadVariableOp":
        return inputs, _resource_read(tf, raw)
    if name == "Fill":
        return tf.cast(inputs, tf.float32), _remaining_model(tf, name, raw)
    if name == "BatchMatMulV2":
        inputs = tf.reshape(inputs, (1, 2, 2))
        other = tf.constant([[[0.2, 0.3], [0.4, 0.6]]], dtype=tf.float64)
        return inputs, lambda x: raw(x=x, y=other, adj_y=True)
    return inputs, _remaining_model(tf, name, raw)


def _assert_slots(tf: Any, left: Any, right: Any) -> None:
    """Compare every real TensorFlow output, including order and device."""

    for actual, expected in zip(tf.nest.flatten(left), tf.nest.flatten(right), strict=True):
        assert actual.dtype == expected.dtype
        assert actual.device == expected.device
        assert actual.shape == expected.shape
        tf.debugging.assert_equal(actual, expected)


@pytest.mark.parametrize("name", _CASES)
def test_tf_native_contract_replays_real_values(name: str) -> None:
    """Check direct execution, trace validation and a full Split invocation."""

    tf = pytest.importorskip("tensorflow")
    with tf.device("/CPU:0"):
        inputs, model = _native_case(tf, name)
        expected = model(inputs)
        trace = tl.trace(model, inputs, backend="tf")
        captures = tuple(capture for capture in trace._tf_op_captures if capture.op_type == name)
        assert captures
        call, records = _prepare_native_call(tf, captures[0])
        with _pause_tf_replay():
            values = call._outputs(call._execute([record.tensor for record in records]))
        _assert_slots(tf, values, tuple(capture.output_tensor for capture in captures))

        result = validate_tf_trace_detailed(trace)
        assert result.failures == ()
        policy = {
            "ReadVariableOp": "pure-resource-read",
            "Pack": "value",
            "Shape": "structural-value",
            "StridedSlice": "structural-value",
            "Identity": "annotation",
            "StopGradient": "annotation",
        }.get(name)
        if policy is None:
            assert result.replayed_histogram[name] == len(captures)
        else:
            assert all(result.classes[capture.label_raw] == policy for capture in captures)
            assert result.replayed_histogram[name] == 0
            if name == "Pack":
                assert result.pure_unverified_histogram[name] == len(captures)

        graph = prepare_graph(model, inputs, backend="tf")
        partition = graph.compute_dag().to_partition(())
        runtime = graph.materialize(graph.analyze(partition))
        _assert_slots(tf, runtime.replay(inputs), expected)
