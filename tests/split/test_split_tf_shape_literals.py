"""TensorFlow shape recipes inspect dimensions, never tensor data or axes."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from v2_helpers import split_request

import torchlens as tl
from torchlens.backends.tf.op_callback_capture import TFInputCapture, TFOpCapture
from torchlens.split import shape_program
from torchlens.split.adapters import tf as tf_adapter


class _Tensor:
    """Minimal TF-shaped tensor that records every attempted host payload read."""

    def __init__(
        self,
        *,
        integer: bool,
        shape: tuple[int, ...],
        payload: Any = None,
    ) -> None:
        """Set metadata independently of an optional readable NumPy payload."""

        self.dtype = SimpleNamespace(is_integer=integer)
        self.shape = shape
        self.payload = payload
        self.reads = 0

    def numpy(self) -> np.ndarray:
        """Reject materialization unless this is an explicitly readable literal."""

        self.reads += 1
        assert self.payload is not None, "Non-shape tensor payload was read."
        return np.asarray(self.payload)


def _capture(op_type: str, input_index: int, tensor: _Tensor) -> TFOpCapture:
    """Wrap one operand in a real callback record without importing TensorFlow."""

    return TFOpCapture(
        label_raw="op_1_1_raw",
        op_type=op_type,
        attrs={},
        output_index=0,
        inputs=(
            TFInputCapture(
                input_index=input_index,
                producer_label_raw=None,
                source_kind="constant/factory",
                source_label_raw=None,
                tensor=tensor,
                ref_key=None,
            ),
        ),
        output_tensor=None,
    )


def _node(capture: Any) -> Any:
    """Provide only the metadata consumed by shape collection and rewriting."""

    return SimpleNamespace(
        target=capture,
        args_template=None,
        kwargs_template=None,
        canonical_id="value:1",
    )


def _segment(monkeypatch: pytest.MonkeyPatch, rewrite: Any) -> Any:
    """Build the real input-replay adapter around a metadata-only TF stand-in."""

    def convert(value: Any, dtype: Any = None) -> _Tensor:
        """Preserve captured objects and materialize only rewritten dimensions."""

        if isinstance(value, _Tensor):
            return value
        return _Tensor(integer=dtype.is_integer, shape=np.asarray(value).shape, payload=value)

    monkeypatch.setattr(tf_adapter, "_tf", lambda: SimpleNamespace(convert_to_tensor=convert))
    segment = tf_adapter._TfGeneratedSegmentBase(
        graph=SimpleNamespace(
            node_by_id={},
            node_id_by_alias={},
            shape_program=SimpleNamespace(rewrite=rewrite),
        ),
        plan=None,
        spec=split_request("50%", backend="tf"),
        node_ids=frozenset(),
    )
    segment._shape_binding = shape_program.ShapeBinding({"B": 3}, {})
    return segment


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("op_type", "input_index", "integer", "shape"),
    [
        ("MatMul", 1, False, (4096, 4096)),
        ("MatMul", 1, True, (4096,)),
        ("ReadVariableOp", 0, False, ()),
        ("Reshape", 0, True, (6,)),
        ("Reshape", 1, False, (2,)),
        ("Reshape", 1, True, (2, 2)),
        ("Fill", 1, True, ()),
        ("ExpandDims", 1, True, ()),
        ("Transpose", 1, True, (3,)),
        ("Mean", 1, True, (1,)),
        ("StridedSlice", 2, True, (3,)),
        ("Tile", 1, True, (3,)),
        ("UnknownOp", 1, True, (2,)),
    ],
)
def test_tf_nonshape_operands_never_read_payloads(
    monkeypatch: pytest.MonkeyPatch,
    op_type: str,
    input_index: int,
    integer: bool,
    shape: tuple[int, ...],
) -> None:
    """Weights, integer data, resources, matrices, and axes cannot be dimensions."""

    tensor = _Tensor(integer=integer, shape=shape)
    capture = _capture(op_type, input_index, tensor)
    node = _node(capture)
    assert shape_program._captured_shape_descriptors(node) == set()
    assert tensor.reads == 0

    def forbidden_rewrite(*args: Any) -> Any:
        """Fail if replay attempts to replace a non-dimension operand."""

        pytest.fail("A non-shape operand reached the dimension rewriter.")

    segment = _segment(monkeypatch, forbidden_rewrite)
    assert segment._tf_inputs(capture, node, {}) == [tensor]
    assert tensor.reads == 0


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("op_type", "input_index"), [("Reshape", 1), ("Fill", 0), ("BroadcastTo", 1)]
)
def test_tf_shape_operands_remain_readable_and_rewritable(
    monkeypatch: pytest.MonkeyPatch, op_type: str, input_index: int
) -> None:
    """An actual integer dimension vector still participates in batch recipes."""

    tensor = _Tensor(integer=True, shape=(2,), payload=[1, 6])
    capture = _capture(op_type, input_index, tensor)
    node = _node(capture)
    assert shape_program._captured_shape_descriptors(node) == {(1, 6)}
    assert tensor.reads == 1

    def rewrite(node_id: str, payload: Any, binding: Any) -> list[int]:
        """Replace only the batch extent in this dimension descriptor."""

        assert node_id == "value:1"
        assert payload == [1, 6]
        return [binding.batch_size, 6]

    segment = _segment(monkeypatch, rewrite)
    rewritten = segment._tf_inputs(capture, node, {})
    assert rewritten[0].payload == [3, 6]
    assert tensor.reads == 2


@pytest.mark.smoke
def test_shape_literal_suffix_scan_is_linear(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mixed lists retain their integer suffix without testing every suffix."""

    inspected_lengths: list[int] = []
    original = shape_program._flat_int_shape

    def inspect(value: Any) -> tuple[int, ...] | None:
        """Count candidate lengths instead of enforcing a timing threshold."""

        inspected_lengths.append(len(value))
        return original(value)

    monkeypatch.setattr(shape_program, "_flat_int_shape", inspect)
    values = [0.5] * 2048 + [1, 6]
    assert shape_program._captured_shape_descriptors(_node(values)) == {(1, 6)}
    assert sum(inspected_lengths) == len(values)


@pytest.mark.heavy
@pytest.mark.tf_backend
def test_tf_literal_reshape_and_axis_replay_across_batches() -> None:
    """Small real TF replay rewrites dimensions while keeping axis 1 unchanged."""

    tf = pytest.importorskip("tensorflow")

    def model(x: Any) -> Any:
        """Use a Python-derived dimension literal and independent axis operands."""

        flat = tf.reshape(x, (x.shape[0], -1))
        expanded = tf.expand_dims(flat, axis=1)
        return tf.transpose(expanded, (0, 2, 1)) * 2.0

    with tf.device("/CPU:0"):
        sample = tf.ones((2, 2, 3), dtype=tf.float32)
        runtime = tl.split.prepare(model, sample, split_request("after:reshape", backend="tf"))
        assert runtime.batch_validation["status"] == "passed", runtime.batch_validation
        for batch in (1, 2, 4):
            values = tf.reshape(tf.range(batch * 6, dtype=tf.float32), (batch, 2, 3))
            actual = runtime.replay(values)
            assert actual.shape == (batch, 6, 1)
            np.testing.assert_allclose(actual.numpy(), model(values).numpy())
