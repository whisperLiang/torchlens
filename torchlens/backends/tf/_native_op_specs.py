"""TensorFlow's native replay signatures and validation policies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

_MISSING = object()

_AttrKind = Literal["str", "int", "bool", "ints", "dtype"]
_ValidationPolicy = Literal[
    "perturb", "value", "structural-value", "annotation", "pure-resource-read"
]


@dataclass(frozen=True)
class _AttrRule:
    """A Python keyword and the native attribute that supplies it."""

    name: str
    kind: _AttrKind
    default: object = _MISSING
    choices: tuple[str, ...] = ()
    capture_names: tuple[str, ...] = ()
    min_items: int = 0


@dataclass(frozen=True)
class _NativeOpSpec:
    """Native wrapper ABI and the capture-validation policy for an op."""

    inputs: tuple[str, ...]
    attrs: tuple[_AttrRule, ...] = ()
    # Variadic ops group the first flattened operands into one list argument.
    min_list_inputs: int = 0
    output_count_attr: str | None = None
    validation: _ValidationPolicy = "perturb"


_NHWC_NCHW = ("NHWC", "NCHW")
_PADDING = ("VALID", "SAME")
_EXPLICIT_PADDING = (*_PADDING, "EXPLICIT")
_EXPLICIT_PADDINGS = _AttrRule("explicit_paddings", "ints", ())
_LAYOUT = _AttrRule("data_format", "str", "NHWC", _NHWC_NCHW)
_STRIDES = _AttrRule("strides", "ints", min_items=4)
_POOL_WINDOW = (_AttrRule("ksize", "ints", min_items=4), _STRIDES)
_CONV_ATTRS = (
    _STRIDES,
    _AttrRule("padding", "str", choices=_EXPLICIT_PADDING),
    _EXPLICIT_PADDINGS,
    _LAYOUT,
    _AttrRule("dilations", "ints", (1, 1, 1, 1), min_items=4),
)
_REDUCTION = _NativeOpSpec(("input", "axis"), (_AttrRule("keep_dims", "bool", False),))
_BINARY = _NativeOpSpec(("x", "y"))
_UNARY = _NativeOpSpec(("x",))
_ACTIVATION = _NativeOpSpec(("features",))

# These are Python raw_ops names. In particular, AvgPool takes "value",
# MaxPool takes "input", and Split takes "axis" (OpDef says "split_dim").
# Type attributes inferred by the wrapper (T, Tidx, ...) are never kwargs.
_NATIVE_OP_SPECS: Mapping[str, _NativeOpSpec] = MappingProxyType(
    {
        "AddV2": _BINARY,
        "AvgPool": _NativeOpSpec(
            ("value",), (*_POOL_WINDOW, _AttrRule("padding", "str", choices=_PADDING), _LAYOUT)
        ),
        "BatchMatMulV2": _NativeOpSpec(
            ("x", "y"),
            tuple(
                _AttrRule(name, "bool", False) for name in ("adj_x", "adj_y", "grad_x", "grad_y")
            ),
        ),
        "BiasAdd": _NativeOpSpec(("value", "bias"), (_LAYOUT,)),
        "BroadcastTo": _NativeOpSpec(("input", "shape")),
        "Cast": _NativeOpSpec(
            ("x",), (_AttrRule("DstT", "dtype"), _AttrRule("Truncate", "bool", False))
        ),
        "ConcatV2": _NativeOpSpec(("values", "axis"), min_list_inputs=2),
        "Conv2D": _NativeOpSpec(
            ("input", "filter"), (*_CONV_ATTRS, _AttrRule("use_cudnn_on_gpu", "bool", True))
        ),
        "DepthwiseConv2dNative": _NativeOpSpec(("input", "filter"), _CONV_ATTRS),
        "Einsum": _NativeOpSpec(("inputs",), (_AttrRule("equation", "str"),), min_list_inputs=1),
        "Elu": _ACTIVATION,
        "Erf": _UNARY,
        "Erfc": _UNARY,
        "Exp": _UNARY,
        "ExpandDims": _NativeOpSpec(("input", "axis")),
        "Fill": _NativeOpSpec(("dims", "value")),
        "GatherV2": _NativeOpSpec(
            ("params", "indices", "axis"), (_AttrRule("batch_dims", "int", 0),)
        ),
        "Greater": _BINARY,
        "Identity": _NativeOpSpec(("input",), validation="annotation"),
        "Less": _BINARY,
        "Log": _UNARY,
        "MatMul": _NativeOpSpec(
            ("a", "b"),
            tuple(
                _AttrRule(name, "bool", False)
                for name in ("transpose_a", "transpose_b", "grad_a", "grad_b")
            ),
        ),
        "MaxPool": _NativeOpSpec(
            ("input",),
            (
                *_POOL_WINDOW,
                _AttrRule("padding", "str", choices=_EXPLICIT_PADDING),
                _EXPLICIT_PADDINGS,
                _AttrRule("data_format", "str", "NHWC", (*_NHWC_NCHW, "NCHW_VECT_C")),
            ),
        ),
        "Maximum": _BINARY,
        "Mean": _REDUCTION,
        "Minimum": _BINARY,
        "Mul": _BINARY,
        "Neg": _UNARY,
        "Pack": _NativeOpSpec(
            ("values",), (_AttrRule("axis", "int", 0),), min_list_inputs=1, validation="value"
        ),
        "Pad": _NativeOpSpec(("input", "paddings")),
        "Pow": _BINARY,
        "Prod": _REDUCTION,
        "ReadVariableOp": _NativeOpSpec(
            ("resource",), (_AttrRule("dtype", "dtype"),), validation="pure-resource-read"
        ),
        "RealDiv": _BINARY,
        "Relu": _ACTIVATION,
        "Relu6": _ACTIVATION,
        "Reshape": _NativeOpSpec(("tensor", "shape")),
        "Rsqrt": _UNARY,
        "SelectV2": _NativeOpSpec(("condition", "t", "e")),
        "Shape": _NativeOpSpec(
            ("input",),
            (
                _AttrRule(
                    "out_type", "dtype", "int32", ("int32", "int64"), ("out_type_enum", "out_type")
                ),
            ),
            validation="structural-value",
        ),
        "Sigmoid": _UNARY,
        "Sin": _UNARY,
        "Slice": _NativeOpSpec(("input", "begin", "size")),
        "Softmax": _NativeOpSpec(("logits",)),
        "Split": _NativeOpSpec(
            ("axis", "value"), (_AttrRule("num_split", "int"),), output_count_attr="num_split"
        ),
        "Sqrt": _UNARY,
        "SquaredDifference": _BINARY,
        "Squeeze": _NativeOpSpec(
            ("input",), (_AttrRule("axis", "ints", (), capture_names=("squeeze_dims", "axis")),)
        ),
        "StopGradient": _NativeOpSpec(("input",), validation="annotation"),
        "StridedSlice": _NativeOpSpec(
            ("input", "begin", "end", "strides"),
            tuple(
                _AttrRule(f"{part}_mask", "int", 0)
                for part in ("begin", "end", "ellipsis", "new_axis", "shrink_axis")
            ),
            validation="structural-value",
        ),
        "Sub": _BINARY,
        "Sum": _REDUCTION,
        "Tanh": _UNARY,
        "Tile": _NativeOpSpec(("input", "multiples")),
        "Transpose": _NativeOpSpec(("x", "perm")),
    }
)
