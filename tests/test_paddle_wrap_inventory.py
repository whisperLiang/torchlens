"""Static Paddle wrapper inventory snapshot tests."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from torchlens import _state
from torchlens.backends import BackendUnsupportedError
from torchlens.backends.paddle import wrappers as paddle_wrappers
from torchlens.backends.paddle.wrappers import PaddleInventory, _PaddleWrapperRegistry

SNAPSHOT_MESSAGE = (
    "Paddle wrapper inventory changed. Classify the new/moved/removed op in "
    "torchlens.backends.paddle.wrappers, then update tests/test_paddle_wrap_inventory.py. "
    "This static snapshot is the correctness guard for same-object no-op and scalar-escape "
    "coverage gaps that dynamic validation cannot see."
)


def _paddle_runtime_or_skip() -> Any:
    """Import Paddle unless TensorFlow has made this process unsafe for it."""

    tensorflow_loaded = any(
        name == "tensorflow" or name.startswith("tensorflow.") for name in sys.modules
    )
    if "paddle" not in sys.modules and tensorflow_loaded:
        pytest.skip("Paddle runtime is unsafe to import after TensorFlow in this process")
    return pytest.importorskip("paddle")


@pytest.fixture(autouse=True)
def _load_paddle() -> None:
    """Load Paddle lazily when a Paddle test actually runs."""

    _paddle_runtime_or_skip()


EXPECTED_WRAPPED = (
    "abs",
    "add",
    "argmax",
    "argmin",
    "assign",
    "c_ops.add",
    "c_ops.batch_norm",
    "c_ops.conv2d",
    "c_ops.depthwise_conv2d",
    "c_ops.depthwise_conv2d_bias",
    "c_ops.hardswish",
    "c_ops.pool2d",
    "c_ops.relu6",
    "cast",
    "clip",
    "concat",
    "cos",
    "divide",
    "einsum",
    "equal",
    "exp",
    "flatten",
    "floor",
    "full",
    "full_like",
    "functional.avg_pool1d",
    "functional.avg_pool2d",
    "functional.batch_norm",
    "functional.conv1d",
    "functional.conv2d",
    "functional.dropout",
    "functional.gelu",
    "functional.hardswish",
    "functional.layer_norm",
    "functional.leaky_relu",
    "functional.linear",
    "functional.max_pool1d",
    "functional.max_pool2d",
    "functional.relu",
    "functional.relu6",
    "functional.sigmoid",
    "functional.silu",
    "functional.softmax",
    "functional.tanh",
    "greater_equal",
    "greater_than",
    "less_equal",
    "less_than",
    "linspace",
    "log",
    "matmul",
    "max",
    "mean",
    "min",
    "mm",
    "multiply",
    "negative",
    "ones",
    "ones_like",
    "pow",
    "prod",
    "reshape",
    "sin",
    "split",
    "sqrt",
    "square",
    "squeeze",
    "stack",
    "std",
    "subtract",
    "sum",
    "tanh",
    "tensor.__add__",
    "tensor.__getitem__",
    "tensor.__matmul__",
    "tensor.__mul__",
    "tensor.__neg__",
    "tensor.__pow__",
    "tensor.__radd__",
    "tensor.__rmatmul__",
    "tensor.__rmul__",
    "tensor.__rpow__",
    "tensor.__rsub__",
    "tensor.__rtruediv__",
    "tensor.__sub__",
    "tensor.__truediv__",
    "tensor._use_gpudnn",
    "tensor.abs",
    "tensor.add",
    "tensor.astype",
    "tensor.cast",
    "tensor.clip",
    "tensor.contiguous",
    "tensor.divide",
    "tensor.exp",
    "tensor.flatten",
    "tensor.log",
    "tensor.matmul",
    "tensor.max",
    "tensor.mean",
    "tensor.min",
    "tensor.multiply",
    "tensor.pow",
    "tensor.prod",
    "tensor.reshape",
    "tensor.rsqrt",
    "tensor.scale",
    "tensor.sqrt",
    "tensor.square",
    "tensor.squeeze",
    "tensor.std",
    "tensor.subtract",
    "tensor.sum",
    "tensor.t",
    "tensor.tile",
    "tensor.transpose",
    "tensor.unsqueeze",
    "tensor.var",
    "tile",
    "to_tensor",
    "transpose",
    "unsqueeze",
    "var",
    "where",
    "zeros",
    "zeros_like",
)

# The July 2026 native-wrapper extension (2e154c53) added eight curated C ops,
# functional.relu6, and Tensor._use_gpudnn without updating this census. The
# Python/Tensor denial inventory below is unchanged. The separate, explicit
# native census was reviewed on Paddle 3.3.1 CPU: mutator-suffixed/state-writing
# APIs, RNG APIs, and distributed global-state APIs are denied BEFORE execution.
# Never derive this expected set from the live registry or its classifier.
EXPECTED_C_OPS_DENIED = tuple(
    "c_ops." + name
    for name in (
        "abs_",
        "acos_",
        "acos_grad_",
        "acosh_",
        "acosh_grad_",
        "adadelta_",
        "adagrad_",
        "adam_",
        "adamax_",
        "adamw_",
        "add_",
        "add_grad_",
        "addmm_",
        "affine_channel_",
        "affine_channel_grad_",
        "all_reduce_",
        "array_write_",
        "asgd_",
        "asin_",
        "asin_grad_",
        "asinh_",
        "asinh_grad_",
        "assign_",
        "assign_out_",
        "assign_out__grad_",
        "assign_value_",
        "atan_",
        "atan_grad_",
        "atanh_",
        "atanh_grad_",
        "average_accumulates_",
        "baddbmm_",
        "batch_norm_",
        "bce_loss_",
        "bce_loss_grad_",
        "bernoulli",
        "bitwise_and_",
        "bitwise_left_shift_",
        "bitwise_not_",
        "bitwise_or_",
        "bitwise_right_shift_",
        "bitwise_xor_",
        "block_multihead_attention_",
        "block_multihead_attention_xpu_",
        "broadcast_",
        "c_allreduce_sum_",
        "c_identity_",
        "c_softmax_with_cross_entropy_grad_",
        "cast_",
        "ceil_",
        "ceil_grad_",
        "celu_grad_",
        "check_finite_and_unscale_",
        "clip_",
        "clip_grad_",
        "coalesce_tensor_",
        "copysign_",
        "copysign_grad_",
        "cos_",
        "cos_grad_",
        "cosh_",
        "cosh_grad_",
        "cross_entropy_with_softmax_",
        "cross_entropy_with_softmax_grad_",
        "cumprod_",
        "cumsum_",
        "dequantize_linear_",
        "digamma_",
        "distributed_fused_lamb_init",
        "distributed_fused_lamb_init_",
        "divide_",
        "dropout_",
        "elu_",
        "elu_grad_",
        "embedding_grad_add_to_",
        "equal_",
        "erf_",
        "erfinv_",
        "exp_",
        "exp_grad_",
        "expm1_",
        "expm1_grad_",
        "exponential_",
        "fake_quantize_dequantize_moving_average_abs_max_",
        "fake_quantize_moving_average_abs_max_",
        "fake_quantize_range_abs_max_",
        "fill_",
        "fill_diagonal_",
        "fill_diagonal_tensor_",
        "fill_diagonal_tensor_grad_",
        "fill_grad_",
        "flatten_",
        "flatten_grad_",
        "floor_",
        "floor_divide_",
        "floor_grad_",
        "fp8_gemm_blockwise_",
        "full_",
        "fused_adam_",
        "fused_multi_transformer_",
        "gammaincc_",
        "gammaln_",
        "gaussian_inplace_",
        "gaussian_inplace_grad_",
        "greater_equal_",
        "greater_than_",
        "group_norm_grad_",
        "hardshrink_grad_",
        "hardsigmoid_grad_",
        "hardswish_grad_",
        "hardtanh_",
        "hardtanh_grad_",
        "i0_",
        "identity_loss_",
        "identity_loss_grad_",
        "increment_",
        "index_add_",
        "index_add_grad_",
        "index_elementwise_put_",
        "index_elementwise_put_with_tensor_",
        "index_put_",
        "l1_norm_",
        "lamb_",
        "lars_momentum_",
        "leaky_relu_",
        "leaky_relu_grad_",
        "lerp_",
        "less_equal_",
        "less_than_",
        "lgamma_",
        "lod_reset_grad_",
        "log10_",
        "log10_grad_",
        "log1p_",
        "log1p_grad_",
        "log2_",
        "log2_grad_",
        "log_",
        "log_grad_",
        "logical_and_",
        "logical_not_",
        "logical_or_",
        "logical_xor_",
        "logit_",
        "logsigmoid_grad_",
        "lu_",
        "lu_grad_",
        "margin_cross_entropy_grad_",
        "masked_fill_",
        "masked_fill_grad_",
        "masked_multihead_attention_",
        "merged_adam_",
        "merged_momentum_",
        "mish_grad_",
        "momentum_",
        "moving_average_abs_max_scale_",
        "mp_allreduce_sum_",
        "multiply_",
        "nadam_",
        "nop_",
        "not_equal_",
        "partial_allgather_",
        "poisson",
        "polygamma_",
        "pow_",
        "pow_grad_",
        "put_along_axis_",
        "quantize_linear_",
        "radam_",
        "randint",
        "random_",
        "random_grad_",
        "random_routing_",
        "reciprocal_",
        "reciprocal_grad_",
        "reduce_",
        "relu6_grad_",
        "relu_",
        "relu_grad_",
        "remainder_",
        "renorm_",
        "reshape_",
        "reshape_grad_",
        "rint_",
        "rint_grad_",
        "rmsprop_",
        "round_",
        "round_grad_",
        "rprop_",
        "rsqrt_",
        "rsqrt_grad_",
        "scale_",
        "scatter_",
        "set_",
        "set_value",
        "set_value_",
        "set_value_with_tensor_",
        "sgd_",
        "share_data_",
        "sigmoid_",
        "sigmoid_cross_entropy_with_logits_",
        "sigmoid_cross_entropy_with_logits_grad_",
        "sigmoid_grad_",
        "silu_",
        "silu_grad_",
        "sin_",
        "sin_grad_",
        "sinh_",
        "sinh_grad_",
        "softmax_",
        "softplus_grad_",
        "softshrink_grad_",
        "softsign_grad_",
        "sparse_batch_norm_",
        "sparse_sync_batch_norm_",
        "sqrt_",
        "sqrt_grad_",
        "square_",
        "square_grad_",
        "squeeze_",
        "squeeze_grad_",
        "subtract_",
        "subtract_grad_",
        "swish_grad_",
        "sync_batch_norm_",
        "sync_calc_stream_",
        "tan_",
        "tan_grad_",
        "tanh_",
        "tanh_grad_",
        "tanh_shrink_grad_",
        "thresholded_relu_",
        "thresholded_relu_grad_",
        "transpose_",
        "tril_",
        "triu_",
        "trunc_",
        "trunc_divide_",
        "uniform",
        "uniform_inplace_",
        "uniform_inplace_grad_",
        "unsqueeze_",
        "unsqueeze_grad_",
        "update_loss_scaling_",
        "where_",
    )
)

EXPECTED_PYTHON_DENIED = (
    "abs_",
    "acos_",
    "acosh_",
    "addmm_",
    "asin_",
    "asinh_",
    "async_save",
    "atan_",
    "atanh_",
    "baddbmm_",
    "bernoulli",
    "bernoulli_",
    "bitwise_and_",
    "bitwise_invert_",
    "bitwise_left_shift_",
    "bitwise_not_",
    "bitwise_or_",
    "bitwise_right_shift_",
    "bitwise_xor_",
    "cast_",
    "cauchy_",
    "clear_async_save_task_queue",
    "copysign_",
    "cos_",
    "cosh_",
    "cumprod_",
    "cumsum_",
    "digamma_",
    "disable_static",
    "div_",
    "divide_",
    "enable_static",
    "equal_",
    "erf_",
    "expm1_",
    "flatten_",
    "floor_divide_",
    "floor_mod_",
    "frac_",
    "functional.elu_",
    "functional.embedding_renorm_",
    "functional.hardtanh_",
    "functional.leaky_relu_",
    "functional.relu_",
    "functional.softmax_",
    "functional.tanh_",
    "functional.thresholded_relu_",
    "gammainc_",
    "gammaincc_",
    "gammaln_",
    "gcd_",
    "geometric_",
    "greater_equal_",
    "greater_than_",
    "hypot_",
    "i0_",
    "index_add_",
    "index_fill_",
    "index_put_",
    "lcm_",
    "ldexp_",
    "less_",
    "less_equal_",
    "less_than_",
    "lgamma_",
    "load",
    "log10_",
    "log1p_",
    "log2_",
    "log_",
    "log_normal_",
    "logical_and_",
    "logical_not_",
    "logical_or_",
    "logical_xor_",
    "logit_",
    "manual_seed",
    "masked_fill_",
    "masked_scatter_",
    "mod_",
    "multigammaln_",
    "multiply_",
    "nan_to_num_",
    "neg_",
    "normal",
    "normal_",
    "not_equal_",
    "poisson",
    "polygamma_",
    "pow_",
    "rand",
    "rand_like",
    "randint",
    "randn",
    "randn_like",
    "remainder_",
    "renorm_",
    "reshape_",
    "save",
    "scatter_",
    "scatter_add_",
    "seed",
    "set_device",
    "sin_",
    "sinc_",
    "sinh_",
    "square_",
    "squeeze_",
    "sub_",
    "subtract_",
    "t_",
    "tan_",
    "tanh_",
    "tensor.__array__",
    "tensor.__bool__",
    "tensor.__dlpack__",
    "tensor.__float__",
    "tensor.__index__",
    "tensor.__int__",
    "tensor.__setitem__",
    "tensor._apply_",
    "tensor._to_dist_",
    "tensor._to_static_var",
    "tensor.abs_",
    "tensor.acos_",
    "tensor.acosh_",
    "tensor.add_",
    "tensor.addmm_",
    "tensor.apply_",
    "tensor.asin_",
    "tensor.asinh_",
    "tensor.atan_",
    "tensor.atanh_",
    "tensor.baddbmm_",
    "tensor.bernoulli_",
    "tensor.bitwise_and_",
    "tensor.bitwise_invert_",
    "tensor.bitwise_left_shift_",
    "tensor.bitwise_not_",
    "tensor.bitwise_or_",
    "tensor.bitwise_right_shift_",
    "tensor.bitwise_xor_",
    "tensor.cast_",
    "tensor.cauchy_",
    "tensor.ceil_",
    "tensor.clamp_",
    "tensor.clip_",
    "tensor.copy_",
    "tensor.copysign_",
    "tensor.cos_",
    "tensor.cosh_",
    "tensor.cumprod_",
    "tensor.cumsum_",
    "tensor.detach_",
    "tensor.digamma_",
    "tensor.div_",
    "tensor.divide_",
    "tensor.equal_",
    "tensor.erfinv_",
    "tensor.exp_",
    "tensor.exponential_",
    "tensor.fill_",
    "tensor.fill_diagonal_",
    "tensor.fill_diagonal_tensor_",
    "tensor.flatten_",
    "tensor.floor_",
    "tensor.floor_divide_",
    "tensor.floor_mod_",
    "tensor.frac_",
    "tensor.gammainc_",
    "tensor.gammaincc_",
    "tensor.gammaln_",
    "tensor.gcd_",
    "tensor.geometric_",
    "tensor.greater_equal_",
    "tensor.greater_than_",
    "tensor.hypot_",
    "tensor.i0_",
    "tensor.index_add_",
    "tensor.index_fill_",
    "tensor.index_put_",
    "tensor.item",
    "tensor.lcm_",
    "tensor.ldexp_",
    "tensor.lerp_",
    "tensor.less_",
    "tensor.less_equal_",
    "tensor.less_than_",
    "tensor.lgamma_",
    "tensor.log10_",
    "tensor.log1p_",
    "tensor.log2_",
    "tensor.log_",
    "tensor.log_normal_",
    "tensor.logical_and_",
    "tensor.logical_not_",
    "tensor.logical_or_",
    "tensor.logical_xor_",
    "tensor.logit_",
    "tensor.masked_fill_",
    "tensor.masked_scatter_",
    "tensor.mod_",
    "tensor.mul_",
    "tensor.multigammaln_",
    "tensor.multiply_",
    "tensor.nan_to_num_",
    "tensor.neg_",
    "tensor.normal_",
    "tensor.not_equal_",
    "tensor.numpy",
    "tensor.polygamma_",
    "tensor.pow_",
    "tensor.put_along_axis_",
    "tensor.random_",
    "tensor.reciprocal_",
    "tensor.reconstruct_from_",
    "tensor.remainder_",
    "tensor.renorm_",
    "tensor.requires_grad_",
    "tensor.reshape_",
    "tensor.resize_",
    "tensor.round_",
    "tensor.rsqrt_",
    "tensor.scale_",
    "tensor.scatter_",
    "tensor.scatter_add_",
    "tensor.set_",
    "tensor.set_value",
    "tensor.sigmoid_",
    "tensor.sin_",
    "tensor.sinc_",
    "tensor.sinh_",
    "tensor.sqrt_",
    "tensor.square_",
    "tensor.squeeze_",
    "tensor.sub_",
    "tensor.subtract_",
    "tensor.t_",
    "tensor.tan_",
    "tensor.tanh_",
    "tensor.tolist",
    "tensor.transpose_",
    "tensor.tril_",
    "tensor.triu_",
    "tensor.trunc_",
    "tensor.uniform_",
    "tensor.unsqueeze_",
    "tensor.where_",
    "tensor.zero_",
    "tolist",
    "transpose_",
    "tril_",
    "triu_",
    "trunc_",
    "uniform",
    "unsqueeze_",
    "where_",
)
EXPECTED_DENIED = tuple(sorted((*EXPECTED_PYTHON_DENIED, *EXPECTED_C_OPS_DENIED)))

ALIAS_NO_OP_APIS = frozenset(
    (
        "functional.dropout",
        "reshape",
        "tensor.astype",
        "tensor.cast",
        "tensor.contiguous",
        "tensor.reshape",
        "tensor._use_gpudnn",
    )
)
TENSOR_ESCAPE_APIS = frozenset(
    (
        "tensor.__array__",
        "tensor.__bool__",
        "tensor.__dlpack__",
        "tensor.__float__",
        "tensor.__index__",
        "tensor.__int__",
        "tensor.item",
        "tensor.numpy",
        "tensor.tolist",
        "tolist",
    )
)
MUTATOR_APIS = frozenset(("tensor.__setitem__", "tensor.copy_", "tensor.set_value"))
RNG_APIS = frozenset(("bernoulli", "normal", "poisson", "rand", "randn", "uniform"))
COARSE_COMPOSITE_APIS = frozenset(
    (
        "functional.avg_pool1d",
        "functional.avg_pool2d",
        "functional.batch_norm",
        "functional.conv1d",
        "functional.conv2d",
        "functional.dropout",
        "functional.layer_norm",
        "functional.linear",
        "functional.max_pool1d",
        "functional.max_pool2d",
        "functional.softmax",
    )
)
STOCHASTIC_COMPOSITE_DENY_APIS = frozenset(("functional.dropout",))


class _NoopBackend:
    """Backend placeholder used only to build wrapper inventory."""


@contextmanager
def _installed_inventory() -> Iterator[PaddleInventory]:
    """Install a private registry and yield its inventory.

    Yields
    ------
    PaddleInventory
        Sorted wrapper inventory built from the live Paddle runtime.
    """

    registry = _PaddleWrapperRegistry()
    registry.wrap(_NoopBackend())
    try:
        yield registry.inventory()
    finally:
        registry.unwrap()


def _assert_inventory_matches_snapshot(inventory: PaddleInventory) -> None:
    """Assert a live inventory matches the pinned static snapshot.

    Parameters
    ----------
    inventory
        Live inventory to compare against the committed snapshot.
    """

    assert inventory.wrapped == EXPECTED_WRAPPED, SNAPSHOT_MESSAGE
    assert inventory.denied == EXPECTED_DENIED, SNAPSHOT_MESSAGE


def test_paddle_wrapper_inventory_matches_static_snapshot() -> None:
    """Pin the exact wrapped and denied Paddle wrapper inventory."""

    with _installed_inventory() as inventory:
        _assert_inventory_matches_snapshot(inventory)


def test_paddle_wrapper_inventory_load_bearing_membership() -> None:
    """Assert static coverage classes remain classified as intended."""

    with _installed_inventory() as inventory:
        wrapped = set(inventory.wrapped)
        denied = set(inventory.denied)

    assert {"std", "var", "tensor.std", "tensor.var"} <= wrapped
    assert wrapped >= ALIAS_NO_OP_APIS
    assert denied >= MUTATOR_APIS
    assert denied >= RNG_APIS
    assert denied >= TENSOR_ESCAPE_APIS
    assert wrapped >= COARSE_COMPOSITE_APIS
    assert wrapped >= STOCHASTIC_COMPOSITE_DENY_APIS
    for op_name in (
        "functional.dropout",
        "functional.identity",
        "reshape",
        "tensor.astype",
        "tensor.cast",
        "tensor.contiguous",
        "tensor.reshape",
    ):
        assert paddle_wrappers.is_alias_allowed_op(op_name)


def test_paddle_same_object_gap_fails_static_inventory_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulate an unwrapped same-object API gap and require the snapshot to fail."""

    patched_tensor_methods = paddle_wrappers._TENSOR_CORE_METHODS - {"astype", "reshape"}
    monkeypatch.setattr(paddle_wrappers, "_TENSOR_CORE_METHODS", patched_tensor_methods)

    with (
        _installed_inventory() as inventory,
        pytest.raises(AssertionError, match="inventory changed"),
    ):
        _assert_inventory_matches_snapshot(inventory)


def test_paddle_native_coverage_gap_fails_static_inventory_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Removing a native observer cannot pass the independently pinned inventory."""

    monkeypatch.setattr(
        paddle_wrappers, "_C_OPS_CORE_OPS", paddle_wrappers._C_OPS_CORE_OPS - {"add"}
    )
    with (
        _installed_inventory() as inventory,
        pytest.raises(AssertionError, match="inventory changed"),
    ):
        _assert_inventory_matches_snapshot(inventory)


def test_every_native_denial_refuses_before_calling_the_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All statically denied C entrypoints refuse without running native code."""

    paddle = _paddle_runtime_or_skip()

    def forbidden_kernel(*args: Any, **kwargs: Any) -> None:
        """Fail if a supposedly denied operation reaches its implementation."""

        raise AssertionError("a denied native kernel was executed")

    for name in EXPECTED_C_OPS_DENIED:
        monkeypatch.setattr(paddle._C_ops, name.removeprefix("c_ops."), forbidden_kernel)
    with _installed_inventory(), _state.active_logging(SimpleNamespace()):
        for name in EXPECTED_C_OPS_DENIED:
            with pytest.raises(BackendUnsupportedError):
                getattr(paddle._C_ops, name.removeprefix("c_ops."))()
