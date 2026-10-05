"""Native Paddle split oracles with another CUDA context current on the same GPU."""

from __future__ import annotations

import pytest
from _paddle_subprocess import run_paddle_subprocess

pytestmark = [pytest.mark.backend_paddle, pytest.mark.heavy]


def test_cpu_split_never_loads_the_cuda_driver() -> None:
    """CPU capture, replay, and training do not depend on CUDA context inspection."""

    run_paddle_subprocess(
        """
        import paddle
        import torchlens as tl
        from unittest.mock import patch

        paddle.set_device('cpu')
        model = paddle.nn.Sequential(
            paddle.nn.Linear(3, 4), paddle.nn.ReLU(), paddle.nn.Linear(4, 2)
        )
        x, y = paddle.ones([2, 3]), paddle.ones([2, 2])
        optimizer = paddle.optimizer.SGD(0.02, parameters=model.parameters())
        with patch('torchlens.backends.paddle._cuda._cuda_driver', side_effect=AssertionError):
            runtime = tl.split.prepare(
                model, x, split_request('after:relu', backend='paddle', trainable=True)
            )
            assert runtime.validate_equivalence(model, (x,))
            boundary = runtime.run_training_prefix(x)
            loss, grads = runtime.train_suffix(boundary, y, optimizer=optimizer)
            assert grads
            assert runtime.backward_prefix(boundary, grads)
        """
    )


@pytest.mark.parametrize("cuda_build", [False, True])
def test_rocm_gpu_scope_never_loads_the_nvidia_driver(cuda_build: bool) -> None:
    """ROCm GPU storage bypasses NVIDIA APIs regardless of the CUDA build flag."""

    run_paddle_subprocess(
        f"""
        from unittest.mock import Mock, patch

        import paddle
        import pytest
        import torchlens as tl
        from torchlens.backends.paddle._cuda import paddle_cuda_scope

        paddle.set_device('cpu')
        model = paddle.nn.Sequential(
            paddle.nn.Linear(3, 4), paddle.nn.ReLU(), paddle.nn.Linear(4, 2)
        )
        x, y = paddle.ones([2, 3]), paddle.ones([2, 2])
        optimizer = paddle.optimizer.SGD(0.02, parameters=model.parameters())

        # A GPUPlace also denotes HIP storage; the CPU host needs no AMD hardware.
        rocm_tensor = Mock(spec=paddle.Tensor)
        rocm_tensor._is_initialized.return_value = True
        rocm_tensor.place.is_gpu_place.return_value = True
        rocm_tensor.data_ptr.return_value = 1
        assert isinstance(rocm_tensor, paddle.Tensor)
        with (
            patch.object(paddle, 'is_compiled_with_rocm', return_value=True),
            patch.object(paddle, 'is_compiled_with_cuda', return_value={cuda_build!r}),
            patch('torchlens.backends.paddle._cuda._cuda_driver', side_effect=AssertionError),
        ):
            with paddle_cuda_scope([rocm_tensor], {{'state': rocm_tensor}}):
                pass
            with pytest.raises(RuntimeError, match='scope failure'):
                with paddle_cuda_scope(rocm_tensor):
                    raise RuntimeError('scope failure')
            rocm_tensor.data_ptr.assert_not_called()

            runtime = tl.split.prepare(
                model, x, split_request('after:relu', backend='paddle', trainable=True)
            )
            assert runtime.validate_equivalence(model, (x,))
            boundary = runtime.run_training_prefix(x)
            loss, grads = runtime.train_suffix(boundary, y, optimizer=optimizer)
            assert grads
            assert runtime.backward_prefix(boundary, grads)
        """
    )


@pytest.mark.serial
def test_gpu_split_restores_foreign_context_and_matches_native_training() -> None:
    """Capture, probing, replay, validation, and SGD honor native storage ownership."""

    run_paddle_subprocess(
        """
        import copy
        import ctypes

        import paddle
        import pytest
        import torchlens as tl
        from torchlens.backends.paddle._cuda import _cuda_driver, paddle_cuda_scope
        from torchlens.backends.paddle.backend import PaddleBackend
        from torchlens.split.errors import SplitUnsupportedError

        paddle.set_device('gpu:0')
        paddle.seed(123)
        model = paddle.nn.Sequential(
            paddle.nn.Linear(3, 4), paddle.nn.ReLU(), paddle.nn.Linear(4, 2)
        )
        reference = copy.deepcopy(model)
        x, y = paddle.ones([2, 3]), paddle.ones([2, 2])
        native_x = x.clone()
        x.stop_gradient = native_x.stop_gradient = False
        expected_gradients = paddle.grad(model(x).sum(), [x, *model.parameters()])
        native_optimizer = paddle.optimizer.SGD(0.02, parameters=reference.parameters())
        native_loss = paddle.nn.functional.mse_loss(reference(native_x), y)
        native_loss.backward()
        native_optimizer.step()
        prefix_optimizer = paddle.optimizer.SGD(0.02, parameters=model[0].parameters())
        suffix_optimizer = paddle.optimizer.SGD(0.02, parameters=model[2].parameters())

        driver = _cuda_driver()
        driver.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        driver.cuDeviceGet.restype = ctypes.c_int
        driver.cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
        driver.cuCtxCreate_v2.restype = ctypes.c_int
        driver.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]
        driver.cuCtxDestroy_v2.restype = ctypes.c_int

        def current_context():
            current = ctypes.c_void_p()
            assert driver.cuCtxGetCurrent(ctypes.byref(current)) == 0
            return current.value

        native_context = current_context()
        device, foreign = ctypes.c_int(), ctypes.c_void_p()
        assert driver.cuDeviceGet(ctypes.byref(device), 0) == 0
        status = driver.cuCtxCreate_v2(ctypes.byref(foreign), 0, device)
        assert status == 0, f'Creating the independent CUDA context failed: driver status {status}'
        assert foreign.value != native_context

        def assert_restored():
            assert current_context() == foreign.value

        def assert_close(left, right):
            with paddle_cuda_scope(left, right):
                assert bool(paddle.allclose(left, right, atol=1e-5, rtol=1e-4).item())
            assert_restored()

        try:
            # A private context on the same GPU reproduces tinygrad's driver behavior.
            runtime = tl.split.prepare(
                model, x, split_request('after:relu', backend='paddle', trainable=True)
            )
            assert_restored()
            assert runtime.trace_graph.shape_program.batch_probe.status == 'passed', (
                runtime.trace_graph.shape_program.batch_probe
            )
            assert runtime.validate_equivalence(model, (x,))
            assert_restored()
            runtime.replay(x)
            assert_restored()

            trace = tl.trace(
                model, x, backend='paddle',
                grad_options=tl.backends.paddle.GradOptions(loss_fn=lambda output: output.sum()),
            )
            assert_restored()
            for name, expected in zip(
                ['inputs.0', *(f'params.{name}' for name, _ in model.named_parameters())],
                expected_gradients, strict=True,
            ):
                assert_close(trace.derived_grads[name].grad, expected)

            def functional(inputs, weight, bias):
                return paddle.nn.functional.relu(
                    paddle.nn.functional.linear(inputs, weight, bias)
                ) * 2

            trace = tl.trace(functional, (x, model[0].weight, model[0].bias), backend='paddle')
            assert_restored()
            assert PaddleBackend().validate_trace(trace) is True
            assert_restored()

            boundary = runtime.run_training_prefix(x)
            assert_restored()
            loss, grads = runtime.train_suffix(boundary, y, optimizer=suffix_optimizer)
            assert_restored()
            runtime.backward_prefix(boundary, grads, optimizer=prefix_optimizer)
            assert_restored()
            assert_close(loss, native_loss)
            assert_close(x.grad, native_x.grad)
            for name, value in model.state_dict().items():
                assert_close(value, reference.state_dict()[name])

            with pytest.raises(RuntimeError):
                with paddle_cuda_scope(x):
                    assert current_context() == native_context
                    raise RuntimeError('scope failure')
            assert_restored()

            node = next(node for node in runtime.trace_graph.nodes if node.target is not None)
            original_target = node.target
            def fail(*args, **kwargs):
                assert current_context() == native_context
                raise RuntimeError('replay failure')
            object.__setattr__(node, 'target', fail)
            try:
                runtime.replay(x)
            except SplitUnsupportedError:
                pass
            else:
                raise AssertionError('replay did not propagate the backend exception')
            finally:
                object.__setattr__(node, 'target', original_target)
            assert_restored()
        finally:
            assert driver.cuCtxSetCurrent(native_context) == 0
            assert driver.cuCtxDestroy_v2(foreign) == 0
        """,
        require_cuda=True,
    )
