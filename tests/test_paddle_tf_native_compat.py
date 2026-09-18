"""Real CPU computation survives either native framework import order."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from importlib.util import find_spec

import pytest

pytestmark = [pytest.mark.heavy, pytest.mark.optional]


@pytest.mark.parametrize("order", [("tensorflow", "paddle"), ("paddle", "tensorflow")])
def test_native_import_order_preserves_forward_and_backward(order: tuple[str, str]) -> None:
    """The lazy guard isolates native symbols, while both autograd engines still work."""

    if any(find_spec(name) is None for name in order):
        pytest.skip("TensorFlow and Paddle extras are required for native coexistence testing")
    if sys.platform != "linux" or not hasattr(os, "RTLD_DEEPBIND"):
        pytest.skip("the extension-local native compatibility guard is Linux-only")
    code = f"""
        import importlib, resource, sys
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        from torchlens._paddle_compat import install_import_guard
        assert install_import_guard()
        flags = sys.getdlopenflags()
        for name in {order!r}:
            importlib.import_module(name)
        import numpy as np
        import tensorflow as tf
        import paddle
        assert sys.getdlopenflags() == flags
        tf.config.set_visible_devices([], 'GPU')
        x = tf.Variable([1.0, 2.0, 3.0])
        with tf.GradientTape() as tape:
            y = tf.reduce_sum(x * x)
        np.testing.assert_array_equal(tape.gradient(y, x).numpy(), [2.0, 4.0, 6.0])
        assert float(y.numpy()) == 14.0
        paddle.set_device('cpu')
        p = paddle.to_tensor([1.0, 2.0, 3.0], stop_gradient=False)
        loss = (p * p).sum()
        loss.backward()
        np.testing.assert_array_equal(p.grad.numpy(), [2.0, 4.0, 6.0])
        assert float(loss.numpy()) == 14.0
        print('native-forward-backward-ok')
    """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "native-forward-backward-ok" in result.stdout
