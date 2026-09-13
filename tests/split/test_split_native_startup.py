"""Check mixed compiler startup and native destruction in cold interpreters."""

from __future__ import annotations

import os
import sys
import textwrap
from importlib.util import find_spec
from pathlib import Path

import pytest

from torchlens.utils._subprocess import run_bounded_subprocess

pytestmark = pytest.mark.heavy


@pytest.mark.parametrize("tf_first", [True, False])
def test_tinygrad_llvm_and_tensorflow_exit_cleanly(tf_first: bool) -> None:
    """Exercise both compilers and require a normal native process exit.

    Parameters
    ----------
    tf_first:
        Whether TensorFlow loads before tinygrad's LLVM library.
    """

    if sys.platform != "linux" or not hasattr(os, "RTLD_DEEPBIND"):
        pytest.skip("LLVM isolation applies to Linux with RTLD_DEEPBIND")
    for name in ("tensorflow", "tinygrad"):
        if find_spec(name) is None:
            pytest.skip(f"optional native runtime {name!r} is not installed")

    source = textwrap.dedent(
        """
        import os
        import resource
        import runpy
        import sys
        from types import SimpleNamespace

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        os.environ['DEBUG'] = '0'
        for name in ('OMP_NUM_THREADS', 'TF_NUM_INTEROP_THREADS', 'TF_NUM_INTRAOP_THREADS'):
            os.environ[name] = '1'
        original_flags = sys.getdlopenflags()
        cleanups = []
        configure = runpy.run_path(sys.argv[1])['pytest_configure']
        configure(SimpleNamespace(option=SimpleNamespace(collectonly=False),
                                  add_cleanup=cleanups.append))
        if sys.argv[2] == 'True':
            import tensorflow as tf
        # Force the real DSO load even if tinygrad has cached all CPU kernels.
        from tinygrad.runtime.autogen import llvm
        assert 'llvm' in llvm.dll._loaded_
        from tinygrad import Tensor
        assert (Tensor([1., 2.], device='CPU') + 10).tolist() == [11., 12.]
        import tensorflow as tf
        import torch
        model = torch.nn.Linear(3, 3)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        model(torch.ones(2, 3)).sum().backward()
        optimizer.step()
        assert tf.zeros_like(tf.constant([1., 2.])).numpy().tolist() == [0., 0.]
        assert tf.nn.relu(tf.constant([-1., 2.])).numpy().tolist() == [0., 2.]

        @tf.function(jit_compile=True, autograph=False)
        def compiled(x):
            return tf.reduce_sum(x * x)

        assert float(compiled(tf.constant([2., 3.])).numpy()) == 13.
        assert sys.getdlopenflags() == original_flags
        for cleanup in reversed(cleanups):
            cleanup()
        print('mixed-native-exit-ok')
        """
    )
    result = run_bounded_subprocess(
        [
            sys.executable,
            "-X",
            "faulthandler",
            "-c",
            source,
            str(Path(__file__).with_name("conftest.py")),
            str(tf_first),
        ],
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, (
        f"native subprocess failed ({result.returncode})\n{result.stdout}\n{result.stderr}"
    )
    assert "mixed-native-exit-ok" in result.stdout
