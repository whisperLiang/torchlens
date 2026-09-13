"""Exercise real native runtimes in cold processes with either import order."""

from __future__ import annotations

import os
import shutil
import sys
import sysconfig
import textwrap
from importlib.util import find_spec
from pathlib import Path

import pytest

from torchlens.utils._subprocess import run_bounded_subprocess

pytestmark = [pytest.mark.heavy, pytest.mark.backend_paddle, pytest.mark.tf_backend]

_REPOSITORY = Path(__file__).resolve().parents[1]
_RUNTIME_HELPERS = textwrap.dedent(
    '''
    def tensorflow_roundtrip(values: list[float]) -> None:
        """Check eager TensorFlow values and gradients against exact integers."""

        import tensorflow as tf

        x = tf.constant(values)
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = tf.reduce_sum(x * x)
        assert float(y.numpy()) == sum(value * value for value in values)
        assert tape.gradient(y, x).numpy().tolist() == [2 * value for value in values]

    def paddle_roundtrip() -> None:
        """Check Paddle values, gradients, and native exception propagation."""

        import paddle

        paddle.set_device("cpu")
        x = paddle.to_tensor([1.0, 2.0], stop_gradient=False)
        y = (x * x).sum()
        y.backward()
        assert float(y.numpy()) == 5.0
        assert x.grad.numpy().tolist() == [2.0, 4.0]
        try:
            paddle.reshape(x, [3])
        except ValueError:
            pass
        else:
            raise AssertionError("Paddle accepted an invalid reshape")

    def assert_native_guard(module_name: str) -> None:
        """Verify CINN builds delegate through the exact guarded native loader."""

        from importlib import import_module
        from importlib.machinery import ExtensionFileLoader

        compat = import_module(module_name)
        spec = sys.modules["paddle.base.libpaddle"].__spec__
        cinn_path = pathlib.Path(spec.origin).parents[1] / "libs" / "libcinnapi.so"
        if cinn_path.is_file():
            assert isinstance(spec.loader, compat._PaddleExtensionLoader)
            assert isinstance(spec.loader.original, ExtensionFileLoader)
            assert spec.loader._handle is not None
    '''
)


def _require_native_runtimes() -> None:
    """Skip only absent runtimes or platforms outside the native guard's scope."""

    if sys.platform != "linux" or not hasattr(os, "RTLD_DEEPBIND"):
        pytest.skip("Paddle LLVM compatibility requires Linux with RTLD_DEEPBIND")
    for name in ("paddle", "tensorflow"):
        if find_spec(name) is None:
            pytest.skip(f"optional native runtime {name!r} is not installed")


def _run_native_child(source: str, *, cwd: Path, args: tuple[str, ...] = ()) -> None:
    """Run one bounded cold interpreter without inheriting installed startup hooks.

    Parameters
    ----------
    source:
        Scenario executed after CPU, thread, and core-dump isolation.
    cwd:
        Working directory for the child process.
    args:
        Additional scenario arguments exposed through ``sys.argv``.
    """

    _require_native_runtimes()
    package_paths = list(dict.fromkeys(sysconfig.get_path(name) for name in ("purelib", "platlib")))
    prelude = textwrap.dedent(
        f"""
        import os
        import pathlib
        import resource
        import sys

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        for name in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "TF_NUM_INTEROP_THREADS", "TF_NUM_INTRAOP_THREADS",
        ):
            os.environ[name] = "1"
        # -S excludes any real installed .pth guard: this child must prove the
        # specific conftest or simulated startup hook under test installs it.
        sys.path.extend({package_paths!r})
        assert all(name not in sys.modules for name in (
            "torch", "torchlens", "tensorflow", "paddle", "_torchlens_paddle_compat",
        ))
        original_dlopen_flags = sys.getdlopenflags()
        """
    )
    result = run_bounded_subprocess(
        [sys.executable, "-S", "-X", "faulthandler", "-c", prelude + source, *args],
        cwd=cwd,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, (
        f"native startup subprocess failed ({result.returncode})\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "native-backend-startup-ok" in result.stdout


@pytest.mark.parametrize("order", ["tf_before_guard", "guard_before_tf", "paddle_before_tf"])
def test_collection_guard_supports_native_import_orders(order: str) -> None:
    """Preserve real import order while checking both runtimes before and after.

    Parameters
    ----------
    order:
        Cold import sequence, including a plugin loading TF before conftest.
    """

    source = _RUNTIME_HELPERS + textwrap.dedent(
        '''
        order = sys.argv[1]
        sys.path[:0] = [str(pathlib.Path.cwd() / "tests"), str(pathlib.Path.cwd())]

        if order == "tf_before_guard":
            # Simulate a pytest plugin that initialized TF before conftest.
            tensorflow_roundtrip([1.0, 2.0])
            assert "paddle" not in sys.modules
            import tensorflow as tf

            @tf.function(jit_compile=True, autograph=False)
            def compiled_before(x: tf.Tensor) -> tf.Tensor:
                """Compile a TensorFlow graph before Paddle loads its LLVM."""

                return tf.reduce_sum(x * x)

            assert float(compiled_before(tf.constant([1.0, 2.0])).numpy()) == 5.0

        import conftest

        assert pathlib.Path(conftest.__file__).resolve() == (
            pathlib.Path.cwd() / "tests" / "conftest.py"
        )
        assert "paddle" not in sys.modules, "conftest must not preload Paddle"
        if order != "tf_before_guard":
            assert "tensorflow" not in sys.modules, "conftest must remain lazy"

        if order == "guard_before_tf":
            tensorflow_roundtrip([1.0, 2.0])
            assert "paddle" not in sys.modules, "TF must be loaded before Paddle"
        elif order == "tf_before_guard":
            assert "tensorflow" in sys.modules

        import paddle

        assert sys.getdlopenflags() == original_dlopen_flags
        if order != "paddle_before_tf":
            assert_native_guard("torchlens._paddle_compat")
        if order == "paddle_before_tf":
            assert "tensorflow" not in sys.modules, "Paddle must be loaded first"
            tensorflow_roundtrip([1.0, 2.0])
        paddle_roundtrip()
        tensorflow_roundtrip([3.0, 4.0])

        if order == "tf_before_guard":
            import torchlens as tl
            from torchlens.backends.paddle import PaddleBackend

            trace = tl.trace(
                lambda x: paddle.nn.functional.relu(x * 2 + 1),
                paddle.to_tensor([-2.0, 3.0]),
                backend="paddle",
            )
            assert trace.num_ops == 3
            assert PaddleBackend().validate_trace(trace) is True
            trace.cleanup()

            @tf.function(jit_compile=True, autograph=False)
            def compiled_after(x: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
                """Force a new LLVM compilation after Paddle has executed."""

                with tf.GradientTape() as tape:
                    tape.watch(x)
                    y = tf.reduce_sum(x * x * x)
                return y, tape.gradient(y, x)

            y, gradient = compiled_after(tf.constant([2.0, 3.0]))
            assert float(y.numpy()) == 35.0
            assert gradient.numpy().tolist() == [12.0, 27.0]
            print("tf-xla-new-compilation-and-torchlens-validation-ok")

        assert sys.getdlopenflags() == original_dlopen_flags
        print("native-backend-startup-ok")
        '''
    )
    _run_native_child(source, cwd=_REPOSITORY, args=(order,))


def test_standalone_pth_guard_is_lazy_and_supports_tensorflow_first(tmp_path: Path) -> None:
    """Load a copied standalone guard through a real isolated ``.pth`` file.

    Parameters
    ----------
    tmp_path:
        Session-private directory for the simulated site-packages and startup files.
    """

    _require_native_runtimes()
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    shutil.copyfile(
        _REPOSITORY / "torchlens" / "_paddle_compat.py",
        site_packages / "_torchlens_paddle_compat.py",
    )
    (site_packages / "_torchlens_paddle_compat.pth").write_text(
        "import _torchlens_paddle_compat; _torchlens_paddle_compat.install_import_guard()\n",
        encoding="utf-8",
    )
    source = _RUNTIME_HELPERS + textwrap.dedent(
        """
        import site

        standalone_site = pathlib.Path(sys.argv[1])
        # A real installation may contain the same module name; this scenario
        # must execute its independent copied module and .pth, never that copy.
        sys.path.insert(0, str(standalone_site))
        site.addsitedir(str(standalone_site))
        assert all(name not in sys.modules for name in (
            "torch", "torchlens", "tensorflow", "paddle",
        )), "the startup hook must use only the standard library"
        import _torchlens_paddle_compat as compat

        assert pathlib.Path(compat.__file__).resolve() == (
            standalone_site / "_torchlens_paddle_compat.py"
        ).resolve()
        assert sum(
            getattr(finder, "guard_id", None) == "torchlens.paddle_llvm.v1"
            for finder in sys.meta_path
        ) == 1
        assert sys.getdlopenflags() == original_dlopen_flags

        tensorflow_roundtrip([1.0, 2.0])
        assert "paddle" not in sys.modules
        paddle_roundtrip()
        assert_native_guard("_torchlens_paddle_compat")
        tensorflow_roundtrip([3.0, 4.0])
        assert sys.getdlopenflags() == original_dlopen_flags
        print("native-backend-startup-ok")
        """
    )
    _run_native_child(source, cwd=tmp_path, args=(str(site_packages),))
