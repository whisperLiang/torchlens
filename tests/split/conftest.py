"""Initialize native compilers before collecting the mixed-backend split suite."""

from __future__ import annotations

import os
import sys
from importlib.util import find_spec
from typing import Any

import pytest


def _isolate_tinygrad_llvm(patches: pytest.MonkeyPatch) -> None:
    """Keep tinygrad's system LLVM from binding to TensorFlow's LLVM globals.

    Parameters
    ----------
    patches:
        Session-owned patches, restored by pytest at shutdown.
    """

    if sys.platform != "linux" or not hasattr(os, "RTLD_DEEPBIND"):
        return
    if find_spec("tinygrad") is None:
        return

    from tinygrad.runtime.support import c

    original = c.DLL.__init__

    def load_library(self: Any, nm: str, *args: Any, **kwargs: Any) -> None:
        """Bind only the LLVM DSO locally; leave other native libraries unchanged.

        Parameters
        ----------
        self:
            tinygrad library handle being initialized.
        nm:
            tinygrad's logical library name.
        *args, **kwargs:
            Original library search paths and ctypes loader options.
        """

        if nm == "llvm":
            kwargs["mode"] = kwargs.get("mode", os.RTLD_LOCAL) | os.RTLD_DEEPBIND
        original(self, nm, *args, **kwargs)

    # The collision can corrupt TF's option registry before its next kernel,
    # or only surface during native static destructors after pytest finishes.
    # Scope this to tinygrad's loader, not process-wide ctypes/dlopen flags.
    patches.setattr(c.DLL, "__init__", load_library)


def pytest_configure(config: pytest.Config) -> None:
    """Load Torch's compiler before TensorFlow can export competing LLVM symbols.

    Parameters
    ----------
    config:
        Active pytest configuration, including collection-only invocations.

    Notes
    -----
    Creating the first Torch optimizer lazily imports Dynamo and Triton. If
    TensorFlow has already been imported during collection, Triton's native
    initializer can corrupt LLVM's option registry and crash either immediately
    or on TensorFlow's next GPU kernel compilation. Keep the supported native
    load order here, before test-module imports; all GPU tests still execute.
    """

    if not config.option.collectonly:
        # The parent and real-model subprocesses share the GPU. Reserving most
        # of its free memory in either TF or JAX starves the next backend even
        # though these tests only need small tensors. Honor explicit overrides.
        environment = pytest.MonkeyPatch()
        if "OMP_NUM_THREADS" not in os.environ and "MKL_NUM_THREADS" not in os.environ:
            import torch

            # Each native framework owns a CPU pool. Small split workloads
            # thrash on many-core hosts when every pool defaults to all cores.
            original_threads = torch.get_num_threads()
            torch.set_num_threads(1)
            config.add_cleanup(lambda: torch.set_num_threads(original_threads))
        for name, value in (
            ("TF_FORCE_GPU_ALLOW_GROWTH", "true"),
            ("XLA_PYTHON_CLIENT_PREALLOCATE", "false"),
            ("OMP_NUM_THREADS", "1"),
            ("MKL_NUM_THREADS", "1"),
            ("TF_NUM_INTRAOP_THREADS", "1"),
            ("TF_NUM_INTEROP_THREADS", "1"),
        ):
            if name not in os.environ:
                environment.setenv(name, value)
        config.add_cleanup(environment.undo)

        from torchlens.utils._torch_compat import get_dynamo_optimized_module_type

        get_dynamo_optimized_module_type(force_probe=True)
        _isolate_tinygrad_llvm(environment)
