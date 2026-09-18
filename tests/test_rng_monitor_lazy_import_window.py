"""The RNG channel monitor warms torch's lazy dynamo import OUTSIDE its window.

The first wrapped op of a capture can trigger torch's own lazy
``import torch._dynamo`` (``torch/_compile.py``), whose import cascade draws
host entropy at module-exec time (``uuid.uuid4()`` in
``torch.distributed._composable.contract``, plus getrandbits / RNG-instance
draws). Fired INSIDE the host-nondeterminism monitor window, those draws
marked ``os.urandom``/getrandbits channels and permanently ceilinged the
FIRST selective runnable-capable capture of the process to UNVERIFIABLE on a
pure deterministic model -- a silent, order-dependent breach of the runnable
contract's "a plain deterministic capture records nothing" pin (hunt-b8 F1).

The poison is once-per-process (a second capture in the same process is
clean), so the authoritative gate runs in a fresh subprocess.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from torchlens import _state
from torchlens.utils import _torch_compat, rng as rng_utils
from torchlens.utils.rng import host_nondeterminism_monitor

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

_FIRST_SELECTIVE_CAPTURE_IS_CLEAN = """
import sys

imports = {"torch._dynamo": [], "torch.backends.opt_einsum": []}


class ObserveTorchImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname in imports:
            rng_module = sys.modules.get("torchlens.utils.rng")
            imports[fullname].append(getattr(rng_module, "_ACTIVE_MONITOR", None) is not None)
        return None


# Torch 2.14 loads opt_einsum from torch.backends during import torch; 2.8
# leaves it lazy. Observe both paths from process startup, without unloading
# modules or requiring a particular torch import order.
finder = ObserveTorchImports()
sys.meta_path.insert(0, finder)

import torch
from torch import nn

import torchlens as tl

assert "torch._dynamo" not in sys.modules, (
    "precondition broken: torch._dynamo was imported before the first capture, "
    "so this child cannot exercise the lazy-import-in-window path"
)
class DeterministicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(4, 4), nn.ReLU())

    def forward(self, x):
        return torch.einsum("bi->bi", self.block(x))


model = DeterministicModel()
x = torch.randn(1, 4)
log = tl.trace(
    model,
    x,
    save=tl.func("relu"),
    capture=tl.options.CaptureOptions(intervention_ready=True),
)
channels = tuple(log._runnable.host_rng_channels)
assert channels == (), (
    "torch's lazy torch._dynamo import cascade fired INSIDE the RNG monitor "
    f"window and poisoned a pure deterministic capture: channels={channels!r}"
)
assert log._runnable.host_rng_unreplayable is False, (
    "deterministic first selective capture settled unreplayable"
)
assert log._runnable.rng_monitor_uncertain is False, (
    "deterministic first capture exhausted the RNG monitor: "
    f"{log._runnable.rng_monitor_uncertain_detail!r}"
)
assert imports["torch.backends.opt_einsum"] == [False], (
    "torch.einsum's dependency entered an armed RNG monitor: "
    f"{imports['torch.backends.opt_einsum']!r}"
)
# find_spec also observes availability probes, so dynamo can have multiple
# lookups before execution. Every lookup must still precede the armed window.
assert imports["torch._dynamo"] and not any(imports["torch._dynamo"]), (
    "torch._dynamo was not resolved outside the RNG monitor: "
    f"{imports['torch._dynamo']!r}"
)
assert "torch._dynamo" in sys.modules, (
    "the capture never triggered (or pre-warmed) the dynamo import; "
    "this gate is vacuous on this torch build"
)
print("OK")
"""


@pytest.mark.heavy
def test_first_selective_runnable_capture_is_not_poisoned_by_lazy_dynamo_import() -> None:
    """Fresh process: first runnable-capable selective capture records nothing."""

    completed = subprocess.run(
        [sys.executable, "-c", _FIRST_SELECTIVE_CAPTURE_IS_CLEAN],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, (
        "first-selective-capture channel-cleanliness child failed:\n"
        f"STDOUT:{completed.stdout}\nSTDERR:{completed.stderr}"
    )
    assert "OK" in completed.stdout


@pytest.mark.smoke
def test_monitor_entry_warms_lazy_torch_imports() -> None:
    """Entering the monitor latches the compat warm flag before any patch."""

    import torch
    from torch import nn

    with host_nondeterminism_monitor(nn.Identity()):
        assert _torch_compat._LAZY_TORCH_IMPORTS_WARMED is True
        assert "torch._dynamo" in sys.modules
        assert "torch.backends.opt_einsum" in sys.modules
    assert torch is not None


@pytest.mark.smoke
@pytest.mark.parametrize("warm_raises", (False, True))
def test_monitor_warmup_is_paused_and_restores_logging(
    monkeypatch: pytest.MonkeyPatch, warm_raises: bool
) -> None:
    """Setup is outside the dispatch ledger; even failed setup restores the gate."""

    observed: list[bool] = []

    def warm_probe() -> None:
        """Observe the logging gate at the actual lazy-import call site."""

        observed.append(_state._logging_enabled)
        if warm_raises:
            raise RuntimeError("import-time probe failed")

    monkeypatch.setattr(rng_utils, "warm_lazy_torch_imports", warm_probe)
    # Isolate entry/exit sequencing from the unrelated process-wide RNG patches.
    monkeypatch.setattr(host_nondeterminism_monitor, "_install_steps", lambda self: ())
    monkeypatch.setattr(_state, "_logging_enabled", True)
    with host_nondeterminism_monitor(None):
        assert observed == [False]
        assert _state._logging_enabled is True
    assert _state._logging_enabled is True


@pytest.mark.smoke
@pytest.mark.parametrize("warm_raises", (False, True))
def test_lazy_import_probes_use_cpu_without_changing_caller_device(
    monkeypatch: pytest.MonkeyPatch, warm_raises: bool
) -> None:
    """A failed warm stays retryable and never leaks its setup device scope."""

    import torch

    observed: list[tuple[str, str]] = []
    original_import = _torch_compat.importlib.import_module

    def import_probe(module_name: str, package: str | None = None) -> object:
        """Stand in for import-time tensor construction on the active device."""

        if module_name not in {"torch._compile", "torch._dynamo", "torch.backends.opt_einsum"}:
            return original_import(module_name, package)
        observed.append((module_name, str(torch.empty(0).device)))
        if warm_raises:
            raise RuntimeError("import-time probe failed")
        return None

    monkeypatch.setattr(_torch_compat, "_LAZY_TORCH_IMPORTS_WARMED", False)
    monkeypatch.setattr(_torch_compat.importlib, "import_module", import_probe)
    before = _torch_compat.get_current_function_mode_stack()
    with torch.device("meta"):
        _torch_compat.warm_lazy_torch_imports()
        assert str(torch.empty(0).device) == "meta"
    assert _torch_compat.get_current_function_mode_stack() == before
    assert observed == [
        ("torch._compile", "cpu"),
        ("torch._dynamo", "cpu"),
        ("torch.backends.opt_einsum", "cpu"),
    ]
    assert _torch_compat._LAZY_TORCH_IMPORTS_WARMED is not warm_raises
