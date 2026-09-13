"""Function-state isolation tests for native batch probes."""

from __future__ import annotations

from types import SimpleNamespace

from torchlens.split.pipeline import _probe_state_scope


def test_non_torch_function_probe_restores_closure_state() -> None:
    """A function probe must not retain mutations to reachable closure values."""

    calls = [0]
    cache = {"last_batch": 1}

    def model() -> None:
        """Mutate both common closure-state shapes."""

        calls[0] += 1
        cache["last_batch"] = calls[0]

    adapter = SimpleNamespace(name="tf", is_tensor=lambda value: False)
    with _probe_state_scope(model, adapter):
        model()

    assert calls == [0]
    assert cache == {"last_batch": 1}
