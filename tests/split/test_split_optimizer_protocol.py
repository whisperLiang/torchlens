"""Framework-independent optimizer state records and registration contracts."""

from __future__ import annotations

import numpy as np
import pytest


def test_tensor_payload_is_independent_and_normalized() -> None:
    """Checksums are stable across endianness and payloads cannot alias live state."""
    from torchlens.split.optimizers import OptimizerStateTensor

    values = np.arange(6, dtype=">f8").reshape(3, 2).T
    tensor = OptimizerStateTensor.from_array("parameter:weight", "momentum", values)
    same = OptimizerStateTensor.from_array("parameter:weight", "momentum", values.astype("<f8"))
    assert tensor.checksum == same.checksum
    assert tensor.shape == (2, 3)
    assert tensor.dtype == "float64"
    values[...] = -1
    np.testing.assert_array_equal(tensor.payload, [[0, 2, 4], [1, 3, 5]])
    with pytest.raises(ValueError):
        tensor.payload.flags.writeable = True


def test_registry_refuses_duplicate_keys_and_ambiguous_matches(monkeypatch) -> None:
    """Registration cannot replace an adapter or silently choose among two matches."""
    from torchlens.split.errors import SplitUnsupportedError
    from torchlens.split.optimizers import (
        get_optimizer_adapter,
        register_optimizer_adapter,
        registry,
    )

    class Adapter:
        """Minimal registry probe with an explicit claim on string optimizers."""

        def __init__(self, name: str) -> None:
            self.name = name

        def supports(self, optimizer: object) -> bool:
            return isinstance(optimizer, str)

    monkeypatch.setattr(registry, "_CUSTOM", {})
    probe = Adapter("one")
    register_optimizer_adapter("probe", probe)
    assert get_optimizer_adapter("probe", "optimizer") is probe
    with pytest.raises(SplitUnsupportedError, match="already registered"):
        register_optimizer_adapter("probe", Adapter("one"))
    with pytest.raises(SplitUnsupportedError, match="no registered adapter"):
        get_optimizer_adapter("probe", object())
    register_optimizer_adapter("probe", Adapter("two"))
    with pytest.raises(SplitUnsupportedError, match="multiple adapters"):
        get_optimizer_adapter("probe", "optimizer")


def test_custom_adapter_requires_factory_on_public_restore(monkeypatch) -> None:
    """A registered wrapper can migrate state using a user-supplied native factory."""

    import torch

    from torchlens.split import SplitFeatures, prepare_graph
    from torchlens.split.errors import SplitUnsupportedError
    from torchlens.split.optimizers import (
        get_optimizer_adapter,
        register_optimizer_adapter,
        registry,
    )

    class Wrapped:
        """A project optimizer that owns one native optimizer internally."""

        def __init__(self, optimizer):
            self.optimizer = optimizer

    model = torch.nn.Linear(3, 2)
    native = torch.optim.Adam(model.parameters())
    delegate = get_optimizer_adapter("torch", native)

    class Adapter:
        """Share Adam state semantics while requiring the application's wrapper factory."""

        name = "wrapped"
        merge_config = delegate.merge_config
        validate_restore = delegate.validate_restore

        def supports(self, optimizer):
            return type(optimizer) is Wrapped

        def export_state(self, optimizer, parameters):
            return delegate.export_state(optimizer.optimizer, parameters)

        def build_optimizer(self, state, parameters, factory):
            assert factory is not None
            fresh = factory(parameters, dict(state.config))
            assert not fresh.optimizer.state
            return fresh

        def restore_state(self, optimizer, state, parameters):
            delegate.restore_state(optimizer.optimizer, state, parameters)

    monkeypatch.setattr(registry, "_CUSTOM", {})
    register_optimizer_adapter("torch", Adapter())
    x = torch.ones(2, 3)
    graph = prepare_graph(model, x, features=SplitFeatures(training=True))
    cut = graph.compute_dag().to_partition(())
    runtime = graph.materialize(graph.analyze(cut))
    runtime.train_suffix(runtime.run_training_prefix(x), torch.zeros(2, 2), optimizer=native)
    optimizers = {"suffix": Wrapped(native)}
    plan = runtime.plan_optimizer_state_migration(cut, optimizers=optimizers)
    assert plan.supported, plan.unsupported_reasons
    snapshot = runtime.export_optimizer_state(optimizers=optimizers, plan=plan)
    target = runtime.repartition(cut)
    with pytest.raises(SplitUnsupportedError, match="require factories"):
        target.restore_optimizer_state(snapshot)

    def factory(parameters, config):
        """Construct a fresh native optimizer for the custom wrapper."""
        assert set(parameters) == {"parameter:weight", "parameter:bias"}
        return Wrapped(torch.optim.Adam(parameters.values(), **config["defaults"]))

    result = target.restore_optimizer_state(snapshot, optimizer_factories={"suffix": factory})
    assert result.restored_slots == 6
    fresh = result.optimizers["suffix"].optimizer
    assert fresh is not native
    for parameter in model.parameters():
        torch.testing.assert_close(
            fresh.state[parameter]["exp_avg"], native.state[parameter]["exp_avg"]
        )
