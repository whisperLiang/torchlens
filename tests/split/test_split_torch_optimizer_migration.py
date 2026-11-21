"""PyTorch optimizer continuity through the public split migration protocol."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from dag_helpers import call_ids, close_call_ids

from torchlens.split import PlacementPlan, SplitFeatures, prepare_graph
from torchlens.split.errors import SplitUnsupportedError


@pytest.mark.parametrize("kind", ["SGD", "Adam", "AdamW"])
def test_torch_migration_preserves_groups_and_native_update_trajectory(kind: str) -> None:
    """Merge unequal parameter groups while preserving each group's update semantics."""
    torch.manual_seed(15)
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 5), torch.nn.Tanh(), torch.nn.Linear(5, 3)
    ).double()
    reference = deepcopy(model)
    x, y = torch.randn(2, 4, dtype=torch.double), torch.randn(2, 3, dtype=torch.double)
    graph = prepare_graph(model, x, features=SplitFeatures(training=True, retain_trace=False))
    dag = graph.compute_dag()
    first = dag.to_partition(close_call_ids(dag, call_ids(graph, modules=frozenset({"1"}))))
    runtime = graph.materialize(graph.analyze(first))
    factory = getattr(torch.optim, kind)
    flags = {"momentum": 0.9} if kind == "SGD" else {"amsgrad": True}

    def group(params, decay, lr):
        return {"params": list(params), "weight_decay": decay, "lr": lr}

    optimizers = {
        "prefix": factory([group(model[0].parameters(), 0.01, 0.001)], lr=0.01, **flags),
        "suffix": factory([group(model[2].parameters(), 0.02, 0.002)], lr=0.01, **flags),
    }
    oracle = factory(
        [
            group(reference[0].parameters(), 0.01, 0.001),
            group(reference[2].parameters(), 0.02, 0.002),
        ],
        lr=0.01,
        **flags,
    )

    for step in range(8):
        if step == 4:
            cut = dag.to_partition(())
            plan = runtime.plan_optimizer_state_migration(cut, optimizers=optimizers)
            assert plan.supported, plan.unsupported_reasons
            snapshot = runtime.export_optimizer_state(optimizers=optimizers, plan=plan)
            runtime = runtime.repartition(cut)
            migrated = runtime.restore_optimizer_state(snapshot)
            assert migrated.restored_slots == (4 if kind == "SGD" else 16)
            optimizers = migrated.optimizers
            groups = optimizers["suffix"].param_groups
            assert [(g["lr"], g["weight_decay"]) for g in groups] == [(0.001, 0.01), (0.002, 0.02)]
        oracle.zero_grad()
        expected = torch.nn.functional.mse_loss(reference(x), y)
        expected.backward()
        oracle.step()
        boundary = runtime.run_training_prefix(x)
        loss, grads = runtime.train_suffix(boundary, y, optimizer=optimizers["suffix"])
        if "prefix" in optimizers:
            runtime.backward_prefix(boundary, grads, optimizer=optimizers["prefix"])
        torch.testing.assert_close(loss, expected, atol=0, rtol=0)
        for left, right in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(left, right, atol=0, rtol=0)


@pytest.mark.cuda
@pytest.mark.parametrize(("source", "target"), [("cpu", "cuda:0"), ("cuda:0", "cpu")])
def test_torch_migration_restores_moments_on_effective_replica(source: str, target: str) -> None:
    """Move Adam moments with a target replica while retaining CPU step counters."""

    if not torch.cuda.is_available():
        pytest.skip("PyTorch CUDA unavailable")
    model = torch.nn.Linear(4, 3).double().to(source)
    inputs = torch.ones(2, 4, dtype=torch.double, device=source)
    graph = prepare_graph(model, inputs, features=SplitFeatures(training=True, retain_trace=False))
    cut = graph.compute_dag().to_partition(())
    runtime = graph.materialize(graph.analyze(cut))
    parameters = [binding["suffix"] for binding in runtime.parameter_bindings().values()]
    optimizer = torch.optim.Adam(parameters)
    runtime.train_suffix(
        runtime.run_training_prefix(inputs),
        torch.zeros(2, 3, dtype=torch.double, device=source),
        optimizer=optimizer,
    )
    plan = runtime.plan_optimizer_state_migration(cut, optimizers={"suffix": optimizer})
    snapshot = runtime.export_optimizer_state(optimizers={"suffix": optimizer}, plan=plan)
    new = runtime.repartition(cut).with_placement(PlacementPlan.across(source, target))
    restored = new.restore_optimizer_state(snapshot).optimizers["suffix"]
    for state_id, bindings in new.parameter_bindings().items():
        value = bindings["suffix"]
        previous = runtime.parameter_bindings()[state_id]["suffix"]
        state = restored.state[value]
        assert state["step"].device.type == "cpu"
        for name in ("exp_avg", "exp_avg_sq"):
            assert state[name].device == torch.device(target)
            torch.testing.assert_close(state[name].cpu(), optimizer.state[previous][name].cpu())
    new.train_suffix(
        new.run_training_prefix(inputs),
        torch.zeros(2, 3, dtype=torch.double, device=target),
        optimizer=restored,
    )
    assert {int(state["step"]) for state in restored.state.values()} == {2}
    assert {int(state["step"]) for state in optimizer.state.values()} == {1}


def test_torch_migration_preserves_bfloat16_and_tensor_learning_rate() -> None:
    """Keep exact low-precision moment bytes and independent tensor hyperparameters."""

    model = torch.nn.Linear(3, 2).to(torch.bfloat16)
    graph = prepare_graph(
        model, torch.ones(1, 3, dtype=torch.bfloat16), features=SplitFeatures(training=True)
    )
    cut = graph.compute_dag().to_partition(())
    runtime = graph.materialize(graph.analyze(cut))
    rate = torch.tensor(0.005, dtype=torch.float64)
    native = torch.optim.Adam(model.parameters(), lr=rate)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    native.step()

    plan = runtime.plan_optimizer_state_migration(cut, optimizers={"suffix": native})
    assert plan.supported, plan.unsupported_reasons
    snapshot = runtime.export_optimizer_state(optimizers={"suffix": native}, plan=plan)
    restored = runtime.repartition(cut).restore_optimizer_state(snapshot).optimizers["suffix"]
    new_rate = restored.param_groups[0]["lr"]
    assert new_rate is not rate
    torch.testing.assert_close(new_rate, rate, atol=0, rtol=0)
    for parameter in model.parameters():
        state = restored.state[parameter]
        assert state["exp_avg"].dtype == torch.bfloat16
        torch.testing.assert_close(
            state["exp_avg"], native.state[parameter]["exp_avg"], atol=0, rtol=0
        )


@pytest.mark.parametrize("mode", ["fresh", "used", "wrong_type", "wrong_config"])
def test_torch_migration_validates_the_native_factory(mode: str) -> None:
    """A factory must preserve source history and supply a compatible fresh optimizer."""

    model = torch.nn.Linear(3, 2)
    inputs = torch.ones(2, 3)
    graph = prepare_graph(model, inputs, features=SplitFeatures(training=True))
    cut = graph.compute_dag().to_partition(())
    runtime = graph.materialize(graph.analyze(cut))
    native = torch.optim.Adam(model.parameters(), lr=0.01)
    runtime.train_suffix(runtime.run_training_prefix(inputs), torch.zeros(2, 2), optimizer=native)
    source = {"suffix": native}
    plan = runtime.plan_optimizer_state_migration(cut, optimizers=source)
    snapshot = runtime.export_optimizer_state(optimizers=source, plan=plan)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    target = runtime.repartition(cut)

    def factory(parameters, configuration):
        """Supply the requested kind of target optimizer for validation."""

        group = configuration["groups"][0]
        ordered = [parameters[key] for key in group["parameters"]]
        if mode == "used":
            return native
        if mode == "wrong_type":
            return torch.optim.SGD(ordered, lr=0.01)
        return torch.optim.Adam(ordered, lr=0.01 if mode == "fresh" else 0.02)

    if mode == "fresh":
        result = target.restore_optimizer_state(snapshot, optimizer_factories={"suffix": factory})
        assert result.optimizers["suffix"] is not native
        assert result.restored_slots == 6
    else:
        with pytest.raises(SplitUnsupportedError, match="Factory returned|factories must"):
            target.restore_optimizer_state(snapshot, optimizer_factories={"suffix": factory})
    assert {int(state["step"]) for state in native.state.values()} == {1}
    for weight, unchanged in zip(model.parameters(), before, strict=True):
        torch.testing.assert_close(weight, unchanged, atol=0, rtol=0)
