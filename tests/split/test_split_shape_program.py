"""Backend-neutral dynamic-batch shape-program contracts."""

from __future__ import annotations

import pytest
import torch
from dag_helpers import close_call_ids

from torchlens.split import SplitFeatures, prepare_graph
from torchlens.split.errors import SplitBoundaryError, SplitUnsupportedError


def test_explicit_multi_input_batch_axes_propagate_through_permute() -> None:
    """Different input axes bind one B and propagate through axis movement."""

    class Model(torch.nn.Module):
        def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            aligned = right.permute(1, 0, 2)
            return (left + aligned).reshape(left.shape[0], -1) * 2.0

    model = Model().eval()
    left = torch.ones(2, 3, 4)
    right = torch.ones(3, 2, 4)
    graph = prepare_graph(
        model,
        (left, right),
        backend="torch",
        features=SplitFeatures(batch_axes={"/args/0": 0, "/args/1": 1}),
    )
    dag = graph.compute_dag()
    partition = dag.to_partition(
        close_call_ids(
            dag,
            frozenset(call.call_id for call in dag.nodes[: (len(dag.nodes) - 1) * 50 // 100 + 1]),
        )
    )
    runtime = graph.materialize(graph.analyze(partition))

    for batch in (1, 3, 5):
        replay_left = torch.ones(batch, 3, 4)
        replay_right = torch.ones(3, batch, 4)
        torch.testing.assert_close(
            runtime.replay(replay_left, replay_right),
            model(replay_left, replay_right),
        )


def test_kwargs_batch_axis_uses_json_pointer_and_runtime_override() -> None:
    """Keyword tensor leaves participate in explicit batch consistency checks."""

    class Model(torch.nn.Module):
        def forward(self, values: torch.Tensor, *, mask: torch.Tensor) -> torch.Tensor:
            return (values * mask).reshape(values.shape[0], -1)

    model = Model().eval()
    values = torch.ones(2, 3, 4)
    mask = torch.ones(2, 3, 4)
    graph = prepare_graph(
        model,
        values,
        backend="torch",
        features=SplitFeatures(batch_axes={"/args/0": 0, "/kwargs/mask": 0}),
        input_kwargs={"mask": mask},
    )
    dag = graph.compute_dag()
    partition = dag.to_partition(
        close_call_ids(
            dag,
            frozenset(call.call_id for call in dag.nodes[: (len(dag.nodes) - 1) * 50 // 100 + 1]),
        )
    )
    runtime = graph.materialize(graph.analyze(partition))

    replay_values = torch.ones(4, 3, 4)
    replay_mask = torch.ones(4, 3, 4)
    torch.testing.assert_close(
        runtime.replay(replay_values, input_kwargs={"mask": replay_mask}),
        model(replay_values, mask=replay_mask),
    )
    with pytest.raises(SplitBoundaryError, match="disagree"):
        runtime.replay(replay_values, input_kwargs={"mask": torch.ones(3, 3, 4)})


def test_explicit_mode_rejects_unique_undeclared_batch_dimension() -> None:
    """A matching dimension alone cannot prove an omitted tensor carries batch."""

    class Model(torch.nn.Module):
        def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            return values * mask

    model = Model().eval()
    with pytest.raises(SplitUnsupportedError, match="undeclared tensor paths"):
        capture_3_graph = prepare_graph(
            model,
            (torch.ones(2, 3), torch.ones(2, 3)),
            backend="torch",
            features=SplitFeatures(batch_axes={"/args/0": 0}),
        )
        capture_3_dag = capture_3_graph.compute_dag()
        capture_3_partition = capture_3_dag.to_partition(
            close_call_ids(
                capture_3_dag,
                frozenset(
                    call.call_id
                    for call in capture_3_dag.nodes[
                        : (len(capture_3_dag.nodes) - 1) * 50 // 100 + 1
                    ]
                ),
            )
        )
        capture_3_graph.materialize(capture_3_graph.analyze(capture_3_partition))


def test_explicit_mode_rejects_ambiguous_undeclared_batch_input() -> None:
    """An omitted tensor with multiple candidate batch axes requires a declaration."""

    class Model(torch.nn.Module):
        def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            return values + mask[:, 0]

    with pytest.raises(SplitUnsupportedError, match="undeclared tensor paths"):
        capture_4_graph = prepare_graph(
            Model().eval(),
            (torch.ones(2, 3), torch.ones(2, 2, 3)),
            backend="torch",
            features=SplitFeatures(batch_axes={"/args/0": 0}),
        )
        capture_4_dag = capture_4_graph.compute_dag()
        capture_4_partition = capture_4_dag.to_partition(
            close_call_ids(
                capture_4_dag,
                frozenset(
                    call.call_id
                    for call in capture_4_dag.nodes[
                        : (len(capture_4_dag.nodes) - 1) * 50 // 100 + 1
                    ]
                ),
            )
        )
        capture_4_graph.materialize(capture_4_graph.analyze(capture_4_partition))


def test_validate_equivalence_uses_prepared_and_override_kwargs() -> None:
    """Equivalence validation invokes both paths with the same keyword arguments."""

    class Model(torch.nn.Module):
        def forward(self, values: torch.Tensor, *, scale: torch.Tensor) -> torch.Tensor:
            return values * scale

    model = Model().eval()
    values = torch.ones(2, 3)
    graph = prepare_graph(
        model,
        values,
        backend="torch",
        features=SplitFeatures(batch_axes={"/args/0": 0, "/kwargs/scale": 0}),
        input_kwargs={"scale": torch.full((2, 3), 2.0)},
    )
    dag = graph.compute_dag()
    partition = dag.to_partition(
        close_call_ids(
            dag,
            frozenset(call.call_id for call in dag.nodes[: (len(dag.nodes) - 1) * 50 // 100 + 1]),
        )
    )
    runtime = graph.materialize(graph.analyze(partition))

    assert runtime.validate_equivalence(model, (values,))
    assert runtime.validate_equivalence(
        model,
        (values,),
        input_kwargs={"scale": torch.full((2, 3), 3.0)},
    )


def test_reshape_retains_batch_product_expression() -> None:
    """A reshape dimension derived from B times a fixed axis is solved at runtime."""

    class Model(torch.nn.Module):
        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return values.reshape(values.shape[0] * values.shape[1], -1) + 1.0

    model = Model().eval()
    example = torch.ones(2, 3, 4)
    graph = prepare_graph(
        model, example, backend="torch", features=SplitFeatures(batch_axes={"/args/0": 0})
    )
    dag = graph.compute_dag()
    partition = dag.to_partition(
        close_call_ids(
            dag,
            frozenset(call.call_id for call in dag.nodes[: (len(dag.nodes) - 1) * 50 // 100 + 1]),
        )
    )
    runtime = graph.materialize(graph.analyze(partition))

    for batch in (1, 4, 5):
        values = torch.ones(batch, 3, 4)
        output = runtime.replay(values)
        assert output.shape == (batch * 3, 4)
        torch.testing.assert_close(output, model(values))


def test_explicit_batch_axes_reject_unknown_and_non_batch_changes() -> None:
    """Invalid declarations and runtime non-batch changes fail structurally."""

    model = torch.nn.Sequential(torch.nn.Flatten(start_dim=1), torch.nn.Linear(12, 2)).eval()
    example = torch.ones(2, 3, 4)
    with pytest.raises(SplitUnsupportedError, match="Unknown batch input paths"):
        capture_7_graph = prepare_graph(
            model, example, backend="torch", features=SplitFeatures(batch_axes={"/args/9": 0})
        )
        capture_7_dag = capture_7_graph.compute_dag()
        capture_7_partition = capture_7_dag.to_partition(
            close_call_ids(
                capture_7_dag,
                frozenset(
                    call.call_id
                    for call in capture_7_dag.nodes[
                        : (len(capture_7_dag.nodes) - 1) * 50 // 100 + 1
                    ]
                ),
            )
        )
        capture_7_graph.materialize(capture_7_graph.analyze(capture_7_partition))

    graph = prepare_graph(
        model, example, backend="torch", features=SplitFeatures(batch_axes={"/args/0": 0})
    )
    dag = graph.compute_dag()
    partition = dag.to_partition(
        close_call_ids(
            dag,
            frozenset(call.call_id for call in dag.nodes[: (len(dag.nodes) - 1) * 50 // 100 + 1]),
        )
    )
    runtime = graph.materialize(graph.analyze(partition))
    with pytest.raises(SplitBoundaryError, match="non-batch dimension"):
        runtime.replay(torch.ones(3, 4, 4))


def test_auto_inference_rejects_nested_ambiguous_inputs() -> None:
    """Conservative auto mode requires explicit paths for nested input trees."""

    class Model(torch.nn.Module):
        def forward(self, values: dict[str, torch.Tensor]) -> torch.Tensor:
            return values["x"] * 2.0

    with pytest.raises(SplitUnsupportedError, match="could not be inferred conservatively"):
        capture_9_graph = prepare_graph(
            Model().eval(),
            {"x": torch.ones(2, 3)},
            backend="torch",
            features=SplitFeatures(batch_axes=None),
        )
        capture_9_dag = capture_9_graph.compute_dag()
        capture_9_partition = capture_9_dag.to_partition(
            close_call_ids(
                capture_9_dag,
                frozenset(
                    call.call_id
                    for call in capture_9_dag.nodes[
                        : (len(capture_9_dag.nodes) - 1) * 50 // 100 + 1
                    ]
                ),
            )
        )
        capture_9_graph.materialize(capture_9_graph.analyze(capture_9_partition))


def test_nested_dict_inputs_rebatch_via_explicit_axes() -> None:
    """A nested input tree rebatches every declared leaf, not just the top tensor."""

    class Model(torch.nn.Module):
        def forward(self, values: dict[str, torch.Tensor]) -> torch.Tensor:
            return values["x"] + values["y"]

    model = Model().eval()
    example = {"x": torch.ones(2, 3), "y": torch.ones(2, 3)}
    graph = prepare_graph(
        model,
        example,
        backend="torch",
        features=SplitFeatures(batch_axes={"/args/0/x": 0, "/args/0/y": 0}),
    )
    dag = graph.compute_dag()
    partition = dag.to_partition(
        close_call_ids(
            dag,
            frozenset(call.call_id for call in dag.nodes[: (len(dag.nodes) - 1) * 50 // 100 + 1]),
        )
    )
    runtime = graph.materialize(graph.analyze(partition))
    assert runtime.traced_batch_size == 1
    for batch in (1, 3, 8):
        values = {"x": torch.ones(batch, 3), "y": torch.ones(batch, 3) * 2}
        torch.testing.assert_close(runtime.replay(values), model(values))
