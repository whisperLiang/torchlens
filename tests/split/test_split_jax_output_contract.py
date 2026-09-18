"""JAX split output reconstruction requires complete declared replay evidence."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
from v2_helpers import split_request

import torchlens as tl
from torchlens.ir.container import DictKey, TupleIndex
from torchlens.split.adapters.jax import JaxGeneratedSuffix, _slice_output_by_path
from torchlens.split.errors import SplitUnsupportedError

pytestmark = [pytest.mark.smoke, pytest.mark.backend_jax]


@pytest.fixture
def output_runtime() -> Any:
    """Build a small JAX runtime with two unambiguously recorded output leaves."""

    jnp = pytest.importorskip("jax.numpy")

    def model(x: Any) -> Any:
        """Return distinct leaves in an integer-keyed nested container."""

        hidden = jnp.maximum(x, 0)
        result = hidden * 2.0
        return {0: result, 1: [result + 1.0]}

    return tl.split.prepare(model, jnp.ones((2, 3)), split_request("after:max", backend="jax"))


@pytest.mark.parametrize("has_intermediate", [False, True])
def test_jax_missing_output_records_never_guess_last_intermediate(
    output_runtime: Any, has_intermediate: bool
) -> None:
    """Neither an empty nor a nonempty overlay establishes an output contract."""

    runtime = output_runtime
    graph = replace(runtime.trace_graph, output_node_ids=())
    suffix = JaxGeneratedSuffix(
        graph=graph,
        plan=runtime.plan,
        spec=runtime.request,
        node_ids=runtime.plan.suffix_node_ids,
    )
    node = graph.compute_nodes[-1]
    overlay = {node.canonical_id: node.op.out} if has_intermediate else {}
    with pytest.raises(SplitUnsupportedError) as error:
        suffix._reconstruct_output(overlay)
    assert error.value.context.reason == "missing output records"


@pytest.mark.parametrize("retain_parent", [False, True])
def test_jax_missing_computed_output_never_substitutes_parent_or_saved_value(
    output_runtime: Any, retain_parent: bool
) -> None:
    """A final equation's operand and historical activation are not its replay result."""

    runtime = output_runtime
    graph = runtime.trace_graph
    node = graph.node_by_id[graph.output_node_ids[-1]]
    assert node.target is not None and node.op.out is not None and node.parents
    parent = graph.node_by_id[graph.node_id_by_alias[node.parents[0]]]
    overlay = {parent.canonical_id: parent.op.out} if retain_parent else {}
    with pytest.raises(SplitUnsupportedError) as error:
        runtime.segments.suffix._output_leaf(node, overlay)
    assert error.value.context.reason == "missing output replay value"


def test_jax_missing_container_spec_never_guesses_integer_key_container_kind(
    output_runtime: Any,
) -> None:
    """Integer paths alone cannot distinguish a dict from a list or tuple."""

    runtime = output_runtime
    graph = replace(
        runtime.trace_graph,
        nodes=tuple(
            replace(node, output_container_spec=None) for node in runtime.trace_graph.nodes
        ),
    )
    suffix = JaxGeneratedSuffix(
        graph=graph,
        plan=runtime.plan,
        spec=runtime.request,
        node_ids=runtime.plan.suffix_node_ids,
    )
    overlay = {node.canonical_id: node.op.out for node in graph.nodes}
    with pytest.raises(SplitUnsupportedError) as error:
        suffix._reconstruct_output(overlay)
    assert error.value.context.reason == "missing output container specification"


def test_jax_repeated_output_without_occurrence_paths_refuses_typed() -> None:
    """Identity-deduplicated capture records cannot prove repeated output positions."""

    jnp = pytest.importorskip("jax.numpy")

    def model(x: Any) -> Any:
        """Return one array twice at different public container paths."""

        hidden = jnp.maximum(x, 0)
        result = hidden * 2.0
        return {0: result, 1: [result, (result + 1.0,)]}

    runtime = tl.split.prepare(model, jnp.ones((2, 3)), split_request("after:max", backend="jax"))
    assert runtime.batch_validation["status"] == "failed"
    with pytest.raises(SplitUnsupportedError) as error:
        runtime.replay(jnp.ones((1, 3)))
    assert error.value.context.reason == "invalid output container records"


def test_jax_container_path_decodes_neutral_sequence_components() -> None:
    """Neutral sequence indices and integer dictionary keys remain distinct components."""

    leaf = object()
    output = {1: [(leaf,)]}
    path = (DictKey(1), TupleIndex(0), TupleIndex(0))
    assert _slice_output_by_path(output, path) is leaf
