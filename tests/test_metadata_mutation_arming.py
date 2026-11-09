"""Planted corruptions for metadata contracts omitted by the original campaign."""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.validation.invariants import METADATA_INVARIANT_CONTRACTS, MetadataInvariantError

pytestmark = pytest.mark.smoke


def _check(name: str, trace: Any) -> None:
    """Run the actual registered contract against a trace or a malformed row fixture."""

    next(contract for contract in METADATA_INVARIANT_CONTRACTS if contract.name == name).check(
        trace
    )


@pytest.fixture
def small_trace() -> tl.Trace:
    """Capture a complete, valid graph before planting one metadata corruption."""

    return tl.trace(
        nn.Sequential(nn.Linear(3, 3), nn.ReLU()),
        torch.ones(2, 3),
        capture=tl.options.CaptureOptions(cache=False, compute_input_output_distances=True),
    )


@pytest.fixture
def recurrent_trace() -> tl.Trace:
    """Capture two passes through one shared parameterized module."""

    class Repeated(nn.Module):
        """Reuse one linear module on two same-shaped passes."""

        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(3, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return the second application of the same linear module."""

            return self.linear(self.linear(x))

    return tl.trace(Repeated(), torch.ones(2, 3), capture=tl.options.CaptureOptions(cache=False))


def test_trace_parameter_count_must_agree_with_independent_records(small_trace: tl.Trace) -> None:
    """A valid capture refuses a parameter count contradicting its parameter rows."""

    _check("trace_self_consistency", small_trace)
    small_trace.num_param_tensors = -1
    with pytest.raises(MetadataInvariantError, match="num_param_tensors"):
        _check("trace_self_consistency", small_trace)


def test_operator_label_inventory_cannot_contain_duplicates(small_trace: tl.Trace) -> None:
    """Repeating a label preserves the inventory length but corrupts its identity."""

    _check("trace_self_consistency", small_trace)
    small_trace.op_labels[1] = small_trace.op_labels[0]
    with pytest.raises(MetadataInvariantError, match="Duplicate op_labels"):
        _check("trace_self_consistency", small_trace)


def test_parameter_partitions_must_sum_to_the_total(small_trace: tl.Trace) -> None:
    """A changed trainable count cannot inherit the unchanged total's validity."""

    _check("trace_self_consistency", small_trace)
    small_trace.num_params_trainable += 1
    with pytest.raises(MetadataInvariantError, match=r"trainable\(.*frozen\("):
        _check("trace_self_consistency", small_trace)


def test_saved_tensor_count_cannot_exceed_the_capture(small_trace: tl.Trace) -> None:
    """A saved-count claim is bounded by the independent retained graph size."""

    _check("trace_self_consistency", small_trace)
    small_trace.num_saved_ops = small_trace.num_tensors + 1
    with pytest.raises(MetadataInvariantError, match="num_saved_ops"):
        _check("trace_self_consistency", small_trace)


def test_special_layer_lists_require_known_labels(small_trace: tl.Trace) -> None:
    """A forged input label must give a metadata error before fuzzy lookup."""

    _check("special_layer_lists", small_trace)
    small_trace.input_layers.append("nonexistent-input")
    with pytest.raises(MetadataInvariantError, match="contains labels not in"):
        _check("special_layer_lists", small_trace)


@pytest.mark.parametrize(
    ("contract", "field", "value", "message"),
    [
        ("non_torch_backward_inert", "grad_fn_order", ["forged"], "grad_fn_order"),
        ("non_torch_primitive_op_inert", "_primitive_op_profile", object(), "primitive profile"),
    ],
)
def test_neutral_backend_cannot_claim_torch_execution_state(
    contract: str, field: str, value: Any, message: str
) -> None:
    """A neutral analysis row rejects unsupported backward or primitive evidence."""

    trace = SimpleNamespace(backend="mlx")
    _check(contract, trace)
    setattr(trace, field, value)
    with pytest.raises(MetadataInvariantError, match=message):
        _check(contract, trace)


def test_neutral_graph_dangling_parent_is_a_named_metadata_error() -> None:
    """Missing graph endpoints refuse before any fuzzy or foreign lookup runs."""

    op = SimpleNamespace(layer_label="op", parents=(), children=())
    trace = SimpleNamespace(layer_list=[op])
    _check("backend_neutral_graph_topology", trace)
    op.parents = ("missing",)
    with pytest.raises(MetadataInvariantError, match="outside trace labels"):
        _check("backend_neutral_graph_topology", trace)


def test_parent_argument_metadata_must_be_a_mapping(small_trace: tl.Trace) -> None:
    """A malformed argument-position table refuses rather than leaking AttributeError."""

    _check("edge_use_parent_arg_consistency", small_trace)
    small_trace.compute_ops[0].parent_arg_positions = ["not a mapping"]
    with pytest.raises(MetadataInvariantError, match="non-mapping parent_arg_positions"):
        _check("edge_use_parent_arg_consistency", small_trace)


def test_argument_position_domain_must_hold_a_mapping(small_trace: tl.Trace) -> None:
    """Malformed positional entries refuse before iteration or edge reconciliation."""

    _check("edge_use_parent_arg_consistency", small_trace)
    small_trace.compute_ops[0].parent_arg_positions = {"args": ["malformed"], "kwargs": {}}
    with pytest.raises(MetadataInvariantError, match="is not a mapping"):
        _check("edge_use_parent_arg_consistency", small_trace)


def test_edge_use_argument_kind_must_be_known(small_trace: tl.Trace) -> None:
    """An otherwise valid edge occurrence cannot claim an unknown argument kind."""

    _check("edge_use_parent_arg_consistency", small_trace)
    op = small_trace.compute_ops[0]
    assert op._edge_uses
    op._edge_uses = (replace(op._edge_uses[0], arg_kind="invalid"), *op._edge_uses[1:])
    with pytest.raises(MetadataInvariantError, match="invalid edge arg_kind"):
        _check("edge_use_parent_arg_consistency", small_trace)


@pytest.mark.parametrize("contract", ["layer_pass_layer_log_xrefs", "pass_count_consistency"])
def test_pass_key_must_equal_its_recorded_pass_index(small_trace: tl.Trace, contract: str) -> None:
    """A real per-pass row cannot disagree with its owning Layer map key."""

    _check(contract, small_trace)
    small_trace.compute_ops[0].pass_index = 2
    with pytest.raises(MetadataInvariantError, match="pass_index"):
        _check(contract, small_trace)


def test_layer_map_key_must_agree_with_its_layer_address(small_trace: tl.Trace) -> None:
    """A valid Layer stored under a different key cannot pass cross-reference checks."""

    _check("layer_pass_layer_log_xrefs", small_trace)
    layer = next(iter(small_trace.layer_logs.values()))
    small_trace.layer_logs = {"forged-key": layer}
    with pytest.raises(MetadataInvariantError, match="Layer key"):
        _check("layer_pass_layer_log_xrefs", small_trace)


def test_reverse_lookup_keys_require_a_forward_entry(small_trace: tl.Trace) -> None:
    """An extra reverse-index key is detected independently of forward entries."""

    _check("lookup_key_consistency", small_trace)
    keys = next(iter(small_trace._layer_num_to_lookup_keys_dict.values()))
    keys.append("forged-reverse-only-key")
    with pytest.raises(MetadataInvariantError, match="not in _lookup_keys_to_layer_num_dict"):
        _check("lookup_key_consistency", small_trace)


def test_forward_lookup_requires_a_reverse_bucket(small_trace: tl.Trace) -> None:
    """A missing reverse bucket gives a named refusal before its entries are read."""

    _check("lookup_key_consistency", small_trace)
    bucket = next(iter(small_trace._lookup_keys_to_layer_num_dict.values()))
    del small_trace._layer_num_to_lookup_keys_dict[bucket]
    with pytest.raises(MetadataInvariantError, match="not in _layer_num_to_lookup_keys_dict"):
        _check("lookup_key_consistency", small_trace)
    assert bucket not in small_trace._layer_num_to_lookup_keys_dict, (
        "validation must not recreate a missing reverse bucket before refusing"
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("min_distance_to_output", 10, "min_distance_to_output"),
        ("has_output_descendant", False, "has_output_descendant"),
    ],
)
def test_output_distance_claims_require_consistent_evidence(
    small_trace: tl.Trace, field: str, value: Any, message: str
) -> None:
    """Computed output distances and reachability flags refuse contradictory claims."""

    assert small_trace.mark_layer_depths
    _check("distance_invariants", small_trace)
    op = small_trace.compute_ops[0]
    assert op.max_distance_to_output < 10
    assert op.output_descendants
    setattr(op, field, value)
    with pytest.raises(MetadataInvariantError, match=message):
        _check("distance_invariants", small_trace)


def test_buffer_inventory_cannot_name_an_absent_layer(small_trace: tl.Trace) -> None:
    """A buffer-layer inventory entry needs a retained layer at that address."""

    _check("buffer_xrefs", small_trace)
    small_trace.buffer_layers.append("nonexistent-buffer-layer")
    with pytest.raises(MetadataInvariantError, match="buffer_layers contains"):
        _check("buffer_xrefs", small_trace)


def test_child_edge_requires_reciprocal_parent(small_trace: tl.Trace) -> None:
    """Dropping the child's parent edge leaves a detectable asymmetric graph."""

    _check("graph_topology", small_trace)
    child = small_trace.compute_ops[0]
    child.parents = ()
    with pytest.raises(MetadataInvariantError, match="does not list .* as parent"):
        _check("graph_topology", small_trace)


def test_input_node_cannot_acquire_even_a_reciprocal_self_parent(small_trace: tl.Trace) -> None:
    """Reciprocal edge consistency never licenses a parent on a model input."""

    _check("graph_topology", small_trace)
    source = next(op for op in small_trace.ops if op.is_input)
    source.parents = (source.layer_label,)
    source.children = (*source.children, source.layer_label)
    with pytest.raises(MetadataInvariantError, match="Input layer .* has parents"):
        _check("graph_topology", small_trace)


def test_parent_argument_domains_are_closed(small_trace: tl.Trace) -> None:
    """A foreign argument domain must refuse even when its contents are empty."""

    _check("graph_topology", small_trace)
    small_trace.compute_ops[0].parent_arg_positions["forged-domain"] = {}
    with pytest.raises(MetadataInvariantError, match="foreign top-level"):
        _check("graph_topology", small_trace)


def test_capture_order_requires_monotonic_indices(small_trace: tl.Trace) -> None:
    """Reordering rows preserves every edge and index but corrupts capture order."""

    _check("graph_ordering", small_trace)
    small_trace.layer_list = list(reversed(small_trace.layer_list))
    with pytest.raises(MetadataInvariantError, match="not monotonically increasing"):
        _check("graph_ordering", small_trace)


def test_retained_orphan_cannot_reenter_the_active_projection(small_trace: tl.Trace) -> None:
    """A retained-island claim on an active label is graph corruption."""

    _check("graph_connectivity", small_trace)
    op = small_trace.compute_ops[0]
    small_trace._orphan_logs = [
        SimpleNamespace(is_orphan=True, label=op.label, layer_label=op.layer_label)
    ]
    with pytest.raises(MetadataInvariantError, match="Retained orphan labels survive"):
        _check("graph_connectivity", small_trace)


def test_loop_member_labels_must_resolve(small_trace: tl.Trace) -> None:
    """A forged recurrence member gives a named failure before ordinary lookup."""

    _check("loop_detection_invariants", small_trace)
    op = small_trace.compute_ops[0]
    op.recurrent_ops = (op.label, "nonexistent-op")
    with pytest.raises(MetadataInvariantError, match="not in op_labels"):
        _check("loop_detection_invariants", small_trace)


def test_layer_equivalence_view_requires_a_set(small_trace: tl.Trace) -> None:
    """Malformed Layer metadata cannot bypass the per-Op equivalence check."""

    op = small_trace.compute_ops[0]
    layer = SimpleNamespace(
        layer_label=op.layer_label, ops={1: op}, equivalent_ops=op.equivalent_ops
    )
    small_trace.layer_logs = {op.layer_label: layer}
    _check("equivalence_symmetry", small_trace)
    layer.equivalent_ops = ["malformed-view"]
    with pytest.raises(
        MetadataInvariantError, match="Layer .* equivalent_ops|Layer .*equivalent_ops"
    ):
        _check("equivalence_symmetry", small_trace)


def test_ungrouped_equivalence_view_cannot_name_an_absent_op() -> None:
    """An analysis row with no expected group still enforces the label namespace."""

    op = SimpleNamespace(label="op", equivalence_class="ungrouped", equivalent_ops={"op"})
    trace = SimpleNamespace(
        op_labels=["op"], op_equivalence_classes={}, layer_list=[op], layer_logs={}
    )
    _check("equivalence_symmetry", trace)
    op.equivalent_ops.add("nonexistent-op")
    with pytest.raises(MetadataInvariantError, match="not in op_labels"):
        _check("equivalence_symmetry", trace)


def test_layer_passes_must_agree_on_their_equivalence_family(small_trace: tl.Trace) -> None:
    """A Layer cannot combine independently valid ops from different families."""

    first, second = small_trace.compute_ops
    assert first.equivalent_ops != second.equivalent_ops
    layer = SimpleNamespace(
        layer_label=first.layer_label, ops={1: first}, equivalent_ops=first.equivalent_ops
    )
    small_trace.layer_logs = {first.layer_label: layer}
    _check("equivalence_symmetry", small_trace)
    layer.ops[2] = second
    with pytest.raises(MetadataInvariantError, match="ops disagree on equivalent_ops"):
        _check("equivalence_symmetry", small_trace)


def test_operator_raw_index_must_be_positive(small_trace: tl.Trace) -> None:
    """A valid computational row refuses a zero raw capture index."""

    _check("op_log_fields", small_trace)
    small_trace.compute_ops[0].raw_index = 0
    with pytest.raises(MetadataInvariantError, match="raw_index=0 < 1"):
        _check("op_log_fields", small_trace)


def test_structural_site_keys_are_validated_on_retained_ops(small_trace: tl.Trace) -> None:
    """A forged site-key string must fail the registered metadata contract."""

    _check("site_key_invariants", small_trace)
    small_trace.compute_ops[0].site_key = "forged-key"
    with pytest.raises(MetadataInvariantError):
        _check("site_key_invariants", small_trace)


def test_negative_capture_duration_refuses_from_recorded_clocks(small_trace: tl.Trace) -> None:
    """Invalid clock order is checked through its derived duration."""

    _check("trace_self_consistency", small_trace)
    small_trace.capture_end_time = small_trace.capture_start_time - 1
    with pytest.raises(MetadataInvariantError, match="capture_duration"):
        _check("trace_self_consistency", small_trace)


@pytest.mark.parametrize("field", ["is_branching", "is_recurrent", "max_layer_op_count"])
def test_derived_structure_claims_cannot_be_forged(small_trace: tl.Trace, field: str) -> None:
    """Structure flags are read-only properties of the underlying graph records."""

    with pytest.raises(AttributeError):
        setattr(small_trace, field, "forged")


def test_module_layer_counts_are_read_only_views(small_trace: tl.Trace) -> None:
    """Module and ModuleCall counts cannot diverge from their own layer inventories."""

    module = small_trace.modules["self"]
    call = next(iter(module.ops.values()))
    assert module.num_layers == len(module.layer_labels)
    assert call.num_layers == len(call.ops)
    with pytest.raises(AttributeError):
        module.num_layers = -1
    with pytest.raises(AttributeError):
        call.num_layers = -1


def test_module_hierarchy_requires_a_root_module(small_trace: tl.Trace) -> None:
    """An empty module accessor cannot masquerade as a complete hierarchy."""

    _check("module_hierarchy", small_trace)
    with pytest.raises(MetadataInvariantError, match="'self' module not found"):
        _check("module_hierarchy", SimpleNamespace(modules={}))


def test_module_pass_keys_must_remain_contiguous(small_trace: tl.Trace) -> None:
    """Changing a module's pass key preserves its size but corrupts its identity."""

    _check("module_hierarchy", small_trace)
    module = small_trace.modules["self"]
    module.ops = {2: next(iter(module.ops.values()))}
    with pytest.raises(MetadataInvariantError, match="pass keys"):
        _check("module_hierarchy", small_trace)


def test_module_child_requires_its_recorded_parent(small_trace: tl.Trace) -> None:
    """A module child listed under self cannot claim a different container parent."""

    _check("module_hierarchy", small_trace)
    root = small_trace.modules["self"]
    child = small_trace.modules[root.address_children[0]]
    assert not child.has_multiple_addresses
    child.address_parent = "uninvoked-container"
    with pytest.raises(MetadataInvariantError, match="child's address_parent"):
        _check("module_hierarchy", small_trace)


def test_module_call_child_requires_a_module_address(small_trace: tl.Trace) -> None:
    """An unresolved call-child link gives a named module-accessor refusal."""

    _check("module_hierarchy", small_trace)
    call = next(iter(small_trace.modules["self"].ops.values()))
    call.call_children.append("nonexistent:1")
    with pytest.raises(MetadataInvariantError, match="call_children .*not in module accessor"):
        _check("module_hierarchy", small_trace)


def test_op_module_path_cannot_name_an_unknown_module(small_trace: tl.Trace) -> None:
    """Conforming the leaf and path claims cannot license a missing module."""

    _check("module_containment_logic", small_trace)
    op = small_trace.compute_ops[0]
    op.modules = ["nonexistent:1"]
    op.module = "nonexistent:1"
    with pytest.raises(MetadataInvariantError, match="unknown module address"):
        _check("module_containment_logic", small_trace)


def test_layer_module_address_must_resolve(small_trace: tl.Trace) -> None:
    """An op cannot claim a module address absent from the captured hierarchy."""

    _check("module_layer_containment", small_trace)
    small_trace.compute_ops[0].module = "nonexistent:1"
    with pytest.raises(MetadataInvariantError, match="not found in module accessor"):
        _check("module_layer_containment", small_trace)


def test_recurrence_pass_map_requires_all_declared_passes() -> None:
    """A counted layer with a missing pass is refused even in neutral row metadata."""

    layer = SimpleNamespace(num_passes=1, ops={1: object()})
    trace = SimpleNamespace(
        is_recurrent=False,
        layer_num_calls={"layer": 1},
        layer_labels=["layer"],
        layer_logs={"layer": layer},
    )
    _check("recurrence_invariants", trace)
    layer.ops = {2: layer.ops[1]}
    with pytest.raises(MetadataInvariantError, match="ops keys"):
        _check("recurrence_invariants", trace)


def test_recurrence_count_inventory_requires_a_retained_layer(small_trace: tl.Trace) -> None:
    """A new recurrence-count key needs a layer even when it declares only one pass."""

    _check("recurrence_invariants", small_trace)
    small_trace.layer_num_calls["nonexistent-layer"] = 1
    with pytest.raises(MetadataInvariantError, match="layer_num_calls key"):
        _check("recurrence_invariants", small_trace)


def test_recurrent_passes_must_share_equivalence_class(recurrent_trace: tl.Trace) -> None:
    """Tied module calls cannot disagree about their recorded structural family."""

    _check("loop_detection_invariants", recurrent_trace)
    passes = [op for op in recurrent_trace.ops if op.func_name == "linear"]
    assert len(passes) == 2
    assert passes[0].layer_label == passes[1].layer_label
    passes[1].equivalence_class = "forged-family"
    with pytest.raises(MetadataInvariantError, match="recurrent_ops type mismatch"):
        _check("loop_detection_invariants", recurrent_trace)


def test_later_recurrent_pass_cannot_claim_a_different_count(recurrent_trace: tl.Trace) -> None:
    """Every pass keeps its own count witness even when group checks were memoized."""

    _check("loop_detection_invariants", recurrent_trace)
    passes = [op for op in recurrent_trace.ops if op.func_name == "linear"]
    assert len(passes) == 2
    passes[1].num_passes = 3
    with pytest.raises(MetadataInvariantError, match="num_passes=3"):
        _check("loop_detection_invariants", recurrent_trace)


def test_recurrent_membership_must_be_symmetric(recurrent_trace: tl.Trace) -> None:
    """A one-sided membership deletion refuses even with a matching local pass count."""

    _check("loop_detection_invariants", recurrent_trace)
    passes = [op for op in recurrent_trace.ops if op.func_name == "linear"]
    assert len(passes) == 2
    passes[1].recurrent_ops = (passes[1].label,)
    passes[1].num_passes = 1
    with pytest.raises(MetadataInvariantError, match="Asymmetric recurrent_ops"):
        _check("loop_detection_invariants", recurrent_trace)


def test_multi_pass_operator_label_must_name_the_pass(small_trace: tl.Trace) -> None:
    """A multipass claim cannot retain an unqualified operator label."""

    _check("op_log_fields", small_trace)
    op = small_trace.compute_ops[0]
    op.num_passes = 2
    op.label = op.layer_label
    with pytest.raises(MetadataInvariantError, match="has no ':'"):
        _check("op_log_fields", small_trace)


def test_cached_influence_geometry_requires_consistent_alignment(small_trace: tl.Trace) -> None:
    """A cached geometry row cannot claim merge alignment absent in its state witness."""

    from torchlens.receptive_field import _engine

    _check("receptive_field_metadata", small_trace)
    solution = _engine.solve(small_trace)
    key, descriptor = next(iter(solution.descriptors.items()))
    descriptors = dict(solution.descriptors)
    descriptors[key] = replace(descriptor, alignment=type(descriptor.alignment).ALIGNED)
    small_trace.__dict__["_receptive_field_solution"] = replace(
        solution, descriptors=MappingProxyType(descriptors)
    )
    with pytest.raises(MetadataInvariantError, match="alignment coherence"):
        _check("receptive_field_metadata", small_trace)
