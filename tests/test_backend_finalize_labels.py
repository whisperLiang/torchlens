"""Cross-backend agreement for the INCIDENTAL bare-label all-keys binding.

There was never a documented contract for what ``layer_dict_all_keys[bare
layer_label]`` resolves to on a multi-pass layer -- bare-label addressing of
multi-pass layers refuses on every path that matters
(``multipass_bare_label_ambiguous``). But the neutral preview finalizer and
the jax backend used to bind the bare label to the FIRST pass while torch's
raw-index artifact resolves to the LAST pass: an undocumented cross-backend
disagreement waiting to be mistaken for behavior. These tests pin the
alignment: every backend's incidental binding is last-pass-wins, and every
pass carries the bare label in its ``lookup_keys`` (torch parity).
"""

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.backends._finalize import _apply_recurrence_relabel_epilogue, _finalize_single_op
from torchlens.postprocess.loop_grouping_adapter import RecurrenceAssignment

pytestmark = pytest.mark.smoke


def _stub_trace() -> SimpleNamespace:
    """Return a minimal trace stub with the lookup indexes finalize touches."""

    return SimpleNamespace(
        layer_list=[],
        layer_dict_main_keys={},
        layer_dict_all_keys={},
        op_labels=[],
        layer_labels=[],
        layer_num_calls={},
        _lookup_keys_to_layer_num_dict={},
        _layer_num_to_lookup_keys_dict=defaultdict(list),
    )


def _assignment(pass_index: int) -> RecurrenceAssignment:
    """Return a 2-pass assignment for the shared ``cell`` layer."""

    return RecurrenceAssignment(
        layer_label="cell",
        recurrent_labels=("cell_raw_0", "cell_raw_1"),
        pass_index=pass_index,
        num_passes=2,
        equivalence_key="eq:cell",
        site_key=None,
    )


def test_neutral_finalizer_bare_label_is_last_pass_and_on_every_pass() -> None:
    """The shared finalizer's bare-label binding is last-pass-wins."""

    trace = _stub_trace()
    first = SimpleNamespace()
    second = SimpleNamespace()
    _finalize_single_op(trace, first, "cell_raw_0", 0, _assignment(1))
    _finalize_single_op(trace, second, "cell_raw_1", 1, _assignment(2))

    assert trace.layer_dict_all_keys["cell:1"] is first
    assert trace.layer_dict_all_keys["cell:2"] is second
    # Incidental raw-index artifact: last pass wins, matching torch.
    assert trace.layer_dict_all_keys["cell"] is second
    # Torch parity: EVERY pass lists the bare label among its lookup keys.
    assert "cell" in first.lookup_keys
    assert "cell" in second.lookup_keys


@pytest.mark.parametrize("grouping", [False, True])
def test_neutral_finalizer_canonicalizes_singletons_and_relations(grouping: bool) -> None:
    """Public labels are final even without grouping; raw sidecars remain addressable.

    Parameters
    ----------
    grouping:
        Whether singleton assignments were produced by recurrence detection.
    """

    trace = _stub_trace()
    raw_labels = ("input_1_1_raw", "relu_1_2_raw", "output_1_3_raw")
    raw_ops = {}
    assignments = {}
    for index, raw in enumerate(raw_labels):
        op = SimpleNamespace(
            parents=list(raw_labels[index - 1 : index]) if index else [],
            children=list(raw_labels[index + 1 : index + 2]),
            input_ancestors={raw_labels[0]},
            root_ancestors={raw_labels[0]},
            output_descendants={raw_labels[-1]},
            parent_arg_positions={"args": {0: raw_labels[index - 1]} if index else {}},
            buffer_source=raw_labels[0],
            equivalence_class=f"eq:{index}",
            conditional_elif_children={0: [raw_labels[-1]]},
            conditional_arm_children={0: {"then": [raw_labels[-1]]}},
            out_versions_by_child={raw_labels[-1]: object()},
        )
        assignment = RecurrenceAssignment(
            layer_label=raw,
            recurrent_labels=(raw,),
            pass_index=1,
            num_passes=1,
            equivalence_key=f"eq:{index}",
            site_key=None,
        )
        assignments[raw] = assignment
        _finalize_single_op(trace, op, raw, index, assignment if grouping else None)
        raw_ops[raw] = op
        assert op._label_raw == raw
        assert op._layer_label_raw == raw
        assert trace.layer_dict_all_keys[raw] is op
        assert trace.layer_dict_all_keys[op.layer_label] is op
        assert trace.layer_dict_main_keys[op.layer_label] is op

    trace._raw_graph_ws = SimpleNamespace(raw_layer_dict=raw_ops)
    trace.input_layers = [raw_labels[0]]
    trace.output_layers = [raw_labels[-1]]
    trace.internal_source_ops = [raw_labels[0]]
    trace.internal_sink_ops = [raw_labels[-1]]
    trace.op_equivalence_classes = {}
    sidecar_mapping = {}
    _apply_recurrence_relabel_epilogue(
        trace, assignments if grouping else None, sidecar_mapping.update
    )
    assert trace.layer_labels == [raw.removesuffix("_raw") for raw in raw_labels]
    assert trace.input_layers == ["input_1_1"]
    assert trace.output_layers == ["output_1_3"]
    assert trace.internal_source_ops == ["input_1_1:1"]
    assert trace.internal_sink_ops == ["output_1_3:1"]
    assert sidecar_mapping == {raw: op.label for raw, op in raw_ops.items()}
    middle = raw_ops[raw_labels[1]]
    assert middle.parents == ["input_1_1:1"]
    assert middle.children == ["output_1_3:1"]
    assert middle.input_ancestors == middle.root_ancestors == {"input_1_1:1"}
    assert middle.output_descendants == {"output_1_3:1"}
    assert middle.parent_arg_positions == {"args": {0: "input_1_1:1"}}
    assert middle.buffer_source == "input_1_1:1"
    assert middle.conditional_elif_children == {0: ["output_1_3:1"]}
    assert middle.conditional_arm_children == {0: {"then": ["output_1_3:1"]}}
    assert list(middle.out_versions_by_child) == ["output_1_3:1"]
    for op in raw_ops.values():
        assert op.equivalent_ops == {op.label}
        assert op.recurrent_ops == [op.label]


def test_neutral_finalizer_preserves_each_output_pass_and_order() -> None:
    """Output endpoints retain a non-final pass and repeated output positions."""

    trace = _stub_trace()
    raw_ops = {}
    assignments = {}
    for index in range(2):
        raw = f"cell_raw_{index}"
        op = SimpleNamespace(parents=[], children=[])
        assignments[raw] = _assignment(index + 1)
        _finalize_single_op(trace, op, raw, index, assignments[raw])
        raw_ops[raw] = op
    trace._raw_graph_ws = SimpleNamespace(raw_layer_dict=raw_ops)
    trace.output_layers = ["cell_raw_1", "cell_raw_0", "cell_raw_1"]
    trace.op_equivalence_classes = {}

    _apply_recurrence_relabel_epilogue(trace, assignments, None)

    assert trace.output_layers == ["cell:2", "cell:1", "cell:2"]
    assert [trace.layer_dict_all_keys[label] for label in trace.output_layers] == [
        raw_ops["cell_raw_1"],
        raw_ops["cell_raw_0"],
        raw_ops["cell_raw_1"],
    ]


def test_torch_bare_label_artifact_is_last_pass() -> None:
    """Pin the torch side of the parity so the agreement cannot drift."""

    class Loop(nn.Module):
        """Three applications of one shared cell."""

        def __init__(self) -> None:
            """Build the shared cell."""

            super().__init__()
            self.cell = nn.Linear(4, 4, bias=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Apply the cell three times."""

            h = x
            for _ in range(3):
                h = torch.relu(self.cell(h))
            return h

    torch.manual_seed(0)
    log = tl.trace(Loop(), torch.randn(2, 4))
    multi_pass_bare = [
        key
        for key, record in log.layer_dict_all_keys.items()
        if isinstance(key, str) and ":" not in key and record.num_passes > 1
    ]
    assert multi_pass_bare, "expected multi-pass layers in the loop trace"
    for key in multi_pass_bare:
        record = log.layer_dict_all_keys[key]
        if key != record.layer_label:
            continue  # raw/address spellings, not the bare layer label
        assert record.pass_index == record.num_passes, (
            f"torch bare-label artifact for {key!r} no longer resolves to the last pass"
        )
