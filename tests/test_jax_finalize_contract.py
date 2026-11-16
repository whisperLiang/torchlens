"""Final JAX labels and the jaxpr site-instance validation contract."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import torchlens as tl
from torchlens.validation._invariants import MetadataInvariantError, check_metadata_invariants
from torchlens.validation._invariants_sites import _check_site_key_uniqueness


@pytest.mark.parametrize("recurrence", [False, True])
def test_jax_final_labels_preserve_raw_aliases_and_replay(recurrence: bool) -> None:
    """Finalization relabels relations without changing capture identities or outputs."""

    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    def model(value: Any) -> Any:
        """Return a small arithmetic chain with a saved intermediate."""

        return jnp.tanh(value + 1)

    with jax.default_device(jax.devices("cpu")[0]):
        value = jnp.ones((2, 3), dtype=jnp.float32)
        trace = tl.trace(
            model,
            (value,),
            backend="jax",
            capture=tl.options.CaptureOptions(recurrence_detection=recurrence),
        )
        assert check_metadata_invariants(trace) is True
        assert trace.validate_forward_pass([]) is True
        np.testing.assert_array_equal(trace.output_ops[0].out, model(value))
        for op in trace._layer_list:
            assert op._label_raw.endswith("_raw")
            assert "_raw" not in op.label
            assert "_raw" not in op.layer_label
            assert trace._layer_dict_all_keys[op._label_raw] is op
            assert trace._layer_dict_all_keys[op.label] is op
            assert op.layer_label in op.lookup_keys
            for labels in (op.parents, op.children, op.graph.root_ancestors, op.input_ancestors):
                for label in labels:
                    assert "_raw" not in label
                    assert label in trace._layer_dict_all_keys
        assert set(trace.jax_capture_index_to_final_op_label.values()) <= set(trace.op_labels)


def test_jax_site_uniqueness_uses_iteration_instance_and_rejects_duplicates() -> None:
    """Repeated sites across iterations are legal; duplicates within one are not."""

    from torchlens._postprocess._site_key import SiteKeyMinter
    from torchlens.backends.jax._site_dialect import _jax_site_components

    minter = SiteKeyMinter()
    ops = []
    for index in range(2):
        op = SimpleNamespace(
            label=f"add:{index + 1}",
            annotations={"jax_source_path": f"root/0:scan/iter:{index}/0:add"},
            module_call_stack=(),
            equivalence_class="repeated-add",
            _label_raw=f"add_{index + 1}_raw",
        )
        axis, instance = _jax_site_components(op)
        op.site_key = minter.mint_at(axis, instance, "add", None)
        ops.append(op)
    assert ops[0].site_key == ops[1].site_key
    _check_site_key_uniqueness(ops, "test")
    # Singleton structural families cannot license a repeated key.
    ops[1].equivalence_class = "singleton-add"
    with pytest.raises(MetadataInvariantError, match="I-S2"):
        _check_site_key_uniqueness(ops, "test")
    ops[1].equivalence_class = ops[0].equivalence_class
    ops[1]._label_raw = ops[0]._label_raw
    with pytest.raises(MetadataInvariantError, match="I-S2"):
        _check_site_key_uniqueness(ops, "test")
