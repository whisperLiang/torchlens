"""Pin the upstream SDPA output reshape, its edges, and exact forward/backward parity."""

from __future__ import annotations

import pytest
import torch
import transformers
from packaging.version import Version

from tests.real_model.r0.families import FAMILY_BY_NAME
from tests.real_model.r0.test_realism_randinit_axes import PRIMARY_LEAF
from torchlens._data_classes._state_adapter import state_items

pytestmark = [pytest.mark.real_model, pytest.mark.real_class]

_SDPA_CALLS = {
    "gpt2": 2,
    "distilgpt2": 2,
    "llama": 2,
    "qwen2": 2,
    "albert": 2,
    "distilbert": 2,
    "bert": 2,
    "vit": 2,
    "clip": 4,
    "whisper": 8,
}


@pytest.mark.parametrize("family", _SDPA_CALLS)
def test_sdpa_edges_output_and_parameter_gradients(family, r0_capture) -> None:
    """Every SDPA edge is retained and capture preserves native outputs and gradients exactly."""
    # The 4.45.2 floor predates DistilBERT SDPA. Assert its actual construction
    # refusal; this is a supported old-band contract, not a skipped capture.
    if family == "distilbert" and not transformers.DistilBertPreTrainedModel._supports_sdpa:
        with pytest.raises(ValueError, match="does not support.*scaled_dot_product_attention"):
            FAMILY_BY_NAME[family].build("sdpa")
        return
    cap = r0_capture(family, "sdpa")
    rows = cap.trace._layer_list
    sdpa_indices = [
        i for i, op in enumerate(rows) if op.func_name == "scaled_dot_product_attention"
    ]
    expected_calls = _SDPA_CALLS[family]
    if family == "whisper":
        expected_calls = len(cap.model.model.encoder.layers) + 2 * len(
            cap.model.model.decoder.layers
        )
    assert len(sdpa_indices) == expected_calls
    reshapes_output = Version(transformers.__version__) >= Version("5.23.0")
    # Before ViT joined the common attention interface, its own SDPA class
    # used permute(0, 2, 1, 3) where the common interface uses transpose(1, 2).
    old_vit = family == "vit" and hasattr(
        transformers.models.vit.modeling_vit, "ViTSdpaSelfAttention"
    )
    expected_consumer = "reshape" if reshapes_output else "permute" if old_vit else "transpose"
    for index in sdpa_indices:
        sdpa, following = rows[index : index + 2]
        sdpa_state = dict(state_items(sdpa))
        following_state = dict(state_items(following))
        assert any(cap.trace.ops[label] is sdpa for label in following_state["parents"])
        assert any(cap.trace.ops[label] is following for label in sdpa_state["children"])
        assert following.func_name == expected_consumer
        if reshapes_output:
            assert len(following_state["parents"]) == 1
            assert following.out.shape == sdpa.out.shape
            assert torch.equal(following.out, sdpa.out)
            transpose = rows[index + 2]
            assert transpose.func_name == "transpose"
            assert any(
                cap.trace.ops[label] is following
                for label in dict(state_items(transpose))["parents"]
            )

    native = getattr(cap.model(*cap.input_args, **cap.input_kwargs), PRIMARY_LEAF[family])
    captured = next(op.out for op in cap.trace.output_ops if op.out.shape == native.shape)
    assert torch.equal(captured, native)
    parameters = tuple(parameter for parameter in cap.model.parameters() if parameter.requires_grad)
    native_grads = torch.autograd.grad(native.sum(), parameters, allow_unused=True)
    captured_grads = torch.autograd.grad(
        captured.sum(), parameters, allow_unused=True, retain_graph=True
    )
    assert len(captured_grads) == len(native_grads)
    for actual, expected in zip(captured_grads, native_grads):
        if expected is None:
            assert actual is None
        else:
            assert actual is not None and torch.equal(actual, expected)
