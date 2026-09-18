"""Offline integration coverage for the supported Transformers 4.x and 5.x majors.

These are real Transformers models, not lookalikes selected by a recipe class name.
The same tests run under either supported major without version-dependent skips.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.validation.invariants import check_metadata_invariants

pytestmark = [pytest.mark.heavy, pytest.mark.optional]


def _tiny_model(family: str, implementation: str) -> tuple[nn.Module, str, str | None]:
    """Build a real, randomly initialized model without consulting the Hub."""

    transformers = pytest.importorskip("transformers")
    if family == "bert":
        config = transformers.BertConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=16,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )
        config._attn_implementation = implementation
        model = transformers.BertModel(config, add_pooling_layer=False)
        return model.eval(), "encoder.layer.0.attention.self", None
    if family == "distilbert":
        config = transformers.DistilBertConfig(
            vocab_size=32,
            dim=16,
            hidden_dim=32,
            n_layers=1,
            n_heads=2,
            max_position_embeddings=16,
            dropout=0.0,
            attention_dropout=0.0,
        )
        config._attn_implementation = implementation
        model = transformers.DistilBertModel(config)
        return model.eval(), "transformer.layer.0.attention", "transformer.layer.0.ffn"
    assert family == "gpt2"
    config = transformers.GPT2Config(
        vocab_size=32,
        n_embd=16,
        n_inner=32,
        n_layer=1,
        n_head=2,
        n_positions=16,
        n_ctx=16,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=False,
    )
    config._attn_implementation = implementation
    model = transformers.GPT2Model(config)
    return model.eval(), "h.0.attn", "h.0.mlp"


def _native_reference(
    model: nn.Module,
    inputs: dict[str, torch.Tensor],
    attention_path: str,
    ffn_path: str | None,
    family: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Collect native outputs and projection values before TorchLens capture."""

    paths = {"attention_output": attention_path}
    if family == "gpt2":
        paths["qkv"] = f"{attention_path}.c_attn"
    else:
        children = ("query", "key", "value") if family == "bert" else ("q_lin", "k_lin", "v_lin")
        paths.update(
            (name, f"{attention_path}.{child}")
            for name, child in zip(("q", "k", "v"), children, strict=True)
        )
    if ffn_path is not None:
        up, down = ("c_fc", "c_proj") if family == "gpt2" else ("lin1", "lin2")
        paths.update(up_out=f"{ffn_path}.{up}", down_out=f"{ffn_path}.{down}", output=ffn_path)
    references: dict[str, torch.Tensor] = {}

    def hook_for(name: str) -> Any:
        """Build a read-only hook for one named native tensor output."""

        def save_output(module: nn.Module, args: tuple[Any, ...], output: Any) -> None:
            """Copy the tensor, leaving the model's output unchanged."""

            tensor = output[0] if isinstance(output, tuple) else output
            references[name] = tensor.detach().clone()

        return save_output

    handles = [
        model.get_submodule(path).register_forward_hook(hook_for(name))
        for name, path in paths.items()
    ]
    try:
        with torch.no_grad():
            output = model(**inputs).last_hidden_state.detach().clone()
    finally:
        for handle in handles:
            handle.remove()
    if family == "gpt2":
        q, k, v = references.pop("qkv").chunk(3, dim=-1)
        references.update(q=q, k=k, v=v)
    return output, references


@pytest.mark.parametrize("family", ["bert", "distilbert", "gpt2"])
@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
def test_real_transformers_capture_and_facets(family: str, implementation: str) -> None:
    """Real eager/SDPA forwards preserve values, graph metadata, and semantic facets."""

    model, attention_path, ffn_path = _tiny_model(family, implementation)
    assert model.config._attn_implementation == implementation
    inputs = {
        "input_ids": torch.tensor([[2, 3, 4, 5, 6], [7, 8, 9, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]),
    }
    expected, references = _native_reference(model, inputs, attention_path, ffn_path, family)
    trace = tl.trace(
        model,
        (),
        input_kwargs=inputs,
        capture=tl.options.CaptureOptions(inference_only=True),
    )
    assert len(trace.output_ops) == 1
    torch.testing.assert_close(trace.output_ops[0].out, expected, rtol=1e-5, atol=1e-6)
    assert check_metadata_invariants(trace) is True
    sdpa_ops = [op for op in trace.ops if "scaled_dot_product_attention" in op.func_name]
    assert bool(sdpa_ops) is (implementation == "sdpa")

    attention = trace.modules[attention_path].facets
    assert attention.n_heads == 2
    assert attention.d_head == 8
    for name in ("q", "k", "v"):
        expected_heads = references[name].reshape(2, 5, 2, 8)
        torch.testing.assert_close(attention[name].value, expected_heads)
        torch.testing.assert_close(attention.head(1)[name].value, expected_heads[:, :, 1])
    torch.testing.assert_close(attention.attn_out.value, references["attention_output"])
    if ffn_path is not None:
        ffn = trace.modules[ffn_path].facets
        for name in ("up_out", "down_out", "output"):
            torch.testing.assert_close(ffn[name].value, references[name])


@pytest.mark.parametrize("route", ["explicit_tokenizer", "autoroute"])
def test_real_fast_tokenizer_text_routes_stay_offline(
    route: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Local fast-tokenizer text routes preserve real masked-LM logits and decoding."""

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    transformers = pytest.importorskip("transformers")
    # Tokenizers is a mandatory Transformers dependency, not an independent extra.
    import tokenizers

    vocabulary = {"[UNK]": 0, "[PAD]": 1, "hello": 2, "world": 3, "tiny": 4}
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        model_max_length=16,
        model_input_names=["input_ids", "attention_mask"],
    )
    tokenizer_dir = tmp_path / "local_tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)
    config = transformers.DistilBertConfig(
        vocab_size=len(vocabulary),
        dim=8,
        hidden_dim=16,
        n_layers=1,
        n_heads=2,
        max_position_embeddings=16,
        dropout=0.0,
        attention_dropout=0.0,
        pad_token_id=vocabulary["[PAD]"],
    )
    config._attn_implementation = "eager"
    model = transformers.DistilBertForMaskedLM(config).eval()
    model.config.name_or_path = str(tokenizer_dir)
    texts = ["hello world", "tiny"]
    encoded = tokenizer(texts, padding=True, return_tensors="pt")
    with torch.no_grad():
        expected = model(**encoded).logits.detach().clone()
    options = {"capture": tl.options.CaptureOptions(inference_only=True), "save_raw_input": True}
    if route == "explicit_tokenizer":
        trace = tl.bridge.hf.trace_text(model, texts, tokenizer=tokenizer, **options)
    else:
        trace = tl.trace(model, texts, **options)
    assert len(trace.output_ops) == 1
    torch.testing.assert_close(trace.output_ops[0].out, expected, rtol=1e-5, atol=1e-6)
    assert check_metadata_invariants(trace) is True
    assert trace.raw_input == texts
    assert trace.input_preprocessor.source == "hf_auto_tokenizer"
    assert trace.input_preprocessor.verified is (route == "autoroute")
    assert trace.input_preprocessor.config["padding"] is True
    assert trace.output_postprocessor.style == "hf_text"
    expected_ids = expected.argmax(dim=-1).tolist()
    assert trace.decoded_output == [
        {
            "batch_item": index,
            "rank": 1,
            "text": tokenizer.decode(ids),
            "token_ids": ids,
        }
        for index, ids in enumerate(expected_ids)
    ]
    assert not hasattr(model, "_torchlens_output_tokenizer")
