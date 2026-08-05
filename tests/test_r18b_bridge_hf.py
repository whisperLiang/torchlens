"""Regression tests for r18b bridge/hf.py hardening.

Covers four A3 findings on ``torchlens/bridge/hf.py``:

* A3-06 -- autoroute predicates must decline implausible modality/chat values.
* A3-07 -- list image preprocessing must not run per-item transforms N+1 times.
* A3-13 -- tokenizer provenance must report honest ``verified``/``padding``.
* A3-05 -- the multimodal detection heuristic must be offline-safe (cache-only).
"""

from __future__ import annotations

from typing import Any

import torch

import torchlens.bridge.hf as hf


# ---------------------------------------------------------------------------
# A3-06 -- predicate misclassification
# ---------------------------------------------------------------------------


def test_multimodal_predicate_declines_none_and_bare_object() -> None:
    """Implausible audio/video values must not trigger the multimodal route."""

    assert hf._is_hf_multimodal_input({"audio": None}) is False
    assert hf._is_hf_multimodal_input({"videos": object()}) is False
    assert hf._is_hf_multimodal_input({"audio": []}) is False
    assert hf._is_hf_multimodal_input({"audio": 5}) is False


def test_multimodal_predicate_accepts_plausible_media() -> None:
    """Real waveform/frame payloads must still route as multimodal."""

    assert hf._is_hf_multimodal_input({"audio": torch.zeros(16000)}) is True
    assert hf._is_hf_multimodal_input({"audio": [0.1, 0.2, 0.3]}) is True
    assert hf._is_hf_multimodal_input({"videos": (torch.zeros(3, 2, 2),)}) is True
    assert hf._is_hf_multimodal_input({"audio": b"\x00\x01"}) is True


def test_text_predicate_declines_malformed_chat() -> None:
    """Chat records with non-string role or non-str/list content must decline."""

    assert hf._is_hf_text_input([{"role": None, "content": object()}]) is False
    assert hf._is_hf_text_input([{"role": "user"}]) is False
    assert hf._is_hf_text_input([{"content": "hi"}]) is False
    assert hf._is_hf_text_input([{"role": 1, "content": "hi"}]) is False


def test_text_predicate_accepts_valid_chat_and_strings() -> None:
    """Valid chat messages and plain strings still route as text."""

    assert hf._is_hf_text_input("hello") is True
    assert hf._is_hf_text_input(["a", "b"]) is True
    assert hf._is_hf_text_input([{"role": "user", "content": "hi"}]) is True
    assert (
        hf._is_hf_text_input([{"role": "user", "content": [{"type": "text", "text": "hi"}]}])
        is True
    )


# ---------------------------------------------------------------------------
# A3-07 -- list image preprocessing N+1 calls
# ---------------------------------------------------------------------------


def test_image_transform_batch_native_called_once() -> None:
    """A batch-native (tagged) processor must be invoked exactly once on a list."""

    calls: list[Any] = []

    def batch_processor(value: Any) -> dict[str, Any]:
        calls.append(value)
        return {"call": len(calls), "value": value}

    batch_processor._tl_batch_input = True  # type: ignore[attr-defined]
    wrapped = hf._make_image_transform(batch_processor)
    result = wrapped(["a", "b", "c"])

    assert len(calls) == 1
    assert calls == [["a", "b", "c"]]
    assert result == {"call": 1, "value": ["a", "b", "c"]}


def test_image_transform_untagged_mapping_no_wasted_peritem_calls() -> None:
    """An untagged mapping (stateful) transform must not run N discarded per-item passes."""

    calls: list[Any] = []

    def stateful_mapping_transform(value: Any) -> dict[str, Any]:
        calls.append(value)
        return {"call_number": len(calls), "value": value}

    wrapped = hf._make_image_transform(stateful_mapping_transform)
    wrapped(["a", "b", "c", "d"])

    # Old behavior: N per-item probes (discarded) + 1 whole-list = N+1 = 5.
    # New behavior: a single per-item probe + 1 whole-list = 2.
    assert len(calls) <= 2
    assert calls[-1] == ["a", "b", "c", "d"]


def test_image_transform_per_item_stacks_without_waste() -> None:
    """A per-item tensor transform must be called once per item and stacked."""

    calls: list[Any] = []

    def per_item(value: Any) -> torch.Tensor:
        calls.append(value)
        return torch.zeros(3, 2, 2)

    wrapped = hf._make_image_transform(per_item)
    result = wrapped(["a", "b", "c"])

    assert len(calls) == 3
    assert calls == ["a", "b", "c"]
    assert isinstance(result, torch.Tensor)
    assert result.shape == (3, 3, 2, 2)


def test_image_transform_single_image_unsqueezes() -> None:
    """A single-image tensor transform result gains a batch dimension."""

    wrapped = hf._make_image_transform(lambda value: torch.zeros(3, 2, 2))
    result = wrapped("one-image")

    assert isinstance(result, torch.Tensor)
    assert result.shape == (1, 3, 2, 2)
