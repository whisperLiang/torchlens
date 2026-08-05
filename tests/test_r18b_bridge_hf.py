"""Regression tests for r18b bridge/hf.py hardening.

Covers four A3 findings on ``torchlens/bridge/hf.py``:

* A3-06 -- autoroute predicates must decline implausible modality/chat values.
* A3-07 -- list image preprocessing must not run per-item transforms N+1 times.
* A3-13 -- tokenizer provenance must report honest ``verified``/``padding``.
* A3-05 -- the multimodal detection heuristic must be offline-safe (cache-only).
"""

from __future__ import annotations

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
