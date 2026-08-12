"""M2: exact semantic-state key-set contract for all 11 record classes.

``state_items`` is the single enumeration every persistence path (save,
pickle, scrub, fork) walks. This contract freezes, per record class and
lifecycle stage, the EXACT key set it yields on the pre-columnar baseline.
When a class moves to columnar-facade storage, its ``__tl_state_items__``
must reproduce THIS key set — a missing key here is the silent-empty
save/load/pickle/fork hole (docs/reference/trace_core_design.md, CRITICAL
state-adapter risk), caught before any artifact test.
"""

from __future__ import annotations

import difflib
import json
import os
import pickle
from pathlib import Path
from typing import Any

import pytest
import torch

import torchlens as tl
from torchlens.data_classes._state_adapter import state_items
from torchlens.data_classes.op import _OP_SLOT_NAMES
from torchlens.data_classes.op import Op

from godobject_oracle.test_aliases import _SEED, _AliasCNN, _RecurrentCell

_GOLDEN_PATH = Path(__file__).resolve().parent / "godobject_oracle" / "goldens" / "state_keysets.json"
_UPDATE_ENV = "TORCHLENS_UPDATE_SURFACE_ORACLE"


def _capture_stage_records() -> dict[str, dict[str, list[str]]]:
    """Build the stage -> class -> sorted state-key-set mapping."""

    torch.manual_seed(_SEED)
    trace = tl.trace(
        _AliasCNN(), torch.linspace(-1.0, 1.0, 16).reshape(1, 1, 4, 4)
    )
    torch.manual_seed(_SEED)
    recurrent = tl.trace(
        _RecurrentCell(), torch.linspace(-1.0, 1.0, 4).reshape(1, 4)
    )
    stages: dict[str, Any] = {
        "live": trace,
        "pickle": pickle.loads(pickle.dumps(trace)),
        "fork": trace.fork(),
        "recurrent": recurrent,
    }

    result: dict[str, dict[str, list[str]]] = {}
    for stage_name, staged in stages.items():
        per_class: dict[str, list[str]] = {}
        records: dict[str, Any] = {"trace": staged}
        records["op"] = list(staged.ops.values())[0]
        for class_key, accessor in (
            ("layer", staged.layers),
            ("module", staged.modules),
            ("module_call", staged.module_calls),
            ("param", staged.params),
        ):
            keys = accessor.keys()
            if keys:
                records[class_key] = accessor[keys[0]]
            else:
                # Absence is contract too (plain pickle drops `modules` at
                # the pre-columnar baseline); freeze it as data.
                per_class[class_key] = ["<no records at this stage>"]
        for class_key, record in records.items():
            per_class[class_key] = sorted(
                {name for name, _ in state_items(record)}
            )
        result[stage_name] = per_class
    return result


@pytest.mark.smoke
def test_state_keysets_match_golden() -> None:
    """state_items key sets are frozen per class and lifecycle stage."""

    actual = json.dumps(_capture_stage_records(), indent=1, sort_keys=True)
    from _oracle_env import resolve_env_golden

    golden_path, record_on_missing = resolve_env_golden(
        _GOLDEN_PATH.parent, _GOLDEN_PATH.name
    )
    if os.environ.get(_UPDATE_ENV) == "1":
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(actual + "\n")
        pytest.skip("updated state-keyset golden")
    if record_on_missing and not golden_path.exists():
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(actual + "\n")
        pytest.skip(
            f"recorded first-run state-keyset golden for this environment: {golden_path}"
        )
    assert golden_path.exists(), (
        f"missing state-keyset golden; generate with {_UPDATE_ENV}=1"
    )
    expected = golden_path.read_text().rstrip("\n")
    if actual != expected:
        diff = "\n".join(
            list(
                difflib.unified_diff(
                    expected.splitlines(),
                    actual.splitlines(),
                    fromfile="golden",
                    tofile="actual",
                    lineterm="",
                )
            )[:80]
        )
        raise AssertionError(f"state key sets diverged:\n{diff}")


@pytest.mark.smoke
def test_declared_fields_dominate_op_state() -> None:
    """Every state key on a live Op is a declared FIELD_POLICY field.

    An Op slot outside the declared schema that carries live state would be
    invisible to the schema-driven columnar adapter; this fails first.
    """

    torch.manual_seed(_SEED)
    trace = tl.trace(
        _AliasCNN(), torch.linspace(-1.0, 1.0, 16).reshape(1, 1, 4, 4)
    )
    declared = set(Op.FIELD_POLICY)
    for op in trace.ops.values():
        undeclared = {name for name, _ in state_items(op)} - declared
        assert not undeclared, (
            f"live Op carries undeclared state: {sorted(undeclared)}"
        )


@pytest.mark.smoke
def test_op_slot_universe_is_declared() -> None:
    """Every Op slot is covered by FIELD_POLICY (no shadow storage)."""

    undeclared_slots = set(_OP_SLOT_NAMES) - set(Op.FIELD_POLICY)
    assert not undeclared_slots, sorted(undeclared_slots)
