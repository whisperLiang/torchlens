"""Capture-OUTCOME corpus for detached torch references — the stage-2 deletion gate.

This module converts the detached-reference corpus from patch-MECHANICS
assertions ("the crawler rewrote slot X to wrapper Y") to capture-OUTCOME
assertions ("the op appears in the trace" / "the miss has exactly this
signature"). Mechanics tests keep guarding the crawler while it exists
(``test_patch_detached_references_coverage.py``); THIS module is the gate any
replacement (the stage-2 rescue net + belt) must keep green, because it pins
what users actually get.

Every row PASSES TODAY: rows in the "covered" sections pin that the op is
captured; rows in the "missed" sections pin the exact current failure
signature (silent mid-graph corruption or the loud output-attribution error).
When stage 2 lands, the missed rows are EXPECTED to start failing in the good
direction (op captured instead of missed) and must be flipped deliberately —
that is the point: no coverage change can happen silently.

Row inventory (safety-net verdict, tri-lab matrices):

- Covered today (converted from mechanics): module-level refs, class attrs
  and function defaults in torch-mentioning source, model instance holders
  (direct / list / dict / ``partial.func`` / ``partial.args`` /
  ``partial.keywords``).
- The 7 crawler-missed classes (silent today): closure cells, staticmethods,
  module-level partials, plain-object attrs, pre-bound tensor methods,
  torch-free-source module class attrs, C-held refs (``lru_cache`` proxy).
- The protocol-invisible class (``from_numpy`` / ``frombuffer`` /
  ``as_subclass``): these assert the BELT/wrapper path — no
  ``TorchFunctionMode`` can EVER see them (measured: zero callbacks), so the
  rescue net cannot cover them and the wrapper+patch path must stay green.
- De-moded composite interiors: a crawler-reachable stale ref inside a
  third-party ``handle_torch_function`` composite body — captured today,
  structurally invisible to any mode (the protocol pops the mode before the
  body runs).
- Worker-thread stale refs: declared unsupported/ceilinged (op logging is
  owner-thread-scoped by design, r43); the thread's op must NOT appear and
  the capture must not crash.
- The mid-graph silent-corruption case (opus matrix; previously uncovered by
  any test): a missed op between two captured ops leaves the consumer with
  ``parents == ()``, ``is_internal_source``, ``capture_verified is None``,
  and ZERO escape diagnostics on the shipped default config. The only signal
  is a generic provenance ``UserWarning``, pinned as part of the signature.
"""

from __future__ import annotations

import functools
import importlib
import sys
import threading
import types
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.backends.torch._tl import is_decorated_function
from torchlens.backends.torch.wrappers import (
    clear_patch_detached_references_cache,
    unwrap_torch,
    wrap_torch,
)


# ---------------------------------------------------------------------------
# Environment: pristine originals, holder construction, rewrap
# ---------------------------------------------------------------------------


class _CorpusEnv:
    """Pre-wrap raw originals plus registration of temp holder modules."""

    def __init__(self) -> None:
        self.relu = torch.relu
        self.sigmoid = torch.sigmoid
        self.tanh = torch.tanh
        self.cos = torch.cos
        self.from_numpy = torch.from_numpy
        self.frombuffer = torch.frombuffer
        self.as_subclass = torch.Tensor.as_subclass
        self.temp_modules: list[str] = []

    def register_module(self, module: types.ModuleType) -> types.ModuleType:
        sys.modules[module.__name__] = module
        self.temp_modules.append(module.__name__)
        return module


@pytest.fixture()
def corpus_env() -> Any:
    """Unwrap torch, expose raw originals, rewrap and clean up afterwards."""
    unwrap_torch()
    clear_patch_detached_references_cache()
    env = _CorpusEnv()
    assert not is_decorated_function(env.relu), "unwrap failed; refs are not pristine"
    try:
        yield env
    finally:
        for name in env.temp_modules:
            sys.modules.pop(name, None)
        clear_patch_detached_references_cache()
        wrap_torch()


def _trace(model: nn.Module, x: torch.Tensor | None = None) -> tl.Trace:
    wrap_torch()
    return tl.trace(model, torch.tensor([0.25, 0.5]) if x is None else x)


def _trace_with_provenance_warning(model: nn.Module) -> tl.Trace:
    """Trace a model whose missed op MUST trigger the provenance warning.

    The generic ``UserWarning`` ("tensor arguments with no graph/source
    provenance") is the ONLY signal today's default config emits for a
    mid-graph miss — no escape diagnostic, no verification downgrade. It is
    pinned here as part of the miss signature so it cannot silently vanish.
    """
    wrap_torch()
    with pytest.warns(UserWarning, match="no graph/source provenance"):
        return tl.trace(model, torch.tensor([0.25, 0.5]))


def _op_names(trace: tl.Trace) -> list[str]:
    return [op.func_name for op in trace.ops]


def _escape_count(trace: tl.Trace) -> int:
    return len(getattr(trace, "escape_diagnostics", []) or [])


def _assert_captured(trace: tl.Trace, *expected: str) -> None:
    """The stale-ref ops made it into the trace (the outcome that matters)."""
    names = _op_names(trace)
    for name in expected:
        assert name in names, f"{name!r} missing from capture: {names}"


def _assert_silent_midgraph_miss(trace: tl.Trace, missing: str, consumer: str) -> None:
    """Pin TODAY's silent-corruption signature for a mid-graph missed op.

    The missed op is simply absent; its consumer is orphaned from the
    dataflow (no parents, marked internal-source); the trace makes no
    verification claim and raises ZERO escape diagnostics on the shipped
    default config. Stage 2 must flip this row to a captured op (rescue) or
    a loud disclosure — flipping it back to silence must be impossible.
    """
    names = _op_names(trace)
    assert missing not in names, f"row premise broken: {missing!r} was captured"
    consumers = [op for op in trace.ops if op.func_name == consumer]
    assert consumers, f"consumer {consumer!r} missing entirely: {names}"
    consumer_op = consumers[0]
    assert consumer_op.parents == ()
    assert consumer_op.is_internal_source
    assert trace.capture_verified is None
    assert _escape_count(trace) == 0


# ---------------------------------------------------------------------------
# Covered today: crawler-reachable holders (converted from mechanics rows)
# ---------------------------------------------------------------------------


def test_module_class_and_default_refs_are_captured(corpus_env: _CorpusEnv) -> None:
    """Module-level, class-attr, and function-default stale refs are traced."""
    mod = types.ModuleType("_tl_outcome_covered_holders")
    corpus_env.register_module(mod)
    exec(
        """
from typing import Any
from torch import relu, sigmoid, tanh

module_ref = relu


class Holder:
    class_ref = sigmoid


def uses_default(x: Any, op: Any = tanh) -> Any:
    return op(x)
""",
        mod.__dict__,
    )

    class Model(nn.Module):
        def forward(self, v: torch.Tensor) -> torch.Tensor:
            return mod.uses_default(mod.Holder.class_ref(mod.module_ref(v)))

    trace = _trace(Model())
    _assert_captured(trace, "relu", "sigmoid", "tanh")


def test_model_instance_holders_are_captured(corpus_env: _CorpusEnv) -> None:
    """Direct / list / dict / partial(.func/.args/.keywords) model holders."""

    def apply_op(op: Callable[..., Any], v: torch.Tensor) -> torch.Tensor:
        return op(v)

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.direct = corpus_env.relu
            self.items = [corpus_env.sigmoid]
            self.mapping = {"op": corpus_env.tanh}
            self.partial_func = functools.partial(corpus_env.cos)
            self.partial_arg = functools.partial(apply_op, corpus_env.sigmoid)
            self.partial_kw = functools.partial(apply_op, op=corpus_env.tanh)

        def forward(self, v: torch.Tensor) -> torch.Tensor:
            y = self.mapping["op"](self.items[0](self.direct(v)))
            y = self.partial_func(y)
            y = self.partial_arg(v=y)
            return self.partial_kw(v=y)

    trace = _trace(Model())
    names = _op_names(trace)
    assert names.count("relu") == 1
    assert names.count("sigmoid") == 2  # list holder + partial.args holder
    assert names.count("tanh") == 2  # dict holder + partial.keywords holder
    assert names.count("cos") == 1


# ---------------------------------------------------------------------------
# The 7 crawler-missed classes: pin TODAY's silent miss (or loud crash)
# ---------------------------------------------------------------------------


def test_closure_cell_ref_is_silently_missed_midgraph(corpus_env: _CorpusEnv) -> None:
    raw_cos = corpus_env.cos

    def make_closure() -> Callable[[torch.Tensor], torch.Tensor]:
        def invoke(v: torch.Tensor) -> torch.Tensor:
            return raw_cos(v)

        return invoke

    closure = make_closure()

    class Model(nn.Module):
        def forward(self, v: torch.Tensor) -> torch.Tensor:
            return torch.relu(closure(torch.sigmoid(v)))

    _assert_silent_midgraph_miss(_trace_with_provenance_warning(Model()), missing="cos", consumer="relu")


def test_staticmethod_ref_is_silently_missed_midgraph(corpus_env: _CorpusEnv) -> None:
    class Holder:
        static = staticmethod(corpus_env.cos)

    class Model(nn.Module):
        def forward(self, v: torch.Tensor) -> torch.Tensor:
            return torch.relu(Holder.static(torch.sigmoid(v)))

    _assert_silent_midgraph_miss(_trace_with_provenance_warning(Model()), missing="cos", consumer="relu")


def test_module_level_partial_is_silently_missed_midgraph(corpus_env: _CorpusEnv) -> None:
    mod = types.ModuleType("_tl_outcome_module_partial")
    corpus_env.register_module(mod)
    mod.partial = functools.partial(corpus_env.cos)

    class Model(nn.Module):
        def forward(self, v: torch.Tensor) -> torch.Tensor:
            return torch.relu(mod.partial(torch.sigmoid(v)))

    _assert_silent_midgraph_miss(_trace_with_provenance_warning(Model()), missing="cos", consumer="relu")


def test_plain_object_attr_is_silently_missed_midgraph(corpus_env: _CorpusEnv) -> None:
    class Plain:
        pass

    holder = Plain()
    holder.op = corpus_env.cos

    class Model(nn.Module):
        def forward(self, v: torch.Tensor) -> torch.Tensor:
            return torch.relu(holder.op(torch.sigmoid(v)))

    _assert_silent_midgraph_miss(_trace_with_provenance_warning(Model()), missing="cos", consumer="relu")


def test_prebound_tensor_method_is_silently_missed_midgraph(corpus_env: _CorpusEnv) -> None:
    owner = torch.tensor([1.0, 2.0])
    bound_add = owner.add  # bound BEFORE wrap; the descriptor inside is raw

    class Model(nn.Module):
        def forward(self, v: torch.Tensor) -> torch.Tensor:
            return torch.relu(bound_add(torch.sigmoid(v)))

    _assert_silent_midgraph_miss(_trace_with_provenance_warning(Model()), missing="add", consumer="relu")


def test_c_held_ref_is_silently_missed_midgraph(corpus_env: _CorpusEnv) -> None:
    """``lru_cache`` C-level storage as the proxy for a compiled extension."""
    raw_cos = corpus_env.cos

    @functools.lru_cache(maxsize=1)
    def held() -> Callable[..., Any]:
        return raw_cos

    held()  # populate the C-held cache slot pre-wrap

    class Model(nn.Module):
        def forward(self, v: torch.Tensor) -> torch.Tensor:
            return torch.relu(held()(torch.sigmoid(v)))

    _assert_silent_midgraph_miss(_trace_with_provenance_warning(Model()), missing="cos", consumer="relu")


def _import_temp_module(tmp_path: Path, mod_name: str, source: str) -> types.ModuleType:
    module_path = tmp_path / f"{mod_name}.py"
    module_path.write_text(source, encoding="utf-8")
    sys.modules.pop(mod_name, None)
    sys.path.insert(0, str(tmp_path))
    importlib.invalidate_caches()
    try:
        return importlib.import_module(mod_name)
    finally:
        sys.path.remove(str(tmp_path))


def test_torch_free_source_class_attr_is_silently_missed_midgraph(
    corpus_env: _CorpusEnv, tmp_path: Path
) -> None:
    """Torch-free FILE-BACKED source: the source gate skips the deep scan."""
    mod = _import_temp_module(tmp_path, "_tl_outcome_torch_free_mid", "class Holder:\n    pass\n")
    corpus_env.temp_modules.append(mod.__name__)
    mod.Holder.op = corpus_env.cos

    class Model(nn.Module):
        def forward(self, v: torch.Tensor) -> torch.Tensor:
            return torch.relu(mod.Holder.op(torch.sigmoid(v)))

    _assert_silent_midgraph_miss(_trace_with_provenance_warning(Model()), missing="cos", consumer="relu")


def test_torch_free_source_output_position_crashes_loudly(
    corpus_env: _CorpusEnv, tmp_path: Path
) -> None:
    """Same missed class in OUTPUT position: today's loud attribution error."""
    mod = _import_temp_module(tmp_path, "_tl_outcome_torch_free_out", "class Holder:\n    pass\n")
    corpus_env.temp_modules.append(mod.__name__)
    mod.Holder.op = corpus_env.cos

    class Model(nn.Module):
        def forward(self, v: torch.Tensor) -> torch.Tensor:
            return mod.Holder.op(torch.sigmoid(v))

    wrap_torch()
    with pytest.raises(RuntimeError, match="could not attribute a model output tensor"):
        tl.trace(Model(), torch.tensor([0.25, 0.5]))


# ---------------------------------------------------------------------------
# Protocol-invisible class: the BELT rows (no mode can EVER see these)
# ---------------------------------------------------------------------------


def test_protocol_invisible_from_numpy_is_captured(corpus_env: _CorpusEnv) -> None:
    mod = types.ModuleType("_tl_outcome_proto_from_numpy")
    corpus_env.register_module(mod)
    mod.op = corpus_env.from_numpy
    arr = np.array([0.25, 0.5], dtype=np.float32)

    class Model(nn.Module):
        def forward(self, v: torch.Tensor) -> torch.Tensor:
            return mod.op(arr) + v

    _assert_captured(_trace(Model()), "from_numpy", "__add__")


def test_protocol_invisible_frombuffer_is_captured(corpus_env: _CorpusEnv) -> None:
    mod = types.ModuleType("_tl_outcome_proto_frombuffer")
    corpus_env.register_module(mod)
    mod.op = corpus_env.frombuffer
    buf = bytearray(np.array([0.25, 0.5], dtype=np.float32).tobytes())

    class Model(nn.Module):
        def forward(self, v: torch.Tensor) -> torch.Tensor:
            return mod.op(buf, dtype=torch.float32) + v

    _assert_captured(_trace(Model()), "frombuffer", "__add__")


def test_protocol_invisible_as_subclass_is_captured(corpus_env: _CorpusEnv) -> None:
    mod = types.ModuleType("_tl_outcome_proto_as_subclass")
    corpus_env.register_module(mod)
    mod.op = corpus_env.as_subclass

    class SubTensor(torch.Tensor):
        pass

    class Model(nn.Module):
        def forward(self, v: torch.Tensor) -> torch.Tensor:
            return mod.op(v, SubTensor)

    _assert_captured(_trace(Model()), "as_subclass")


# ---------------------------------------------------------------------------
# De-moded composite interior: captured today, invisible to any mode
# ---------------------------------------------------------------------------


def test_demoded_composite_interior_ref_is_captured(corpus_env: _CorpusEnv) -> None:
    """A crawler-reachable stale ref inside a protocol composite body.

    The composite has torch's own ``has_torch_function`` /
    ``handle_torch_function`` shape, so under ANY TorchFunctionMode its body
    runs with the mode popped — a future net can never see ``interior``.
    Today the crawler patches the module global and the op IS captured; this
    row is the belt's coverage rationale for reachable composite interiors.
    """
    mod = types.ModuleType("_tl_outcome_composite_interior")
    corpus_env.register_module(mod)
    mod.__dict__.update(
        {
            "has_torch_function": torch.overrides.has_torch_function,
            "handle_torch_function": torch.overrides.handle_torch_function,
            "interior": corpus_env.cos,
        }
    )
    exec(
        """
def protocol_composite(x):
    if has_torch_function((x,)):
        return handle_torch_function(protocol_composite, (x,), x)
    return interior(x)
""",
        mod.__dict__,
    )

    class Model(nn.Module):
        def forward(self, v: torch.Tensor) -> torch.Tensor:
            return torch.relu(mod.protocol_composite(v))

    _assert_captured(_trace(Model()), "cos", "relu")


# ---------------------------------------------------------------------------
# Worker-thread refs: declared unsupported/ceilinged (owner-thread scoping)
# ---------------------------------------------------------------------------


def test_worker_thread_ref_is_not_logged_and_capture_survives(corpus_env: _CorpusEnv) -> None:
    """Op logging is owner-thread-scoped by design (r43): the worker thread's
    op must NOT enter the trace (no false attribution), and the capture must
    complete. Modes are thread-local too, so stage 2 inherits this ceiling —
    the row pins the declared behavior, not a gap to fix."""
    raw_cos = corpus_env.cos

    class Model(nn.Module):
        def forward(self, v: torch.Tensor) -> torch.Tensor:
            result: dict[str, torch.Tensor] = {}

            def work() -> None:
                result["r"] = raw_cos(v.detach())

            worker = threading.Thread(target=work)
            worker.start()
            worker.join()
            return torch.relu(v) + result["r"].sum()

    # The thread-made tensor re-enters the owner thread with no provenance,
    # so the same disclosure warning fires as for any mid-graph miss.
    trace = _trace_with_provenance_warning(Model())
    names = _op_names(trace)
    assert "cos" not in names
    assert "relu" in names and "sum" in names and "__add__" in names


# ---------------------------------------------------------------------------
# The mid-graph silent-corruption row (opus matrix: previously uncovered)
# ---------------------------------------------------------------------------


def test_midgraph_silent_corruption_full_signature(corpus_env: _CorpusEnv) -> None:
    """The complete TODAY-signature of a silent mid-graph escape, in one row.

    ``sigmoid -> [missed cos] -> relu``: the trace contains sigmoid and relu,
    no cos, relu is a parentless internal-source op whose only internal-source
    ancestor is itself, the trace claims nothing (``capture_verified is
    None``), and the shipped default config emits ZERO escape diagnostics —
    i.e. the user gets a plausible-looking but corrupted graph with no signal
    anything is wrong. This is the row stage 2 exists to flip."""
    raw_cos = corpus_env.cos

    def make_closure() -> Callable[[torch.Tensor], torch.Tensor]:
        def invoke(v: torch.Tensor) -> torch.Tensor:
            return raw_cos(v)

        return invoke

    closure = make_closure()

    class Model(nn.Module):
        def forward(self, v: torch.Tensor) -> torch.Tensor:
            return torch.relu(closure(torch.sigmoid(v)))

    trace = _trace_with_provenance_warning(Model())
    assert _op_names(trace) == ["none", "sigmoid", "relu", "none"]
    relu_op = [op for op in trace.ops if op.func_name == "relu"][0]
    assert relu_op.parents == ()
    assert relu_op.is_internal_source
    assert relu_op.internal_source_ancestors == frozenset({relu_op.label.split(":")[0]})
    assert trace.capture_verified is None
    assert _escape_count(trace) == 0
