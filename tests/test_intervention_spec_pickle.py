"""Pickle / ``tl.save`` parity for intervention-spec-carrying traces.

Pre-fix, plain ``pickle`` of a Trace captured with ``intervene=`` failed
(``zero_ablate.<locals>.factory`` and ``when.<locals>._predicate`` are local
closures) while ``tl.save`` of the same trace succeeded -- a serialization
split on one object. Two aligned fixes:

* ``HelperSpec`` pickle hooks drop the builtin factory closure and rebuild it
  at restore through the SAME builtin registry ``tl.load`` uses
  (``rebuild_builtin_helper``); ``opaque_audit`` factories drop to the
  canonical factory-less audit-only form.
* ``Trace.__getstate__`` serializes the capture-session predicate carriers
  (``_stop_directive``, ``_capture_config``, ``_predicate_save_options`` --
  all ``FieldPolicy.DROP``) to their loaded-artifact form (absent / None),
  which every post-capture consumer already tolerates.
"""

import copy
import pickle
from collections.abc import Iterator

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention.types import HelperSpec


class _Net(nn.Module):
    """Linear + relu net."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the layer and relu."""

        return torch.relu(self.fc(x))


@pytest.fixture(scope="module")
def intervened_trace() -> Iterator[tl.Trace]:
    """Return a trace captured with a builtin-helper intervention."""

    torch.manual_seed(0)
    trace = tl.trace(
        _Net(),
        torch.randn(4, 8),
        save=tl.func("relu"),
        intervene=tl.when(tl.func("relu"), tl.zero_ablate()),
    )
    try:
        yield trace
    finally:
        trace.cleanup()


@pytest.mark.smoke
def test_builtin_helper_spec_pickle_rebuilds_factory() -> None:
    """A builtin ``HelperSpec`` pickles; the restored factory works."""

    spec = tl.zero_ablate()
    restored = pickle.loads(pickle.dumps(spec))
    assert isinstance(restored, HelperSpec)
    assert restored.helper_name == spec.helper_name
    assert restored.factory is not None
    hook = restored()
    out = hook(torch.ones(3), hook=None)
    assert torch.equal(out, torch.zeros(3))


@pytest.mark.smoke
def test_builtin_helper_spec_pickle_preserves_args() -> None:
    """Constructor arguments survive the rebuild (identity source)."""

    spec = tl.scale(0.5)
    restored = pickle.loads(pickle.dumps(spec))
    assert restored.args == spec.args
    hook = restored()
    out = hook(torch.ones(3), hook=None)
    assert torch.allclose(out, torch.full((3,), 0.5))


@pytest.mark.smoke
def test_helper_spec_deepcopy_keeps_factory_identity() -> None:
    """``copy.deepcopy`` keeps its pre-pickle-hook factory-sharing semantics."""

    spec = tl.zero_ablate()
    duplicate = copy.deepcopy(spec)
    assert duplicate.factory is spec.factory


@pytest.mark.smoke
def test_intervened_trace_pickles_like_it_saves(intervened_trace: tl.Trace, tmp_path) -> None:
    """The RED-capable parity case: pickle succeeds where tl.save succeeds."""

    blob = pickle.dumps(intervened_trace)
    restored = pickle.loads(blob)
    assert len(restored.layer_list) == len(intervened_trace.layer_list)
    # The saved (intervened) activation payload survives the round-trip.
    assert torch.equal(restored["relu_1_2"].ops[0].out, intervened_trace["relu_1_2"].ops[0].out)
    # tl.save of the very same object keeps working (the parity claim).
    path = str(tmp_path / "iv.tlspec")
    tl.save(intervened_trace, path)
    assert tl.load(path)["relu_1_2"] is not None


@pytest.mark.smoke
def test_intervened_fork_pickles(intervened_trace: tl.Trace) -> None:
    """A fork carrying the deep-copied intervention spec also pickles."""

    fork = intervened_trace.fork()
    restored = pickle.loads(pickle.dumps(fork))
    assert len(restored.layer_list) == len(fork.layer_list)


@pytest.mark.smoke
def test_pickled_trace_matches_loaded_predicate_carrier_form(
    intervened_trace: tl.Trace,
) -> None:
    """Pickle serializes the predicate carriers to the loaded-artifact form."""

    restored = pickle.loads(pickle.dumps(intervened_trace))
    assert "_stop_directive" not in restored.__dict__
    assert "_capture_config" not in restored.__dict__
    assert restored.__dict__.get("_predicate_save_options") is None
