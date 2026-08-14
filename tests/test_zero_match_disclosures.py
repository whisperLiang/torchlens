"""Zero-match disclosure for the halt selector slot and preview backends.

``halt=`` was the ONE selector slot in the public surface outside the
zero-match disclosure family: a typo'd op/module name silently ran the FULL
forward (spending the memory/latency the halt was meant to bound) and handed
back the model's real outputs where the caller expected a frontier, with
``outcome.status == complete`` as the only implicit signal. The tf/mlx
intervene zero-match legs are the preview half of the torch-side 7969aca8
fix (fired accounting written but never read on tf; no counter at all on
mlx).
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import torchlens as tl

pytestmark = pytest.mark.smoke


def _model() -> nn.Module:
    """Tiny two-op model."""

    return nn.Sequential(nn.Linear(3, 3), nn.ReLU())


def test_halt_zero_match_warns_on_completed_capture() -> None:
    """A halt selector that never fires discloses instead of staying silent."""

    with pytest.warns(UserWarning, match="halt selector .* matched zero sites"):
        log = tl.trace(_model(), torch.randn(2, 3), halt=tl.func("nosuchopzzz"))
    assert log.outcome.status.name == "COMPLETE"


def test_halt_that_fires_does_not_warn() -> None:
    """A genuinely-halting selector emits no zero-match warning."""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        log = tl.trace(_model(), torch.randn(2, 3), halt=tl.func("linear"))
    assert log.outcome.status.name == "HALTED"
    assert not any("matched zero sites" in str(item.message) for item in caught)


def test_value_dependent_halt_callable_is_not_judged() -> None:
    """A non-selector halt callable legitimately never firing stays silent."""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tl.trace(_model(), torch.randn(2, 3), halt=lambda ctx: False)
    assert not any(
        "halt selector" in str(item.message) and "matched zero" in str(item.message)
        for item in caught
    )


def test_tf_zero_fire_site_warns_in_reachability_audit() -> None:
    """A planned tf site with zero fires warns at the post-forward audit.

    ``fired_site_labels`` was written by the fire path and read by NOTHING;
    a selector matching zero captured ops left the audit's ``unreachable``
    list empty and returned cleanly, and module sites got no audit at all.
    The audit itself is import-safe without TensorFlow, so this leg runs on
    torch-only hosts.
    """

    from torchlens.backends.tf.interventions import (
        TFInterventionPlan,
        TFInterventionSite,
        audit_tf_site_reachability,
    )

    fired_site = TFInterventionSite(
        plan_id="site_0",
        predicate=lambda ctx: None,
        selector=lambda ctx: False,
        decision=None,
        hook=lambda tensor: tensor,
        level="op",
    )
    silent_module_site = TFInterventionSite(
        plan_id="site_1",
        predicate=lambda ctx: None,
        selector=lambda ctx: False,
        decision=None,
        hook=lambda tensor: tensor,
        level="module",
    )
    plan = TFInterventionPlan(
        sites=(fired_site, silent_module_site),
        op_sites=(fired_site,),
        module_sites=(silent_module_site,),
    )
    plan.fired_site_labels.append(("site_0", "relu_1_1"))
    session = SimpleNamespace(events=SimpleNamespace(op_events=[]))

    with pytest.warns(UserWarning, match="site_1 fired at zero sites"):
        audit_tf_site_reachability(plan, session)

    # Every planned site fired: the audit stays silent.
    plan.fired_site_labels.append(("site_1", "module_1"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        audit_tf_site_reachability(plan, session)
    assert not any("fired at zero sites" in str(item.message) for item in caught)
