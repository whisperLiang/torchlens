"""Round-5 RNG seal closures (grind-r5 b8 R57).

* ``Generator.spawn()`` advanced no digested state (it mutates only
  ``seed_seq._n_children_spawned``), so a spawn+child-draw on a digest-rooted
  model generator escaped with ``channels=[] / uncertain=False`` -- a clean
  false-VERIFIED. Spawn state is now folded into every Generator/BitGenerator
  digest, and bare ``SeedSequence`` holders are digested too.
* A held pre-window alias of an implicit-now converter called with an
  explicit literal ``None`` (``localtime(None)``) read the clock but escaped
  the value-blind positional-count check unmarked.
* ``_skip_retired_hooks`` followed the restored SLOT's predecessor attr even
  when the dead link was the owner's OTHER hook, restoring the wrong chain.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch
from torch import nn

from torchlens.utils.rng import _skip_retired_hooks, host_nondeterminism_monitor

_HELD_LOCALTIME = time.localtime  # pre-window held alias (module import time)


class _SpawningModel(nn.Module):
    """Model holding a digest-rooted generator that spawns mid-forward."""

    def __init__(self) -> None:
        super().__init__()
        self.gen = np.random.default_rng(1234)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


@pytest.mark.smoke
def test_generator_spawn_changes_state_digest() -> None:
    """The digest must witness spawn(): it is verdict-steering hidden state."""

    generator = np.random.default_rng(5)
    before = host_nondeterminism_monitor._digest_rng_instance(generator)
    generator.spawn(1)
    after = host_nondeterminism_monitor._digest_rng_instance(generator)
    assert before != after, "Generator.spawn() left the state digest unchanged"


@pytest.mark.smoke
def test_bit_generator_spawn_changes_state_digest() -> None:
    """Same seal for a bare model-held BitGenerator."""

    bit_generator = np.random.PCG64(7)
    before = host_nondeterminism_monitor._digest_rng_instance(bit_generator)
    bit_generator.spawn(1)
    after = host_nondeterminism_monitor._digest_rng_instance(bit_generator)
    assert before != after, "BitGenerator.spawn() left the state digest unchanged"


@pytest.mark.smoke
def test_seed_sequence_holder_is_digestable_and_spawn_witnessed() -> None:
    """A bare model-held SeedSequence is a spawnable entropy root."""

    seed_seq = np.random.SeedSequence(42)
    before = host_nondeterminism_monitor._digest_rng_instance(seed_seq)
    seed_seq.spawn(1)
    after = host_nondeterminism_monitor._digest_rng_instance(seed_seq)
    assert before != after, "SeedSequence.spawn() left the state digest unchanged"


@pytest.mark.smoke
def test_model_held_generator_spawn_draw_is_witnessed_in_window() -> None:
    """The r5 probe scenario: spawn a child from a model-held generator and
    draw from it inside the window -- the window must NOT settle clean."""

    model = _SpawningModel()
    with host_nondeterminism_monitor(model) as result:
        child = model.gen.spawn(1)[0]
        child.standard_normal()
    assert result.channels or result.uncertain, (
        "spawn+child-draw on a digest-rooted model generator settled "
        "channels=[] / uncertain=False (false-VERIFIED escape)"
    )


@pytest.mark.smoke
def test_held_alias_localtime_none_is_marked() -> None:
    """``held_localtime(None)`` reads the clock exactly like ``held_localtime()``."""

    with host_nondeterminism_monitor(nn.Identity()) as result:
        _HELD_LOCALTIME(None)
    assert any("localtime" in channel for channel in result.channels), (
        f"held localtime(None) escaped unmarked: {sorted(result.channels)!r}"
    )


@pytest.mark.smoke
def test_held_alias_localtime_literal_timestamp_stays_a_transform() -> None:
    """A provably non-None literal keeps the pure-transform classification."""

    with host_nondeterminism_monitor(nn.Identity()) as result:
        _HELD_LOCALTIME(1700000000.0)
    assert not any("localtime" in channel for channel in result.channels)
    assert not result.uncertain


@pytest.mark.smoke
def test_held_alias_localtime_computed_argument_flags_uncertainty() -> None:
    """A computed argument is runtime-dependent: neither a clock-draw claim
    nor a clean pass is provable, so the window flags uncertainty."""

    timestamp = float(len("x")) * 1700000000.0
    with host_nondeterminism_monitor(nn.Identity()) as result:
        _HELD_LOCALTIME(timestamp)
    assert result.uncertain, (
        "a computed explicit-time argument settled certain; the value could "
        "have been None at runtime"
    )


@pytest.mark.smoke
def test_skip_retired_hooks_follows_the_dead_links_own_chain() -> None:
    """A dead THREADING hook found while restoring the sys slot must resolve
    through the dead owner's threading predecessor, not the sys one."""

    class _DeadOwner:
        pass

    def dead_threading_hook(frame, event, arg):
        return None

    def sentinel_threading_predecessor(frame, event, arg):
        return None

    def sentinel_sys_predecessor(frame, event, arg):
        return None

    owner = _DeadOwner()
    owner._hooks_retired = True
    owner._threading_hook = dead_threading_hook
    owner._sys_hook = None
    owner._previous_threading_profile = sentinel_threading_predecessor
    owner._previous_sys_profile = sentinel_sys_predecessor
    dead_threading_hook._tl_owner = owner

    resolved = _skip_retired_hooks(dead_threading_hook, "_previous_sys_profile")
    assert resolved is sentinel_threading_predecessor, (
        "restoring the sys slot through a dead threading hook resolved the "
        "dead owner's SYS predecessor instead of its threading chain"
    )
