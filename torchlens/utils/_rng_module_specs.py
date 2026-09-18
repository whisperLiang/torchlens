"""Static endpoint dispositions for the feature-detected Torch RNG surface."""

# Per-module endpoint specs, feature-detected at build. ``get_rng_state`` family rows are
# deliberately structurally_covered (NO monitor row): the returned state TENSOR is already
# covered by the r39 tensor->host escape belt (branch-on-state-bytes ->
# INCOMPLETE_SCALAR_ESCAPE, never VERIFIED; a store-only read stays VERIFIED), and a row
# would over-ceiling ``torch.utils.checkpoint(preserve_rng_state=True)``, which
# round-trips VERIFIED+ATTESTED today (r65 probe za4).
_TORCH_RNG_CORE_SPEC: tuple[tuple[str, str, str], ...] = (
    ("seed", "entropy", "draws OS entropy and reseeds the global engine"),
    ("manual_seed", "mutation", "in-forward host mutation of the global engine"),
    ("initial_seed", "replayable_read", "scalar read fully determined by the capture seed"),
    ("set_rng_state", "mutation", "in-forward host mutation of the global engine"),
    ("get_rng_state", "structurally_covered", "state-tensor return; r39 escape belt"),
)
_TORCH_RNG_DEVICE_SPEC: tuple[tuple[str, str, str], ...] = _TORCH_RNG_CORE_SPEC + (
    ("seed_all", "entropy", "draws OS entropy and reseeds every device engine"),
    ("manual_seed_all", "mutation", "in-forward host mutation of every device engine"),
    ("set_rng_state_all", "mutation", "in-forward host mutation of every device engine"),
    ("get_rng_state_all", "structurally_covered", "state-tensor return; r39 escape belt"),
)
_TORCH_RNG_ACCELERATOR_SPEC: tuple[tuple[str, str, str], ...] = (
    ("initial_seed", "replayable_read", "scalar read fully determined by the capture seed"),
    ("get_rng_state", "structurally_covered", "state-tensor return; r39 escape belt"),
    ("get_rng_state_all", "structurally_covered", "state-tensor return; r39 escape belt"),
)
_TORCH_RNG_MODULE_SPECS: tuple[tuple[str, tuple[tuple[str, str, str], ...]], ...] = (
    ("torch", _TORCH_RNG_CORE_SPEC),
    ("torch.random", _TORCH_RNG_CORE_SPEC),
    ("torch.cuda", _TORCH_RNG_DEVICE_SPEC),
    ("torch.cuda.random", _TORCH_RNG_DEVICE_SPEC),
    ("torch.mps", _TORCH_RNG_CORE_SPEC),
    # r67 C1 (hon1-F6): ``torch.mtia`` carries the full feature-detected device RNG
    # spec -- on torch 2.8 only ``get_rng_state``/``set_rng_state`` resolve, and any
    # torch upgrade that grows the mtia surface lights up through the same
    # ``hasattr`` feature detection instead of a hand-list edit.
    ("torch.mtia", _TORCH_RNG_DEVICE_SPEC),
    ("torch.xpu", _TORCH_RNG_DEVICE_SPEC),
    # r67 C1: ``torch.xpu.random`` re-exports the xpu RNG surface exactly like
    # ``torch.cuda.random`` does for cuda; found by the independent no-list module
    # discovery immunizer (the same shared-blind-spot class as mtia).
    ("torch.xpu.random", _TORCH_RNG_DEVICE_SPEC),
    # The accelerator frontend delegates to the active backend's default generators;
    # it owns no separate default_generators tuple and exposes only these readers.
    ("torch.accelerator.random", _TORCH_RNG_ACCELERATOR_SPEC),
)
