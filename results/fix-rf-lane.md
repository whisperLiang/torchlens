# LANE fix-rf — RF adaptive-pool envelope bound (F1) + EXACT tightness oracle (F2)

Branch `fix/rf` off main `fd18accc`, worktree `~/.claude/worktrees/torchlens-fix-rf`.
Source audit: `deephunt-rf.md` (read first). Sanity: `torchlens.__file__` resolved to the
worktree for every gate; fresh `TORCHLENS_CACHE_DIR` + private `--basetemp` +
`CUDA_VISIBLE_DEVICES=""` + `OMP_NUM_THREADS=8` throughout.

## F1 (BUG, medium) — adaptive-pool descriptor envelope under-covered for ratio > 2 — FIXED

- Commit `c77c6eeb` `fix(receptive_field): contain the adaptive-pool descriptor envelope
  for ratios above 2` (`torchlens/receptive_field/rules/conv_pool.py` +
  `tests/test_rf_adaptive_envelope.py`).
- Root cause confirmed exactly as audited: `adaptive_pool` emitted edges
  `((r, -1), (r, +1))`; the true bin ends at `ceil((o+1)*r) - 1`, so the constant `+1`
  hi intercept under-covers by up to `r - 2` once `r = in/out > 2`. Reproduced live:
  `adaptive_avg_pool1d(15->3)` published `size=(3,)` `UPPER_BOUND` vs true window 5.
- Fix: hi edge intercept `r - 1/out` — the TIGHTEST sound linear bound (the audit's
  primary suggestion was `r`; both boundaries have denominator dividing `out`, so
  `ceil((o+1)*r) - 1 <= r*o + r - 1/out` for every ratio; the two intercepts produce the
  identical integer-hull size, but the tighter one dominates under downstream fractional
  composition). Lo edge `-1` was already sound (`floor(o*r) > o*r - 1`). `exact=False`
  (UPPER_BOUND) unchanged — no new exactness claimed. Global-pool `output_size=1` branch
  untouched. Per-unit `.at()`/`check()`/projective were never affected (exact
  `map_adaptive_pool_index_set` callback).
- Regression (red-capable, verified red pre-fix): `tests/test_rf_adaptive_envelope.py` —
  analytic bin-width oracle anchored to autograd, then containment goldens for the
  auditor's cases: 1d `(15,3),(21,4),(17,5),(16,6),(10,4),(9,2)`, 2d `(21,17)->3/4/5`
  (both axes), `adaptive_max_pool1d` (shared rule), and the composed
  `conv1d(k3,p1) after adaptive_avg_pool1d(15->3)` whole-graph case (pre-fix 13 vs true
  15, gradient-verified). Pre-fix red confirmed: `assert 3 >= 5` failed on `[15-3]`.
  Post-fix sizes are honest upper bounds (e.g. 7 for true 5) with status UPPER_BOUND —
  containment asserted as `>=`, no golden loosened, no rebaseline needed.

## F2 (tripwire scope gap, low-medium) — no tightness oracle for EXACT claims — SHIPPED (test-level oracle)

- Commit `53bee21f` `test(receptive_field): add a saturating-model tightness oracle for
  EXACT claims` (`tests/test_rf_exact_tightness.py` +
  `torchlens/receptive_field/_validation.py` docstring honesty fix).
- Design call (tractability): gradient support is a LOWER bound on influence (dead units,
  argmax pooling, zero weights), so an in-`check()` FAIL gate on nonzero slack would
  false-alarm on honest EXACT rules over real models — that would break the tripwire in
  the other direction. The sound, tractable oracle is a TEST battery over the built-in
  registry under SATURATING models (all-ones weights, uniform average pooling), where
  gradient support saturates true influence: every claimed-EXACT per-axis endpoint must
  be attained (`slack_per_axis == 0`) at interior (unclipped) units.
- Battery: conv2d k3p1, dilated conv1d k5d2, strided conv2d, avg_pool2d, avg_pool1d
  (max pooling deliberately excluded — argmax-sparse gradients make slack honest).
  Red-capability control: a planted 9x9-for-true-5x5 EXACT conv rule PASSes one-sided
  containment AND the adjoint corner cross-check (both directions consult the same lying
  registry — the audit's exact rfprobe3 finding), but the slack oracle exposes it
  (`slack_per_axis[-2:] == (4, 4)`). Undersized-rule control keeps the containment
  tripwire provably armed (FAIL). Adaptive-pool descriptor/per-unit split pinned
  (descriptor UPPER_BOUND exempt from tightness; per-unit box exact with zero slack).
- Docstring fix: `_exact_box_adjoint_violations` no longer overstates its power — it is a
  direction-coherence oracle, blind to symmetric overclaims; tightness enforcement is
  named as the test battery.
- QUEUED (design call for JMT): an opt-in user-facing tightness gate
  (e.g. `check(..., require_tight=True)` refusing nonzero slack on EXACT boxes) is a
  public-API kwarg → needs glossary lockstep + explicit approval; not shipped here.

## Tripwire discipline

No validation check weakened, no tolerance broadened, no golden rebaselined. Both changes
STRENGTHEN honesty: F1 makes a published containing bound actually contain; F2 adds a new
oracle class and keeps both prior controls (undersized FAIL, containment PASS) armed.

## Gate counts (worktree venv `~/projects/torchlens/.venv`, torch 2.13.0+cu130, CPU)

| Gate | Result |
|---|---|
| `tests/test_rf_adaptive_envelope.py` pre-fix | 1 failed (`3 >= 5` on `[15-3]`, red as required), 1 passed before `-x` stop |
| `tests/test_rf_adaptive_envelope.py` post-fix | 12 passed |
| `tests/test_rf_exact_tightness.py` | 9 passed |
| Core RF gate (at/crossval/geometric/table/rules/integration/hardening/remediation/getitem/axis-perm + new) | 229 passed, 5 skipped |
| Full RF sweep (all `tests/test_rf_*.py` + `test_r18rf_render_flow.py` + `test_r19b_rf_gradient_verify.py` + `test_r19f_rf_view_table.py`) | 406 passed, 6 skipped, 0 failed (47s) |
| `ruff check` (receptive_field + new tests) | clean |
| `mypy` (conv_pool.py, _validation.py) | clean on touched files; 2 pre-existing errors in untouched `_save_budget.py` / `backends/_finalize.py` (present on main) |

## Commits

- `c77c6eeb` fix(receptive_field): contain the adaptive-pool descriptor envelope for ratios above 2
- `53bee21f` test(receptive_field): add a saturating-model tightness oracle for EXACT claims

O1 (grid_sample/unfold rules someday) and O2 (unreachable negative-step getitem branch)
from the audit are observations, out of lane scope, not acted on.
