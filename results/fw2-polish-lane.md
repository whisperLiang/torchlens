# FW2-POLISH lane result (fixwave-2, 2026-08-14)

Branch `fw2/polish` off main `05a2fc51`. Lane scope per `results/fixwave-2-plan.md`
section 2 FW2-POLISH: B6-VIZ (R19-1..7) + B7-ASSERT viz asserts (R24-2/3) + R40-1/2/3
+ B7-DEADCONFIG (R47-1/3/5..10, R42-1/2/3/5/6/9, R52-4) + SF-03 viz sites + B8-37 +
b5-ledger viz C901 item. TRIPWIRE LOCKED honored: no check weakened, no tolerance
widened; the one golden rebaseline (rank semantic golden) has its root cause fixed in
the SAME commit and used the golden's documented regen flag.

## Commits (9)

| sha | scope |
|---|---|
| f6c80914 | R47-1/3/5/6/7/8/9/10 dead-config deletion + closed-vocab knob refusal + inert_option_fields deprecation family |
| 13cadf42 | R42-1/2/6 dead HAS_* retirement, R42-3/5 dead pair/single-mentions, R42-9 dup parser, R52-4 frozenset |
| ed89b8b8 | R24-2/3 collapse/sibling honesty guards assert -> raise |
| 4531879d | R40-1/2/3a/3b viewer reap, bounded notebook render, start_new_session, warn_parallel hardening |
| cf330d5b | SF-03: the 7 viz B028 warn sites -> user_stacklevel() |
| 04284e78 | R19-1 fold-uniformity signature + R19-2 per-pass rank geometry (+ golden rebaseline, root cause in-commit) |
| 5249dec2 | R19-3 empty-husk suppression + R19-4 dashed-cluster truthfulness + R19-5 segment span disclosure |
| b8ffdd9d | R19-6 image paths relativized to one imagepath attr; marker-lint split; stub kwarg |
| (tail)   | R42-5 orphaned `_filter_conditional_elif_children` deletion + this results file |

## Item disposition

EXECUTED (fixed here, with pins):
- **R19-1** fold `+N more` uniformity fingerprint now carries per-layer op-type sequence
  + canonical func_config digest. Seeds red pre-fix (dilation-2 conv, tanh-for-relu);
  DOT byte-difference pin; uniform plateaus (incl. differing representative) still fold.
- **R19-2** rank path: node identity from renderer-unique name (recurrent passes no
  longer collapse to one node_data entry); bboxes keyed by full pass-qualified region
  key both sides; sorted iteration. Distinct-bbox pin red-proven against the legacy
  collapse; in-process byte-determinism pin; PYTHONHASHSEED subprocess sweep (seeds
  0/1/2, one sha256) as the plan-mandated golden. Rank semantic golden rebaselined via
  `TORCHLENS_UPDATE_RANK_RENDER_IR=1` (drift = the fix: node ids now pass-qualified).
- **R19-3** module clusters whose whole subtree is empty are never emitted (the descent
  branch opened husks before the leaf guard). 8-block collapse="max" probe: eight empty
  boxes (4 dashed) pre-fix, zero after; `collapse="none"` renders unchanged.
- **R19-4** input-connectivity marks every module containing a connected edge endpoint,
  every edge (tautological break + LCA-only gating removed). Red-proven pin: nested
  module with only boundary-crossing edges rendered dashed in unrolled while solid in
  rolled; plus an unrolled-vs-rolled agreement pin.
- **R19-5** op segments placed above their ops' module homes disclose the spanned
  modules in the label (`... -- spans @0:1, @1:1, +2 more`); within-module segments
  unchanged (both pinned; verified in integration DOT).
- **R19-6** node `image=` attrs are relative to the visualizer scratch root; the
  mkdtemp path appears exactly once as the graph-level `imagepath` attribute (injected
  post-node-build because the dir is lazily created; dot + rank paths). End-to-end PNG
  render verified; pin asserts no absolute paths in image attrs.
- **R24-2/3** OptimizerResult/segment-parity cardinality guards and the sibling-order
  structural backstops raise RuntimeError (shipped-default label-honesty paths no longer
  disarmed by `python -O`); -O-survival pin added.
- **R40-1** viewer children retained + reaped (`_VIEWER_PROCS`, census-classified);
  **R40-2** notebook display renders through the bounded runner (120s + the CLI paths'
  typed timeout/failure raises) instead of unbounded `dot.pipe()`/`display(dot)`;
  **R40-3a** `start_new_session` on all six render/viewer spawn sites;
  **R40-3b** `warn_parallel` ignores the user-assignable process name
  (`parent_process()` + import-PID stamp catches raw forks) and the rank carve-out
  refuses a fork-INHERITED initialized-group flag (first-observer PID stamp,
  census-classified). Spoof + inheritance pins with a positive rank control.
- **R42-1** six ambient-context HAS_* flags retired (public pre-2.1 torch surfaces;
  probes, snapshot ternaries, `_require` calls deleted); **R42-6** the
  HAS_AUTOCAST_DEVICE_TYPE_ARG duplicate alias deleted; **R52-4** capability membership
  via companion frozenset (tuple stays the ordered snapshot authority).
- **R42-2 (bug half)** `get_variable_function_names` wrong-namespace fallback
  (`torch.__all__`) now degrades to an empty roster with the flag flip visible.
- **R42-3** themes `legend_lines`/`semantic_class_attrs` dead pair deleted; **R42-5**
  `reconstructed_sdpa_value`, `_input_container_structure_capability`, and (orphaned
  from the capture lane's queue, collision-free) `_filter_conditional_elif_children`
  deleted; **R42-9** one shared `TORCHLENS_COLLAPSE_STRICT` parser.
- **R47-1** the nine write-only option fields deleted; keywords warn for one window via
  the registered `inert_option_fields` deprecation family (census pin 9); **R47-5** the
  `_module_containment_engine` knob + full plumbing deleted; **R47-3** the three
  postprocess audit knobs refuse unrecognized values (typo can no longer disarm/reroute
  an audit); **R47-6/7** dependency + suppress-knob documented; **R47-8** dead reserved
  `cleanup_class` axis deleted (`storage` axis is LIVE via the schema-bindings gate —
  struck); **R47-9** write-only `invisible` param deleted across call sites (fence
  deviation note below); **R47-10** the three deleted-knob tombstones no longer spell
  the dead env-var name.
- **SF-03** the seven visualization B028 sites resolve stacklevel via
  `user_stacklevel()`. `ruff --select B028` on `torchlens/visualization/` is clean.

VERIFIED ALREADY FIXED AT TIP (struck, evidence in-line):
- **B8-37** `_deprecations.py` stacklevel: b4-D shipped the `user_stacklevel()`
  mechanism with caller-attribution tests.
- **R19-7a** duplicate cluster re-emission / empty `{}` bodies: probe on tip (4-block
  and 8-block models, collapse none+max) found zero duplicate headers and zero empty
  bodies; the empty-subtree guard now also forecloses the class.
- **b5 viz C901 item** (`_compute_selected_node_lines` depth-11): already a flat
  single-level dispatch at tip.

COULD NOT REPRODUCE AT TIP (documented, no code change):
- **R19-7c** repr 0x-address leak into DOT tooltips: probed with an opaque-object
  output member; no `0x` hits in DOT. Tooltip sites are strings-only today.

SCOPED OUT (residuals, with reasoning — not silently dropped):
- **R19-7a residue**: clusters that DO have content elsewhere can still be re-entered
  as empty `{}` bodies by later recursion paths (OracleNested keeps two benign
  `cluster_encoder_pass1 {}` re-entries; graphviz merges same-name subgraphs, so
  output is correct but noisy). Full dedup needs a one-open-per-cluster restructure
  of the recursion; next viz wave. Also note: a module focus view no longer shows an
  empty box for the focused-OUT sibling (it was styled as a false "no input ancestor"
  claim); if a "sibling exists" hint is wanted it should be an honest marker, not a
  connectivity-styled empty cluster.
- **R19-7b** dedup'd boundary-crossing distinct-dataflow edges lack multiplicity
  disclosure: needs restructuring of the incremental edge-emission loop (edges are
  written to the Digraph as encountered; disclosure requires a counting pre-pass or
  post-edit of emitted bodies). LOW tier; recorded for the next viz wave.
- **R42-2 (deletion half)** the five private-API lazy probes and their None-guard
  fallbacks (HAS_TENSORBASE_CLASS etc., ~90 LOC) are deliberately KEPT: they probe
  fragile `torch._C` surfaces, and probe-behind-HAS_*-flag with graceful degradation is
  the repo's LOCKED torch-compat doctrine (CLAUDE.md). Deleting the fallbacks would
  make a future private-API removal crash instead of degrade. If the maintainers want
  them gone it is a doctrine change, not a debloat.

FENCE DEVIATION (disclosed): R47-9's full fix required deleting the dead `invisible=`
kwarg at 8 call sites in sibling `_completeness_*` files not in this lane's write list.
All owning lanes were terminal/merged at edit time (live branches `fw2/wrap` and
`fw2/b4pslice` touch only wrappers.py / postprocess-contracts — verified disjoint).

## Gates

- **Net LOC (package `torchlens/`)**: debloat scope (R47+R42 commits over options.py,
  postprocess/__init__.py, field_policy.py, _torch_compat.py, themes.py,
  reconstruction.py, container.py, cleanup.py, user_funcs.py, ir/workspaces.py,
  completeness files): **net NEGATIVE (~-250)**. Whole-lane package delta:
  **+99 (566 ins / 467 del)** — POSITIVE, i.e. the strict
  whole-lane reading of the gate is MISSED. Cause: the lane bundles the B6-VIZ honesty
  items, whose mandated fixes (bounded notebook render, per-pass geometry, span
  disclosure, raise-based guards, warn_parallel hardening) are additive by nature and
  outweigh the sanctioned deletions. No further deletion had adjudicated evidence, and
  deleting unevidenced code to hit a metric is exactly the failure mode the tripwire
  doctrine forbids. Flagged for orchestrator adjudication rather than gamed.
  Tests add ~+900 LOC of red-proven pins (pins were the other half of the gate).
- **Goldens**: PYTHONHASHSEED sweep (subprocess, 3 seeds, single sha256) and
  fold-honesty byte-comparison landed as pins (`tests/test_viz_rank_geometry.py`,
  `tests/test_viz_fold_honesty.py`); rank semantic golden rebaselined with root cause
  fixed in the same commit via its documented regen flag.
- **Targeted suites** (worktree venv, fresh TORCHLENS_CACHE_DIR, private basetemp,
  CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS=8), 30-file consolidated run:
  **426 passed, 5 skipped, 5 failed** — 4 of the 5 fails are the pre-existing reds
  below; the 5th was the viz render-identity oracle, whose drift IS the adjudicated
  R19 rendering change (empty dashed husks gone from OracleBatchNorm/OracleNested;
  duplicate empty cluster re-entries reduced) — regenerated via its documented
  `TORCHLENS_UPDATE_VIZ_RENDER_ORACLE=1` flag with the root cause fixed in this same
  lane, then verified green (**1 passed** standalone; suite re-tally 427/4).
- `ruff check` clean on every touched file; `mypy` on touched core files shows only the
  two pre-existing errors also present at main tip (`_save_budget.py:375`,
  `backends/_finalize.py:1102`).

## Pre-existing reds encountered (NOT this lane's; all reproduced at bare main 05a2fc51)

Beyond the two dispatch-ledgered ones (test_r18n_bundle_save_atomic, test_facets_p4):
1. `test_module_containment_equality.py::test_module_containment_snapshot[raw_hook_replacement_logged]` — snapshot drift at tip.
2. `test_trace_field_invariant.py::test_trace_field_set_subset_of_user_facing` — unexpected `_op_accessor_cache` Trace field.
3. `test_postprocess_dag.py::test_read_enforcement_green_on_default_capture` — step 9 undeclared `_label_raw` read.
4. `test_auto_collapse_metrics.py::test_signal_tally_latency_under_budget` — latency budget red under load.
5. `test_output_aesthetics.py::test_generate_aesthetic_report` — missing reports dir under a fresh TORCHLENS_CACHE_DIR.

These are deselected/reported, never masked; 1-3 look like fixwave integration
follow-ups (2's field wants a census/type classification; 3 is a postprocess contract
diff owed by whoever introduced the step-9 `_label_raw` read).
