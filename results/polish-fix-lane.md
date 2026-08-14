# POLISH-FIX lane result — fix R19 regressions + reviewed rebaselines

Branch `fw2/polish`, worktree `~/.claude/worktrees/torchlens-fw2-polish`.
Scope: the 10 reds in the lane's viz/options/render territory gate, split
class A (real R19-6 regression) / class B (golden drift needing reviewed
rebaseline).

## Class A — real regression: R19-6 broke SVG image embedding (8 reds)

ROOT CAUSE. R19-6 (b8ffdd9d) emits node `image=` attrs RELATIVE to the
visualizer scratch root, with the root supplied once as the graph-level
`imagepath` attribute. Graphviz resolves the relative paths fine for its own
rendering, but it copies the RAW relative href into SVG output verbatim. The
TorchLens SVG post-pass `_inline_svg_local_images` (data-URI embedding)
resolved hrefs against `Path.cwd()` only, so every relativized image failed
`read_bytes()` and was swapped for the "preview unavailable" placeholder —
zero `<image>` tags, zero data URIs, repgeom/feature-map/aesthetic renders
all imageless.

FIX (in `torchlens/visualization/_render_flow.py` + `_render_dot.py`):
`_resolve_svg_image_path` takes an `image_root` and resolves relative hrefs
against it first (falling back to CWD when unset or the rooted file does not
exist — matching Graphviz's own imagepath search order, and keeping
user-supplied CWD-relative images working). The root — the same
`trace._visualizer_dir` that becomes the DOT `imagepath` — is threaded
through `_inline_svg_local_images`, `_inline_svg_file_local_images`,
`_render_graph_only_svg`, and `_write_composed_code_panel` from the three
render call sites (saved-SVG inline, notebook display, code-panel compose).
R19-6's confinement is fully preserved: DOT bytes are unchanged by this fix,
and the R19-6 guard test `test_image_node_attrs_are_visualizer_relative`
stays green.

Greened:
- tests/test_repgeom.py::test_mds_scatter_draw_uses_one_contained_image_per_annotated_node
- tests/test_repgeom.py::test_mds_scatter_draw_embeds_data_uri_and_survives_save_load
- tests/test_repgeom.py::test_rdm_node_spec_draw_uses_one_contained_image_and_survives_save_load
- tests/test_repgeom.py::test_scree_node_spec_draw_uses_one_contained_image_and_survives_save_load
- tests/test_feature_maps_c2.py::test_feature_map_node_spec_renders_one_image_and_overlay_differs_from_fallback
  (code fix + assertion update: the test pinned the ABSOLUTE image path in
  the DOT, which is the exact byte R19-6 removes; it now pins the confined
  contract — relative `image=`, one `imagepath=` root, absolute path absent)
- tests/test_output_aesthetics.py::test_generate_aesthetic_report

Two auto_collapse reds in class A were NOT image-related:

- test_v2_max_op_segment_renders_dashed_box_and_contracts_edges: stale
  against intended R19-3/5 behavior. Max collapse of SegmentToyNet now
  renders ZERO module clusters (segments swallow every module's ops; the
  empty labeled husks are gone) so the old edge-before-first-cluster
  ordering probe hit StopIteration. Test updated to pin the new truth:
  contracted segment edge present, `subgraph cluster_` ABSENT, and the
  R19-5 `-- spans` disclosure on the top-level segment label.
- test_signal_tally_latency_under_budget: NOT an R19 regression — proven by
  A/B (HEAD vs main, identical: cold 117ms, warm ~33ms; budget 100ms).
  The cold first call pays per-trace op-facade hydration (M5 columnar
  first-read, profiled: `__getattribute__`/`cell_get` dominate), which the
  single-shot measurement charged to the tally. Fixed the measurement: one
  untimed warm-up call, then the timed tally (analyze_collapse does not
  cache its result — warm runs still do the full work, ~33ms, so the budget
  still trips on a real 3x tally regression).

## Class B — reviewed golden rebaselines (2 reds)

### tests/fixtures/s5_render_golden_manifest.json (S5 render-identity)

The drift decomposes into TWO layers, both fully explained line-by-line:

1. manifest -> branch base (main 44362ab9): the manifest was ALREADY stale
   on main — red at the branch base itself, deterministic (double-render
   byte-identical). Only the three `max` DOT hashes differed (+30/+30/+20
   bytes). PROVEN byte-exact to be r21's injective segment identity
   (e0e087b1, adjudicated on main): regex-reverting the segment node names
   in the fresh DOTs (`{a}passN__segment__{b}passM` -> `{a}__segment__{b}pass1`)
   reproduces the manifest hashes EXACTLY for all three cases. Plans
   identical. The manifest had not been regenerated since 2440bc09.
2. base -> HEAD: the intended R19 change, every diff line mechanically
   classified into exactly three classes: R19-3 empty-husk cluster removals
   (attribute-only husks and fully-empty duplicate re-entries; remaining
   attr-less `subgraph X {}` re-entry shells are DOT-source no-ops for
   clusters that render from their content entry), R19-4 dashed->solid on
   modules whose edges cross their boundary (resnet18 layer1..4 under
   auto), R19-5 `-- spans` segment label disclosure. Collapse-plan reprs
   byte-identical base vs HEAD.

Regenerated from a fresh HEAD render (fresh cache, deterministic);
`tests/test_s5_render_identity_harness.py` green (2 passed).

### docs/images/collapse gallery (schema-lockstep generated artifact)

Six SVGs drifted (mode-none, mode-auto, schedule-0/25/50/75). Every diff
line classified: cluster-id renumbering (`clust N` shifts after R19-3
removes husk clusters from the DOT) plus, in mode-auto only, four module
boxes losing `stroke-dasharray` — the R19-4 dashed->solid connectivity fix.
No geometry, label, or node changes. Regenerated via
`python scripts/render_collapse_reference.py`; gate green.

## Gate

- The 10 original reds, one run: **10 passed** (exit 0).
- Full territory gate `pytest tests/ -k "collapse or viz or aesthetic or
  options or fold or draw or render or graphviz or repgeom or feature_map"
  -p no:randomly` (fresh TORCHLENS_CACHE_DIR, private basetemp,
  CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS=8, torchlens.__file__ ->
  worktree): **PENDING — filled below after run**
- ruff check + format clean on touched files; mypy: the 2 remaining errors
  are pre-existing in untouched files (`_save_budget.py`, `_finalize.py`,
  other lanes' territory).
