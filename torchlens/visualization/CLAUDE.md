# visualization/ - Graph Rendering and Visual Helpers

## Forward Rendering Pipeline

Forward `Trace.draw()` resolves its request once, then follows one renderer-neutral pipeline:

```
SourceGraph -> NodeUniverse -> RenderIR -> renderers/{base,graphviz}
```

- `source_graph.py` normalizes the trace walk: focus, buffer visibility, `skip_fn`, and edge occurrences.
- `node_universe.py` projects that source graph into visible structural units and projected endpoints.
  Collapse planning uses this same universe through `collapse_plan.py`.
- `render_ir.py` decorates those units with resolved nodes, edges, regions, ordering constraints, and
  backend-ready statements. Renderers receive this immutable IR, not TorchLens trace objects.
- `renderers/base.py` defines the renderer protocol and capability checks; `renderers/graphviz.py`
  serializes and executes Graphviz. The rank layout backend also consumes the resolved IR.

`Trace.draw()` dispatches directly to `_render_dot.py`; backward and combined entrypoints dispatch
to `_render_entrypoints.py`. The graphviz renderer is the primary backend;
`vis_node_placement="auto"` selects dot or the rank
layout according to the resolved graph cost. Forward sibling ordering is a Graphviz-only post-layout
operation and conservatively no-ops outside its supported forward/unrolled/dot cases.

## Related Surfaces

`node_spec.py`, `themes.py`, `modes.py`, and `overlays.py` provide node presentation decisions.
`code_panel.py` adds captured source beside Graphviz output. `bundle_diff.py`, `fastlog_preview.py`,
and `fastlog_live.py` support specialized visualization workflows.

## Internal Helper Modules

The `_render_dot.py` entry point is split across sibling helper modules; all are internal:

| Module | Purpose |
|--------|---------|
| `request.py` | Resolved visualization requests and output targets (`ResolvedRenderRequest`, `RenderTarget`, `RenderContext`) |
| `_render_common.py` | Shared render types, constants, and imports for Graphviz rendering |
| `_render_flow.py` | Focus, skip, container, and sibling setup helpers |
| `_render_nodes.py` | Node construction and raw value helpers |
| `_render_edges.py` | Edge and endpoint helpers |
| `_render_leaf.py` | Backward/grad-fn leaf placement and module inference |
| `_render_ordering.py` | Sibling-ordering scope decision, plain-layout verification, and the DOT rank-group post-pass |
| `_render_regions.py` | Nested module-cluster (region) subgraph emission and empty-subtree pruning |
| `_svg_compose.py` | SVG post-processing (image inlining, viewBox normalization) and code-panel composition |
| `_render_utils.py` | Internal Graphviz helpers shared across rendering paths (subprocess execution, HTML escaping) |
| `_label_format.py` | Node label formatting helpers |
| `_edge_multiplicity.py` | Rendered-edge multiplicity disclosure (dedupe registry, honest `xN` edge labels) |
| `_condensed_flow.py` | Child condensed-flow-graph construction for smart module collapse |
| `_rank_layout_internal/`, `_summary_internal/` | Rank-layout backend internals and `summary()` internals |
| `renderers/` | Renderer protocol (`base.py`) and the Graphviz backend (`graphviz.py`) |

Smart collapse: `auto_collapse.py` (analysis + fold discovery), `collapse_optimizer.py` (v2
frontier selection; owns the `COLLAPSE_OPTIMIZER_MAX_OPS` compute ceiling), and
`collapse_plan.py` (plan/schedule projection) form the engine.

Backward and combined graph entrypoints live in `_render_entrypoints.py`. Their grad-function source
normalizers produce the same `RenderIRNode`, `RenderIREdge`, and `RenderIRRegion` records as the forward
pipeline, including backward-pass regions for unrolled graphs, and dispatch through the same renderer
protocol and Graphviz backend. Experimental Dagua remains opt-in under `torchlens.experimental.dagua`.
