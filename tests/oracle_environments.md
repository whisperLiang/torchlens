# Reviewed oracle environments

The canonical golden environment remains Python 3.10 / torch 2.13.0.
Python 3.11 / torch 2.13.0 is also enrolled for the non-slow, environment-governed
oracles. Visualization goldens additionally require graphviz 0.21 and, where
declared by the family, pydot 4.0.1.

The 14 Python 3.11 golden files were independently compared against the canonical
files before enrollment. Every payload was byte-identical, including tensor
digests, public surfaces, state keysets, DOT text, and rank-render semantics.
The original `.tlspec` fixtures and canonical golden files were not regenerated.
Each new environment directory carries a `PROVENANCE` record with the reason,
generator, runtime, and source-checkout identity.

`test_oracle_attestation.py::test_reviewed_environment_golden_inventory` checks
that all enrolled files are present, provenance-stamped, and not gitignored.
Unknown environments remain ignored and fail closed locally; no automatic
recording or canonical fallback was added.

The rebuilt unified environment also enrolls Python 3.11 / torch 2.8.0. The
verified installation is Python 3.11.15 / torch 2.8.0+cu128, with torchvision
0.23.0, torchaudio 2.8.0, Transformers 5.17.0, RFDETR 1.8.3, TensorFlow 2.21.0,
and Keras 3.15.1. The oracle directory key uses the Torch release version,
without its CUDA build suffix.
Eleven of its fourteen golden payloads are byte-identical to the canonical files.
The remaining three surface snapshots (`plain_cnn`, `train_batchnorm`,
`tiny_transformer`) differ with the upstream Torch runtime: module docstrings and
the native Transformer operation graph change. All three are independently
byte-identical to snapshots from the unchanged TorchLens HEAD package
`25b9dc9817a8a5740aa031d5e64e8defdbb8b034` running in this new environment.
Neither canonical payloads nor frozen `.tlspec` fixtures were overwritten.
Legacy artifact tests explicitly assert the expected cross-minor loading warning.

## Collapse-gallery render environment

The checked-in smart-collapse SVG gallery additionally depends on the native
Linux Graphviz **2.43.0** renderer and `fonts-dejavu-core`; the Python `graphviz`
package version is a separate dependency. System font installation can otherwise
change text extents and node positions even with unchanged TorchLens code.

[`scripts/render_collapse_reference.py`](../scripts/render_collapse_reference.py)
temporarily selects
[`scripts/collapse_reference_fonts.conf`](../scripts/collapse_reference_fonts.conf)
when rendering or checking this gallery. The configuration maps `Times`,
`Times-Roman`, and `Times New Roman` to **DejaVu Serif**; Pango may expand the
Graphviz name before matching. The original `FONTCONFIG_FILE` setting is restored
on both success and failure. This pin is local to the reference generator and
does not modify system font configuration or ordinary TorchLens rendering.

With that environment, all 14 SVGs reproduce the committed files byte-for-byte.
The existing SVGs were not regenerated to accept the font-dependent geometry
change. `tests/test_collapse_reference_fonts.py` checks environment restoration,
and the schema-lockstep gallery gate reruns the generator's `--check` comparison.

## Verification

Activate the project virtual environment first. Rendering tests also require the
Graphviz executables `dot` and `neato` on `PATH`; the Python `graphviz` package
alone does not provide them. The repository's test extra pins pydot 4.0.1.

```bash
TORCHLENS_ORACLE_ENFORCE=1 TORCHLENS_RENDER_BYTE_ORACLE=1 python -m pytest \
  tests/godobject_oracle/ tests/surface_oracle/ \
  tests/test_state_keyset_contract.py \
  tests/test_rank_render_ir_semantic_goldens.py \
  tests/test_viz_render_identity_oracle.py \
  tests/test_oracle_attestation.py tests/test_golden_governance_lint.py \
  -m "not rare and not slow" --tb=short
```

The separate, non-mutating gallery check is:

```bash
python scripts/render_collapse_reference.py --check
```

This is an enforcing run, with no update/record flags. A deliberate enrollment or
update must still supply `TORCHLENS_GOLDEN_REASON`, run the owning generator in a
fresh interpreter, review the resulting differences, and rerun without mutation
flags. The generating run reports SKIP, not a successful comparison.
