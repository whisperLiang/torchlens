"""Provocation + reachability meta-gate over the frozen refusal vocabularies (R25-1).

Every existing lockstep gate checks SHAPE (doc row <-> declaration); none checked
REACHABILITY (a member with no raise site) or PROVOCATION (a code no test ever
asserts), which is exactly how dead members and unprovoked tripwires survive
forever — ``PathDivergenceError`` was provoked ~100x while a swap of any of its
12 divergence CODES would have failed zero tests.

Three enforcement layers, all static file scans (smoke-tier, no capture):

1. REACHABILITY — every ``RunnableErrorCode`` / ``MergedErrorCode`` member must
   be referenced in ``torchlens/`` beyond its declaration. A dead member goes
   red (the ``merged_export_unsupported`` precedent: declaration-only for a
   full release while refused surfaces raised different codes).
2. PROVOCATION RATCHET — every vocabulary code must be referenced by some test,
   except the frozen ``UNPROVOKED_BASELINE`` (historical debt: may only
   SHRINK — delete entries as provocations land; adding is a conscious public
   decision) and the reasoned ``ENV_GATED_ALLOWLIST`` (codes whose provocation
   needs hardware/topology this suite cannot assume).
3. Anti-vacuity + red-capability — the scanners must find real universes and
   must be demonstrably able to fail.

A NEW code (enum member or contract row) therefore cannot ship unprovoked:
it is not in the baseline, so this gate refuses until a provoking test exists
or a reasoned allowlist/baseline entry is consciously added in the same change.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from torchlens.merged import MergedErrorCode
from torchlens.runnable import RunnableErrorCode

pytestmark = pytest.mark.smoke

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PACKAGE_ROOT = _REPO_ROOT / "torchlens"
_TESTS_ROOT = _REPO_ROOT / "tests"
_CONTRACT_DOC = _REPO_ROOT / "docs" / "reference" / "error_refusal_contract.md"

_DOC_ROW_PATTERN = re.compile(r"^\| `([a-z0-9_]+)` \|")

# Codes whose provocation genuinely requires an environment this suite cannot
# assume (multi-rank process groups, CUDA devices, wall-clock nondeterminism).
# Every entry carries the reason; an entry whose reason stops holding must be
# moved to a provocation test, never silently kept.
ENV_GATED_ALLOWLIST: dict[str, str] = {}

# Historical provocation debt, frozen at FW2-ERRORS lane time. SHRINK-ONLY:
# when a provoking test lands for a code, DELETE its row here in the same
# change. Adding a row is a public-vocabulary governance decision, not a fix
# for a red gate.
UNPROVOKED_BASELINE: frozenset[str] = frozenset(
    {
        "annotation_backend_unsupported",
        "annotation_namespace_invalid",
        "annotation_tensor_not_portable",
        "artifact_kind_mismatch",
        "artifact_save_level_unsupported",
        "auto_environment_unsupported",
        "backend_capability_conformance",
        "backend_error",
        "backward_capture_conflict",
        "backward_graph_unavailable",
        "backward_pass_filter_invalid",
        "batch_items_invalid",
        "batch_render_invalid",
        "bundle_diff_layout_invalid",
        "bundle_diff_members_invalid",
        "bundle_member_payload_missing",
        "bundle_member_unknown",
        "bundle_shape_mismatch",
        "bundle_stack_incomplete",
        "bundle_statistic_invalid",
        "code_panel_model_collected",
        "code_panel_side_invalid",
        "collapse_plan_unavailable",
        "compiled_callable_unsupported",
        "container_leaf_not_saved",
        "container_not_reconstructable",
        "container_selector_requires_registry",
        "container_selector_unresolved",
        "container_value_source_invalid",
        "dagua_renderer_not_opted_in",
        "decoded_output_not_classification",
        "decoded_output_unavailable",
        "fsdp_capture_unsupported",
        "gradient_not_saved",
        "gradient_pass_ambiguous",
        "graphviz_render_failed",
        "intervening_cluster_invalid",
        "intervention_action_type_invalid",
        "intervention_engine_invalid",
        "intervention_helper_unknown",
        "intervention_tensor_unsupported",
        "layer_pass_ambiguous",
        "layers_not_logged",
        "link_format_invalid",
        "max_pairs_invalid",
        "metric_name_invalid",
        "metric_shape_mismatch",
        "metric_tensor_type_invalid",
        "metric_type_invalid",
        "model_type_unsupported",
        "module_call_ambiguous",
        "module_focus_empty",
        "module_focus_not_found",
        "node_label_field_invalid",
        "op_lookup_pass_out_of_range",
        "op_lookup_pass_required",
        "option_group_type_invalid",
        "output_device_invalid",
        "output_sink_conflict",
        "recording_events_not_retained",
        "recording_failed_not_convertible",
        "recording_halt_frontier_missing",
        "recording_multipass_not_convertible",
        "recording_option_duplicate",
        "renderer_capability_unsupported",
        "run_fast_divergence_policy_invalid",
        "run_fast_requires_inputs",
        "run_input_missing",
        "run_legacy_options_conflict",
        "run_source_model_collected",
        "save_mode_invalid",
        "save_payload_level_conflict",
        "selector_function_pattern_type_invalid",
        "skip_fn_boundary_invalid",
        "stack_ordinals_duplicate",
        "stack_ordinals_unavailable",
        "stack_output_not_tensor",
        "storage_argument_conflict",
        "summary_fields_invalid",
        "summary_option_conflict",
        "sweep_intervention_conflict",
        "sweep_names_length_mismatch",
        "sweep_site_missing",
        "sweep_site_type_invalid",
        "sweep_values_empty",
        "sweep_values_missing",
        "tensor_connection_labels_missing",
        "trace_not_finished",
        "visualization_direction_invalid",
        "visualization_mode_invalid",
        "visualization_renderer_invalid",
    }
)


def _iter_python_texts(root: Path) -> list[tuple[Path, str]]:
    """Read every Python file under ``root`` once."""

    return [(path, path.read_text()) for path in sorted(root.rglob("*.py"))]


def _documented_codes() -> set[str]:
    """Return the contract-doc code universe."""

    codes: set[str] = set()
    for line in _CONTRACT_DOC.read_text().splitlines():
        match = _DOC_ROW_PATTERN.match(line.strip())
        if match is not None:
            codes.add(match.group(1))
    return codes


def _vocabulary() -> dict[str, str]:
    """Return every governed code mapped to its vocabulary of origin."""

    universe: dict[str, str] = {}
    for member in RunnableErrorCode:
        universe[member.value] = "RunnableErrorCode"
    for member in MergedErrorCode:
        universe[member.value] = "MergedErrorCode"
    for code in _documented_codes():
        universe.setdefault(code, "error_refusal_contract")
    return universe


def _test_referenced_codes(universe: set[str]) -> set[str]:
    """Return every code referenced by some test, by literal or member name.

    A code counts as provoked when its string literal appears in a test file,
    or its enum MEMBER name does (tests that assert via
    ``RunnableErrorCode.X.value`` never spell the literal).
    """

    member_names = {member.value: member.name for member in RunnableErrorCode}
    member_names.update({member.value: member.name for member in MergedErrorCode})

    referenced: set[str] = set()
    texts = _iter_python_texts(_TESTS_ROOT)
    for code in universe:
        needle_name = member_names.get(code)
        for path, text in texts:
            if path.name == Path(__file__).name:
                continue  # this gate's own tables never count as provocation
            if code in text or (needle_name is not None and needle_name in text):
                referenced.add(code)
                break
    return referenced


def test_universe_is_nonempty() -> None:
    """Anti-vacuity: the scanners find real vocabularies."""

    universe = _vocabulary()
    assert len(universe) > 150, f"vocabulary scan collapsed: {len(universe)} codes"
    assert any(origin == "RunnableErrorCode" for origin in universe.values())
    assert any(origin == "MergedErrorCode" for origin in universe.values())
    assert any(origin == "error_refusal_contract" for origin in universe.values())


def test_every_enum_member_is_reachable_in_source() -> None:
    """No frozen enum member is declaration-only (dead vocabulary)."""

    texts = _iter_python_texts(_PACKAGE_ROOT)
    dead: list[str] = []
    for enum_cls, declaring in (
        (RunnableErrorCode, _PACKAGE_ROOT / "runnable.py"),
        (MergedErrorCode, _PACKAGE_ROOT / "merged" / "_enums.py"),
    ):
        for member in enum_cls:
            reachable = False
            for path, text in texts:
                if path == declaring:
                    continue
                if member.name in text or member.value in text:
                    reachable = True
                    break
            if not reachable:
                dead.append(f"{enum_cls.__name__}.{member.name}")
    assert dead == [], (
        "frozen error-code members with no reference outside their declaration "
        f"(dead vocabulary — raise or remove, with a doc change): {dead}"
    )


def test_every_code_is_provoked_or_consciously_ledgered() -> None:
    """The shrink-only provocation ratchet over the full vocabulary."""

    universe = set(_vocabulary())
    referenced = _test_referenced_codes(universe)
    unprovoked = universe - referenced - set(ENV_GATED_ALLOWLIST)

    newly_unprovoked = unprovoked - UNPROVOKED_BASELINE
    assert newly_unprovoked == set(), (
        "codes with no provoking test reference and no baseline/allowlist row "
        "(new vocabulary must ship provoked): "
        f"{sorted(newly_unprovoked)}"
    )

    stale_baseline = UNPROVOKED_BASELINE - unprovoked
    assert stale_baseline == set(), (
        "baseline rows whose codes are now provoked — delete these rows to "
        f"lock in the ratchet progress: {sorted(stale_baseline)}"
    )

    stale_allowlist = {
        code for code in ENV_GATED_ALLOWLIST if code not in universe or code in referenced
    }
    assert stale_allowlist == set(), (
        f"allowlist rows that are stale (unknown or now provoked): {sorted(stale_allowlist)}"
    )


def test_provocation_scanner_is_red_capable() -> None:
    """A code absent from every test is actually reported (gate can fail)."""

    probe = "zz_probe_code_that_no_test_references_zz"
    referenced = _test_referenced_codes({probe})
    assert referenced == set()
