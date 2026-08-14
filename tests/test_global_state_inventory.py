"""Trust-lane inventory and exception restoration for mutable module globals."""

from __future__ import annotations

import ast
import os
import sys
import threading
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import _state
from torchlens.backends.torch import _tl as torch_tl, completeness_witness, rescue
from torchlens.capture import projections, trace as capture_trace

_SCOPED_CAPTURE_STATE = frozenset(
    {
        ("torchlens/_state.py", "_active_hook_plan"),
        ("torchlens/_state.py", "_active_intervention_spec"),
        ("torchlens/_state.py", "_active_owner_thread_id"),
        ("torchlens/_state.py", "_active_record_spans"),
        ("torchlens/_state.py", "_active_trace"),
        ("torchlens/_state.py", "_capture_replay_templates"),
        ("torchlens/_state.py", "_dynamo_warning_emitted"),
        ("torchlens/_state.py", "_func_call_id_counter"),
        ("torchlens/_state.py", "_function_call_counts"),
        ("torchlens/_state.py", "_function_call_models"),
        ("torchlens/_state.py", "_functorch_warning_emitted"),
        ("torchlens/_state.py", "_logging_enabled"),
        ("torchlens/_state.py", "_relationship_input_id"),
        ("torchlens/_state.py", "_relationship_input_shape_hash"),
        ("torchlens/_state.py", "_relationship_model_class"),
        ("torchlens/_state.py", "_relationship_model_id"),
        ("torchlens/_state.py", "_relationship_weight_fingerprint"),
        ("torchlens/_state.py", "_tagged_buffer_ids"),
        ("torchlens/_trace_core/op_store.py", "_CLONE_SCOPE_DEPTH"),
        ("torchlens/backends/mlx/wrappers.py", "_ACTIVE_TAP_OBSERVER"),
        ("torchlens/backends/paddle/wrappers.py", "_ACTIVE_TAP_OBSERVER"),
        ("torchlens/backends/tinygrad/backend.py", "_ACTIVE_TINYGRAD_MODULE_STACK"),
        ("torchlens/backends/torch/_completeness_finalize.py", "_ACTIVE_WITNESS_STATE"),
        ("torchlens/backends/torch/_tl.py", "_ACTIVE_LABEL_SESSION"),
        ("torchlens/backends/torch/buffer_writes.py", "_WITNESS_MARKER_STATE"),
        ("torchlens/backends/torch/completeness_witness.py", "_CAPTURED_STORAGE_PTRS"),
        ("torchlens/backends/torch/completeness_witness.py", "_DISPATCH_TENSOR_ORIGINS"),
        ("torchlens/backends/torch/completeness_witness.py", "_RUNNABLE_LEDGER_FACTS"),
        ("torchlens/capture/projections.py", "_active_recording_state"),
        ("torchlens/capture/trace.py", "_ACTIVE_CAPTURE_BACKEND"),
        ("torchlens/experimental/__init__.py", "_STOP_AFTER_SITE"),
        ("torchlens/utils/introspection.py", "_FUNC_CALL_LOCATION"),
        ("torchlens/utils/rng.py", "_ACTIVE_MONITOR"),
        # Live monitor patches per (id(holder), name), spliced back on window
        # unwind -- a survivor past monitor exit is exactly the leak class this
        # row exists to catch (d327e3aa's non-LIFO restore bug).
        ("torchlens/utils/rng.py", "_PATCH_STACKS"),
        # Accumulate/drain fence for in-flight cpu_async D2H copies (R36-1):
        # armed per copy on the wrapper hot path, drained at the capture
        # finalize seam and on the failure-scrub arms.
        ("torchlens/utils/tensor_utils.py", "_CPU_ASYNC_PENDING_EVENTS"),
        ("torchlens/utils/tensor_utils.py", "_DEFER_BUSY"),
        ("torchlens/utils/tensor_utils.py", "_DEFER_PENDING"),
        ("torchlens/utils/tensor_utils.py", "_DEFER_STATE_PTRS"),
        ("torchlens/utils/tensor_utils.py", "_DEFER_WINDOW_DEPTH"),
    }
)
"""Per-capture state: set during one capture and cleared or restored on exit.

A leak here is a correctness bug -- the next capture inherits a live owner, an
open session, or a stale window depth. ``test_mid_capture_failure_restores_process_state``
pins the high-risk members against exception and interruption paths.
"""

_INSTALL_STATE_AND_CACHES = frozenset(
    {
        ("torchlens/_state.py", "_decorated_func_mapper"),
        ("torchlens/_state.py", "_decorated_to_orig"),
        ("torchlens/_state.py", "_orig_to_decorated"),
        ("torchlens/_state.py", "_prepared_models"),
        ("torchlens/_state.py", "_prepared_root_by_module"),
        ("torchlens/_state.py", "_stale_prepared_roots"),
        ("torchlens/backends/torch/_tl.py", "_RETIRED_LABEL_SESSION"),
        # F3b's lazy public-impl metadata sync (3c7ed93e): a one-way
        # synced-yet? sentinel flipped on first successful wrap, install-class.
        ("torchlens/user_funcs.py", "_public_impl_metadata_synced"),
        ("torchlens/backends/torch/backward.py", "_AUTOGRAD_WRAPPERS_INSTALLED"),
        ("torchlens/backends/torch/backward.py", "_ORIGINAL_AUTOGRAD_BACKWARD"),
        ("torchlens/backends/torch/backward.py", "_ORIGINAL_AUTOGRAD_GRAD"),
        ("torchlens/backends/torch/backward.py", "_ORIGINAL_SAVED_TENSORS_HOOKS_ENTER"),
        ("torchlens/backends/torch/backward.py", "_ORIGINAL_SAVED_TENSORS_HOOKS_INIT"),
        ("torchlens/backends/torch/backward.py", "_SAVED_TENSORS_HOOKS_INIT_PATCHED"),
        ("torchlens/backends/torch/belt.py", "_ledger"),
        ("torchlens/backends/torch/belt.py", "_member_map"),
        ("torchlens/backends/torch/belt.py", "_report"),
        ("torchlens/backends/torch/belt.py", "_swept_module_ids"),
        # Sweep-epoch bookkeeping companions to _swept_module_ids (8ba75e99):
        # a dead-weakref dirty bit and the sys.modules size at the last
        # complete sweep, reset with the belt install state.
        ("torchlens/backends/torch/belt.py", "_swept_modules_dirty"),
        ("torchlens/backends/torch/belt.py", "_swept_sys_modules_size"),
        ("torchlens/backends/torch/completeness_witness.py", "_AUTHORIZED_INTERNAL_CALLER_CODE"),
        (
            "torchlens/backends/torch/completeness_witness.py",
            "_AUTHORIZED_INTERNAL_CALLER_CODE_IDS",
        ),
        ("torchlens/backends/torch/escape_detection.py", "_TABLES"),
        # (holder, attribute, original) rows for every installed identity shim;
        # popped by the shim uninstall, so it is install bookkeeping, not capture
        # state.
        ("torchlens/backends/torch/identity_shims.py", "_installed"),
        ("torchlens/backends/torch/wrappers.py", "_DEVICE_CONSTRUCTOR_NAMES"),
        ("torchlens/backends/torch/wrappers.py", "_DeviceContext"),
        # One-way "decorate_all_once() ran to COMPLETION" sentinel; deliberately
        # never reset by unwrap_torch() (partial-decoration recovery keys on it).
        ("torchlens/backends/torch/wrappers.py", "_FULL_DECORATION_COMPLETED"),
        ("torchlens/backends/torch/wrappers.py", "_torchvision_ops_ensured"),
        ("torchlens/capture/arg_positions.py", "_schema_corrections_applied"),
        ("torchlens/distributed/_lifecycle.py", "_STATE"),
    }
)
"""Wrapper install / uninstall bookkeeping and prepared-model registration.

Process-lifetime by design: these hold the torch originals, the decorated-callable
id maps, the belt ledger, and the prepared-model registry. They must survive
between captures and be restored by ``unwrap_torch()`` / ``release_model()``,
not reset per capture.
"""

_WARN_ONCE_STATE = frozenset(
    {
        ("torchlens/_capture_state_helpers.py", "_COMPILED_FORCED_EAGER_WARNED"),
        ("torchlens/_capture_state_helpers.py", "_COMPILED_MODEL_UNWRAP_WARNED"),
        ("torchlens/_capture_state_helpers.py", "_VALIDATION_DEEPCOPY_WARNING_TYPES"),
        ("torchlens/_deprecations.py", "_WARNED_DEPRECATIONS"),
        ("torchlens/_io/bundle.py", "_NONPERSISTENT_DISCLOSURE_WARNED"),
        ("torchlens/_io/bundle.py", "_UNATTESTABLE_ACTIVATION_DISCLOSURE_WARNED"),
        ("torchlens/backends/tf/_tf_compat.py", "_warned_missing_capabilities"),
        ("torchlens/backends/torch/buffer_writes.py", "_PARAM_BYTE_WITNESS_NOT_ARMED"),
        ("torchlens/backends/torch/completeness_witness.py", "_HOST_ESCAPE_OBSERVER_FAILED"),
        ("torchlens/backends/torch/ops.py", "_UNSUPPORTED_OUTPUT_CONTAINER_WARNED"),
        ("torchlens/data_classes/op.py", "_WARNED_REFERENCE_SAVE_MODE"),
        ("torchlens/distributed/_lifecycle.py", "_AUTO_ARM_WARNED"),
        ("torchlens/fastlog/_storage_resolver.py", "_WARNED_REFERENCE_SAVE_MODE"),
        ("torchlens/utils/_torch_compat.py", "_warned_missing_capabilities"),
        ("torchlens/utils/introspection.py", "_col_offset_cache_warned"),
        ("torchlens/validation/_stock_layer_grads.py", "_PASS_INDEX_PARSE_WARNED"),
        ("torchlens/visualization/_render_dot.py", "_SIBLING_ORDER_WARNING_EMITTED"),
        ("torchlens/visualization/auto_collapse.py", "_COUNT_MISMATCH_WARNING_EMITTED"),
    }
)
"""Once-per-process disclosure sentinels.

Each suppresses a repeat warning. They are ORDER-COUPLING for tests: whichever
test fires the warning first consumes it, so a suite that asserts on the warning
must reset them (the reset fixture is owned by ``tests/conftest.py``). Any new
sentinel landing here without a reset is an order-dependence bug waiting to
happen.
"""

_CAPABILITY_PROBE_STATE = frozenset(
    {
        ("torchlens/utils/_torch_compat.py", "HAS_C10D_ABORT_PG"),
        ("torchlens/utils/_torch_compat.py", "HAS_DISABLE_TORCH_FUNCTION"),
        ("torchlens/utils/_torch_compat.py", "HAS_DISPATCH_MODE_STACK_QUERY"),
        ("torchlens/utils/_torch_compat.py", "HAS_DTENSOR_SHARD_GEOMETRY"),
        ("torchlens/utils/_torch_compat.py", "HAS_FAKE_TENSOR_MODE"),
        ("torchlens/utils/_torch_compat.py", "HAS_JIT_SCHEMA_ENUMERATION"),
        ("torchlens/utils/_torch_compat.py", "HAS_TENSORBASE_CLASS"),
        ("torchlens/utils/_torch_compat.py", "HAS_VARIABLE_FUNCTIONS_CLASS"),
        ("torchlens/utils/_torch_compat.py", "HAS_C10D_GROUP_REGISTRY"),
        ("torchlens/utils/_torch_compat.py", "HAS_C10D_GROUP_SEQ"),
        ("torchlens/utils/_torch_compat.py", "HAS_DEVICE_MESH"),
        ("torchlens/utils/_torch_compat.py", "HAS_DTENSOR"),
        ("torchlens/utils/_torch_compat.py", "HAS_DYNAMO_COMPILE_COUNTERS"),
        ("torchlens/utils/_torch_compat.py", "HAS_DYNAMO_IS_COMPILING"),
        ("torchlens/utils/_torch_compat.py", "HAS_DYNAMO_OPTIMIZED_MODULE"),
        ("torchlens/utils/_torch_compat.py", "HAS_DYNAMO_ORIG_CALLABLE_MARKER"),
        ("torchlens/utils/_torch_compat.py", "HAS_FP8_DTYPES"),
        ("torchlens/utils/_torch_compat.py", "HAS_FSDP_WRAPPER"),
        ("torchlens/utils/_torch_compat.py", "HAS_PIPELINING"),
        ("torchlens/utils/_torch_compat.py", "HAS_TRACING_TENSOR_TYPES"),
        ("torchlens/utils/_torch_compat.py", "_C10D_ABORT_PG_PROBED"),
        ("torchlens/utils/_torch_compat.py", "_DISABLE_TORCH_FUNCTION_CLS"),
        ("torchlens/utils/_torch_compat.py", "_DISABLE_TORCH_FUNCTION_PROBED"),
        ("torchlens/utils/_torch_compat.py", "_DISPATCH_MODE_STACK_FN"),
        ("torchlens/utils/_torch_compat.py", "_DISPATCH_MODE_STACK_PROBED"),
        ("torchlens/utils/_torch_compat.py", "_C10D_GROUP_REGISTRY_PROBED"),
        ("torchlens/utils/_torch_compat.py", "_C10D_GROUP_SEQ_PROBED"),
        ("torchlens/utils/_torch_compat.py", "_DEVICE_MESH_PROBED"),
        ("torchlens/utils/_torch_compat.py", "_DEVICE_MESH_TYPE"),
        ("torchlens/utils/_torch_compat.py", "_DTENSOR_PROBED"),
        ("torchlens/utils/_torch_compat.py", "_DTENSOR_SHARD_GEOMETRY_FN"),
        ("torchlens/utils/_torch_compat.py", "_DTENSOR_SHARD_GEOMETRY_PROBED"),
        ("torchlens/utils/_torch_compat.py", "_DTENSOR_TYPE"),
        ("torchlens/utils/_torch_compat.py", "_DYNAMO_COMPILE_COUNTERS"),
        ("torchlens/utils/_torch_compat.py", "_DYNAMO_COMPILE_COUNTERS_PROBED"),
        ("torchlens/utils/_torch_compat.py", "_DYNAMO_IS_COMPILING_FN"),
        ("torchlens/utils/_torch_compat.py", "_DYNAMO_IS_COMPILING_PROBED"),
        ("torchlens/utils/_torch_compat.py", "_DYNAMO_OPTIMIZED_MODULE_PROBED"),
        ("torchlens/utils/_torch_compat.py", "_DYNAMO_OPTIMIZED_MODULE_TYPE"),
        ("torchlens/utils/_torch_compat.py", "_DYNAMO_ORIG_CALLABLE_MARKER_PROBED"),
        ("torchlens/utils/_torch_compat.py", "_FAKE_TENSOR_MODE_CLS"),
        ("torchlens/utils/_torch_compat.py", "_FAKE_TENSOR_MODE_PROBED"),
        ("torchlens/utils/_torch_compat.py", "_FP8_DTYPES"),
        ("torchlens/utils/_torch_compat.py", "_FP8_DTYPES_PROBED"),
        ("torchlens/utils/_torch_compat.py", "_FSDP_WRAPPER_PROBED"),
        ("torchlens/utils/_torch_compat.py", "_FSDP_WRAPPER_TYPE"),
        ("torchlens/utils/_torch_compat.py", "_JIT_SCHEMA_ENUMERATION_FN"),
        ("torchlens/utils/_torch_compat.py", "_JIT_SCHEMA_ENUMERATION_PROBED"),
        ("torchlens/utils/_torch_compat.py", "_PIPELINING_PROBED"),
        ("torchlens/utils/_torch_compat.py", "_PIPELINING_TYPES"),
        ("torchlens/utils/_torch_compat.py", "_TENSORBASE_CLASS"),
        ("torchlens/utils/_torch_compat.py", "_TENSORBASE_CLASS_PROBED"),
        ("torchlens/utils/_torch_compat.py", "_TOP_SAVED_TENSORS_DEFAULT_HOOKS_ARGS"),
        ("torchlens/utils/_torch_compat.py", "_TRACING_TENSOR_TYPES"),
        ("torchlens/utils/_torch_compat.py", "_TRACING_TENSOR_TYPES_PROBED"),
        ("torchlens/utils/_torch_compat.py", "_VARIABLE_FUNCTIONS_CLASS"),
        ("torchlens/utils/_torch_compat.py", "_VARIABLE_FUNCTIONS_CLASS_PROBED"),
        ("torchlens/utils/rng.py", "_cuda_rng_unusable"),
        ("torchlens/utils/tensor_utils.py", "_cuda_available"),
    }
)
"""Feature-detection memos for the running torch build.

Written once by a lazy probe and then immutable for the process. They are facts
of the runtime, never capture state, and must never be reset to force a
behavioral branch -- ``CLAUDE.md`` forbids version parsing precisely because
these flags are the sanctioned mechanism.
"""

_DIAGNOSTIC_AUDIT_STATE = frozenset(
    {
        ("torchlens/_trace_core/op_store.py", "_AUDIT_CLONE_READS"),
        ("torchlens/_trace_core/op_store.py", "_AUDIT_COLLECTORS"),
        ("torchlens/_trace_core/op_store.py", "_AUDIT_FINGERPRINTS"),
        ("torchlens/_trace_core/op_store.py", "_AUDIT_READS"),
        ("torchlens/_trace_core/op_store.py", "_AUDIT_ROW_RELEASES"),
        ("torchlens/_trace_core/op_store.py", "_AUDIT_WRITE_EFFECTS"),
        ("torchlens/postprocess/__init__.py", "RECORDED_STEP_CLONE_READS"),
        ("torchlens/postprocess/__init__.py", "RECORDED_STEP_EFFECTIVE_WRITES"),
        ("torchlens/postprocess/__init__.py", "RECORDED_STEP_READS"),
        ("torchlens/postprocess/__init__.py", "RECORDED_STEP_WRITES"),
        # Last-run validation readbacks (B8-42 / R33-2): the internal
        # validation trace never escapes tl.validate, so the first failure
        # and the peak observation mirror into these slots, cleared at each
        # run entry. Diagnostics only -- never part of a verdict.
        ("torchlens/validation/diagnostics.py", "_LAST_RUN_FAILURE"),
        ("torchlens/validation/diagnostics.py", "_LAST_RUN_PEAKS"),
    }
)
"""Diagnostic side-channels: audit instrumentation and last-run readbacks.

The audit rows are armed only by an environment variable and stay empty in
production runs (``TORCHLENS_POSTPROCESS_ASSERTIONS`` /
``TORCHLENS_POSTPROCESS_READ_AUDIT`` off); they accumulate within one armed
window and are scoped by the executor's begin/end pass. The validation
last-run slots are overwritten per run and never steer a verdict.
"""

_WEAK_SUBJECT_TABLES = frozenset(
    {
        ("torchlens/_state.py", "_log_registry"),
        ("torchlens/backends/torch/backward.py", "_BACKWARD_TRACE_SLOTS"),
        ("torchlens/backends/torch/completeness_witness.py", "_ALIAS_MUTATION_CANDIDATE_LABELS"),
        ("torchlens/backends/torch/completeness_witness.py", "_DATA_ALIAS_MUTATION_TRACES"),
        (
            "torchlens/backends/torch/completeness_witness.py",
            "_HOST_ESCAPE_BOOL_CONSUMER_LOCATIONS",
        ),
        ("torchlens/backends/torch/completeness_witness.py", "_HOST_ESCAPE_BOOL_SOURCE_LABELS"),
        ("torchlens/backends/torch/completeness_witness.py", "_HOST_ESCAPE_CROSS_THREAD_CAPTURED"),
        ("torchlens/backends/torch/completeness_witness.py", "_HOST_ESCAPE_LABEL_LEAF_ORIGINS"),
        ("torchlens/backends/torch/completeness_witness.py", "_HOST_ESCAPE_MUTABLE_WRITEBACK"),
        ("torchlens/backends/torch/completeness_witness.py", "_HOST_ESCAPE_RAW_POINTER"),
        ("torchlens/backends/torch/completeness_witness.py", "_HOST_ESCAPE_SOURCE_LABELS"),
        (
            "torchlens/backends/torch/completeness_witness.py",
            "_HOST_ESCAPE_STATE_METADATA_OBSERVATIONS",
        ),
        ("torchlens/backends/torch/completeness_witness.py", "_HOST_ESCAPE_STATE_METADATA_READS"),
        ("torchlens/backends/torch/completeness_witness.py", "_HOST_ESCAPE_STATE_SOURCE_LABELS"),
        ("torchlens/backends/torch/completeness_witness.py", "_HOST_ESCAPE_STATE_SOURCE_NAMES"),
        ("torchlens/backends/torch/completeness_witness.py", "_HOST_ESCAPE_UNATTRIBUTABLE_BOOL"),
        ("torchlens/backends/torch/completeness_witness.py", "_HOST_ESCAPE_UNATTRIBUTABLE_OPAQUE"),
        ("torchlens/backends/torch/completeness_witness.py", "_INPUT_METADATA_VIEW_READ"),
        ("torchlens/backends/torch/completeness_witness.py", "_LAYOUT_ANCESTRY_CLEAN"),
        ("torchlens/backends/torch/completeness_witness.py", "_PRUNED_ALIAS_MUTATION_LABELS"),
        ("torchlens/backends/torch/completeness_witness.py", "_PRUNED_RNG_CONTROL_LABELS"),
        ("torchlens/backends/torch/completeness_witness.py", "_RUNNABLE_INPUT_STORAGE_SITES"),
        ("torchlens/backends/torch/completeness_witness.py", "_STATE_METADATA_FACTS"),
        ("torchlens/backends/torch/completeness_witness.py", "_STORAGE_REBIND_BARRIER_LABELS"),
        # Per-module-namespace container-slot memo (8ba75e99), keyed weakly by
        # the module object with a len(namespace)-based staleness check.
        ("torchlens/backends/torch/model_prep.py", "_module_namespace_container_slots"),
        ("torchlens/backends/torch/model_prep.py", "_source_line_cache"),
        # Implicit-backward task ordinals keyed weakly by their owning trace;
        # entries die with the trace.
        ("torchlens/backends/torch/tensor_tracking.py", "_IMPLICIT_BACKWARD_TASK_IDS"),
        ("torchlens/backends/torch/wrappers.py", "_COW_STATE_PTRS_CACHE"),
        ("torchlens/data_classes/_compaction.py", "_COMPACTED_TRACES"),
        ("torchlens/data_classes/_nonfinite.py", "_MEMOS"),
        ("torchlens/partial/__init__.py", "_FAILED_CAPTURE_RESULTS"),
        ("torchlens/visualization/auto_collapse.py", "_ANALYSIS_CACHE"),
        ("torchlens/visualization/auto_collapse.py", "_OP_ADJACENCY_INDEX_CACHE"),
        ("torchlens/visualization/code_panel.py", "_SOURCE_MEMO"),
        ("torchlens/visualization/collapse_optimizer.py", "_BOX_UNITS_CACHE"),
        ("torchlens/visualization/collapse_optimizer.py", "_RESULT_CACHE"),
        ("torchlens/visualization/collapse_optimizer.py", "_SCHEDULE_CACHE"),
    }
)
"""Side tables keyed WEAKLY by their subject (trace, tensor, model, code).

Entries die with the subject, so these can neither pin memory nor leak state
across captures. The test below verifies mechanically that every member really
is bound to a ``weakref`` container -- a member that silently becomes a strong
dict would otherwise keep its whole subject graph alive.
"""

_PUBLIC_REGISTRATION_STATE = frozenset(
    {
        ("torchlens/backends/registry.py", "_REGISTRY"),
        ("torchlens/capture/flops.py", "_CUSTOM_OP_RULES"),
        ("torchlens/ir/container.py", "_CONTAINER_REGISTRY"),
        ("torchlens/receptive_field/_rules.py", "_BUILTIN_RF_RULES"),
        ("torchlens/receptive_field/_rules.py", "_RF_RULES"),
        ("torchlens/receptive_field/_rules.py", "_RF_RULES_EPOCH"),
        ("torchlens/semantic/facets.py", "_BUILTIN_REGISTRY"),
        ("torchlens/semantic/facets.py", "_REGISTRY"),
        ("torchlens/semantic/facets.py", "_REGISTRY_VERSION"),
        ("torchlens/semantic/facets.py", "_TRANSFORMERLENS_ALIASES_ENABLED"),
    }
)
"""Process state a PUBLIC API mutates: registries, rule tables, feature toggles.

Distinct from a cache because a user call CHANGES capture behavior for the rest
of the process (a registered facet, an RF rule, an enabled alias set). These are
the members whose leakage across a test session can silently change results, so
they carry an epoch/version counter where downstream caches must invalidate.
"""

_PROCESS_CACHES = frozenset(
    {
        ("torchlens/_input_walk.py", "_STOCK_NP_SCALAR_CACHE"),
        ("torchlens/_io/bundle.py", "_NESTED_BLOB_KINDS"),
        ("torchlens/_io/payload_codec.py", "_CODECS"),
        ("torchlens/_io/rehydrate.py", "_REHYDRATE_KINDS"),
        ("torchlens/_io/runnable.py", "_DATACLASS_FIELD_NAMES"),
        ("torchlens/_io/runnable.py", "_SPARSE_CORE_NODE_KINDS"),
        ("torchlens/_io/runnable.py", "_TORCH_SYMBOL_NAMES"),
        ("torchlens/_io/runnable.py", "_TORCH_SYMBOL_NAMESPACE_SIZE"),
        ("torchlens/_io/scrub.py", "_SCRUB_VALUE_KINDS"),
        ("torchlens/_io/state_keys.py", "_CACHE_GENERATION"),
        ("torchlens/_state.py", "_arg_names"),
        ("torchlens/_state.py", "_dir_cache"),
        ("torchlens/_state.py", "_dynamic_arg_specs"),
        ("torchlens/_state.py", "_naming_counters"),
        ("torchlens/_training_validation.py", "_NON_GRAD_DTYPES"),
        ("torchlens/backends/torch/backward.py", "_BACKWARD_GRAD_FN_REGISTRY"),
        ("torchlens/backends/torch/completeness_witness.py", "_FRAMEWORK_FILENAME_VERDICTS"),
        ("torchlens/backends/torch/model_prep.py", "_module_class_metadata_cache"),
        ("torchlens/backends/torch/ops.py", "_CAPTURE_PRODUCER_POLICIES"),
        ("torchlens/capture/arg_positions.py", "FUNC_ARG_SPECS"),
        ("torchlens/capture/projections.py", "_CAPTURE_POLICY_CACHE"),
        ("torchlens/capture/projectors.py", "_REFRESH_SOURCES"),
        ("torchlens/capture/salient_args.py", "_EXTRACTORS"),
        ("torchlens/constants.py", "_TORCHVISION_FUNCS_CACHE"),
        ("torchlens/data_classes/op.py", "_RELATION_CELL_ENCODINGS"),
        ("torchlens/data_classes/trace.py", "_MODEL_LOG_DEFAULT_FILL"),
        ("torchlens/partial/__init__.py", "_FAILED_CAPTURE_REGISTRY"),
        ("torchlens/postprocess/ast_branches.py", "_file_cache"),
        ("torchlens/receptive_field/_engine.py", "_SCHEMA_OPERAND_SLOTS_CACHE"),
        ("torchlens/utils/introspection.py", "_COL_OFFSET_CACHE"),
        # Import-time derived ULP tolerance table, lazily extended for dtypes
        # outside _REPLAY_ULP_HEADROOM; clearing only re-derives (pure finfo
        # arithmetic), so it is a memo, not capability state.
        ("torchlens/utils/tensor_utils.py", "_DTYPE_FLOAT_TOLERANCES"),
        # Fork-inheritance discriminator for warn_parallel (r-b6 R40-3b): the
        # PID that first observed an initialized process group. Clearing it
        # only re-stamps on the next capture entry; it never steers anything
        # but the child-process refusal.
        ("torchlens/utils/display.py", "_DIST_GROUP_OBSERVED_PID"),
        # Live viewer child handles (r-b6 R40-1): retained solely so each
        # launch can reap already-exited viewers. Clearing it costs at most
        # one unreaped zombie per cleared entry until process exit — hygiene,
        # never correctness.
        ("torchlens/visualization/_render_utils.py", "_VIEWER_PROCS"),
    }
)
"""Process-lifetime memos holding strong references.

Correctness-neutral (a cleared cache only costs time) but they are the class
that PINS objects, so each strong key/value must be justified: prefer a weak
table when the key is a user object, and clear the cache in the capture epilogue
when its entries are session-scoped (see ``_module_class_metadata_cache``).
"""

MUTATING_METHODS = frozenset(
    {
        "append",
        "appendleft",
        "add",
        "clear",
        "discard",
        "extend",
        "insert",
        "move_to_end",
        "pop",
        "popitem",
        "popleft",
        "remove",
        "setdefault",
        "sort",
        "update",
        "__setitem__",
    }
)
"""Method names whose call mutates a container in place."""

MUTABLE_FACTORIES = frozenset(
    {
        "ChainMap",
        "Counter",
        "OrderedDict",
        "WeakKeyDictionary",
        "WeakSet",
        "WeakValueDictionary",
        "defaultdict",
        "deque",
        "dict",
        "list",
        "set",
    }
)
"""Callables whose result is a mutable container."""


def _package_python_paths(repo: Path) -> list[Path]:
    """Return every Python source file in the package.

    The inventory used to scan only ``_state.py`` plus ``capture/``,
    ``validation/`` and ``backends/torch/``, which governed a MINORITY of the
    surface it claimed: the process-global per-capture tap observers in the
    mlx/paddle backends, the mutable facet toggle, the RF rule registry, the
    warn-once sentinels in ``_io``/``visualization``/``distributed`` and the
    whole ``_torch_compat`` capability block were all out of scope, so a new
    unclassified global could land in them without failing anything.

    Parameters
    ----------
    repo:
        Repository root.

    Returns
    -------
    list[Path]
        Sorted package Python source paths.
    """

    return sorted((repo / "torchlens").rglob("*.py"))


def _module_level_mutable_bindings(tree: ast.Module) -> dict[str, str]:
    """Return module-level names bound to a mutable container, with their source.

    Parameters
    ----------
    tree:
        Parsed module.

    Returns
    -------
    dict[str, str]
        Name -> unparsed initializer for each module-level mutable binding.
    """

    bindings: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        mutable = isinstance(
            value, (ast.Dict, ast.Set, ast.List, ast.DictComp, ast.SetComp, ast.ListComp)
        )
        if isinstance(value, ast.Call):
            factory = value.func
            factory_name = (
                factory.id
                if isinstance(factory, ast.Name)
                else factory.attr
                if isinstance(factory, ast.Attribute)
                else None
            )
            mutable = mutable or factory_name in MUTABLE_FACTORIES
        if not mutable:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                bindings[target.id] = ast.unparse(value)
    return bindings


def _names_mutated_in_place(trees: dict[str, ast.Module]) -> set[str]:
    """Return every name mutated in place anywhere in the package.

    Both spellings count, because module state is routinely mutated through an
    imported module alias: a bare ``_CACHE[key] = value`` in the owning module
    AND an ``_state._dir_cache[key] = value`` from another one. The attribute
    spelling is matched by ATTRIBUTE NAME, which can over-include a same-named
    attribute on an unrelated object; over-inclusion only ever asks for one more
    classification row, whereas under-inclusion is the blind spot this closes.

    Parameters
    ----------
    trees:
        Relative path -> parsed module for the whole package.

    Returns
    -------
    set[str]
        Names observed under an in-place mutation.
    """

    mutated: set[str] = set()
    for tree in trees.values():
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.Delete)):
                targets = (
                    node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
                )
                for target in targets:
                    if not isinstance(target, ast.Subscript):
                        continue
                    base = target.value
                    if isinstance(base, ast.Name):
                        mutated.add(base.id)
                    elif isinstance(base, ast.Attribute):
                        mutated.add(base.attr)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in MUTATING_METHODS
            ):
                base = node.func.value
                if isinstance(base, ast.Name):
                    mutated.add(base.id)
                elif isinstance(base, ast.Attribute):
                    mutated.add(base.attr)
    return mutated


def _mutable_module_state(repo: Path) -> dict[tuple[str, str], str]:
    """Return every mutable module global in the package, with its initializer.

    Two detectors, because either alone has a structural blind spot:

    * ``global`` declarations catch REBINDING (``_flag = True``) but can never
      see a container mutated in place -- ``_CACHE[key] = value`` needs no
      ``global`` statement at all, so the whole cache/registry class was
      invisible to the previous gate.
    * Module-level mutable bindings catch the container class, qualified by
      package-wide evidence that something actually mutates them, so frozen
      lookup tables are not dragged in.

    Parameters
    ----------
    repo:
        Repository root.

    Returns
    -------
    dict[tuple[str, str], str]
        ``(relative path, name)`` -> unparsed initializer (empty for names known
        only from a ``global`` declaration).
    """

    trees = {
        path.relative_to(repo).as_posix(): ast.parse(path.read_text())
        for path in _package_python_paths(repo)
    }
    mutated_names = _names_mutated_in_place(trees)
    state: dict[tuple[str, str], str] = {}
    for relative, tree in trees.items():
        bindings = _module_level_mutable_bindings(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Global):
                for name in node.names:
                    state.setdefault((relative, name), bindings.get(name, ""))
        for name, initializer in bindings.items():
            if name in mutated_names:
                state[(relative, name)] = initializer
    return state


def _capture_scope_snapshot() -> dict[str, Any]:
    """Return high-risk per-capture global state for restoration checks.

    Returns
    -------
    dict[str, Any]
        Values that must be identical before and after a failed capture.
    """

    return {
        "logging_enabled": _state._logging_enabled,
        "active_trace": _state._active_trace,
        "active_owner_thread_id": _state._active_owner_thread_id,
        "nonowner_belt_armed": _state._nonowner_belt_armed,
        "active_fast_run_collector": _state._active_fast_run_collector,
        "active_hook_plan": _state._active_hook_plan,
        "active_intervention_spec": _state._active_intervention_spec,
        "capture_replay_templates": _state._capture_replay_templates,
        "relationship_model_id": _state._relationship_model_id,
        "relationship_model_class": _state._relationship_model_class,
        "relationship_weight_fingerprint": _state._relationship_weight_fingerprint,
        "relationship_input_id": _state._relationship_input_id,
        "relationship_input_shape_hash": _state._relationship_input_shape_hash,
        "runnable_ledger_armed": _state._runnable_ledger_armed,
        "active_label_session": torch_tl._ACTIVE_LABEL_SESSION,
        "active_witness_state": completeness_witness._ACTIVE_WITNESS_STATE,
        "rescue_active": rescue._rescue_is_active(),
        "active_recording_state": projections._active_recording_state,
        "active_capture_backend": capture_trace._ACTIVE_CAPTURE_BACKEND,
    }


class _InjectedBaseFailure(BaseException):
    """BaseException subclass used to exercise the interruption cleanup arm."""


class _RaiseMidCapture(nn.Module):
    """Run one logged op before raising a selected exception class."""

    def __init__(self, error_type: type[BaseException]) -> None:
        """Store the exception type raised during ``forward``.

        Parameters
        ----------
        error_type:
            Exception class to raise after one tensor operation.
        """

        super().__init__()
        self.error_type = error_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one operation and then raise the injected exception.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            This path never returns.

        Raises
        ------
        BaseException
            Always raises ``self.error_type`` after the logged operation.
        """

        _ = torch.relu(x)
        raise self.error_type("injected mid-capture failure")


class _BlockingCapture(nn.Module):
    """Hold a public capture open while a concurrent capture is attempted."""

    def __init__(self, entered: threading.Event, release: threading.Event) -> None:
        """Store synchronization events for the capture overlap.

        Parameters
        ----------
        entered:
            Event set after the first logged operation runs.
        release:
            Event that permits the model to finish.
        """

        super().__init__()
        self.entered = entered
        self.release = release

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Log one operation and wait until the contender is refused.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Logged activation after the release event is set.
        """

        out = torch.relu(x)
        self.entered.set()
        if not self.release.wait(timeout=5.0):
            raise TimeoutError("concurrent capture test did not release the owner")
        return out


class _PauseFromForeignThread(nn.Module):
    """Log ops before and after a NON-OWNER thread enters ``pause_logging()``.

    The foreign thread models the reachable-without-concurrent-capture case: a
    thread merely ANALYZING an older Trace (``tl.save``, validation, an ``.out``
    transform) enters the same process-global pause the owner's forward relies on.

    The foreign pause is HELD OPEN across the owner's second op group, which is
    the realistic shape: an analysis thread's ``pause_logging()`` body spans a
    whole save / validation pass, not a single statement.
    """

    def __init__(self, ops_per_side: int) -> None:
        """Store how many logged ops run on each side of the foreign pause.

        Parameters
        ----------
        ops_per_side:
            Number of logged operations before, during, and after the pause.
        """

        super().__init__()
        self.ops_per_side = ops_per_side
        self.foreign_paused = threading.Event()
        self.foreign_release = threading.Event()
        self.foreign_error: list[BaseException] = []

    def _foreign_pause(self) -> None:
        """Hold ``pause_logging()`` open from a non-owner thread."""

        try:
            with _state.pause_logging():
                self.foreign_paused.set()
                if not self.foreign_release.wait(timeout=10.0):
                    raise TimeoutError("owner never released the foreign pause")
        except BaseException as error:  # pragma: no cover - reported by the test
            self.foreign_error.append(error)
        finally:
            self.foreign_paused.set()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run ops before, during, and after a held foreign pause.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Activation after all three op groups ran.
        """

        for _ in range(self.ops_per_side):
            x = torch.relu(x)
        foreign = threading.Thread(target=self._foreign_pause)
        foreign.start()
        assert self.foreign_paused.wait(timeout=10.0), "foreign pause never entered"
        # These ops run while a NON-OWNER thread holds the pause open.
        for _ in range(self.ops_per_side):
            x = torch.relu(x)
        self.foreign_release.set()
        foreign.join(timeout=10.0)
        assert not foreign.is_alive(), "foreign pause thread did not exit"
        # And these run after the foreign thread restored what it saved.
        for _ in range(self.ops_per_side):
            x = torch.relu(x)
        return x


_WEAKLY_HELD = frozenset(
    {
        ("torchlens/_capture_state_helpers.py", "_VALIDATION_DEEPCOPY_WARNING_TYPES"),
        # Kind memos re-keyed weakly by the value TYPE so dynamically created
        # classes stay collectable; lifecycle class stays _PROCESS_CACHES (the
        # ledger is orthogonal: weakness is a storage fact).
        ("torchlens/_io/rehydrate.py", "_REHYDRATE_KINDS"),
        ("torchlens/_io/scrub.py", "_SCRUB_VALUE_KINDS"),
        ("torchlens/_state.py", "_log_registry"),
        ("torchlens/_state.py", "_prepared_models"),
        ("torchlens/_state.py", "_prepared_root_by_module"),
        ("torchlens/_state.py", "_stale_prepared_roots"),
        ("torchlens/backends/torch/backward.py", "_BACKWARD_TRACE_SLOTS"),
        ("torchlens/backends/torch/buffer_writes.py", "_PARAM_BYTE_WITNESS_NOT_ARMED"),
        ("torchlens/backends/torch/completeness_witness.py", "_ALIAS_MUTATION_CANDIDATE_LABELS"),
        ("torchlens/backends/torch/completeness_witness.py", "_CAPTURED_STORAGE_PTRS"),
        ("torchlens/backends/torch/completeness_witness.py", "_DATA_ALIAS_MUTATION_TRACES"),
        ("torchlens/backends/torch/completeness_witness.py", "_DISPATCH_TENSOR_ORIGINS"),
        (
            "torchlens/backends/torch/completeness_witness.py",
            "_HOST_ESCAPE_BOOL_CONSUMER_LOCATIONS",
        ),
        ("torchlens/backends/torch/completeness_witness.py", "_HOST_ESCAPE_BOOL_SOURCE_LABELS"),
        ("torchlens/backends/torch/completeness_witness.py", "_HOST_ESCAPE_CROSS_THREAD_CAPTURED"),
        ("torchlens/backends/torch/completeness_witness.py", "_HOST_ESCAPE_LABEL_LEAF_ORIGINS"),
        ("torchlens/backends/torch/completeness_witness.py", "_HOST_ESCAPE_MUTABLE_WRITEBACK"),
        ("torchlens/backends/torch/completeness_witness.py", "_HOST_ESCAPE_OBSERVER_FAILED"),
        ("torchlens/backends/torch/completeness_witness.py", "_HOST_ESCAPE_RAW_POINTER"),
        ("torchlens/backends/torch/completeness_witness.py", "_HOST_ESCAPE_SOURCE_LABELS"),
        (
            "torchlens/backends/torch/completeness_witness.py",
            "_HOST_ESCAPE_STATE_METADATA_OBSERVATIONS",
        ),
        ("torchlens/backends/torch/completeness_witness.py", "_HOST_ESCAPE_STATE_METADATA_READS"),
        ("torchlens/backends/torch/completeness_witness.py", "_HOST_ESCAPE_STATE_SOURCE_LABELS"),
        ("torchlens/backends/torch/completeness_witness.py", "_HOST_ESCAPE_STATE_SOURCE_NAMES"),
        ("torchlens/backends/torch/completeness_witness.py", "_HOST_ESCAPE_UNATTRIBUTABLE_BOOL"),
        ("torchlens/backends/torch/completeness_witness.py", "_HOST_ESCAPE_UNATTRIBUTABLE_OPAQUE"),
        ("torchlens/backends/torch/completeness_witness.py", "_INPUT_METADATA_VIEW_READ"),
        ("torchlens/backends/torch/completeness_witness.py", "_LAYOUT_ANCESTRY_CLEAN"),
        ("torchlens/backends/torch/completeness_witness.py", "_PRUNED_ALIAS_MUTATION_LABELS"),
        ("torchlens/backends/torch/completeness_witness.py", "_PRUNED_RNG_CONTROL_LABELS"),
        ("torchlens/backends/torch/completeness_witness.py", "_RUNNABLE_INPUT_STORAGE_SITES"),
        ("torchlens/backends/torch/completeness_witness.py", "_RUNNABLE_LEDGER_FACTS"),
        ("torchlens/backends/torch/completeness_witness.py", "_STATE_METADATA_FACTS"),
        ("torchlens/backends/torch/completeness_witness.py", "_STORAGE_REBIND_BARRIER_LABELS"),
        ("torchlens/backends/torch/model_prep.py", "_module_namespace_container_slots"),
        ("torchlens/backends/torch/model_prep.py", "_source_line_cache"),
        ("torchlens/backends/torch/tensor_tracking.py", "_IMPLICIT_BACKWARD_TASK_IDS"),
        ("torchlens/backends/torch/wrappers.py", "_COW_STATE_PTRS_CACHE"),
        ("torchlens/data_classes/_compaction.py", "_COMPACTED_TRACES"),
        ("torchlens/data_classes/_nonfinite.py", "_MEMOS"),
        ("torchlens/partial/__init__.py", "_FAILED_CAPTURE_RESULTS"),
        ("torchlens/visualization/auto_collapse.py", "_ANALYSIS_CACHE"),
        ("torchlens/visualization/auto_collapse.py", "_OP_ADJACENCY_INDEX_CACHE"),
        ("torchlens/visualization/code_panel.py", "_SOURCE_MEMO"),
        ("torchlens/visualization/collapse_optimizer.py", "_BOX_UNITS_CACHE"),
        ("torchlens/visualization/collapse_optimizer.py", "_RESULT_CACHE"),
        ("torchlens/visualization/collapse_optimizer.py", "_SCHEDULE_CACHE"),
    }
)
"""Every inventory member whose container is bound WEAKLY, across all classes.

Orthogonal to the lifecycle classes on purpose: weakness is a storage fact, and
``_prepared_models`` is both a weak registry AND install state, while
``_VALIDATION_DEEPCOPY_WARNING_TYPES`` is both weak AND a warn-once sentinel.
Freezing the weak set separately means a member that silently changes from
``WeakKeyDictionary()`` to ``{}`` -- and so starts pinning its whole subject
graph -- fails a gate instead of hiding behind a reassuring lifecycle row.
"""


_LIFECYCLE_CLASSES = (
    _SCOPED_CAPTURE_STATE,
    _INSTALL_STATE_AND_CACHES,
    _WARN_ONCE_STATE,
    _CAPABILITY_PROBE_STATE,
    _DIAGNOSTIC_AUDIT_STATE,
    _WEAK_SUBJECT_TABLES,
    _PUBLIC_REGISTRATION_STATE,
    _PROCESS_CACHES,
)
"""Every lifecycle class, in declaration order. The union must be exact."""


@pytest.mark.smoke
def test_global_state_inventory_is_classified_and_shrink_only() -> None:
    """Every mutable module global in the PACKAGE has exactly one lifecycle class.

    Scope and detection are both wider than they were. The inventory covered
    ``_state.py`` plus three subpackages, and classified only ``global``
    declarations -- so the majority of the surface it claimed to govern was
    unreachable by it, and the entire mutated-in-place class (``_CACHE[key] =
    value`` needs no ``global`` statement) was structurally invisible even
    in-lane. See ``_package_python_paths`` and ``_mutable_module_state``.
    """

    repo = Path(__file__).resolve().parents[1]
    classified = set().union(*_LIFECYCLE_CLASSES)

    assert sum(len(category) for category in _LIFECYCLE_CLASSES) == len(classified), (
        "global lifecycle classes overlap"
    )
    observed = _mutable_module_state(repo)
    missing = sorted(set(observed) - classified)
    stale = sorted(classified - set(observed))
    assert not missing, (
        f"unclassified mutable module state (assign it a lifecycle class above): {missing}"
    )
    assert not stale, f"inventory rows no longer present in the package: {stale}"


@pytest.mark.smoke
def test_weakly_held_state_is_exactly_the_declared_ledger() -> None:
    """The weak/strong split of every inventory member is frozen and exact.

    This is what keeps the inventory from being a rubber stamp. A lifecycle row
    says what a global is FOR; this says how it HOLDS its subjects, which is the
    difference between a side table that dies with its trace and one that pins
    every captured graph in the process. Both directions are checked, so neither
    weakening nor strengthening a container can pass unreviewed.
    """

    repo = Path(__file__).resolve().parents[1]
    observed = _mutable_module_state(repo)
    weakly_held = {
        entry for entry, initializer in observed.items() if "weakref.Weak" in initializer
    }

    became_strong = sorted(_WEAKLY_HELD - weakly_held)
    became_weak = sorted(weakly_held - _WEAKLY_HELD)
    assert not became_strong, (
        "declared-weak module state is no longer bound to a weakref container "
        f"(it now pins its subjects): {became_strong}"
    )
    assert not became_weak, (
        "module state became weak without updating the ledger (welcome, but the "
        f"row has to move): {became_weak}"
    )
    assert set().union(*_LIFECYCLE_CLASSES) >= _WEAKLY_HELD, (
        "weak-ledger rows missing from every lifecycle class"
    )
    assert _WEAK_SUBJECT_TABLES <= _WEAKLY_HELD, (
        "a member of the weak-subject-table CLASS is not in the weak ledger"
    )


@pytest.mark.parametrize("error_type", [RuntimeError, _InjectedBaseFailure])
def test_mid_capture_failure_restores_process_state(
    error_type: type[BaseException],
) -> None:
    """Ordinary and interruption failures leave no stale capture owner.

    Parameters
    ----------
    error_type:
        Failure class injected after one recorded operation.
    """

    tl.trace(nn.ReLU(), torch.ones(2))
    before = _capture_scope_snapshot()

    with pytest.raises(error_type, match="injected mid-capture failure"):
        tl.trace(_RaiseMidCapture(error_type), torch.ones(2))

    assert _capture_scope_snapshot() == before
    recovered = tl.trace(nn.ReLU(), torch.ones(2))
    assert any(op.func_name == "relu" for op in recovered.compute_ops)


def test_concurrent_public_capture_refuses_without_corruption() -> None:
    """Overlapping public captures fail loudly and leave the owner intact."""

    tl.trace(nn.ReLU(), torch.ones(2))
    before = _capture_scope_snapshot()
    entered = threading.Event()
    release = threading.Event()
    owner_errors: list[BaseException] = []

    def run_owner() -> None:
        """Run the capture that owns process-global logging state."""

        try:
            tl.trace(_BlockingCapture(entered, release), torch.ones(2))
        except BaseException as error:
            owner_errors.append(error)

    owner = threading.Thread(target=run_owner)
    owner.start()
    assert entered.wait(timeout=5.0), "owner capture never reached its forward"
    try:
        with pytest.raises(_state.ReentrantTraceError, match="not re-entrant"):
            tl.trace(nn.ReLU(), torch.ones(2))
    finally:
        release.set()
        owner.join(timeout=5.0)

    assert not owner.is_alive(), "owner capture did not finish after release"
    assert owner_errors == []
    assert _capture_scope_snapshot() == before


def test_foreign_thread_pause_does_not_blind_the_owner_capture() -> None:
    """A non-owner ``pause_logging()`` never drops the owner's later ops.

    ``_PauseLogging.__enter__`` used to clear the process-global toggle
    unconditionally, so ANY thread pausing (even one only analyzing an old
    Trace) silently truncated a live capture from that instant on. The owner
    check now lives in the context manager itself, covering every call site.
    """

    ops_per_side = 3
    model = _PauseFromForeignThread(ops_per_side)
    trace = tl.trace(model, torch.ones(2))

    assert model.foreign_error == [], f"foreign pause raised {model.foreign_error!r}"
    relu_ops = [op for op in trace.compute_ops if op.func_name == "relu"]
    assert len(relu_ops) == 3 * ops_per_side, (
        "ops logged while a non-owner thread held pause_logging() are missing: "
        f"the foreign pause blinded the capture (saw {len(relu_ops)} of "
        f"{3 * ops_per_side} relu ops)"
    )
    assert _state._logging_enabled is False
    assert _state._active_owner_thread_id is None


def test_capture_admission_runs_under_the_admission_lock() -> None:
    """Admission blocks while another thread holds the admission lock.

    ``active_logging``'s refusal check and its publication of ``_active_trace`` /
    ``_active_owner_thread_id`` / ``_logging_enabled`` are separate bytecodes.
    Unlocked, two threads entering together can both pass the check, and the
    loser then overwrites the winner's owner id — after which every op the winner
    logs is dropped by the wrapper's owner-thread fast path and its Trace is
    silently short, with no error anywhere. The check and the publication
    therefore have to happen under one lock; this asserts that directly, because
    the racing window itself is only a few bytecodes wide and a probabilistic
    probe cannot gate it reliably.
    """

    before = _capture_scope_snapshot()
    entered = threading.Event()
    blocked_for_lock = threading.Event()
    admitted = threading.Event()
    failures: list[BaseException] = []

    def admit() -> None:
        """Enter and immediately leave one capture session."""

        try:
            with _state.active_logging(cast("Any", object())):
                admitted.set()
        except BaseException as error:  # pragma: no cover - reported by the test
            failures.append(error)
            admitted.set()

    with _state._capture_admission_lock:
        entered.set()
        worker = threading.Thread(target=admit)
        worker.start()
        # The worker cannot reach the check, let alone publish, while the lock
        # is held here. If admission ran outside the lock it would sail through.
        blocked_for_lock.wait(timeout=0.5)
        assert not admitted.is_set(), (
            "active_logging admitted a capture while the admission lock was "
            "held: the check-then-publish sequence is not serialized"
        )
        assert _state._active_trace is None
        assert _state._active_owner_thread_id is None

    assert admitted.wait(timeout=10.0), "admission never completed after release"
    worker.join(timeout=10.0)
    assert not worker.is_alive(), "admission worker hung"
    assert failures == [], f"admission failed after the lock was released: {failures!r}"
    assert _capture_scope_snapshot() == before


def test_contended_admission_never_publishes_partial_owner_state() -> None:
    """Under contention, an admitted session owns the globals for its whole body.

    Complements the lock test above: whatever the interleaving, a thread that is
    admitted must see its OWN owner id and trace for the entire session, and
    every other thread must get the documented refusal rather than a corrupted
    half-published state. Concurrent capture stays unsupported by design; this
    pins the admission mechanism's behavior under contention.
    """

    before = _capture_scope_snapshot()
    contenders = 4
    rounds = 25
    admitted = 0
    refused = 0
    stolen: list[tuple[int, int | None]] = []
    unexpected: list[BaseException] = []
    lock = threading.Lock()
    # Bytecode-level interleaving is what the admission lock excludes; make the
    # scheduler switch as often as possible so contention is real here.
    prior_switch_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        for _ in range(rounds):
            barrier = threading.Barrier(contenders)

            def contend() -> None:
                """Enter ``active_logging`` at the same instant as the others."""

                nonlocal admitted, refused
                token = object()
                barrier.wait(timeout=10.0)
                try:
                    with _state.active_logging(cast("Any", token)):
                        mine = threading.get_ident()
                        for _ in range(200):
                            owner = _state._active_owner_thread_id
                            if owner != mine or _state._active_trace is not token:
                                with lock:
                                    stolen.append((mine, owner))
                                break
                except _state.ReentrantTraceError:
                    with lock:
                        refused += 1
                except BaseException as error:  # pragma: no cover - test signal
                    with lock:
                        unexpected.append(error)
                else:
                    with lock:
                        admitted += 1

            threads = [threading.Thread(target=contend) for _ in range(contenders)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=20.0)
            assert not any(thread.is_alive() for thread in threads), "contender hung"
    finally:
        sys.setswitchinterval(prior_switch_interval)

    assert unexpected == [], f"unexpected admission failure: {unexpected!r}"
    assert stolen == [], (
        "an admitted capture's owner globals were overwritten by a racing "
        f"contender (mine, observed_owner) pairs: {stolen!r}"
    )
    assert admitted + refused == contenders * rounds
    assert admitted >= 1, "no capture was admitted at all"
    assert _capture_scope_snapshot() == before


def test_unwrap_torch_refuses_during_an_active_capture() -> None:
    """Mid-capture ``unwrap_torch()`` is a typed refusal, not a silent truncation.

    Reachable single-threaded: from a forward hook, an ``activation_transform``,
    or any user callback running inside the traced forward. Removing the
    wrappers there left the rest of the forward unlogged and returned a
    truncated Trace with no error at all.
    """

    from torchlens.backends.torch.wrappers import unwrap_torch

    seen: list[BaseException] = []

    class _UnwrapMidForward(nn.Module):
        """Attempt an unwrap between two logged operations."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run one op, try to unwrap, then run another op.

            Parameters
            ----------
            x:
                Input tensor.

            Returns
            -------
            torch.Tensor
                Activation after both operations.
            """

            x = torch.relu(x)
            try:
                unwrap_torch()
            except BaseException as error:
                seen.append(error)
            return torch.relu(x)

    trace = tl.trace(_UnwrapMidForward(), torch.ones(2))

    assert len(seen) == 1, "unwrap_torch() mid-capture did not refuse"
    error = seen[0]
    assert isinstance(error, tl.errors.CaptureContextError)
    assert error.fields["code"] == "unwrap_during_active_capture"
    relu_ops = [op for op in trace.compute_ops if op.func_name == "relu"]
    assert len(relu_ops) == 2, "the refused unwrap still truncated the capture"

    # The wrappers survived the refusal: the next capture needs no re-wrap.
    recovered = tl.trace(nn.ReLU(), torch.ones(2))
    assert any(op.func_name == "relu" for op in recovered.compute_ops)


def test_child_process_capture_refusal_is_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The child-process guard raises a typed, actionable refusal.

    It used to raise a bare ``RuntimeError`` whose message began with
    "WARNING:" — unbranchable by callers and mislabelled as a warning while it
    was in fact a hard refusal. Its docstring also implied it refused THREAD
    concurrency, which it never did (that class is covered by atomic admission,
    the non-owner pause no-op, and DataParallel unwrapping).
    """

    import multiprocessing as mp

    from torchlens.utils.display import warn_parallel

    class _FakeChild:
        """Stand in for a non-rank, non-daemonic child process."""

        name = "Process-1"
        daemon = False

    monkeypatch.setattr(mp, "current_process", lambda: _FakeChild())
    # r-b6 R40-3b: the guard no longer trusts the user-assignable process
    # name — child detection keys on ``parent_process()`` (plus the raw-fork
    # PID stamp), so the fake must present a parent to read as a child.
    monkeypatch.setattr(mp, "parent_process", lambda: _FakeChild())

    with pytest.raises(tl.errors.CaptureContextError) as refusal:
        warn_parallel()

    assert refusal.value.fields["code"] == "child_process_capture_unsupported"
    assert refusal.value.fields["process_name"] == "Process-1"
    assert not str(refusal.value).startswith("WARNING:")


def test_child_process_guard_ignores_spoofed_main_process_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child named "MainProcess" is still refused (r-b6 R40-3b).

    ``process.name`` is a user-assignable constructor kwarg
    (``mp.Process(name="MainProcess")``); the historical name-keyed check let
    such a child capture and corrupt the per-interpreter toggle state.
    """

    import multiprocessing as mp

    from torchlens.utils.display import warn_parallel

    class _SpoofedChild:
        """A child process wearing the main process's name."""

        name = "MainProcess"
        daemon = False

    monkeypatch.setattr(mp, "current_process", lambda: _SpoofedChild())
    monkeypatch.setattr(mp, "parent_process", lambda: _SpoofedChild())

    with pytest.raises(tl.errors.CaptureContextError) as refusal:
        warn_parallel()
    assert refusal.value.fields["code"] == "child_process_capture_unsupported"


def test_child_process_guard_refuses_fork_inherited_group_stamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An inherited initialized-group stamp does not read as a rank (r-b6 R40-3b).

    A forked child of a rank inherits ``dist.is_initialized() == True``; the
    rank carve-out must not readmit it. The PID stamp of the process that
    first observed the group is the discriminator.
    """

    import multiprocessing as mp

    import torchlens.utils.display as display_mod
    from torchlens.utils.display import warn_parallel

    class _FakeChild:
        """A non-daemonic child claiming rank via an inherited group flag."""

        name = "Process-2"
        daemon = False

    class _FakeDist:
        """Distributed module stub reporting an initialized group."""

        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def is_initialized() -> bool:
            return True

    monkeypatch.setattr(mp, "current_process", lambda: _FakeChild())
    monkeypatch.setattr(mp, "parent_process", lambda: _FakeChild())
    fake_dist = _FakeDist()
    monkeypatch.setattr(torch, "distributed", fake_dist)
    monkeypatch.setitem(sys.modules, "torch.distributed", fake_dist)  # type: ignore[arg-type]

    # Positive control: a process that stamps its OWN pid reads as a rank, so
    # the refusal below can only come from the inheritance discriminator.
    monkeypatch.setitem(display_mod._DIST_GROUP_OBSERVED_PID, "pid", os.getpid())
    warn_parallel()

    # The "parent rank" observed the group under a different PID; the fork
    # child inherits that stamp and must be refused.
    monkeypatch.setitem(display_mod._DIST_GROUP_OBSERVED_PID, "pid", os.getpid() + 1)
    with pytest.raises(tl.errors.CaptureContextError) as refusal:
        warn_parallel()
    assert refusal.value.fields["code"] == "child_process_capture_unsupported"


def test_main_process_capture_is_never_refused() -> None:
    """The guard is a no-op in the main process, including from a worker thread."""

    from torchlens.utils.display import warn_parallel

    warn_parallel()
    errors: list[BaseException] = []

    def call_from_thread() -> None:
        """A capture on a worker thread is legitimate and must not be refused."""

        try:
            warn_parallel()
        except BaseException as error:  # pragma: no cover - reported by the test
            errors.append(error)

    worker = threading.Thread(target=call_from_thread)
    worker.start()
    worker.join(timeout=5.0)
    assert errors == [], f"a main-process worker thread was refused: {errors!r}"


def test_interrupted_partial_diagnostics_still_restore_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An interruption during partial-trace recovery still tears the session down.

    The failed-forward epilogue builds best-effort partial diagnostics and then
    restores the model. Both arms of that construction catch ``Exception``, so a
    ``KeyboardInterrupt`` raised inside it escaped straight past
    ``cleanup_model_session`` — leaving the user's model with TorchLens-forced
    ``requires_grad``, ``tl_*`` attributes, and an installed buffer tracker.
    """

    from torchlens import partial as partial_module

    class _FailingModel(nn.Module):
        """Register a frozen parameter and then fail the forward."""

        def __init__(self) -> None:
            """Build a model with one explicitly frozen parameter."""

            super().__init__()
            self.weight = nn.Parameter(torch.ones(2), requires_grad=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run one logged op and then raise.

            Parameters
            ----------
            x:
                Input tensor.

            Returns
            -------
            torch.Tensor
                This path never returns.

            Raises
            ------
            ValueError
                Always, after one logged operation.
            """

            _ = torch.relu(x * self.weight)
            raise ValueError("forward failure with interrupted diagnostics")

    def _interrupt_partial_construction(*_args: Any, **_kwargs: Any) -> Any:
        """Interrupt partial-trace construction the way Ctrl-C would."""

        raise KeyboardInterrupt("interrupted during partial construction")

    monkeypatch.setattr(partial_module.PartialTrace, "from_trace", _interrupt_partial_construction)

    model = _FailingModel()
    before = _capture_scope_snapshot()

    with pytest.raises(KeyboardInterrupt):
        tl.trace(model, torch.ones(2))

    assert model.weight.requires_grad is False, (
        "the interrupted epilogue left TorchLens-forced requires_grad on a frozen parameter"
    )
    assert not [name for name in vars(model) if name.startswith("tl_")], (
        "the interrupted epilogue left tl_* session metadata on the model"
    )
    assert _capture_scope_snapshot() == before
    # The process is still usable for the next capture.
    recovered = tl.trace(nn.ReLU(), torch.ones(2))
    assert any(op.func_name == "relu" for op in recovered.compute_ops)
