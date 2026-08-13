"""Loaded-sparse transaction execution and allocation checks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, cast

import torch

from . import _state
from ._runnable_state import (
    PreparedRunnableState,
    RunResourceCeiling,
    runnable_tensor_byte_digest,
)
from .errors import (
    RunCapabilityUnavailableError,
    RuntimeSignatureDriftError,
)
from .intervention.replay import _CallConeNode, _walk_call_cone
from .runnable import (
    ActivationPayloadLayerDescriptor,
    ContractCheck,
    DivergencePolicy,
    InputAttestationFingerprint,
    NumericAttestationStatus,
    PathFaithfulness,
    ReadinessReport,
    ReadinessStatus,
    RunnableCallDescriptor,
    RunnableErrorCode,
    RunProvider,
    RunResult,
    SparseRunDescriptor,
    StateSource,
    TensorSlotRole,
)
from .utils.rng import (
    restore_host_rng,
    snapshot_host_rng,
)

if TYPE_CHECKING:
    from ._runnable_execution import (
        _ambient_execution_context_restored,
        _bind_call_outputs,
        _call_execution_context_entered,
        _call_witness_checks,
        _container_spec_reconstruction_lossy,
        _contract_check,
        _control_witness_source_slot_ids,
        _declared_nondeterministic_sources,
        _execute_sparse_call,
        _finalize_provider_run,
        _first_failed_live_input_check,
        _host_rng_unreproduced,
        _input_derived_layout_stale,
        _mode_sensitive_op_unwitnessed,
        _nondeterministic_value_sources,
        _numeric_attestation_check,
        _output_container_spec,
        _output_not_reproduced,
        _path_faithfulness,
        _populate_source_slots,
        _post_execution_contract_checks,
        _pre_call_contract_checks,
        _raise_failed_contract_as_divergence,
        _raise_numeric_attestation_failure,
        _raw_activation_slot_ids,
        _reconstruct_live_output,
        _reconstruct_output,
        _run_fork_name,
        _seed_run_generators,
        _seeded_fork_devices,
        _split_mixed_inputs,
        _state_contract_checks,
        _tensor_derived_scalar_stale,
        _tensor_derived_scalar_witness_slot_ids,
        _unbound_state_escape_stale,
        _uninit_taint_reaches,
    )

__all__ = (
    "_execute_loaded_sparse_transaction",
    "run_live_trace",
    "_live_runtime_input_leaves",
)


def _execute_loaded_sparse_transaction(
    trace: Any,
    inputs: Any,
    *,
    seed: int | None,
    divergence_policy: DivergencePolicy,
    descriptor: SparseRunDescriptor,
    readiness: ReadinessReport,
    callables: Mapping[str, Callable[..., Any]],
    slot_values: dict[str, torch.Tensor],
    ceiling: RunResourceCeiling,
    input_byte_digests: Mapping[str, str],
    input_fingerprints: Mapping[str, InputAttestationFingerprint],
    input_checks: tuple[ContractCheck, ...],
    input_alias_unresolved: bool,
    prepared_state: PreparedRunnableState,
    fork: Any,
) -> RunResult:
    """Execute one sparse transaction whose caller owns rollback on escape."""

    # r59/r61: the ONE aggregate resource ceiling for this transaction -- constructed
    # by ``run_loaded_sparse_trace`` before input binding, threaded here -- bounds
    # realized output count (per-call projection + bind backstop) and re-materialization
    # clone bytes at every snapshot site. The run-prep retention floor
    # (prepare_runnable_state) already refused a guaranteed-OOM retained population
    # before we got here.
    # Escape-source witness slots must be digested at their production point (before
    # any later in-place op mutates the live tensor), matching the save-side digest.
    escape_witness_slot_ids = _tensor_derived_scalar_witness_slot_ids(descriptor)
    witness_source_snapshots: dict[str, torch.Tensor] = {}
    _populate_source_slots(
        fork,
        descriptor,
        slot_values,
        ceiling=ceiling,
        witness_slot_ids=escape_witness_slot_ids,
        witness_source_snapshots=witness_source_snapshots,
    )
    contract_checks: list[ContractCheck] = [
        *input_checks,
        *_state_contract_checks(descriptor, slot_values),
    ]

    # ``contract_checks`` only ever grows by appends and ``ContractCheck`` is frozen,
    # so the earliest failed check is found by visiting each check exactly once across
    # the whole transaction instead of rescanning the cumulative list at every call
    # boundary (O(calls^2) ``passed`` reads on long replays). Same earliest check,
    # same raise, same policy handling as scanning the full list front-to-back.
    scan_cursor = 0
    first_failed_check: ContractCheck | None = None

    def first_failed_contract_so_far() -> ContractCheck | None:
        """Earliest failed check appended so far, or ``None`` while all pass.

        Each check is visited at most once across the whole transaction, which
        is sound only because ``contract_checks`` grows by append and
        ``ContractCheck`` is frozen: no already-scanned check can later fail.
        """

        nonlocal scan_cursor, first_failed_check
        while first_failed_check is None and scan_cursor < len(contract_checks):
            check = contract_checks[scan_cursor]
            scan_cursor += 1
            if not check.passed:
                first_failed_check = check
        return first_failed_check

    def raise_first_divergence_incremental() -> None:
        """Raise the earliest failed contract check as a divergence, if any failed.

        A no-op under :attr:`DivergencePolicy.RETURN_DIVERGED`, where the
        failure is carried in the poisoned result instead of raised.
        """

        failed = first_failed_contract_so_far()
        if failed is None or divergence_policy is DivergencePolicy.RETURN_DIVERGED:
            return
        _raise_failed_contract_as_divergence(failed, fork=fork)

    raise_first_divergence_incremental()
    # r35 corr2_5: PRE-EXECUTION state digests -- eligibility compares each
    # state slot's capture-start bytes, not whatever a mutating call left
    # behind. Computed only when an activation archive exists to attest.
    state_byte_digests: dict[str, str] = {}
    if isinstance(descriptor.payload_layers.activations, ActivationPayloadLayerDescriptor):
        for state_slot in descriptor.tensor_slots:
            if state_slot.state_binding is None or state_slot.slot_id not in slot_values:
                continue
            try:
                state_byte_digests[state_slot.slot_id] = runnable_tensor_byte_digest(
                    slot_values[state_slot.slot_id]
                )
            except Exception:
                # Undigestable state cannot attest; the absent entry reads as a
                # mismatch in the eligibility partition (fail-safe).
                continue
    call_outputs: dict[str, Any] = {}
    attestation_slot_ids = _raw_activation_slot_ids(descriptor)
    attestation_slot_values: dict[str, torch.Tensor] = {}
    # r55 C3: registry lookup for the per-call allocation preflight (namespace /
    # qualname drive the size-driving classification).
    registry_by_id = {entry.registry_id: entry for entry in descriptor.callable_registry}
    # Run-invariant ``descriptor.tensor_slots`` indexes for the per-call output bind.
    # The descriptor is frozen for the whole replay, so these are built ONCE here
    # instead of once per call (they were O(calls x slots) rebuilds).
    slots_by_id = {slot.slot_id: slot for slot in descriptor.tensor_slots}
    state_slot_ids = frozenset(
        slot.slot_id
        for slot in descriptor.tensor_slots
        if slot.role in {TensorSlotRole.PARAMETER, TensorSlotRole.BUFFER}
    )
    # Producer slot id -> its version-alias slot ids (``version_of`` AND
    # ``producer_slot_id`` both naming the producer), in ``tensor_slots`` order, so
    # the per-output bind stages aliases by O(1) lookup instead of scanning every
    # slot for every produced output (O(outputs x slots)).
    version_alias_ids: dict[str, tuple[str, ...]] = {}
    for slot in descriptor.tensor_slots:
        if slot.version_of is not None and slot.version_of == slot.producer_slot_id:
            version_alias_ids[slot.version_of] = (
                *version_alias_ids.get(slot.version_of, ()),
                slot.slot_id,
            )

    # r35 corr2_4: the fork/restore set follows the SEEDING PRIMITIVE, never the
    # bound-input overlay -- every visible CUDA device is forked when CUDA is
    # initialized or the descriptor's capture metadata names a CUDA device
    # (including produced-only intermediates and RNG-source slots), and the
    # executor seeds ONLY the CPU generator plus each forked CUDA generator, so
    # no unforked generator (an unmentioned CUDA device, MPS/XPU) is ever
    # touched by a seeded run.
    devices = _seeded_fork_devices(descriptor, seed)
    cuda_initialized_before = (
        torch.cuda.is_available() and torch.cuda.is_initialized() if seed is not None else None
    )
    host_rng_unreproduced = _host_rng_unreproduced(descriptor, seed)
    # Faithful original replay of a host-RNG capture (matching seed): reseed every
    # engine (torch + Python + NumPy) to the captured seed so the recorded taken
    # path is reproduced exactly, not just torch's generator. Preserve and restore
    # the caller's host RNG so run() never leaks a reseed into ambient global state.
    reseed_host = (
        seed is not None and descriptor.rng_profile.host_rng_consumed and not host_rng_unreproduced
    )
    host_rng_saved = snapshot_host_rng() if reseed_host else None
    rng_context = torch.random.fork_rng(devices=devices) if seed is not None else nullcontext()
    try:
        # Decision E: the recorded capture-scoped ambient backend context is
        # restored transactionally around the whole run (finally-restored on
        # every exit); each resolved call additionally enters its own recorded
        # per-call context tightly (see execute_call below).
        with (
            _ambient_execution_context_restored(descriptor.ambient_context),
            rng_context,
            _state.pause_logging(),
        ):
            if seed is not None:
                _seed_run_generators(cast(int, seed), devices, reseed_host=reseed_host)

            def execute_call(call_node: _CallConeNode) -> None:
                """Execute and stage one dependency-ready sparse call."""

                call = cast(RunnableCallDescriptor, call_node)
                call_checks, before_versions = _pre_call_contract_checks(
                    descriptor,
                    call,
                    slot_values,
                )
                contract_checks.extend(call_checks)
                raise_first_divergence_incremental()
                try:
                    with _call_execution_context_entered(call.execution_context):
                        output = _execute_sparse_call(
                            call,
                            callables[call.call_id],
                            slot_values,
                            registry_entry=registry_by_id.get(call.registry_id),
                            ceiling=ceiling,
                        )
                except RuntimeSignatureDriftError:
                    # r39 corr2_4: under ``return_diverged`` an admitted-but-INEXECUTABLE
                    # divergent input (wrong feature shape/dtype/device) reaches the resolved
                    # callable and throws. If an input contract check ALREADY failed, this is
                    # input DIVERGENCE, not resolved-callable signature drift -- roll back and
                    # raise the typed ``PathDivergenceError`` carrying the first failed check.
                    # Genuine drift (no prior failed check) keeps ``RuntimeSignatureDriftError``.
                    failed = first_failed_contract_so_far()
                    if failed is not None:
                        _raise_failed_contract_as_divergence(failed, fork=fork)
                    raise
                call_outputs[call.call_id] = output
                contract_checks.extend(
                    _bind_call_outputs(
                        call,
                        output,
                        slot_values,
                        fork,
                        ceiling=ceiling,
                        before_versions=before_versions,
                        attestation_slot_ids=attestation_slot_ids,
                        attestation_slot_values=attestation_slot_values,
                        slots=slots_by_id,
                        state_slot_ids=state_slot_ids,
                        version_alias_ids=version_alias_ids,
                        witness_slot_ids=escape_witness_slot_ids,
                        witness_source_snapshots=witness_source_snapshots,
                    )
                )
                contract_checks.extend(_call_witness_checks(descriptor, call, slot_values))
                raise_first_divergence_incremental()

            _walk_call_cone(descriptor.calls, execute_call)
    finally:
        if host_rng_saved is not None:
            restore_host_rng(host_rng_saved)

    if (
        seed is not None
        and not devices
        and torch.cuda.is_available()
        and torch.cuda.is_initialized() != bool(cuda_initialized_before)
    ):
        # r35 corr2_4 tripwire: CUDA initialized DURING a seeded run whose fork
        # set excluded it -- the descriptor's capture device summary missed a
        # CUDA consumer, so run-local RNG isolation cannot be guaranteed. This
        # is an internal summary bug, never silently ignored.
        _state._unregister_log(fork)
        raise RuntimeError(
            "Internal invariant violation: CUDA became initialized during a "
            "seeded sparse run whose descriptor named no CUDA device; the "
            "capture device summary is incomplete."
        )

    output = _reconstruct_output(
        descriptor, slot_values, fork, ceiling=ceiling, call_outputs=call_outputs
    )
    contract_checks.extend(
        _post_execution_contract_checks(
            descriptor,
            inputs=inputs,
            output=output,
            slot_values=slot_values,
            fork=fork,
        )
    )
    raise_first_divergence_incremental()
    mode_sensitive_op_unwitnessed = _mode_sensitive_op_unwitnessed(descriptor)
    tensor_derived_scalar_stale = _tensor_derived_scalar_stale(
        descriptor, slot_values, witness_source_snapshots
    )
    unbound_state_escape_stale = _unbound_state_escape_stale(descriptor, slot_values)
    # r73 F1: compared against the RAW user input tree (pre-clone leaves; the run
    # executed on defensive clones, so runtime strides are unchanged here).
    input_derived_layout_stale = _input_derived_layout_stale(descriptor, inputs)
    # r53 hon_2: ONE load-side classifier settles declared nondeterministic value
    # sources; the branch ceiling, the attestation gate, and the report signal
    # all consult it (the r52 raise-vs-not_applicable inconsistency is
    # structurally unrepresentable).
    value_source_taint = _nondeterministic_value_sources(descriptor)
    nondeterministic_control_source = _uninit_taint_reaches(
        value_source_taint, _control_witness_source_slot_ids(descriptor)
    )
    declared_nondeterministic_sources = _declared_nondeterministic_sources(
        descriptor, value_source_taint
    )
    output_container_spec = _output_container_spec(fork)
    container_reconstruction_lossy = _container_spec_reconstruction_lossy(output_container_spec)
    output_not_reproduced = _output_not_reproduced(descriptor, output_container_spec)
    # r35 I3 (corr2_7): settle the PROVISIONAL path verdict from ALL non-numeric
    # contract checks and static/dynamic ceilings FIRST; numeric attestation is
    # strictly downstream of it. A verdict that is not VERIFIED -- including one
    # inherited monotonically from a prior poisoned run of the source Trace --
    # makes attestation NOT_APPLICABLE before any archive byte is read, so
    # ATTESTED can never coexist with DIVERGED/UNVERIFIABLE/poisoned, and every
    # FUTURE contract check automatically caps attestation through this same
    # derivation (no parallel Boolean flag list).
    provisional_verdict, provisional_mismatch = _path_faithfulness(
        descriptor,
        contract_checks,
        host_rng_unreproduced=host_rng_unreproduced,
        tensor_derived_scalar_stale=tensor_derived_scalar_stale,
        unbound_state_escape_stale=unbound_state_escape_stale,
        container_reconstruction_lossy=container_reconstruction_lossy,
        output_not_reproduced=output_not_reproduced,
        mode_sensitive_op_unwitnessed=mode_sensitive_op_unwitnessed,
        input_alias_unresolved=input_alias_unresolved,
        nondeterministic_control_source=nondeterministic_control_source,
        input_derived_layout_stale=input_derived_layout_stale,
    )
    eligibility_verdict = provisional_verdict
    inherited_status = fork._runnable.path_faithfulness
    if (
        isinstance(inherited_status, PathFaithfulness)
        and inherited_status is not PathFaithfulness.VERIFIED
        and eligibility_verdict is PathFaithfulness.VERIFIED
    ):
        eligibility_verdict = inherited_status
    numeric_attestation, attestation_check = _numeric_attestation_check(
        descriptor,
        prepared_state,
        slot_values=slot_values,
        attestation_slot_values=attestation_slot_values,
        input_byte_digests=input_byte_digests,
        input_fingerprints=input_fingerprints,
        state_byte_digests=state_byte_digests,
        trace=trace,
        provisional_verdict=eligibility_verdict,
    )
    if attestation_check is not None:
        contract_checks.append(attestation_check)
        if not attestation_check.passed:
            _raise_numeric_attestation_failure(fork, attestation_check)
    return _finalize_provider_run(
        fork=fork,
        output=output,
        readiness=readiness,
        state_source=prepared_state.state_source,
        initializer_policy_version=prepared_state.initializer_policy_version,
        seed=prepared_state.seed,
        random_filled_slot_ids=prepared_state.random_filled_slot_ids,
        contract_checks=tuple(contract_checks),
        provisional_path_faithfulness=provisional_verdict,
        provisional_mismatch=provisional_mismatch,
        numeric_attestation=numeric_attestation,
        divergence_policy=divergence_policy,
        nondeterministic_sources=declared_nondeterministic_sources,
    )


def run_live_trace(
    trace: Any,
    inputs: Any,
    *,
    seed: int | None,
    on_divergence: DivergencePolicy | str = DivergencePolicy.RAISE,
) -> RunResult:
    """Run the live-model refresh provider on a transactional fork.

    r37 corr2-5: the live provider finalizes through the SAME spine as the sparse
    provider -- ``mark_trace_path_status`` (monotonic Trace poison), the shared
    divergence-policy enforcement, and the one ``_run_report`` finalizer (poison
    derived solely from the faithfulness lattice). A lossy live reconstruction
    therefore returns a POISONED report and a monotonically marked Trace that
    every faithful consumer (``to_pandas``, export, chaining) refuses.

    Parameters
    ----------
    trace:
        Live Trace retaining its source-model weak reference.
    inputs:
        New forward input accepted by the existing ``save_new_outs`` path.
    seed:
        Optional refresh seed.
    on_divergence:
        Divergence policy threaded from the public ``run`` surface.

    Returns
    -------
    RunResult
        Structured output, refreshed fork, and live-provider report.

    Raises
    ------
    RunCapabilityUnavailableError
        If the live source model is no longer available.
    """

    divergence_policy = DivergencePolicy(on_divergence)
    source_ref = getattr(trace, "_source_model_ref", None)
    model = source_ref() if source_ref is not None else None
    if model is None:
        raise RunCapabilityUnavailableError(
            "The live Trace no longer retains its source model.",
            code=RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value,
            provider=RunProvider.LIVE,
        )
    prior_log_ids = {id(log) for log in _state.list_logs()}
    fork = trace._fork_trace(name=_run_fork_name(trace))
    try:
        input_args = inputs
        input_kwargs = None
        if (
            isinstance(inputs, Mapping)
            and {"args", "kwargs"}.issubset(inputs)
            and set(inputs).issubset({"args", "kwargs"})
        ):
            args, kwargs = _split_mixed_inputs(inputs)
            input_args = list(args)
            input_kwargs = dict(kwargs)
        # r41 hon1_3 (corr2_4 parity): PRE-compute the soft input-contract checks
        # BEFORE the forward -- a failing forward may in-place-mutate an input leaf
        # (``resize_``) before the failing op, so a post-hoc metadata read could
        # misclassify. The precompute has ZERO admission power: a divergent-but-
        # executable input (changed batch / seq-len) still runs and may honestly
        # settle VERIFIED (fresh-refresh semantics); classification happens only at
        # native-failure time below.
        first_failed = _first_failed_live_input_check(trace, input_args, input_kwargs)
        try:
            fork.save_new_outs(model, input_args, input_kwargs=input_kwargs, random_seed=seed)
        except Exception as exc:  # not BaseException: KeyboardInterrupt/SystemExit stay raw
            if first_failed is not None:
                # An admitted-but-inexecutable DIVERGENT input surfaces as the typed
                # PathDivergenceError carrying the first failed input check, with the
                # native error chained as ``__cause__`` (corr2_4 on both providers).
                # A native failure on a NON-divergent input re-raises raw below -- a
                # genuinely failing model is not a divergence.
                _raise_failed_contract_as_divergence(first_failed, fork=None, cause=exc)
            raise
        output, faithful = _reconstruct_live_output(fork)
        # A lossy output container (computed non-field/non-key state, __slots__, or a
        # data-descriptor field) cannot be faithfully rebuilt, so it is UNVERIFIABLE here
        # too -- never a false VERIFIED on the live-refresh provider.
        if _container_spec_reconstruction_lossy(_output_container_spec(fork)):
            faithful = False
        readiness = ReadinessReport(
            status=ReadinessStatus.READY,
            provider=RunProvider.LIVE,
            backend=str(getattr(trace, "backend", "torch")),
            capability="live_model_fast_capture",
            resolver_records=(),
            state_sources_available=(StateSource.LIVE_MODEL_STATE,),
            witness_completeness=None,
            diagnostics=(),
        )
        # Honesty gate: only a faithfully reconstructed output (exact container type
        # and non-tensor leaves) is VERIFIED. An output we could only approximate
        # from naive leaf paths is UNVERIFIABLE, never blessed with a wrong object.
        provisional = PathFaithfulness.VERIFIED if faithful else PathFaithfulness.UNVERIFIABLE
        return _finalize_provider_run(
            fork=fork,
            output=output,
            readiness=readiness,
            state_source=StateSource.LIVE_MODEL_STATE,
            initializer_policy_version=None,
            seed=seed,
            random_filled_slot_ids=(),
            contract_checks=(
                _contract_check(
                    "live_output_reconstruction",
                    faithful,
                    RunnableErrorCode.OUTPUT_STRUCTURE_MISMATCH,
                    "Live output could not be faithfully reconstructed from its "
                    "captured container contract.",
                ),
            ),
            provisional_path_faithfulness=provisional,
            provisional_mismatch=None,
            numeric_attestation=NumericAttestationStatus.NOT_PRESENT,
            divergence_policy=divergence_policy,
        )
    except BaseException:
        _state._unregister_log(fork)
        for log in _state.list_logs():
            if id(log) not in prior_log_ids:
                _state._unregister_log(log)
        raise


def _live_runtime_input_leaves(input_args: Any, input_kwargs: Any) -> list[torch.Tensor] | None:
    """Flatten runtime live-refresh inputs to ordered tensor leaves, or ``None``.

    Mirrors the capture-side flatten (``backend.fetch_label_move_input_tensors``:
    ``get_vars_of_type_from_obj`` per positional arg, then per kwarg value, at
    cycle-safe finite input walk) so leaf ORDER pairs 1:1 with the capture's recorded
    ``input_layers`` ordering. Returns ``None`` on any traversal failure -- the
    caller then skips classification entirely (never masks the native error).
    """

    from .utils.introspection import INPUT_SEARCH_DEPTH_LIMIT, get_vars_of_type_from_obj

    try:
        if isinstance(input_args, (list, tuple)):
            args_list = list(input_args)
        else:
            args_list = [input_args]
        leaves: list[torch.Tensor] = []
        for arg in args_list:
            unresolved: list[str] = []
            leaves.extend(
                get_vars_of_type_from_obj(
                    arg,
                    torch.Tensor,
                    search_depth=INPUT_SEARCH_DEPTH_LIMIT,
                    depth_exceeded_paths=unresolved,
                )
            )
            if unresolved:
                return None
        if input_kwargs:
            for value in input_kwargs.values():
                unresolved = []
                leaves.extend(
                    get_vars_of_type_from_obj(
                        value,
                        torch.Tensor,
                        search_depth=INPUT_SEARCH_DEPTH_LIMIT,
                        depth_exceeded_paths=unresolved,
                    )
                )
                if unresolved:
                    return None
        return leaves
    except Exception:
        return None
