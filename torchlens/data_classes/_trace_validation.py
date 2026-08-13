"""Trace validation and replay mixin."""

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union, cast

import torch
from torch import nn

if TYPE_CHECKING:
    from ..validation.status import ValidationReplayStatus
    from .trace import Trace

    _TraceMixinBase = Trace
else:
    _TraceMixinBase = object
from .._deprecations import MISSING, MissingType, warn_deprecated_alias
from .._errors import ArgumentConflictError, InvalidArgumentError, RecordBindingError
from ..options import ReplayOptions, merge_replay_options
from ..runnable import DivergencePolicy, RunProvider, RunResult
from .cleanup import (
    _LIST_FIELDS_TO_CLEAN,
    _clear_entry_attributes,
    _label_for_reference_removal,
    _remove_log_entry_references,
    _scrub_conditional_fields_after_removal,
    _scrub_per_op_equivalence_lists,
    cleanup,
)
from .op import Op


def _materialize_layer_mirrors_for_removed(
    trace: "Trace",
    removed_entries: Iterable[Op],
) -> None:
    """Materialize Layer mirror fields whose representative op is being removed.

    A Layer's M8 mirror descriptors read through to its first-pass op; husking
    that op would otherwise change what a still-held Layer reads. Materializing
    first preserves the dict-era post-removal surface (the copies existed at
    this point in the dict era). Layers whose representative op survives keep
    mirroring — their reads are unaffected by the removal.
    """

    layer_logs = trace.__dict__.get("layer_logs")
    if not layer_logs:
        return
    from .layer import _layer_rep_op, materialize_layer_mirrors

    removed_ids = {id(entry) for entry in removed_entries}
    for layer_log in layer_logs.values():
        rep = _layer_rep_op(layer_log)
        if rep is not None and id(rep) in removed_ids:
            materialize_layer_mirrors(layer_log)


_USE_STORED_TRANSFORM = object()
_JAX_VALIDATION_REPLAY_BACKEND = "jax"
_MLX_VALIDATION_REPLAY_BACKEND = "mlx"
_TINYGRAD_VALIDATION_REPLAY_BACKEND = "tinygrad"


def _warn_stateful_live_run_once(trace: Any, model: nn.Module) -> None:
    """Warn once when a live rerun has an obvious model-state mutation risk.

    Parameters
    ----------
    trace:
        Source Trace carrying the once-only warning marker.
    model:
        Live model that is about to be re-executed.
    """

    if trace.__dict__.get("_stateful_run_warning_emitted", False):
        return
    if not model.training:
        return
    running_stat_risk: tuple[str, tuple[str, ...]] | None = None
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.modules.batchnorm._BatchNorm) or not module.training:
            continue
        buffers = getattr(module, "_buffers", {})
        running_stat_buffers = tuple(
            name
            for name in ("running_mean", "running_var", "num_batches_tracked")
            if buffers.get(name) is not None
        )
        if bool(getattr(module, "track_running_stats", False)) and running_stat_buffers:
            running_stat_risk = (module_name or "<root>", running_stat_buffers)
            break
    if running_stat_risk is None:
        return
    import warnings

    module_name, buffer_names = running_stat_risk
    warnings.warn(
        "run() detected training-mode BatchNorm running-stat buffers "
        f"{', '.join(buffer_names)} on module {module_name!r}; re-executing the live model can "
        "mutate them. Use eval() for immutable feature extraction or clone the model explicitly "
        "when isolated training-mode state is required.",
        UserWarning,
        stacklevel=3,
    )
    trace.__dict__["_stateful_run_warning_emitted"] = True


def _loaded_non_torch_validation_replay_unavailable(trace: Any) -> bool:
    """Return whether loaded non-torch replay validation cannot run.

    Parameters
    ----------
    trace:
        Trace-like object being checked.

    Returns
    -------
    bool
        True for loaded non-torch traces whose backend runtime replay
        capture lists were stripped by portable save.
    """

    if not bool(getattr(trace, "_loaded_from_bundle", False)):
        return False
    backend = str(getattr(trace, "backend", "torch"))
    if backend == _MLX_VALIDATION_REPLAY_BACKEND:
        return True
    if backend == _JAX_VALIDATION_REPLAY_BACKEND:
        return not bool(getattr(trace, "jax_equation_captures", ()))
    if backend == _TINYGRAD_VALIDATION_REPLAY_BACKEND:
        return not bool(getattr(trace, "tinygrad_uop_captures", ()))
    return False


class TraceValidationMixin(_TraceMixinBase):
    def save_new_outs(
        self: "Trace",
        model: torch.nn.Module,
        input_args: torch.Tensor | List[Any],
        input_kwargs: Optional[Dict[Any, Any]] = None,
        layers_to_save: str | List[str] = "all",
        grad_layers_to_save: str | List[str] | None = "all",
        random_seed: Optional[int] = None,
        backward_ready: bool | None = None,
    ) -> None:
        """Re-run the model with new inputs, saving only outs.

        Parameters
        ----------
        model, input_args, input_kwargs, layers_to_save, grad_layers_to_save, random_seed, backward_ready:
            Forwarded unchanged to
            :func:`torchlens.capture.trace.save_new_outs`.
        """
        from ..capture.outcome import require_capture_capability
        from ..capture.trace import save_new_outs as _impl
        from .._capture_state_helpers import unwrap_compiled_model

        # N3/N5: a live refresh re-drives the FULL native forward against the
        # recorded graph, which a halted/failed/unproven capture cannot honor.
        require_capture_capability(self, "live_replay")
        model = unwrap_compiled_model(model)

        return _impl(
            self,
            model=model,
            input_args=input_args,
            input_kwargs=input_kwargs,
            layers_to_save=layers_to_save,
            grad_layers_to_save=grad_layers_to_save,
            random_seed=random_seed,
            backward_ready=backward_ready,
        )

    def validate_saved_outs(
        self: "Trace",
        ground_truth_output_tensors: List[torch.Tensor],
        verbose: bool = False,
        validate_metadata: bool = True,
    ) -> Union[bool, "ValidationReplayStatus"]:
        """Deprecated alias for :meth:`validate_forward_pass`.

        Parameters
        ----------
        ground_truth_output_tensors, verbose, validate_metadata:
            Forwarded unchanged to :meth:`validate_forward_pass`.

        Returns
        -------
        bool or ValidationReplayStatus
            ``True`` if validation succeeds. Loaded non-torch traces whose
            runtime replay captures were stripped return an explicit
            unavailable status.
        """
        warn_deprecated_alias(
            "Trace.validate_saved_outs",
            "Trace.validate_forward_pass",
        )
        return self.validate_forward_pass(
            ground_truth_output_tensors=ground_truth_output_tensors,
            verbose=verbose,
            validate_metadata=validate_metadata,
        )

    def validate_forward_pass(
        self: "Trace",
        ground_truth_output_tensors: List[torch.Tensor],
        verbose: bool = False,
        validate_metadata: bool = True,
    ) -> Union[bool, "ValidationReplayStatus"]:
        """Validate saved outs against ground-truth model outputs.

        Parameters
        ----------
        ground_truth_output_tensors, verbose, validate_metadata:
            Forwarded unchanged to
            :func:`torchlens.validation.core.validate_saved_outs`.

        Returns
        -------
        bool or ValidationReplayStatus
            ``True`` if validation succeeds. Loaded non-torch traces whose
            runtime replay captures were stripped return an explicit
            unavailable status instead of a pass/fail bool.
        """
        from ..backends import get_backend_spec
        from ..capture.outcome import require_capture_capability
        from ..runnable import refuse_poisoned_trace

        refuse_poisoned_trace(self, "validation")
        # N2: refusing ENTRY for failed/unproven captures is not a check
        # exemption -- the tripwire bodies stay byte-untouched, the halted
        # exemption neither widens nor narrows, and legacy UNATTESTED
        # artifacts deliberately keep entry OPEN (the tripwire stays armed).
        require_capture_capability(self, "validation_entry")
        status = self.validation_replay_status
        if bool(getattr(self, "_loaded_from_bundle", False)) and not status.available:
            setattr(self, "_validation_replay_status", status)
            return status
        spec = get_backend_spec(getattr(self, "backend", "torch"))
        validation_result = spec.validate_trace(
            self,
            ground_truth_output_tensors=ground_truth_output_tensors,
            verbose=verbose,
            validate_metadata=validate_metadata,
        )
        if spec.name == "torch":
            from ..validation.status import ValidationReplayStatus

            if isinstance(validation_result, ValidationReplayStatus):
                setattr(self, "_validation_replay_status", validation_result)
                if validation_result.state in {"passed", "failed"}:
                    return validation_result.passed
        return validation_result

    @property
    def validation_replay_status(self: "Trace") -> "ValidationReplayStatus":
        """Return replay-validation availability or last completed result.

        Returns
        -------
        ValidationReplayStatus
            Status object distinguishing live replay validation from loaded
            traces whose runtime replay captures were stripped during save.
        """

        from ..backends import get_backend_spec
        from ..validation.status import ValidationReplayStatus

        cached_status = getattr(self, "_validation_replay_status", None)
        if isinstance(cached_status, ValidationReplayStatus):
            return cached_status
        backend = str(getattr(self, "backend", "torch"))
        if _loaded_non_torch_validation_replay_unavailable(self):
            return ValidationReplayStatus.unavailable_loaded_runtime_stripped(
                backend=backend,
                payload_load_status=getattr(self, "payload_load_status", None),
            )
        spec = get_backend_spec(backend)
        if not spec.capabilities.validation_replay:
            return ValidationReplayStatus.unavailable_unsupported(backend=backend)
        return ValidationReplayStatus.available_live(backend=backend)

    def push(
        self: "Trace",
        strict: bool | MissingType = MISSING,
        hooks: dict[Any, Any] | None | MissingType = MISSING,
        differentiable: bool | MissingType = MISSING,
        replay: ReplayOptions | None = None,
    ) -> "Trace":
        """Push the edit downstream through the recorded graph (DAG replay).

        Parameters
        ----------
        strict:
            Whether divergence warnings should raise.
        hooks:
            Optional mapping from selector-like targets to hook callables.
        differentiable:
            If true, return a new Trace whose replayed tensors remain
            differentiable from fresh replay-frontier leaves.

        Returns
        -------
        Trace
            This model log, mutated in place.
        """

        from ..capture.outcome import require_capture_capability

        require_capture_capability(self, "live_replay")
        replay_options = merge_replay_options(
            replay=replay,
            strict=strict,
            hooks=hooks,
            differentiable=differentiable,
        )

        from ..intervention.replay import push as _impl

        return _impl(self, replay=replay_options)

    def replay(
        self: "Trace",
        strict: bool | MissingType = MISSING,
        hooks: dict[Any, Any] | None | MissingType = MISSING,
        differentiable: bool | MissingType = MISSING,
        replay: ReplayOptions | None = None,
    ) -> "Trace":
        """Deprecated alias for :meth:`push`.

        Parameters
        ----------
        strict, hooks, differentiable, replay:
            Forwarded unchanged to :meth:`push`.

        Returns
        -------
        Trace
            This model log, mutated in place.
        """

        from .._deprecations import warn_deprecated_alias

        warn_deprecated_alias("Trace.replay", "Trace.push")
        return self.push(strict=strict, hooks=hooks, differentiable=differentiable, replay=replay)

    def push_from(
        self: "Trace",
        site: Any,
        strict: bool | MissingType = MISSING,
        replay: ReplayOptions | None = None,
    ) -> "Trace":
        """Push downstream from a pre-mutated site.

        Parameters
        ----------
        site:
            Layer pass or selector resolving to one origin. The origin's
            current out is preserved and used as the override.
        strict:
            Whether divergence warnings should raise.

        Returns
        -------
        Trace
            This model log, mutated in place.
        """

        from ..capture.outcome import require_capture_capability

        require_capture_capability(self, "live_replay")
        replay_options = merge_replay_options(replay=replay, strict=strict)

        from ..intervention.replay import push_from as _impl

        return _impl(self, site, replay=replay_options)

    def replay_from(
        self: "Trace",
        site: Any,
        strict: bool | MissingType = MISSING,
        replay: ReplayOptions | None = None,
    ) -> "Trace":
        """Deprecated alias for :meth:`push_from`.

        Parameters
        ----------
        site, strict, replay:
            Forwarded unchanged to :meth:`push_from`.

        Returns
        -------
        Trace
            This model log, mutated in place.
        """

        from .._deprecations import warn_deprecated_alias

        warn_deprecated_alias("Trace.replay_from", "Trace.push_from")
        return self.push_from(site, strict=strict, replay=replay)

    def run(
        self: "Trace",
        model: Any = None,
        x: Any = None,
        *,
        inputs: Any | MissingType = MISSING,
        seed: int | None = None,
        fast: bool = False,
        on_divergence: DivergencePolicy = DivergencePolicy.RAISE,
        append: bool | MissingType = MISSING,
        chunk_size: int | None | MissingType = MISSING,
        chunk_paths: Any | None = None,
        strict: bool | MissingType = MISSING,
        replay: ReplayOptions | None = None,
        transform: Callable[[Any], Any] | bool | object = _USE_STORED_TRANSFORM,
        output_transform: Callable[[Any], Any] | bool | object = _USE_STORED_TRANSFORM,
    ) -> "Trace | RunResult":
        """Execute this Trace through its live or loaded provider.

        Parameters
        ----------
        model:
            Model to execute through TorchLens decorated wrappers. When omitted,
            the live model captured by this ``Trace`` is reused if still available.
        x:
            Forward input. If ``model`` is omitted, the first positional argument
            is treated as the new user input.
        inputs:
            Unified provider input tree. Supplying this keyword returns a
            transactional :class:`RunResult` and leaves this Trace unchanged.
        seed:
            Optional deterministic live refresh, random-state, and runtime RNG seed.
        fast:
            Explicit stateful static-loop mode. The live provider runs native ``forward``
            with targeted module/function collection and a per-call path/shape guard. The
            loaded provider performs one ordinary verified run, then reuses staged state
            and compiled argument binders. Unlike the default transactional provider, later
            fast iterations reuse one result Trace in place.
        on_divergence:
            Strict divergence behavior or the sole poison-return opt-in.
        append:
            If true, append a compatible chunk along batch dimension 0.
        chunk_size:
            If supplied, split positional tensor input into chunks of this size,
            run the first chunk normally, then append remaining chunks.
        chunk_paths:
            Optional explicit tensor leaf paths to split.
        strict:
            Whether graph-shape divergence should raise instead of warn.
        transform:
            Stored-transform sentinel, ``False`` to bypass, or explicit input
            transform callable for this run.
        output_transform:
            Stored-transform sentinel, ``False`` to bypass, or explicit output
            transform callable for this run.

        Returns
        -------
        Trace or RunResult
            A unified transactional result for ``inputs=`` and loaded sparse
            providers. Legacy ``run(model, x)`` intervention reruns retain their
            compatibility return until that surface is migrated.

        Notes
        -----
        A live-provider run re-executes the retained model object. TorchLens warns
        once when it detects training-mode BatchNorm running-stat buffers, which the
        forward pass can mutate. Custom mutable attributes such as caches and user
        counters cannot be detected generically. Clone the model explicitly when isolated
        state is required. Live-state mutation can also change the captured graph and
        trigger the normal graph-change tripwire.
        """

        if seed is not None:
            # r77 nit + r79 hardening: validate ``seed`` at the run door so junk
            # raises the typed precondition lane instead of escaping as torch's
            # raw ``RuntimeError``. r79 extends the r77 non-int check to the two
            # escapes r78 found: ``bool`` (an int subclass that
            # ``Generator.manual_seed`` rejects) and an int outside torch's
            # accepted long range (pybind overflow). The failed call is
            # transactional either way (global torch RNG untouched).
            from .._runnable_state import validate_run_seed

            validate_run_seed(seed)
        readiness = self._runnable.readiness
        loaded_provider = getattr(readiness, "provider", None)
        use_unified_provider = inputs is not MISSING or (
            not isinstance(model, nn.Module)
            and loaded_provider
            in {
                RunProvider.LOADED_SPARSE,
                RunProvider.LOADED_ANALYSIS,
            }
        )
        if use_unified_provider:
            if inputs is not MISSING:
                if model is not None or x is not None:
                    raise ArgumentConflictError(
                        "Pass inputs= without the legacy model/x arguments",
                        code="run_legacy_arguments_conflict",
                        remedy="pass only inputs= on the unified run surface",
                    )
                run_inputs = inputs
            else:
                if x is not None:
                    raise ArgumentConflictError(
                        "Loaded sparse run accepts one input tree",
                        code="run_legacy_arguments_conflict",
                        remedy="pass one input tree, preferably via inputs=",
                    )
                run_inputs = model
            if any(value is not MISSING for value in (append, chunk_size, strict)) or (
                chunk_paths is not None or replay is not None
            ):
                raise ArgumentConflictError(
                    "Sparse/unified run does not accept legacy rerun options",
                    code="run_legacy_options_conflict",
                    remedy=(
                        "drop append/chunk_size/strict/chunk_paths/replay from the "
                        "unified run call"
                    ),
                )
            if fast and DivergencePolicy(on_divergence) is not DivergencePolicy.RAISE:
                raise InvalidArgumentError(
                    "fast=True always fails closed and requires on_divergence='raise'",
                    code="run_fast_divergence_policy_invalid",
                    remedy="use on_divergence='raise' with fast=True, or drop fast=",
                    argument="on_divergence",
                )
            from ..capture.outcome import require_capture_capability

            if loaded_provider is RunProvider.LOADED_SPARSE:
                # N3 (loaded-sparse split): HALTED stays ALLOWED here -- the
                # loaded provider executes exactly the recorded taken-path
                # prefix DAG under pause_logging(), so the live-replay failure
                # mode cannot occur; failed/unproven captures still refuse.
                require_capture_capability(self, "loaded_sparse_run")
                if fast:
                    from .._fast_run import run_fast_loaded_trace

                    return run_fast_loaded_trace(self, run_inputs, seed=seed)
                from .._runnable_execution import run_loaded_sparse_trace

                return run_loaded_sparse_trace(
                    self,
                    run_inputs,
                    seed=seed,
                    on_divergence=on_divergence,
                )
            if loaded_provider is RunProvider.LOADED_ANALYSIS:
                from .._runnable_execution import raise_analysis_run_unavailable

                raise_analysis_run_unavailable(self)
            # N3/N5 (live provider): the live run -- fast=True included --
            # re-drives the full native forward, which halted/failed/unproven
            # captures cannot honor.
            require_capture_capability(self, "live_replay")
            from .._runnable_execution import run_live_trace

            source_ref = getattr(self, "_source_model_ref", None)
            live_model = source_ref() if source_ref is not None else None
            if live_model is not None:
                _warn_stateful_live_run_once(self, live_model)

            if fast:
                from .._fast_run import run_fast_live_trace

                return run_fast_live_trace(self, run_inputs, seed=seed)

            return run_live_trace(
                self,
                run_inputs,
                seed=seed,
                on_divergence=on_divergence,
            )

        if fast:
            raise ArgumentConflictError(
                "fast=True is available only with the unified inputs= surface",
                code="run_fast_requires_inputs",
                remedy="call trace.run(inputs=..., fast=True) instead of the legacy surface",
                argument="fast",
            )

        # N3/N5 (legacy live rerun surface): same live-provider rule.
        from ..capture.outcome import require_capture_capability

        require_capture_capability(self, "live_replay")

        run_model: nn.Module | None
        if isinstance(model, nn.Module):
            run_model = model
            user_input = x
        else:
            source_ref = getattr(self, "_source_model_ref", None)
            user_input = model
            if x is not None:
                raise ArgumentConflictError(
                    "Pass either run(model, x) or run(new_user_input), not both",
                    code="run_legacy_arguments_conflict",
                    remedy="pass run(model, x) or run(new_user_input), never both forms",
                )
            transformed_input = self._apply_rerun_transform(user_input, transform=transform)
            run_model = source_ref() if source_ref is not None else None
            if run_model is None:
                raise RecordBindingError(
                    "This Trace does not retain a live model reference",
                    code="run_source_model_collected",
                    remedy="pass the model explicitly as trace.run(model, input)",
                )
        replay_options = merge_replay_options(
            replay=replay,
            append=append,
            chunk_size=chunk_size,
            strict=strict,
        )
        if isinstance(model, nn.Module):
            transformed_input = self._apply_rerun_transform(user_input, transform=transform)
        _warn_stateful_live_run_once(self, run_model)

        from ..intervention.rerun import run as _impl

        resolved_output_transform = self._resolve_rerun_output_transform(output_transform)
        result = _impl(
            self,
            run_model,
            transformed_input,
            replay=replay_options,
            chunk_paths=chunk_paths,
            output_transform=resolved_output_transform,
        )
        # Atomic swap rebuilds Trace state; restore raw_input to the new
        # user-supplied value so visualization / save-load report the
        # current input rather than the prior trace's.
        result.raw_input = user_input
        return result

    def rerun(
        self: "Trace",
        model: Any = None,
        x: Any = None,
        *,
        append: bool | MissingType = MISSING,
        chunk_size: int | None | MissingType = MISSING,
        chunk_paths: Any | None = None,
        strict: bool | MissingType = MISSING,
        replay: ReplayOptions | None = None,
        transform: Callable[[Any], Any] | bool | object = _USE_STORED_TRANSFORM,
        output_transform: Callable[[Any], Any] | bool | object = _USE_STORED_TRANSFORM,
    ) -> "Trace":
        """Deprecated alias for :meth:`run`.

        Parameters
        ----------
        model, x, append, chunk_size, chunk_paths, strict, replay, transform, output_transform:
            Forwarded unchanged to :meth:`run`.

        Returns
        -------
        Trace
            This model log, mutated in place after a validated atomic swap.
        """

        from .._deprecations import warn_deprecated_alias

        warn_deprecated_alias("Trace.rerun", "Trace.run")
        return self.run(
            model,
            x,
            append=append,
            chunk_size=chunk_size,
            chunk_paths=chunk_paths,
            strict=strict,
            replay=replay,
            transform=transform,
            output_transform=output_transform,
        )

    def _apply_rerun_transform(
        self: "Trace",
        user_input: Any,
        *,
        transform: Callable[[Any], Any] | bool | object,
    ) -> Any:
        """Apply the stored or explicit input transform for ``rerun``.

        Parameters
        ----------
        user_input:
            New user input supplied to ``rerun``.
        transform:
            Sentinel to reuse the stored transform, ``False`` to bypass, or an
            explicit callable to use for this rerun.

        Returns
        -------
        Any
            Model-ready rerun input.
        """

        stored_transform = getattr(self, "_transform", None)
        if transform is _USE_STORED_TRANSFORM and stored_transform is not None:
            return stored_transform(user_input)
        if transform is False:
            return user_input
        if callable(transform):
            return transform(user_input)
        return user_input

    def _resolve_rerun_output_transform(
        self: "Trace",
        output_transform: Callable[[Any], Any] | bool | object,
    ) -> Callable[[Any], Any] | None:
        """Resolve the output transform callable for ``rerun``.

        Parameters
        ----------
        output_transform:
            Sentinel to reuse the stored output transform, ``False`` to bypass,
            or an explicit callable to use for this rerun.

        Returns
        -------
        Callable[[Any], Any] | None
            Output transform to apply to the fresh model output, or ``None``.
        """

        stored_transform = getattr(self, "_output_transform", None)
        if output_transform is _USE_STORED_TRANSFORM:
            return stored_transform
        if output_transform is False:
            return None
        if callable(output_transform):
            return output_transform
        return None

    def check_metadata_invariants(self: "Trace") -> bool:
        """Run metadata invariant checks on this completed model log.

        Returns
        -------
        bool
            ``True`` if all invariants pass.
        """
        # N2: refusing ENTRY for failed/unproven captures is not a check
        # exemption -- the tripwire bodies stay byte-untouched, and legacy
        # UNATTESTED artifacts deliberately keep entry OPEN.
        from ..capture.outcome import require_capture_capability

        require_capture_capability(self, "validation_entry")
        from ..validation.invariants import check_metadata_invariants as _impl

        return _impl(self)

    def cleanup(self: "Trace") -> None:
        """Delete log data, break cycles, and free cached GPU memory.

        Returns
        -------
        None
            This method mutates the model log in place.
        """
        return cleanup(self)

    def release_param_refs(self: "Trace", *, allow_iter_rehydrate: bool = False) -> None:
        """Release live ``nn.Parameter`` references held by ParamLogs.

        Parameters
        ----------
        allow_iter_rehydrate:
            If ``True``, iterating ``param_logs`` may lazily restore live
            references from the source model. Public explicit releases leave
            this disabled.

        Returns
        -------
        None
            This method mutates ParamLogs in place.
        """
        if hasattr(self.param_logs, "_rehydrate_on_iter"):
            self.param_logs._rehydrate_on_iter = False
        for param_log in self.param_logs.values():
            param_log.release_param_ref()
        if hasattr(self.param_logs, "_rehydrate_on_iter"):
            self.param_logs._rehydrate_on_iter = allow_iter_rehydrate

    def _postprocess(
        self: "Trace",
        output_tensors: List[torch.Tensor],
        output_tensor_addresses: List[str],
    ) -> None:
        """Run postprocessing on a completed raw capture pass.

        Parameters
        ----------
        output_tensors:
            Output tensors returned by the model.
        output_tensor_addresses:
            Hierarchical addresses for those outputs.
        """
        from ..postprocess import postprocess as _impl

        self._postprocessing_active = True
        try:
            return _impl(
                self,
                output_tensors=output_tensors,
                output_tensor_addresses=output_tensor_addresses,
            )
        finally:
            self._postprocessing_active = False

    def _run_and_log_inputs_through_model(
        self: "Trace",
        model: torch.nn.Module,
        input_args: torch.Tensor | List[Any],
        input_kwargs: Optional[Dict[Any, Any]] = None,
        layers_to_save: Optional[str | List[str | int]] = "all",
        grad_layers_to_save: Optional[str | List[str | int]] = "all",
        random_seed: Optional[int] = None,
        postprocess: bool = True,
    ) -> Any:
        """Run a forward pass and capture it into this model log.

        Parameters
        ----------
        model, input_args, input_kwargs, layers_to_save, grad_layers_to_save, random_seed:
            Forwarded unchanged to
            :func:`torchlens.capture.trace.run_and_log_inputs_through_model`.
        """
        from ..capture.trace import run_and_log_inputs_through_model as _impl

        return _impl(
            self,
            model=model,
            input_args=input_args,
            input_kwargs=input_kwargs,
            layers_to_save=layers_to_save,
            grad_layers_to_save=grad_layers_to_save,
            random_seed=random_seed,
            postprocess=postprocess,
        )

    def log_backward(self: "Trace", loss: torch.Tensor, **backward_kwargs: Any) -> "Trace":
        """Run backward from ``loss`` while capturing first-class backward metadata.

        Parameters
        ----------
        loss:
            Tensor whose ``grad_fn_handle`` roots the backward graph.
        **backward_kwargs:
            Keyword arguments forwarded to ``torch.Tensor.backward``.

        Returns
        -------
        Trace
            This model log, for chaining.
        """
        from ..backends import BackendUnsupportedError, get_backend_spec
        from ..capture.outcome import require_capture_capability

        # N3: the backward projection assumes a structurally complete captured
        # forward graph; HALTED stays allowed (the autograd graph of a halted
        # capture IS the captured prefix).
        require_capture_capability(self, "backward")
        spec = get_backend_spec(getattr(self, "backend", "torch"))
        if not spec.capabilities.backward_capture:
            raise BackendUnsupportedError(
                f"Backend {spec.name!r} does not support backward capture. "
                "Use trace.derived_grads when this backend exposes leaf-level "
                "derived gradients."
            )
        from ..backends.torch.backward import log_backward as _impl

        return cast("Trace", _impl(self, loss, **backward_kwargs))

    def backward(self: "Trace", loss: torch.Tensor, **backward_kwargs: Any) -> "Trace":
        """Run backward from ``loss`` and populate this Trace with backward metadata.

        Parameters
        ----------
        loss:
            Tensor whose ``grad_fn_handle`` roots the backward graph.
        **backward_kwargs:
            Keyword arguments forwarded to ``torch.Tensor.backward``.

        Returns
        -------
        Trace
            This Trace, for chaining.
        """

        return self.log_backward(loss, **backward_kwargs)

    def recording_backward(self: "Trace") -> Any:
        """Return a context manager that captures user-managed backward calls.

        Returns
        -------
        Any
            Backward recording context manager.
        """
        from ..backends import BackendUnsupportedError, get_backend_spec
        from ..capture.outcome import require_capture_capability

        require_capture_capability(self, "backward")
        spec = get_backend_spec(getattr(self, "backend", "torch"))
        if not spec.capabilities.backward_capture:
            raise BackendUnsupportedError(
                f"Backend {spec.name!r} does not support backward capture. "
                "Use trace.derived_grads when this backend exposes leaf-level "
                "derived gradients."
            )
        from ..backends.torch.backward import recording_backward as _impl

        return _impl(self)

    def disarm_triggers(self: "Trace") -> None:
        """Detach this Trace from global autograd backward interception.

        Returns
        -------
        None
            Future plain ``loss.backward()`` or ``torch.autograd.*`` calls will
            not record into this Trace.
        """
        from ..backends.torch.backward import disarm_triggers as _impl

        _impl(self)

    def _remove_log_entry(
        self: "Trace",
        log_entry: Op,
        remove_references: bool = True,
    ) -> None:
        """Remove a single layer-pass entry and scrub graph references.

        Parameters
        ----------
        log_entry:
            Entry to remove.
        remove_references:
            Whether to scrub all graph references to the removed entry.
        """
        tensor_label = _label_for_reference_removal(log_entry, self._tracing_finished)
        _materialize_layer_mirrors_for_removed(self, (log_entry,))
        if remove_references:
            _remove_log_entry_references(self, tensor_label)
        _clear_entry_attributes(log_entry)

    def _batch_remove_log_entries(
        self: "Trace",
        entries_to_remove: Iterable[Op],
        remove_references: bool = True,
    ) -> None:
        """Remove multiple layer-pass entries using single-pass filtering.

        Parameters
        ----------
        entries_to_remove:
            Entries to remove.
        remove_references:
            Whether to scrub all graph references to the removed entries.
        """
        entries_to_remove = list(entries_to_remove)
        surviving_entries = [entry for entry in self if entry not in entries_to_remove]
        _materialize_layer_mirrors_for_removed(self, entries_to_remove)

        labels_to_remove = set()
        for entry in entries_to_remove:
            labels_to_remove.add(_label_for_reference_removal(entry, self._tracing_finished))

        if not remove_references:
            for entry in entries_to_remove:
                _clear_entry_attributes(entry)
            return

        _scrub_conditional_fields_after_removal(self, labels_to_remove, surviving_entries)

        for field_name in _LIST_FIELDS_TO_CLEAN:
            collection = getattr(self, field_name)
            collection[:] = [label for label in collection if label not in labels_to_remove]

        self.conditional_branch_edges = [
            edge
            for edge in self.conditional_branch_edges
            if edge[0] not in labels_to_remove and edge[1] not in labels_to_remove
        ]

        for param_group, tensor_labels in list(self.layers_with_params.items()):
            self.layers_with_params[param_group] = [
                label for label in tensor_labels if label not in labels_to_remove
            ]
        self.layers_with_params = {
            param_group: tensor_labels
            for param_group, tensor_labels in self.layers_with_params.items()
            if len(tensor_labels) > 0
        }

        for equiv_group, equivalent_label_set in list(self.op_equivalence_classes.items()):
            equivalent_label_set -= labels_to_remove
        self.op_equivalence_classes = {
            equiv_group: equivalent_label_set
            for equiv_group, equivalent_label_set in self.op_equivalence_classes.items()
            if len(equivalent_label_set) > 0
        }

        _scrub_per_op_equivalence_lists(surviving_entries, labels_to_remove)

        for entry in entries_to_remove:
            _clear_entry_attributes(entry)
