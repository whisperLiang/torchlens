"""Runtime compatibility reporting for model/input pairs."""

from __future__ import annotations

import ast
import inspect
import itertools
import multiprocessing
import textwrap
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn

from torchlens._distributed import DistributedFinding, detect_distributed_state
from torchlens._robustness import _iter_tensors
from torchlens.utils._torch_compat import (
    get_dynamo_optimized_module_type,
    get_fp8_dtypes,
    get_fx_graph_module_type,
    get_torch_capability_snapshot,
)

Status = Literal["pass", "known_broken", "scope", "not_tested"]
Severity = Literal["ok", "info", "warning", "error"]


@dataclass(frozen=True)
class CompatRow:
    """One row in the TorchLens compatibility truth table.

    Parameters
    ----------
    key:
        Stable machine-readable row key.
    label:
        Human-readable row label.
    status:
        Compatibility status for this model/input pair.
    severity:
        User-facing severity for the row.
    detected:
        Whether the row's condition was detected in the supplied model/input.
    details:
        Explanation of the finding.
    suggestion:
        Suggested workaround or next action.
    """

    key: str
    label: str
    status: Status
    severity: Severity
    detected: bool
    details: str
    suggestion: str = ""


@dataclass(frozen=True)
class CompatReport:
    """Compatibility report returned by :func:`torchlens.compat.report`.

    Parameters
    ----------
    model_type:
        Qualified type name for the inspected model.
    torch_version:
        PyTorch version used for the inspection.
    rows:
        Truth-table rows for the inspected model/input pair.
    """

    model_type: str
    torch_version: str
    rows: tuple[CompatRow, ...]

    def row(self, key: str) -> CompatRow:
        """Return one report row by stable key.

        Parameters
        ----------
        key:
            Row key to look up.

        Returns
        -------
        CompatRow
            Matching report row.

        Raises
        ------
        KeyError
            If ``key`` is not present in this report.
        """

        for report_row in self.rows:
            if report_row.key == key:
                return report_row
        raise KeyError(key)

    def to_markdown(self) -> str:
        """Render this report as a GitHub-flavored Markdown table.

        Returns
        -------
        str
            Markdown representation of the report.
        """

        lines = [
            f"### TorchLens compatibility report for `{self.model_type}`",
            "",
            f"- PyTorch: `{self.torch_version}`",
            f"- Rows: {len(self.rows)}",
            "",
            "| Row | Status | Severity | Detected | Details | Suggestion |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for row in self.rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _escape_markdown(row.label),
                        f"`{row.status}`",
                        f"`{row.severity}`",
                        "yes" if row.detected else "no",
                        _escape_markdown(row.details),
                        _escape_markdown(row.suggestion),
                    ]
                )
                + " |"
            )
        return "\n".join(lines)

    def show(self) -> str:
        """Render this report as a fixed-width text table.

        Returns
        -------
        str
            Text table suitable for terminals and notebook display.
        """

        headers = ("Row", "Status", "Severity", "Detected", "Details", "Suggestion")
        body = [
            (
                row.label,
                row.status,
                row.severity,
                "yes" if row.detected else "no",
                row.details,
                row.suggestion,
            )
            for row in self.rows
        ]
        widths = _column_widths([headers, *body])
        lines = [
            f"TorchLens compatibility report for {self.model_type}",
            f"PyTorch: {self.torch_version}",
            "",
            _format_table_line(headers, widths),
            _format_table_line(tuple("-" * width for width in widths), widths),
        ]
        lines.extend(_format_table_line(row, widths) for row in body)
        return "\n".join(lines)


def report(model: nn.Module, input: Any) -> CompatReport:  # noqa: A002
    """Probe a model/input pair against TorchLens compatibility rows.

    Parameters
    ----------
    model:
        Model or wrapper to inspect.
    input:
        Example model input. The report inspects the input tree but does not
        execute ``model(input)``.

    Returns
    -------
    CompatReport
        Structured compatibility report.
    """

    rows = (
        _hf_transformers_row(model),
        _accelerate_dispatch_row(model),
        _accelerate_offload_row(model),
        _bitsandbytes_row(model),
        _tied_parameters_row(model),
        _multi_gpu_rng_row(),
        _data_parallel_row(model),
        _ddp_row(model),
        _fsdp_row(model),
        *_distributed_rows(model, input),
        _deepspeed_row(model),
        _torch_compile_row(model),
        _fx_row(model),
        _torch_capabilities_row(),
        _lightning_row(model),
        _functorch_row(model),
        _quantized_row(model, input),
        _fp8_dtype_row(model, input),
        _device_context_row(),
        _single_thread_row(),
    )
    return CompatReport(
        model_type=_qualified_type_name(model),
        torch_version=str(torch.__version__),
        rows=rows,
    )


def _escape_markdown(value: str) -> str:
    """Escape table separators in Markdown cells.

    Parameters
    ----------
    value:
        Cell value.

    Returns
    -------
    str
        Escaped cell value.
    """

    return value.replace("|", "\\|").replace("\n", " ")


def _column_widths(rows: Sequence[Sequence[str]]) -> tuple[int, ...]:
    """Compute fixed-width table column sizes.

    Parameters
    ----------
    rows:
        Table rows.

    Returns
    -------
    tuple[int, ...]
        Width for each column.
    """

    column_count = len(rows[0])
    return tuple(max(len(row[index]) for row in rows) for index in range(column_count))


def _format_table_line(row: Sequence[str], widths: Sequence[int]) -> str:
    """Format one fixed-width table row.

    Parameters
    ----------
    row:
        Cell values.
    widths:
        Column widths.

    Returns
    -------
    str
        Formatted row.
    """

    return "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))


def _qualified_type_name(value: Any) -> str:
    """Return a stable qualified type name.

    Parameters
    ----------
    value:
        Object to identify.

    Returns
    -------
    str
        ``module.qualname`` for ``value``'s type.
    """

    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _iter_modules(model: nn.Module) -> Iterable[nn.Module]:
    """Yield modules from ``model`` with a defensive fallback.

    Parameters
    ----------
    model:
        Module to inspect.

    Returns
    -------
    Iterable[nn.Module]
        Module iterator.
    """

    try:
        return tuple(model.modules())
    except Exception:
        return (model,)


def _class_identity(value: Any) -> str:
    """Return lowercase class/module identity text for heuristic checks.

    Parameters
    ----------
    value:
        Object to identify.

    Returns
    -------
    str
        Lowercase qualified type identity.
    """

    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}".lower()


def _class_in_namespace(value: Any, module_prefixes: Sequence[str]) -> bool:
    """Return whether ``value``'s type or a base lives in a listed module namespace.

    Detection anchors on real ``__module__`` provenance across the full MRO rather
    than on a substring of a single class name. A class merely *named* like a
    framework wrapper but defined in a user module does not match, while a genuine
    subclass of a framework base class does. This mirrors the module-path anchoring
    used by :func:`_is_quantized_module`.

    Parameters
    ----------
    value:
        Object whose type MRO is inspected.
    module_prefixes:
        Module-path namespaces (for example ``"transformers"``). A prefix matches a
        module that equals it or is a dotted descendant of it, so ``"transformers"``
        matches ``transformers.modeling_utils`` but not ``transformersx``.

    Returns
    -------
    bool
        True if any MRO base is defined under a listed namespace.
    """

    prefixes = tuple(prefix.lower() for prefix in module_prefixes)
    try:
        mro = type(value).__mro__
    except Exception:
        return False
    for klass in mro:
        module = (getattr(klass, "__module__", "") or "").lower()
        if any(module == prefix or module.startswith(f"{prefix}.") for prefix in prefixes):
            return True
    return False


def _hf_transformers_row(model: nn.Module) -> CompatRow:
    """Build the Hugging Face Transformers wrapper row.

    Parameters
    ----------
    model:
        Model to inspect.

    Returns
    -------
    CompatRow
        Report row.
    """

    detected = _class_in_namespace(model, ("transformers",))
    details = (
        "Hugging Face-style module detected; eager forward capture is supported when the "
        "model is not compiled, offloaded, or sharded."
        if detected
        else "No Hugging Face Transformers wrapper detected."
    )
    suggestion = (
        "Use torchlens.compat.from_huggingface for offline-first loading when helpful."
        if detected
        else ""
    )
    return CompatRow(
        "hf_transformers",
        "HF Transformers wrapper",
        "pass",
        "info" if detected else "ok",
        detected,
        details,
        suggestion,
    )


def _accelerate_dispatch_row(model: nn.Module) -> CompatRow:
    """Build the Accelerate device-map dispatch row.

    Parameters
    ----------
    model:
        Model to inspect.

    Returns
    -------
    CompatRow
        Report row.
    """

    device_map = getattr(model, "hf_device_map", None)
    detected = bool(device_map)
    status: Status = "known_broken" if detected else "pass"
    details = (
        "Accelerate device_map dispatch detected; parameters may materialize on different "
        "devices during forward, so TorchLens cannot validate a single coherent eager trace."
        if detected
        else "No Accelerate device_map dispatch detected."
    )
    return CompatRow(
        "accelerate_device_map_auto",
        "Accelerate device_map='auto'",
        status,
        "error" if detected else "ok",
        detected,
        details,
        "Run on a single materialized device before logging." if detected else "",
    )


def _accelerate_offload_row(model: nn.Module) -> CompatRow:
    """Build the Accelerate CPU/disk offload row.

    Parameters
    ----------
    model:
        Model to inspect.

    Returns
    -------
    CompatRow
        Report row.
    """

    detected = False
    for module in _iter_modules(model):
        hook = getattr(module, "_hf_hook", None)
        if hook is None:
            continue
        # Offload is signalled by the hook's own offload flags. execution_device is
        # present for plain single-device dispatch too, and using its truthiness
        # both false-positives (offload=False + a device) and false-negatives
        # (device index 0 is falsy), so it is not an offload signal.
        if bool(getattr(hook, "offload", False)) or bool(getattr(hook, "offload_buffers", False)):
            detected = True
            break
    status: Status = "known_broken" if detected else "pass"
    details = (
        "Accelerate CPU/disk offload hooks detected; lazy parameter movement can bypass "
        "TorchLens' assumptions about tensor identity and device placement."
        if detected
        else "No Accelerate CPU/disk offload hooks detected."
    )
    return CompatRow(
        "accelerate_cpu_disk_offload",
        "Accelerate CPU/disk offload",
        status,
        "error" if detected else "ok",
        detected,
        details,
        "Disable offload or log a fully materialized copy." if detected else "",
    )


def _bitsandbytes_row(model: nn.Module) -> CompatRow:
    """Build the bitsandbytes quantization row.

    Parameters
    ----------
    model:
        Model to inspect.

    Returns
    -------
    CompatRow
        Report row.
    """

    detected = bool(getattr(model, "is_loaded_in_8bit", False)) or bool(
        getattr(model, "is_loaded_in_4bit", False)
    )
    if not detected:
        detected = any("bitsandbytes" in _class_identity(module) for module in _iter_modules(model))
    status: Status = "known_broken" if detected else "pass"
    details = (
        "bitsandbytes 8-bit/4-bit modules detected; custom parameter wrappers and kernels "
        "are outside TorchLens' dense eager tensor contract."
        if detected
        else "No bitsandbytes 8-bit/4-bit modules detected."
    )
    return CompatRow(
        "bitsandbytes_8bit_4bit",
        "bitsandbytes 8-bit/4-bit",
        status,
        "error" if detected else "ok",
        detected,
        details,
        "Log an unquantized reference model when exact out metadata is required."
        if detected
        else "",
    )


def _tied_parameters_row(model: nn.Module) -> CompatRow:
    """Build the tied/shared parameter row.

    Parameters
    ----------
    model:
        Model to inspect.

    Returns
    -------
    CompatRow
        Report row.
    """

    seen: dict[int, str] = {}
    duplicates: list[str] = []
    inspected = True
    try:
        for name, parameter in _iter_named_parameters_no_dedup(model):
            param_id = id(parameter)
            if param_id in seen:
                duplicates.append(f"{seen[param_id]}={name}")
            else:
                seen[param_id] = name
    except Exception:
        inspected = False
    detected = bool(duplicates)
    if not inspected:
        details = "Parameter enumeration failed; tied/shared parameters could not be inspected."
    elif detected:
        details = (
            "Shared parameter objects detected; TorchLens tracks parameter identity and should "
            f"preserve tied-edge metadata ({', '.join(duplicates[:3])})."
        )
    else:
        details = "No tied/shared parameter objects detected."
    return CompatRow(
        "tied_parameters",
        "Tied/shared parameters",
        "pass",
        "info" if detected else "ok",
        detected,
        details,
        "",
    )


def _iter_named_parameters_no_dedup(model: nn.Module) -> Iterable[tuple[str, nn.Parameter]]:
    """Yield ``(name, parameter)`` pairs preserving shared-object duplicates.

    ``named_parameters(remove_duplicate=False)`` is the primary source. If the model
    overrides ``named_parameters`` with a signature that rejects that keyword
    (older or custom signatures), fall back to a non-deduplicating walk over
    registered parameter slots so tied objects stay visible instead of silently
    collapsing into one entry (which would make ties invisible and the row a false
    ``pass``).

    Parameters
    ----------
    model:
        Model whose parameters are enumerated.

    Yields
    ------
    tuple[str, torch.nn.Parameter]
        Qualified parameter name and parameter object, duplicates preserved.
    """

    try:
        yield from model.named_parameters(remove_duplicate=False)
        return
    except TypeError:
        pass
    try:
        modules = tuple(model.named_modules(remove_duplicate=False))
    except TypeError:
        modules = tuple(model.named_modules())
    for module_name, module in modules:
        for param_name, parameter in getattr(module, "_parameters", {}).items():
            if parameter is None:
                continue
            yield (f"{module_name}.{param_name}" if module_name else param_name), parameter


def _multi_gpu_rng_row() -> CompatRow:
    """Build the multi-GPU RNG row.

    Returns
    -------
    CompatRow
        Report row.
    """

    try:
        device_count = torch.cuda.device_count()
    except Exception:
        device_count = 0
    detected = device_count > 1
    details = (
        f"{device_count} CUDA devices visible; TorchLens snapshots/restores RNG state for all "
        "CUDA devices via torch.cuda.get_rng_state_all()."
        if detected
        else f"{device_count} CUDA device(s) visible; multi-GPU RNG replay was not exercised."
    )
    return CompatRow(
        "multi_gpu_rng",
        "Multi-GPU RNG",
        "pass",
        "info" if detected else "ok",
        detected,
        details,
        "",
    )


def _data_parallel_row(model: nn.Module) -> CompatRow:
    """Build the ``nn.DataParallel`` row.

    Parameters
    ----------
    model:
        Model to inspect.

    Returns
    -------
    CompatRow
        Report row.
    """

    detected = isinstance(model, nn.DataParallel)
    status: Status = "pass"
    details = (
        "nn.DataParallel detected. TorchLens unwraps .module for rank-local single-process "
        "capture; threaded replica internals are not expanded."
        if detected
        else "nn.DataParallel not detected."
    )
    return CompatRow(
        "data_parallel",
        "nn.DataParallel",
        status,
        "info" if detected else "ok",
        detected,
        details,
        "Call torchlens.trace(model.module, x) if you need to bypass the wrapper explicitly."
        if detected
        else "",
    )


def _ddp_row(model: nn.Module) -> CompatRow:
    """Build the DistributedDataParallel row.

    Parameters
    ----------
    model:
        Model to inspect.

    Returns
    -------
    CompatRow
        Report row.
    """

    detected = _class_in_namespace(model, ("torch.nn.parallel.distributed",))
    details = (
        "DistributedDataParallel detected; TorchLens unwraps the rank-local .module and "
        "captures that eager module."
        if detected
        else "DistributedDataParallel not detected."
    )
    return CompatRow(
        "distributed_data_parallel",
        "DistributedDataParallel",
        "pass",
        "info" if detected else "ok",
        detected,
        details,
        "",
    )


def _fsdp_row(model: nn.Module) -> CompatRow:
    """Build the FSDP row.

    Parameters
    ----------
    model:
        Model to inspect.

    Returns
    -------
    CompatRow
        Report row.
    """

    detected = _class_in_namespace(
        model, ("torch.distributed.fsdp", "torch.distributed._composable.fsdp")
    )
    status: Status = "scope" if detected else "pass"
    details = (
        "FSDP detected; sharded parameter materialization is outside TorchLens' launch scope."
        if detected
        else "FSDP not detected."
    )
    return CompatRow(
        "fsdp",
        "FSDP",
        status,
        "warning" if detected else "ok",
        detected,
        details,
        "Log a rank-local unsharded copy before FSDP wrapping." if detected else "",
    )


# Stable row order and labels for the distributed-detection block. Each key
# matches a DistributedFinding.kind so the report and the capture-entry refusal
# can never drift apart.
_DISTRIBUTED_ROW_SPECS: tuple[tuple[str, str, str], ...] = (
    (
        "dtensor",
        "DTensor / sharded tensors",
        "No DTensor or sharded tensor state detected by the bounded entry scan. "
        "It covers registered state, builtin/instance-__dict__ input containers, and plain "
        "module attributes; descriptor-only or slots-only containers and tensors created "
        "inside forward remain outside entry-time detection.",
    ),
    ("device_mesh", "Device mesh", "No device mesh detected."),
    (
        "tensor_parallel",
        "Tensor parallel (TP)",
        "No tensor-parallel state or direct TP-namespace forward hook detected by the bounded "
        "entry scan; user-wrapped or opaque hook callables remain outside structural detection.",
    ),
    (
        "pipeline_parallel",
        "Pipeline parallel (PP)",
        "No pipeline-parallel stage or schedule detected by the bounded instance-state scan; "
        "descriptor-only or slots-only holders remain opaque.",
    ),
)


def _distributed_rows(model: nn.Module, input_value: Any) -> tuple[CompatRow, ...]:
    """Build the DTensor / device-mesh / TP / PP rows.

    Parameters
    ----------
    model:
        Model to inspect.
    input_value:
        Example input tree to inspect for distributed tensors.

    Returns
    -------
    tuple[CompatRow, ...]
        One row per distributed condition, in stable order, whether or not the
        condition was detected.

    Notes
    -----
    Detection is shared verbatim with the capture-entry refusal in
    :func:`torchlens._distributed.check_distributed_capture`, so a row reporting
    a refusing condition and the error the user then hits cannot disagree.
    """

    findings = {finding.kind: finding for finding in detect_distributed_state(model, input_value)}
    return tuple(
        _distributed_row(key, label, clear_details, findings.get(key))
        for key, label, clear_details in _DISTRIBUTED_ROW_SPECS
    )


def _distributed_row(
    key: str,
    label: str,
    clear_details: str,
    finding: DistributedFinding | None,
) -> CompatRow:
    """Build one distributed-detection row from an optional finding.

    Parameters
    ----------
    key:
        Stable row key, equal to the matching ``DistributedFinding.kind``.
    label:
        Human-readable row label.
    clear_details:
        Details text used when the condition was not detected.
    finding:
        Detected finding, or ``None`` when the condition is absent.

    Returns
    -------
    CompatRow
        Report row.
    """

    if finding is None:
        return CompatRow(key, label, "pass", "ok", False, clear_details, "")
    details = finding.detail
    sites = finding.describe_sites()
    if sites:
        details = f"{details} Sites: {sites}."
    if not finding.exact:
        details = (
            f"{details} Detected structurally (by type namespace), because this torch build "
            "did not expose the exact class for an isinstance check."
        )
    if finding.refuses_capture:
        details = (
            f"{details} torchlens.trace() refuses this model with "
            "DistributedCaptureUnsupportedError rather than returning a wrong trace."
        )
        return CompatRow(key, label, "scope", "error", True, details, finding.suggestion)
    return CompatRow(key, label, "scope", "warning", True, details, finding.suggestion)


def _deepspeed_row(model: nn.Module) -> CompatRow:
    """Build the DeepSpeed row.

    Parameters
    ----------
    model:
        Model to inspect.

    Returns
    -------
    CompatRow
        Report row.
    """

    detected = _class_in_namespace(model, ("deepspeed",))
    status: Status = "scope" if detected else "pass"
    details = (
        "DeepSpeed engine detected; ZeRO/offload execution is outside TorchLens' launch scope."
        if detected
        else "DeepSpeed not detected."
    )
    return CompatRow(
        "deepspeed",
        "DeepSpeed",
        status,
        "warning" if detected else "ok",
        detected,
        details,
        "Log the underlying eager module outside DeepSpeed." if detected else "",
    )


def _torch_compile_row(model: nn.Module) -> CompatRow:
    """Build the ``torch.compile`` row.

    Parameters
    ----------
    model:
        Model to inspect.

    Returns
    -------
    CompatRow
        Report row.
    """

    from .._capture_state_helpers import compiled_plain_callable_sites
    from ..utils import _torch_compat

    optimized_module_type = get_dynamo_optimized_module_type()
    optimized_detected = optimized_module_type is not None and isinstance(
        model, optimized_module_type
    )
    plain_callable_sites = compiled_plain_callable_sites(model)
    detected = optimized_detected or bool(plain_callable_sites)
    # The coexistence contract shipped by the rung-2 stance integration (torch
    # >= 2.6): compiled callables run their original eager Python during
    # capture with zero graph breaks, compiled caches stay intact with at most
    # one bounded recompile on the next compiled call afterward, and
    # unwrap_torch() reverts torch for free.
    stance_available = bool(_torch_compat.HAS_SET_STANCE)
    contract = (
        "Coexistence contract: zero graph breaks during capture, at most one bounded "
        "recompile on the next compiled call afterward, and unwrap_torch() reverts "
        "torch for free. Captured values are eager-path values, not compiled-path "
        "numerics."
    )
    if stance_available:
        status: Status = "pass"
        if optimized_detected:
            details = (
                "torch.compile OptimizedModule detected; capture traces the eager source "
                "module and runs compiled callables under "
                "torch.compiler.set_stance('force_eager'), so interiors are fully logged "
                f"with ordinary verified semantics. {contract}"
            )
        elif plain_callable_sites:
            details = (
                "torch.compile callable detected on a plain module attribute at "
                f"{', '.join(plain_callable_sites)}. Capture runs it through its original "
                f"eager Python under set_stance, so its interior IS logged. {contract}"
            )
        else:
            details = (
                "No OptimizedModule or direct plain-attribute compiled callable detected. "
                "On this torch (set_stance available), compiled callables reached during "
                "capture -- including globals/free-function references outside this "
                "structural preflight -- run their original eager Python and are logged."
            )
        suggestion = (
            "Verify the contract with tl.debug.count_compiles(); correlate Dynamo graph "
            "breaks with tl.debug.graph_breaks(). For compiled-artifact introspection "
            "use the ecosystem tools (torch DebugMode, tlparse, the profiler, depyf)."
            if detected
            else ""
        )
        return CompatRow(
            "torch_compile",
            "torch.compile",
            status,
            "ok",
            detected,
            details,
            suggestion,
        )
    status = "scope" if detected else "pass"
    if optimized_detected:
        details = (
            "torch.compile OptimizedModule detected; compiled graph capture is outside "
            "TorchLens' primary scope."
        )
    elif plain_callable_sites:
        details = (
            "torch.compile callable detected on a plain module attribute at "
            f"{', '.join(plain_callable_sites)}. Capture marks its compiled interior incomplete, "
            "including on warm-cache execution."
        )
    else:
        details = (
            "No OptimizedModule or direct plain-attribute compiled callable detected. Compiled "
            "callables reached only through globals/free-function references remain outside "
            "this structural preflight."
        )
    return CompatRow(
        "torch_compile",
        "torch.compile",
        status,
        "warning" if detected else "ok",
        detected,
        details,
        "Log the original eager model (torch >= 2.6 captures compiled callables eagerly "
        "via set_stance). Correlate Dynamo graph breaks with tl.debug.graph_breaks(); "
        "for compiled-code context use torch DebugMode, tlparse, or torchlens.bridge.depyf."
        if detected
        else "",
    )


def _fx_row(model: nn.Module) -> CompatRow:
    """Build the FX GraphModule row.

    Parameters
    ----------
    model:
        Model to inspect.

    Returns
    -------
    CompatRow
        Report row.
    """

    graph_module_type = get_fx_graph_module_type()
    detected = graph_module_type is not None and isinstance(model, graph_module_type)
    status: Status = "scope" if detected else "pass"
    details = (
        "torch.fx.GraphModule detected; TorchLens launch support targets eager nn.Module "
        "execution rather than FX IR parity."
        if detected
        else "FX GraphModule not detected."
    )
    return CompatRow(
        "fx_graph_module",
        "FX GraphModule",
        status,
        "warning" if detected else "ok",
        detected,
        details,
        "Log the pre-FX eager module or use torchlens.compat.from_fx for migration help."
        if detected
        else "",
    )


def _lightning_row(model: nn.Module) -> CompatRow:
    """Build the Lightning training-step row.

    Parameters
    ----------
    model:
        Model to inspect.

    Returns
    -------
    CompatRow
        Report row.
    """

    has_training_step = callable(getattr(model, "training_step", None))
    is_lightning = _class_in_namespace(model, ("pytorch_lightning", "lightning.pytorch"))
    is_train_mode = bool(getattr(model, "training", False))
    detected = has_training_step and is_lightning and is_train_mode
    status: Status = "known_broken" if detected else "pass"
    details = (
        "LightningModule training_step detected while the module is in training mode; mid-loop "
        "trainer capture is not a supported TorchLens entry point."
        if detected
        else "Lightning training_step not detected in an active training-mode LightningModule."
    )
    return CompatRow(
        "lightning_training_step",
        "Lightning training_step mid-loop",
        status,
        "error" if detected else "ok",
        detected,
        details,
        "Use torchlens.callbacks.lightning.LayerProfilerCallback or log a plain forward."
        if detected
        else "",
    )


def _functorch_row(model: nn.Module) -> CompatRow:
    """Build the vmap/functorch row.

    Parameters
    ----------
    model:
        Model to inspect.

    Returns
    -------
    CompatRow
        Report row.
    """

    detected = _forward_references_functorch(model)
    status: Status = "known_broken" if detected else "pass"
    details = (
        "forward source references vmap/functorch; TorchLens skips logging inside active "
        "functorch transforms and will produce an incomplete log."
        if detected
        else (
            "No static vmap/functorch marker detected in forward source. Functional tensors "
            "already present in inspectable input/state containers are refused at entry; private "
            "functional tensors created inside forward remain outside that preflight."
        )
    )
    return CompatRow(
        "vmap_functorch",
        "vmap/functorch",
        status,
        "error" if detected else "ok",
        detected,
        details,
        "Log the non-vmap module separately when a complete operation trace is required."
        if detected
        else "",
    )


def _forward_references_functorch(model: nn.Module) -> bool:
    """Return whether ``model.forward`` references vmap/functorch in executable code.

    The forward source is parsed into an AST and searched for real name/attribute
    references (``vmap``, ``functorch``, or the ``torch.func`` submodule). Comments
    and docstrings are ignored, so prose that merely mentions vmap (for example a
    docstring saying the model does *not* use vmap) does not trip detection.

    Parameters
    ----------
    model:
        Model whose ``forward`` source is inspected.

    Returns
    -------
    bool
        True only when forward code references a functorch/vmap marker.
    """

    try:
        source = inspect.getsource(model.forward)
    except (OSError, TypeError):
        return False
    try:
        tree = ast.parse(textwrap.dedent(source))
    except (SyntaxError, ValueError):
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in ("vmap", "functorch"):
            return True
        if isinstance(node, ast.Attribute):
            if node.attr == "vmap":
                return True
            if (
                node.attr == "func"
                and isinstance(node.value, ast.Name)
                and node.value.id == "torch"
            ):
                return True
    return False


def _quantized_row(model: nn.Module, input_value: Any) -> CompatRow:
    """Build the quantized tensor/model row.

    Parameters
    ----------
    model:
        Model to inspect.
    input_value:
        Input tree to inspect.

    Returns
    -------
    CompatRow
        Report row.
    """

    quantized_input = any(
        getattr(tensor, "is_quantized", False) for tensor in _iter_tensors(input_value)
    )
    quantized_module = any(_is_quantized_module(module) for module in _iter_modules(model))
    detected = quantized_input or quantized_module
    status: Status = "known_broken" if detected else "pass"
    details = (
        "Quantized tensors/modules detected. tensor_nanequal no longer crashes on quantized "
        "tensors, but full quantized-model out metadata remains best-effort."
        if detected
        else "No quantized tensors or quantized nn modules detected."
    )
    return CompatRow(
        "quantized_tensor",
        "Quantized tensors/modules",
        status,
        "error" if detected else "ok",
        detected,
        details,
        "Use a float reference model for bugs involving exact out validation." if detected else "",
    )


def _fp8_dtype_row(model: nn.Module, input_value: Any) -> CompatRow:
    """Build the fp8 (``float8_*``) dtype row.

    Parameters
    ----------
    model:
        Model to inspect.
    input_value:
        Input tree to inspect.

    Returns
    -------
    CompatRow
        Report row.

    Notes
    -----
    Reports parameters, buffers, and inputs only -- the same pre-capture surface every
    other row inspects. An fp8 tensor produced *inside* the forward (the common case,
    since fp8 is usually a cast of a float32 activation) cannot be seen from here, so
    the row's ``detected=False`` never claims a capture contains no fp8, and the
    passing detail says which scopes were checked.
    """

    fp8_dtypes = get_fp8_dtypes(force_probe=True)
    if not fp8_dtypes:
        return CompatRow(
            "fp8_dtype",
            "fp8 (float8_*) tensors",
            "pass",
            "ok",
            False,
            "This torch build exposes no float8 dtypes.",
        )
    fp8_input = any(tensor.dtype in fp8_dtypes for tensor in _iter_tensors(input_value))
    inspected_state = True
    fp8_state = False
    try:
        fp8_state = any(
            tensor.dtype in fp8_dtypes
            for tensor in itertools.chain(model.parameters(), model.buffers())
        )
    except Exception:  # noqa: BLE001 - a model may override enumeration and raise
        inspected_state = False
    detected = fp8_input or fp8_state
    if not detected:
        # Same fail-open-honestly contract as the tied-parameters row: say that the
        # scope could not be read rather than reporting a clean pass over it.
        unread = (
            ""
            if inspected_state
            else " Parameter/buffer enumeration failed, so model state was NOT inspected."
        )
        return CompatRow(
            "fp8_dtype",
            "fp8 (float8_*) tensors",
            "pass" if inspected_state else "not_tested",
            "ok" if inspected_state else "info",
            False,
            "No float8 parameters, buffers, or inputs detected (an fp8 cast performed "
            f"inside the forward is not visible before capture).{unread}",
        )
    where = " and ".join(
        label for label, hit in (("inputs", fp8_input), ("parameters/buffers", fp8_state)) if hit
    )
    return CompatRow(
        "fp8_dtype",
        "fp8 (float8_*) tensors",
        "scope",
        "warning",
        True,
        f"float8 tensors detected in {where}. Capture, metadata, and validation replay "
        "handle them: torch implements no isinf/nan_to_num/allclose/isfinite/reduction "
        "kernels for fp8, so TorchLens widens those comparisons to float32, which is "
        "exact for every fp8 bit pattern. Saving an fp8 activation to a portable "
        "`.tlspec` is refused with a typed error, because safetensors has no fp8 "
        "transport this release.",
        "Nothing to change for in-RAM analysis. To persist an fp8 activation, cast it "
        "to float32/bfloat16 before the save, or save at metadata level.",
    )


def _is_quantized_module(module: nn.Module) -> bool:
    """Return whether a module appears to come from PyTorch quantization namespaces.

    Parameters
    ----------
    module:
        Module to inspect.

    Returns
    -------
    bool
        True for quantized/QAT module classes.
    """

    identity = _class_identity(module)
    prefixes = (
        "torch.ao.nn.quantized",
        "torch.nn.quantized",
        "torch.ao.nn.intrinsic.quantized",
        "torch.ao.nn.qat",
        "torch.nn.qat",
    )
    return any(identity.startswith(prefix) for prefix in prefixes)


def _device_context_row() -> CompatRow:
    """Build the DeviceContext factory row.

    Returns
    -------
    CompatRow
        Report row.
    """

    return CompatRow(
        "device_context_factory",
        "DeviceContext factory injection",
        "pass",
        "ok",
        False,
        "Factory functions honor active torch.device(...) contexts during active logging.",
        "",
    )


def _torch_capabilities_row() -> CompatRow:
    """Build the runtime capability snapshot row.

    Returns
    -------
    CompatRow
        Report row summarizing private runtime capability probes.
    """

    snapshot = _runtime_capability_snapshot()
    missing = [name for name, available in snapshot.items() if not available]
    status: Status = "not_tested" if missing else "pass"
    severity: Severity = "warning" if missing else "ok"
    details = "Runtime capabilities: " + _format_capability_snapshot(snapshot)
    if missing:
        details += "; missing=" + ", ".join(missing)
    suggestion = (
        "Run torchlens.utils.doctor() for the same snapshot; missing flags indicate graceful "
        "degradation of private runtime integration points."
        if missing
        else ""
    )
    return CompatRow(
        "torch_capabilities",
        "Runtime capability snapshot",
        status,
        severity,
        bool(missing),
        details,
        suggestion,
    )


def _runtime_capability_snapshot() -> dict[str, bool]:
    """Return all runtime compatibility capability flags.

    Returns
    -------
    dict[str, bool]
        Mapping from capability flag names to availability.
    """

    snapshot = get_torch_capability_snapshot()
    try:
        from torchlens.backends.tf._tf_compat import get_tf_capability_snapshot
    except ImportError:
        return snapshot
    snapshot.update(get_tf_capability_snapshot())
    return snapshot


def _format_capability_snapshot(snapshot: dict[str, bool]) -> str:
    """Format capability flags as a stable comma-separated list.

    Parameters
    ----------
    snapshot:
        Capability flags to format.

    Returns
    -------
    str
        Stable ``name=value`` list.
    """

    return ", ".join(f"{name}={available}" for name, available in sorted(snapshot.items()))


def _single_thread_row() -> CompatRow:
    """Build the single-thread design row.

    Returns
    -------
    CompatRow
        Report row.
    """

    current_process = multiprocessing.current_process().name
    current_thread = threading.current_thread().name
    detected = current_process != "MainProcess" or current_thread != "MainThread"
    status: Status = "known_broken" if detected else "pass"
    details = (
        f"Running in {current_process}/{current_thread}; TorchLens capture is single-threaded "
        "and process-global."
        if detected
        else "Running in MainProcess/MainThread; this matches TorchLens' single-thread design."
    )
    return CompatRow(
        "single_thread_design",
        "Single-thread design",
        status,
        "error" if detected else "ok",
        detected,
        details,
        "Run capture from the main process and main thread." if detected else "",
    )


__all__ = ["CompatReport", "CompatRow", "Severity", "Status", "report"]
