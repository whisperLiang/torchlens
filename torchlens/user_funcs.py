"""Public API entry points for TorchLens.

This module contains every user-facing function:
  - ``log_forward_pass``  — the main entry point (runs model, returns ModelLog)
  - ``validate_forward_pass`` — replay-based correctness check
  - ``show_model_graph`` — visualization convenience wrapper
  - ``get_model_metadata`` — metadata-only convenience wrapper (deprecated path)
  - ``validate_batch_of_models_and_inputs`` — bulk validation harness

**Two-pass strategy** (``log_forward_pass`` with selective layers):
When the user requests specific layers (not "all" or "none"), TorchLens must
first run an exhaustive pass to discover the full graph structure — only then can
it resolve user-friendly layer names/indices to internal layer numbers.  A second
fast pass replays the model, saving only the requested activations.  This is why
``log_forward_pass`` has two branches: the simple path (save all/none) and the
two-pass path (save specific layers).
"""

import collections.abc
import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from .utils.introspection import get_vars_of_type_from_obj
from .utils.rng import set_random_seed
from .utils.display import warn_parallel, _vprint
from .utils.arg_handling import safe_copy_args, safe_copy_kwargs, normalize_input_args
from .utils.collections import assign_to_sequence_or_dict, index_nested
from .utils.tensor_utils import tensor_nanequal
from .data_classes.model_log import (
    ModelLog,
)


def _unwrap_data_parallel(model: nn.Module) -> nn.Module:
    """Unwrap nn.DataParallel to get the underlying module."""
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


def _move_tensors_to_device(obj, device):
    """Recursively move tensors in a nested structure (lists, tuples, dicts) to *device*.

    Handles common dict-like types (OrderedDict, HuggingFace BatchEncoding, etc.)
    by attempting to reconstruct the original container type after moving values.
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, (list, tuple)):
        moved = [_move_tensors_to_device(item, device) for item in obj]
        return type(obj)(moved) if not isinstance(obj, tuple) else tuple(moved)
    elif isinstance(obj, collections.abc.MutableMapping):
        # Handles dict, UserDict, BatchEncoding, OrderedDict, etc.
        moved = {k: _move_tensors_to_device(v, device) for k, v in obj.items()}
        if type(obj) is dict:
            return moved
        try:
            return type(obj)(moved)
        except Exception:
            return moved
    return obj


def _run_model_and_save_specified_activations(
    model: nn.Module,
    input_args: Union[torch.Tensor, List[Any]],
    input_kwargs: Dict[Any, Any],
    layers_to_save: Optional[Union[str, List[Union[int, str]]]] = "all",
    keep_unsaved_layers: bool = True,
    output_device: str = "same",
    activation_postfunc: Optional[Callable] = None,
    mark_input_output_distances: bool = False,
    detach_saved_tensors: bool = False,
    save_function_args: bool = False,
    save_gradients: bool = False,
    random_seed: Optional[int] = None,
    num_context_lines: int = 7,
    optimizer=None,
    save_source_context: bool = False,
    save_rng_states: bool = False,
    detect_loops: bool = True,
    verbose: bool = False,
) -> ModelLog:
    """Run a forward pass with logging enabled, returning a populated ModelLog.

    This is the single internal entry point that creates a ModelLog, configures it,
    and delegates to ``ModelLog._run_and_log_inputs_through_model`` which handles
    model preparation, the exhaustive (and optionally fast) forward pass, and all
    postprocessing.

    Args:
        model: PyTorch model.
        input_args: Positional arguments to model.forward(); a single tensor or list.
        input_kwargs: Keyword arguments to model.forward().
        layers_to_save: Which layers to save activations for ('all', 'none'/None, or a list).
        keep_unsaved_layers: If False, layers without saved activations are pruned from the log.
        output_device: Device for saved tensors: 'same' (default), 'cpu', or 'cuda'.
        activation_postfunc: Optional transform applied to each activation before storage
            (e.g., channel-wise averaging to reduce memory).
        mark_input_output_distances: Compute BFS distances from input/output layers.
            Expensive for large graphs — off by default.
        detach_saved_tensors: If True, saved tensors are detached from the autograd graph.
        save_function_args: If True, store the non-tensor arguments to each function call.
            Required for validation replay (``validate_saved_activations``).
        save_gradients: If True, register backward hooks to capture gradients.
        random_seed: Fixed RNG seed for reproducibility (important for stochastic models).
        num_context_lines: Number of source-code context lines stored per function call.
        optimizer: Optional optimizer — used to tag which parameters have optimizers attached.
        detect_loops: If True (default), run full isomorphic subgraph expansion to
            detect repeated patterns (loops). If False, only group operations that
            share the same parameters — much faster for very large graphs.
        verbose: If True, print timed progress messages at each major pipeline stage.

    Returns:
        Fully-populated ModelLog.
    """
    # Auto-detect model device from its first parameter and move inputs to match.
    # This prevents silent device-mismatch errors when the model is on CUDA but
    # the user passes CPU tensors (a common mistake).
    model_device = next((p.device for p in model.parameters()), None)
    if model_device is not None:
        input_args = _move_tensors_to_device(input_args, model_device)
        if input_kwargs is not None:
            input_kwargs = _move_tensors_to_device(input_kwargs, model_device)

    model_name = str(type(model).__name__)
    model_log = ModelLog(
        model_name,
        output_device,
        activation_postfunc,
        keep_unsaved_layers,
        save_function_args,
        save_gradients,
        detach_saved_tensors,
        mark_input_output_distances,
        num_context_lines,
        optimizer,
        save_source_context,
        save_rng_states,
        detect_loops,
        verbose,
    )
    model_log._run_and_log_inputs_through_model(
        model, input_args, input_kwargs, layers_to_save, random_seed
    )
    return model_log


def log_forward_pass(
    model: nn.Module,
    input_args: Union[torch.Tensor, List[Any], Tuple[Any]],
    input_kwargs: Optional[Dict[Any, Any]] = None,
    layers_to_save: Optional[Union[str, List]] = "all",
    keep_unsaved_layers: bool = True,
    output_device: str = "same",
    activation_postfunc: Optional[Callable] = None,
    mark_input_output_distances: bool = False,
    detach_saved_tensors: bool = False,
    save_function_args: bool = False,
    save_gradients: bool = False,
    save_source_context: bool = False,
    save_rng_states: bool = False,
    vis_mode: str = "none",
    vis_nesting_depth: int = 1000,
    vis_outpath: str = "graph.gv",
    vis_save_only: bool = False,
    vis_fileformat: str = "pdf",
    vis_buffer_layers: bool = False,
    vis_direction: str = "bottomup",
    vis_graph_overrides: Optional[Dict] = None,
    vis_node_overrides: Optional[Dict] = None,
    vis_nested_node_overrides: Optional[Dict] = None,
    vis_edge_overrides: Optional[Dict] = None,
    vis_gradient_edge_overrides: Optional[Dict] = None,
    vis_module_overrides: Optional[Dict] = None,
    vis_node_placement: str = "auto",
    vis_renderer: str = "graphviz",
    vis_theme: str = "torchlens",
    random_seed: Optional[int] = None,
    num_context_lines: int = 7,
    optimizer=None,
    detect_loops: bool = True,
    unwrap_when_done: bool = False,
    verbose: bool = False,
) -> ModelLog:
    """Run a forward pass through *model*, log every operation, and return a ModelLog.

    This is the primary user-facing entry point for TorchLens.  It intercepts every
    tensor-producing operation during ``model.forward()``, records metadata and
    (optionally) saves activations, then returns a ``ModelLog`` that provides
    dict-like access to every layer's data.

    Torch functions are automatically wrapped on the first call and stay wrapped
    afterward.  Pass ``unwrap_when_done=True`` to restore the original torch
    callables after logging completes.

    **Layer selection** (``layers_to_save``):

    - ``'all'`` (default) — save activations for every layer.
    - ``'none'`` / ``None`` / ``[]`` — save no activations (metadata only).
    - A list containing any mix of:
      1. Layer name, e.g. ``'conv2d_1_1'`` (all passes).
      2. Pass-qualified label, e.g. ``'conv2d_1_1:2'`` (second pass only).
      3. Module address, e.g. ``'features.0'`` (output of that module).
      4. Integer index (ordinal position; negative indices work).
      5. Substring filter, e.g. ``'conv2d'`` (all matching layers).

    When specific layers are requested, a **two-pass strategy** is used: first an
    exhaustive pass discovers the full graph structure (needed to resolve names),
    then ``save_new_activations`` replays the model in fast mode to save only the
    requested layers.  For ``'all'`` or ``'none'``, a single pass suffices.

    Args:
        model: PyTorch model.
        input_args: Positional args for ``model.forward()``; a single tensor or list.
        input_kwargs: Keyword args for ``model.forward()``.
        layers_to_save: Which layers to save activations for (see above).
        keep_unsaved_layers: If False, layers without saved activations are removed from
            the returned ModelLog (they still exist during processing).
        output_device: Device for stored tensors: ``'same'``, ``'cpu'``, or ``'cuda'``.
        activation_postfunc: Optional function applied to each activation before saving.
        mark_input_output_distances: Compute BFS distances from inputs/outputs (expensive).
        detach_saved_tensors: If True, detach saved tensors from the autograd graph.
        save_function_args: Store non-tensor args for each function call (needed for
            ``validate_saved_activations``).
        save_gradients: Capture gradients during a subsequent backward pass.
        save_source_context: If True, record the Python call stack for each
            tensor operation and capture module source code (file, line, signatures).
            Default False for speed; enable for debugging and code inspection.
        save_rng_states: If True, capture RNG states before each operation (needed for
            validation replay of stochastic ops like dropout). Auto-enabled when
            ``validate_forward_pass`` is used. Default False for speed.
        vis_mode: ``'none'`` (default), ``'rolled'``, or ``'unrolled'`` visualization.
        vis_nesting_depth: Max module nesting depth shown in visualization.
        vis_outpath: Output file path for the graph visualization.
        vis_save_only: If True, save the visualization file without displaying it.
        vis_fileformat: Image format (``'pdf'``, ``'png'``, ``'jpg'``, etc.).
        vis_buffer_layers: Include buffer layers in the visualization.
        vis_direction: Layout direction: ``'bottomup'``, ``'topdown'``, or ``'leftright'``.
        vis_graph_overrides: Graphviz graph-level attribute overrides.
        vis_node_overrides: Graphviz node attribute overrides.
        vis_nested_node_overrides: Graphviz attribute overrides for nested (module) nodes.
        vis_edge_overrides: Graphviz edge attribute overrides.
        vis_gradient_edge_overrides: Graphviz attribute overrides for gradient edges.
        vis_module_overrides: Graphviz subgraph (module cluster) attribute overrides.
        vis_node_placement: Layout engine: ``'auto'`` (default), ``'dot'``, ``'elk'``,
            or ``'sfdp'``.  ``'auto'`` uses dot for small graphs and ELK/sfdp for large.
        vis_renderer: Visualization backend: ``'graphviz'`` (default) or ``'dagua'``.
        vis_theme: Named Dagua theme when ``vis_renderer='dagua'``.
        random_seed: Fixed RNG seed for reproducibility with stochastic models.
        num_context_lines: Lines of source context to capture per function call.
        optimizer: Optional optimizer to annotate which params are being optimized.
        detect_loops: If True (default), run full isomorphic subgraph expansion.
        unwrap_when_done: If True, restore original torch callables after logging.
            Default False — torch stays wrapped for subsequent calls.
        verbose: If True, print timed progress messages at each major pipeline stage.

    Returns:
        A ``ModelLog`` containing layer activations (if requested) and full metadata.
    """
    # DataParallel is not supported — unwrap and warn if present.
    warn_parallel()
    model = _unwrap_data_parallel(model)

    if vis_mode not in ["none", "rolled", "unrolled"]:
        raise ValueError("Visualization option must be either 'none', 'rolled', or 'unrolled'.")

    if output_device not in ["same", "cpu", "cuda"]:
        raise ValueError("output_device must be either 'same', 'cpu', or 'cuda'.")

    if type(layers_to_save) is str:
        layers_to_save = layers_to_save.lower()

    if layers_to_save in ["all", "none", None, []]:
        # --- SINGLE-PASS path ---
        # "all" or "none": no name resolution needed, so one pass suffices.
        model_log = _run_model_and_save_specified_activations(
            model=model,
            input_args=input_args,  # type: ignore[arg-type]
            input_kwargs=input_kwargs,  # type: ignore[arg-type]
            layers_to_save=layers_to_save,
            keep_unsaved_layers=keep_unsaved_layers,
            output_device=output_device,
            activation_postfunc=activation_postfunc,
            mark_input_output_distances=mark_input_output_distances,
            detach_saved_tensors=detach_saved_tensors,
            save_function_args=save_function_args,
            save_gradients=save_gradients,
            random_seed=random_seed,
            num_context_lines=num_context_lines,
            optimizer=optimizer,
            save_source_context=save_source_context,
            save_rng_states=save_rng_states,
            detect_loops=detect_loops,
            verbose=verbose,
        )
    else:
        # --- TWO-PASS path ---
        # Pass 1 (exhaustive): Run with layers_to_save=None and keep_unsaved_layers=True
        # so the full graph is discovered and all layer labels are assigned.  No
        # activations are saved yet — this pass is purely for metadata/structure.
        if verbose:
            print("[torchlens] Two-pass mode: Pass 1 (exhaustive, metadata only)")
        model_log = _run_model_and_save_specified_activations(
            model=model,
            input_args=input_args,  # type: ignore[arg-type]
            input_kwargs=input_kwargs,  # type: ignore[arg-type]
            layers_to_save=None,
            keep_unsaved_layers=True,
            output_device=output_device,
            activation_postfunc=activation_postfunc,
            mark_input_output_distances=mark_input_output_distances,
            detach_saved_tensors=detach_saved_tensors,
            save_function_args=save_function_args,
            save_gradients=save_gradients,
            random_seed=random_seed,
            num_context_lines=num_context_lines,
            optimizer=optimizer,
            save_source_context=save_source_context,
            save_rng_states=save_rng_states,
            detect_loops=detect_loops,
            verbose=verbose,
        )
        # Pass 2 (fast): Now that layer labels exist, resolve the user's requested
        # layers and replay the model, saving only the matching activations.
        _vprint(model_log, "Two-pass mode: Pass 2 (fast, saving requested layers)")
        model_log.keep_unsaved_layers = keep_unsaved_layers
        model_log.save_new_activations(
            model=model,
            input_args=input_args,  # type: ignore[arg-type]
            input_kwargs=input_kwargs,
            layers_to_save=layers_to_save,  # type: ignore[arg-type]
            random_seed=random_seed,
        )

    # Print final summary.
    _vprint(
        model_log,
        f"Done: {len(model_log.layer_logs)} layers, "
        f"{model_log.num_tensors_saved} saved, "
        f"{model_log.total_activation_memory_str}",
    )

    # Visualize if desired.
    if vis_mode != "none":
        model_log.render_graph(
            vis_mode,
            vis_nesting_depth,
            vis_outpath,
            vis_graph_overrides,
            vis_node_overrides,
            vis_nested_node_overrides,
            vis_edge_overrides,
            vis_gradient_edge_overrides,
            vis_module_overrides,
            vis_save_only,
            vis_fileformat,
            vis_buffer_layers,
            vis_direction,
            vis_node_placement=vis_node_placement,
            vis_renderer=vis_renderer,
            vis_theme=vis_theme,
        )

    if unwrap_when_done:
        from .decoration import unwrap_torch

        unwrap_torch()

    return model_log


def get_model_metadata(
    model: nn.Module,
    input_args: Union[torch.Tensor, List[Any], Tuple[Any]],
    input_kwargs: Optional[Dict[Any, Any]] = None,
) -> ModelLog:
    """Return model metadata without saving any activations.

    Equivalent to ``log_forward_pass(model, input_args, input_kwargs, layers_to_save=None,
    mark_input_output_distances=True)``.  Prefer using ``log_forward_pass`` directly —
    this wrapper exists for backward compatibility and may be removed in a future release.

    Args:
        model: PyTorch model to inspect.
        input_args: Positional args for ``model.forward()``.
        input_kwargs: Keyword args for ``model.forward()``.

    Returns:
        ModelLog with full metadata but no saved activations.
    """
    model_log = log_forward_pass(
        model,
        input_args,
        input_kwargs,
        layers_to_save=None,
        mark_input_output_distances=True,
    )
    return model_log


def show_model_graph(
    model: nn.Module,
    input_args: Union[torch.Tensor, List, Tuple],
    input_kwargs: Optional[Dict[Any, Any]] = None,
    vis_mode: str = "unrolled",
    vis_nesting_depth: int = 1000,
    vis_outpath: str = "graph.gv",
    vis_graph_overrides: Optional[Dict] = None,
    vis_node_overrides: Optional[Dict] = None,
    vis_nested_node_overrides: Optional[Dict] = None,
    vis_edge_overrides: Optional[Dict] = None,
    vis_gradient_edge_overrides: Optional[Dict] = None,
    vis_module_overrides: Optional[Dict] = None,
    vis_save_only: bool = False,
    vis_fileformat: str = "pdf",
    vis_buffer_layers: bool = False,
    vis_direction: str = "bottomup",
    vis_node_placement: str = "auto",
    vis_renderer: str = "graphviz",
    vis_theme: str = "torchlens",
    random_seed: Optional[int] = None,
    detect_loops: bool = True,
    verbose: bool = False,
) -> None:
    """Convenience wrapper: visualize the computational graph without saving activations.

    Runs an exhaustive forward pass (no activations saved) to discover the graph
    structure, renders the visualization, then cleans up the ModelLog.  For more
    control, use ``log_forward_pass`` with ``vis_mode`` set and access the ModelLog
    directly.

    Args:
        model: PyTorch model.
        input_args: Positional args for ``model.forward()``.
        input_kwargs: Keyword args for ``model.forward()``.
        vis_mode: ``'rolled'`` or ``'unrolled'`` (``'none'`` is accepted but a no-op).
        vis_nesting_depth: Max module nesting depth shown (default 1000 = all).
        vis_outpath: Output file path for the visualization.
        vis_save_only: If True, save without displaying.
        vis_fileformat: Image format (``'pdf'``, ``'png'``, ``'jpg'``, etc.).
        vis_buffer_layers: Include buffer layers in the visualization.
        vis_direction: ``'bottomup'``, ``'topdown'``, or ``'leftright'``.
        vis_renderer: Visualization backend: ``'graphviz'`` (default) or ``'dagua'``.
        vis_theme: Named Dagua theme when ``vis_renderer='dagua'``.
        random_seed: Fixed RNG seed for stochastic models.

    Returns:
        None.
    """
    model = _unwrap_data_parallel(model)
    if not input_kwargs:
        input_kwargs = {}

    if vis_mode not in ["none", "rolled", "unrolled"]:
        raise ValueError("Visualization option must be either 'none', 'rolled', or 'unrolled'.")

    model_log = _run_model_and_save_specified_activations(
        model=model,
        input_args=input_args,  # type: ignore[arg-type]
        input_kwargs=input_kwargs,
        layers_to_save=None,
        activation_postfunc=None,
        mark_input_output_distances=False,
        detach_saved_tensors=False,
        save_gradients=False,
        random_seed=random_seed,
        detect_loops=detect_loops,
        verbose=verbose,
    )
    # Render in a try/finally so temporary tl_ attributes on the model are
    # always cleaned up, even if Graphviz rendering raises.
    try:
        model_log.render_graph(
            vis_mode,
            vis_nesting_depth,
            vis_outpath,
            vis_graph_overrides,
            vis_node_overrides,
            vis_nested_node_overrides,
            vis_edge_overrides,
            vis_gradient_edge_overrides,
            vis_module_overrides,
            vis_save_only,
            vis_fileformat,
            vis_buffer_layers,
            vis_direction,
            vis_node_placement=vis_node_placement,
            vis_renderer=vis_renderer,
            vis_theme=vis_theme,
        )
    finally:
        model_log.cleanup()


def validate_forward_pass(
    model: nn.Module,
    input_args: Union[torch.Tensor, List[Any], Tuple[Any]],
    input_kwargs: Optional[Dict[Any, Any]] = None,
    random_seed: Union[int, None] = None,
    verbose: bool = False,
    validate_metadata: bool = True,
) -> bool:
    """Validate that saved activations faithfully reproduce the model's output.

    **How it works:**

    1. Run model.forward() *without* TorchLens to get ground-truth output tensors.
    2. Run ``log_forward_pass`` with ``save_function_args=True`` and ``layers_to_save='all'``
       to capture every activation and its creating function's arguments.
    3. Call ``ModelLog.validate_saved_activations`` which replays the forward pass
       layer-by-layer from saved activations, checking that the output matches
       ground truth.  It also injects random activations and verifies the output
       changes (proving the saved activations are actually used, not just ignored).
    4. If ``validate_metadata=True``, run comprehensive invariant checks on all
       metadata cross-references (graph edges, module containment, labels, etc.).

    **Why save_function_args=True is required:**  The validation replay re-executes
    each function using its saved non-tensor arguments (e.g., stride, padding for
    conv2d).  Without them, replay cannot reconstruct the correct computation.

    Args:
        model: PyTorch model.
        input_args: Input for which to validate the saved activations.
        input_kwargs: Keyword arguments for model forward pass.
        random_seed: Fixed RNG seed for reproducibility (auto-generated if None).
        verbose: If True, print detailed error messages on validation failure.
        validate_metadata: If True (default), also run metadata invariant checks.

    Returns:
        True if all validation checks pass, False otherwise.
    """
    warn_parallel()
    model = _unwrap_data_parallel(model)
    # Fix a random seed so both the ground-truth run and the logged run see
    # identical randomness (critical for models with dropout, etc.).
    if random_seed is None:
        random_seed = random.randint(1, 4294967294)
    set_random_seed(random_seed)
    input_args = normalize_input_args(input_args, model)
    if not input_kwargs:
        input_kwargs = {}
    # Deep-copy inputs so the ground-truth forward pass doesn't mutate the
    # originals (some models modify inputs in-place).
    input_args_copy = safe_copy_args(input_args)
    input_kwargs_copy = safe_copy_kwargs(input_kwargs)

    model_device = next((p.device for p in model.parameters()), None)
    if model_device is not None:
        input_args_copy = _move_tensors_to_device(input_args_copy, model_device)
        input_kwargs_copy = _move_tensors_to_device(input_kwargs_copy, model_device)

    # Step 1: Get ground-truth outputs by running the model *outside* TorchLens.
    # Save state_dict first because requires_grad forcing during logging can
    # alter parameter metadata; we restore it afterward.
    state_dict = model.state_dict()
    ground_truth_output_all = get_vars_of_type_from_obj(
        model(*input_args_copy, **input_kwargs_copy),
        torch.Tensor,
        search_depth=5,
        return_addresses=True,
        allow_repeats=True,
    )
    # Deduplicate by structural address to match how capture/trace.py extracts
    # outputs (same tensor returned in multiple positions is counted once).
    addresses_used = []
    ground_truth_output_tensors = []
    for entry in ground_truth_output_all:
        if entry[1] in addresses_used:
            continue
        ground_truth_output_tensors.append(entry[0])
        addresses_used.append(entry[1])
    model.load_state_dict(state_dict)

    # Step 2: Run the model *through* TorchLens, saving all activations.
    # save_function_args=True is essential — the replay needs each function's
    # non-tensor arguments to re-execute the computation from saved activations.
    model_log = _run_model_and_save_specified_activations(
        model=model,
        input_args=input_args,
        input_kwargs=input_kwargs,
        layers_to_save="all",
        keep_unsaved_layers=True,
        activation_postfunc=None,
        mark_input_output_distances=False,
        detach_saved_tensors=False,
        save_gradients=False,
        save_function_args=True,
        random_seed=random_seed,
        save_rng_states=True,
    )
    # Step 3: Validate by replaying the forward pass from saved activations.
    try:
        activations_are_valid = model_log.validate_saved_activations(
            ground_truth_output_tensors, verbose, validate_metadata=validate_metadata
        )
    finally:
        model_log.cleanup()
        del model_log
    return activations_are_valid


def validate_saved_activations(
    model: nn.Module,
    input_args: Union[torch.Tensor, List[Any], Tuple[Any]],
    input_kwargs: Optional[Dict[Any, Any]] = None,
    random_seed: Union[int, None] = None,
    verbose: bool = False,
) -> bool:
    """Deprecated: use ``validate_forward_pass`` instead."""
    import warnings

    warnings.warn(
        "validate_saved_activations is deprecated, use validate_forward_pass instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return validate_forward_pass(
        model, input_args, input_kwargs, random_seed=random_seed, verbose=verbose
    )


def validate_batch_of_models_and_inputs(
    models_and_inputs_dict: Dict[str, Dict[str, Union[str, Callable, Dict]]],
    out_path: str,
    redo_model_if_already_run: bool = True,
) -> pd.DataFrame:
    """Batch-validate multiple models, writing incremental results to a CSV.

    For each model/input pair, calls ``validate_forward_pass`` and appends the
    result to a running CSV at *out_path*.  If the CSV already exists, previously
    validated models can be skipped (controlled by *redo_model_if_already_run*).

    Args:
        models_and_inputs_dict: Mapping of model_name to a dict with keys:
            - ``model_category`` (str): grouping label (e.g. 'torchvision').
            - ``model_loading_func`` (callable): zero-arg function returning an nn.Module.
            - ``model_sample_inputs`` (dict[str, input]): named sample inputs.
        out_path: File path for the results CSV (created if absent, appended otherwise).
        redo_model_if_already_run: Re-validate models already present in the CSV.

    Returns:
        DataFrame with columns: model_category, model_name, input_name, validation_success.
    """
    if os.path.exists(out_path):
        current_csv = pd.read_csv(out_path)
    else:
        current_csv = pd.DataFrame.from_dict(
            {
                "model_category": [],
                "model_name": [],
                "input_name": [],
                "validation_success": [],
            }
        )
    models_already_run = current_csv["model_name"].unique()
    for model_name, model_info in tqdm(models_and_inputs_dict.items(), desc="Validating models"):
        print(f"Validating model {model_name}")
        if model_name in models_already_run and not redo_model_if_already_run:
            continue
        model_category = model_info["model_category"]
        model_loading_func = model_info["model_loading_func"]
        model = model_loading_func()
        model_sample_inputs = model_info["model_sample_inputs"]
        for input_name, input_data in model_sample_inputs.items():
            validation_success = validate_forward_pass(model, input_data)
            current_csv = pd.concat(
                [
                    current_csv,
                    pd.DataFrame(
                        [
                            {
                                "model_category": model_category,
                                "model_name": model_name,
                                "input_name": input_name,
                                "validation_success": validation_success,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
        current_csv.to_csv(out_path, index=False)
        del model
    return current_csv


def _deep_clone_tensors(val: Any) -> Any:
    """Recursively clone all tensors in a nested structure of lists/tuples/dicts."""
    if isinstance(val, torch.Tensor):
        return val.detach().clone()
    elif isinstance(val, (list, tuple)):
        cloned = [_deep_clone_tensors(v) for v in val]
        return type(val)(cloned)
    elif isinstance(val, dict):
        return {k: _deep_clone_tensors(v) for k, v in val.items()}
    return val


def _apply_value_to_args(
    args_list: Union[List[Any], Dict[str, Any]],
    pos: Union[int, str, tuple],
    value: Any,
) -> None:
    """
    Helper function to apply a computed value to the correct position in args
    (handles nested tuple/dict structures).
    """
    if not isinstance(pos, tuple):
        args_list[pos] = value
    else:
        # Mirrors validation replay semantics for nested arg positions.
        args_list[pos[0]] = assign_to_sequence_or_dict(args_list[pos[0]], pos[1], value)


def _index_nested(obj: Any, path: Tuple[Any, ...]) -> Any:
    """Index a nested tuple/list/dict by following ``path`` keys in order."""
    out = obj
    for key in path:
        out = out[key]
    return out


def _find_tensor_path(container: Any, target: torch.Tensor) -> Optional[Tuple[Any, ...]]:
    """Find the nested path of ``target`` tensor within ``container``.

    Comparison uses exact tensor equality semantics with NaN-aware handling.
    """
    if isinstance(container, torch.Tensor):
        if tensor_nanequal(container, target, allow_tolerance=False):
            return ()
        return None

    if isinstance(container, (list, tuple)):
        for idx, val in enumerate(container):
            subpath = _find_tensor_path(val, target)
            if subpath is not None:
                return (idx,) + subpath
        return None

    if isinstance(container, dict):
        for key, val in container.items():
            subpath = _find_tensor_path(val, target)
            if subpath is not None:
                return (key,) + subpath
        return None

    return None


def _resolve_parent_value_for_child(
    model_log: "ModelLog",
    parent_label: str,
    child_label: str,
    parent_replay_value: Any,
    route_cache: Dict[Tuple[str, str], Optional[Tuple[Any, ...]]],
) -> Any:
    """Resolve which portion of a parent's replay output should feed a specific child."""
    cache_key = (parent_label, child_label)
    if cache_key not in route_cache:
        parent_layer = model_log[parent_label]
        cached_path: Optional[Tuple[Any, ...]] = None
        if child_label in parent_layer.children_tensor_versions:
            child_saved_value = parent_layer.children_tensor_versions[child_label]
            if isinstance(child_saved_value, torch.Tensor):
                cached_path = _find_tensor_path(parent_layer.activation, child_saved_value)
        route_cache[cache_key] = cached_path

    path = route_cache[cache_key]
    if path is None:
        return parent_replay_value
    return _index_nested(parent_replay_value, path)


def _reconstruct_output_from_template(
    template: Any,
    computed_activations: Dict[str, Any],
    model_log: "ModelLog",
) -> Any:
    """Reconstruct model output with the original container format from a lightweight template."""
    if isinstance(template, tuple) and len(template) == 2 and template[0] == "__tl_output_ref__":
        layer_label = template[1]
        if layer_label in computed_activations:
            return computed_activations[layer_label]
        raw_to_final = getattr(model_log, "_raw_to_final_layer_labels", {})
        final_label = raw_to_final.get(layer_label, layer_label)
        return computed_activations[final_label]

    if isinstance(template, list):
        return [
            _reconstruct_output_from_template(v, computed_activations, model_log) for v in template
        ]

    if isinstance(template, dict):
        if "__tl_tuple_type__" in template and "items" in template:
            items = [
                _reconstruct_output_from_template(v, computed_activations, model_log)
                for v in template["items"]
            ]
            tuple_type = template["__tl_tuple_type__"]
            if tuple_type is tuple:
                return tuple(items)
            try:
                return tuple_type(*items)
            except Exception:
                return tuple_type(items)

        if "__tl_dict_type__" in template and "items" in template:
            rebuilt_items = [
                (k, _reconstruct_output_from_template(v, computed_activations, model_log))
                for k, v in template["items"]
            ]
            dict_type = template["__tl_dict_type__"]
            rebuilt_dict = {k: v for k, v in rebuilt_items}
            if dict_type is dict:
                return rebuilt_dict

            # Try multiple strategies to reconstruct the dict-like object
            # Strategy 1: Try calling with **kwargs (works for many dataclass-like objects)
            try:
                return dict_type(**rebuilt_dict)
            except Exception:
                pass

            # Strategy 2: Try calling with dict as single arg (works for some dict subclasses)
            try:
                return dict_type(rebuilt_dict)
            except Exception:
                pass

            # Strategy 3: Try creating empty instance and updating (works for OrderedDict, etc.)
            try:
                instance = dict_type()
                instance.update(rebuilt_dict)
                return instance
            except Exception:
                pass

            # Strategy 4: For transformers ModelOutput-like classes, try positional args
            # These classes often accept values in the order they were defined
            try:
                return dict_type(*rebuilt_dict.values())
            except Exception:
                pass

            # Fallback: return plain dict with a warning attribute
            # This preserves data even if we can't reconstruct the exact type
            return rebuilt_dict

        return {
            k: _reconstruct_output_from_template(v, computed_activations, model_log)
            for k, v in template.items()
        }

    return template


def replay_forward_pass(
    model_log: "ModelLog",
    new_input: Union[torch.Tensor, List[Any]],
    new_input_kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Replay the computation using the saved graph structure and new input.

    Args:
        model_log: ModelLog object returned by log_forward_pass (must have
                   save_function_args=True to capture non-tensor args).
        new_input: New input tensor(s) to substitute for the original.
        new_input_kwargs: Optional kwargs for the input layer.

    Returns:
        The final output tensor from the replayed forward pass.
    """
    if new_input_kwargs is None:
        new_input_kwargs = {}

    computed_activations: Dict[str, Any] = {}
    parent_child_route_cache: Dict[Tuple[str, str], Optional[Tuple[Any, ...]]] = {}
    output: Any = None

    for layer_pass_log in model_log.layer_list:
        layer_label = layer_pass_log.layer_label

        # Clone tensors in captured_args/kwargs to prevent in-place modification crashes in batch-norm layers
        raw_args = (
            layer_pass_log.captured_args
            if getattr(layer_pass_log, "captured_args", None) is not None
            else []
        )
        raw_kwargs = (
            layer_pass_log.captured_kwargs
            if getattr(layer_pass_log, "captured_kwargs", None) is not None
            else {}
        )

        args_list = _deep_clone_tensors(list(raw_args))
        kwargs_dict = _deep_clone_tensors(dict(raw_kwargs))

        parent_arg_locs = getattr(layer_pass_log, "parent_layer_arg_locs", {})

        for arg_pos, parent_label in parent_arg_locs.get("args", {}).items():
            if parent_label in computed_activations:
                parent_value = _resolve_parent_value_for_child(
                    model_log,
                    parent_label,
                    layer_label,
                    computed_activations[parent_label],
                    parent_child_route_cache,
                )
                _apply_value_to_args(args_list, arg_pos, parent_value)
            elif layer_pass_log.is_input_layer:
                _apply_value_to_args(args_list, arg_pos, new_input)

        for kwarg_name, parent_label in parent_arg_locs.get("kwargs", {}).items():
            if parent_label in computed_activations:
                kwargs_dict[kwarg_name] = _resolve_parent_value_for_child(
                    model_log,
                    parent_label,
                    layer_label,
                    computed_activations[parent_label],
                    parent_child_route_cache,
                )
            elif layer_pass_log.is_input_layer:
                kwargs_dict[kwarg_name] = new_input_kwargs.get(
                    kwarg_name, kwargs_dict.get(kwarg_name)
                )

        func = getattr(layer_pass_log, "func_applied", None)
        if func is not None:
            output = func(*args_list, **kwargs_dict)
            if output is None:
                # Some in-place ops (e.g., tensor.__setitem__) return None.
                # In those cases, the mutated first arg is the semantic output.
                if getattr(layer_pass_log, "func_is_inplace", False) and len(args_list) > 0:
                    output = args_list[0]
                else:
                    output = _deep_clone_tensors(layer_pass_log.activation)
            if layer_pass_log.iterable_output_index is not None:
                output = index_nested(output, layer_pass_log.iterable_output_index)
        else:
            if layer_pass_log.is_input_layer:
                output = _deep_clone_tensors(new_input)
            else:
                output = _deep_clone_tensors(layer_pass_log.activation)

        computed_activations[layer_label] = output

    output_template = getattr(model_log, "_output_structure_template", None)
    if output_template is not None:
        return _reconstruct_output_from_template(output_template, computed_activations, model_log)

    if model_log.output_layers:
        if len(model_log.output_layers) == 1:
            return computed_activations[model_log.output_layers[0]]
        return [computed_activations[layer_label] for layer_label in model_log.output_layers]

    return output


def split_graph(
    model_log: "ModelLog",
    split_layer_indices: Union[int, List[int]],
) -> Tuple[List[str], List[str], List[str]]:
    """Split a computational graph at specified layer indices.

    Args:
        model_log: ModelLog with full graph structure (must have save_function_args=True)
        split_layer_indices: Layer index/indices where to split (supports negative indexing)

    Returns:
        Tuple of (subgraph1_labels, subgraph2_labels, split_point_labels):
        - subgraph1_labels: Layers from inputs up to the maximum split point (inclusive)
        - subgraph2_labels: Layers after the maximum split point to outputs
        - split_point_labels: Outputs of subgraph1 needed by subgraph2
    """
    if isinstance(split_layer_indices, int):
        split_layer_indices = [split_layer_indices]

    total_layers = len(model_log.layer_list)
    split_indices = []
    for idx in split_layer_indices:
        if idx < 0:
            idx = total_layers + idx
        if idx < 0 or idx >= total_layers:
            raise ValueError(f"Split index {idx} out of range for {total_layers} layers")
        split_indices.append(idx)

    max_split_idx = max(split_indices)

    # Subgraph 1 is everything up to the max split index (inclusive)
    # Subgraph 2 is everything after it.
    subgraph1_labels = [layer.layer_label for layer in model_log.layer_list[:max_split_idx + 1]]
    subgraph2_labels = [layer.layer_label for layer in model_log.layer_list[max_split_idx + 1:]]

    # The features that need to be passed between subgraph1 and subgraph2
    # are all layers in subgraph1 that have at least one child in subgraph2.
    # We also include any output layers that might be in subgraph1.
    subgraph2_set = set(subgraph2_labels)
    
    split_point_labels = []
    for layer_label in subgraph1_labels:
        layer = model_log[layer_label]
        child_labels = getattr(layer, "child_layers", [])
        
        # Check if the node is needed by subgraph2 or is a model output
        has_child_in_sg2 = any(c in subgraph2_set for c in child_labels)
        is_output = layer_label in model_log.output_layers
        
        if has_child_in_sg2 or is_output:
            split_point_labels.append(layer_label)

    return subgraph1_labels, subgraph2_labels, split_point_labels


def replay_subgraph(
    model_log: "ModelLog",
    subgraph_labels: List[str],
    input_values: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    """Replay a subgraph with provided input values.

    Args:
        model_log: Original ModelLog
        subgraph_labels: Labels of layers to replay (in topological order)
        input_values: Dict mapping layer labels to their input tensor values

    Returns:
        Dict mapping layer labels to their computed outputs
    """
    from .utils.tensor_utils import safe_copy

    computed = {}
    # Deep copy input values to avoid mutation
    for k, v in input_values.items():
        computed[k] = safe_copy(v) if isinstance(v, torch.Tensor) else v

    label_to_layer = {layer.layer_label: layer for layer in model_log.layer_list}
    parent_child_route_cache: Dict[Tuple[str, str], Optional[Tuple[Any, ...]]] = {}

    for label in subgraph_labels:
        if label in computed:
            continue

        layer = label_to_layer[label]

        # Clone captured args/kwargs to prevent in-place modification issues
        raw_args = getattr(layer, "captured_args", None)
        raw_kwargs = getattr(layer, "captured_kwargs", None)

        args_list = list(raw_args) if raw_args is not None else []
        kwargs_dict = dict(raw_kwargs) if raw_kwargs is not None else {}

        # Deep clone tensors in args/kwargs
        args_list = _deep_clone_tensors(args_list)
        kwargs_dict = _deep_clone_tensors(kwargs_dict)

        # Substitute parent values
        parent_arg_locs = getattr(layer, "parent_layer_arg_locs", {})
        for arg_pos, parent_label in parent_arg_locs.get("args", {}).items():
            if parent_label in computed:
                parent_value = _resolve_parent_value_for_child(
                    model_log, parent_label, label, computed[parent_label], parent_child_route_cache
                )
                if isinstance(arg_pos, tuple):
                    args_list[arg_pos[0]] = assign_to_sequence_or_dict(
                        args_list[arg_pos[0]], arg_pos[1], parent_value
                    )
                else:
                    args_list[arg_pos] = parent_value

        for kwarg_name, parent_label in parent_arg_locs.get("kwargs", {}).items():
            if parent_label in computed:
                parent_value = _resolve_parent_value_for_child(
                    model_log, parent_label, label, computed[parent_label], parent_child_route_cache
                )
                kwargs_dict[kwarg_name] = parent_value

        # Execute function
        func = getattr(layer, "func_applied", None)
        if func is not None:
            output = func(*args_list, **kwargs_dict)
            if output is None:
                if getattr(layer, "func_is_inplace", False) and len(args_list) > 0:
                    output = args_list[0]
                else:
                    output = safe_copy(layer.activation) if layer.activation is not None else None
            if layer.iterable_output_index is not None:
                output = index_nested(output, layer.iterable_output_index)
        else:
            # For layers without func_applied (like input layers)
            if layer.is_input_layer and label not in input_values:
                # This shouldn't happen if input_values is correct
                output = safe_copy(layer.activation) if layer.activation is not None else None
            else:
                output = safe_copy(layer.activation) if layer.activation is not None else None

        computed[label] = output

    return computed


def split_and_replay_graph(
    model_log: "ModelLog",
    split_layer_indices: Union[int, List[int]],
    new_input: Union[torch.Tensor, List[Any]],
    new_input_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Any]:
    """Split graph at specified layers and replay both subgraphs with new input.

    This function demonstrates that splitting the graph and replaying both parts
    produces the same result as replaying the full graph. Useful for:
    - Model partitioning across devices
    - Intermediate feature extraction
    - Debugging specific model sections

    Args:
        model_log: ModelLog with full graph (must have save_function_args=True)
        split_layer_indices: Layer index/indices where to split
        new_input: New input tensor(s) for replay
        new_input_kwargs: Optional keyword arguments for input

    Returns:
        Tuple of (intermediate_features, final_output):
        - intermediate_features: Dict of split point activations from subgraph1
        - final_output: Final model output from subgraph2
    """
    if new_input_kwargs is None:
        new_input_kwargs = {}

    # Split the graph
    subgraph1_labels, subgraph2_labels, split_labels = split_graph(model_log, split_layer_indices)

    # Find input layers (layers with is_input_layer=True)
    input_layers = {}
    for layer in model_log.layer_list:
        if getattr(layer, "is_input_layer", False):
            input_layers[layer.layer_label] = new_input

    # If no input layers found, use first layer as fallback
    if not input_layers:
        input_layers[model_log.layer_list[0].layer_label] = new_input

    # Replay subgraph1 (input to split points)
    subgraph1_outputs = replay_subgraph(
        model_log,
        subgraph1_labels,
        input_layers,
    )

    # Extract intermediate features at split points
    intermediate_features = {label: subgraph1_outputs[label] for label in split_labels}

    # Replay subgraph2 (split points to output)
    subgraph2_outputs = replay_subgraph(
        model_log,
        subgraph2_labels,
        intermediate_features,
    )

    # Reconstruct final output from both halves. Some models can expose outputs
    # that are already produced in subgraph1 when the split index is near the end.
    combined_outputs = {**subgraph1_outputs, **subgraph2_outputs}

    # Reconstruct final output
    output_template = getattr(model_log, "_output_structure_template", None)
    if output_template is not None:
        final_output = _reconstruct_output_from_template(
            output_template, combined_outputs, model_log
        )
    elif model_log.output_layers:
        if len(model_log.output_layers) == 1:
            final_output = combined_outputs[model_log.output_layers[0]]
        else:
            final_output = [combined_outputs[label] for label in model_log.output_layers]
    else:
        final_output = None

    return intermediate_features, final_output
