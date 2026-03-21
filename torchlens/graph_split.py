"""Graph replay and split helpers.

Keeps replay/split implementation out of the public API module and uses a
single execution engine for both full-graph replay and subgraph replay.
"""

import inspect
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from .utils.collections import assign_to_sequence_or_dict, index_nested
from .utils.tensor_utils import tensor_nanequal

if TYPE_CHECKING:
    from .data_classes.model_log import ModelLog


def _get_replay_cache(model_log: "ModelLog") -> Dict[str, Any]:
    """Lazily create and return replay-related caches stored on the ModelLog."""
    cache = getattr(model_log, "_graph_split_cache", None)
    if cache is None:
        cache = {
            "label_to_layer": {layer.layer_label: layer for layer in model_log.layer_list},
            "parent_child_route_cache": {},
            "split_boundary_cache": {},
        }
        setattr(model_log, "_graph_split_cache", cache)
    return cache


def _deep_clone_tensors(val: Any) -> Any:
    """Recursively clone all tensors in a nested structure of lists/tuples/dicts."""
    if isinstance(val, torch.Tensor):
        return val.detach().clone()
    if isinstance(val, (list, tuple)):
        cloned = [_deep_clone_tensors(v) for v in val]
        return type(val)(cloned)
    if isinstance(val, dict):
        return {k: _deep_clone_tensors(v) for k, v in val.items()}
    return val


def _clone_tensors_preserve_graph(val: Any) -> Any:
    """Recursively clone tensors while preserving their autograd connections."""
    return _clone_tensors_preserve_graph_with_memo(val, {})


def _clone_tensors_preserve_graph_with_memo(val: Any, memo: Dict[int, Any]) -> Any:
    """Clone tensors while preserving graph links and shared-reference topology."""
    obj_id = id(val)
    if obj_id in memo:
        return memo[obj_id]

    if isinstance(val, torch.Tensor):
        if isinstance(val, torch.nn.Parameter):
            memo[obj_id] = val
            return val
        cloned_tensor = val.clone()
        memo[obj_id] = cloned_tensor
        return cloned_tensor

    if isinstance(val, list):
        cloned_list: List[Any] = []
        memo[obj_id] = cloned_list
        cloned_list.extend(_clone_tensors_preserve_graph_with_memo(v, memo) for v in val)
        return cloned_list

    if isinstance(val, tuple):
        placeholder: List[Any] = []
        memo[obj_id] = placeholder
        cloned_tuple = type(val)(
            _clone_tensors_preserve_graph_with_memo(v, memo) for v in val
        )
        memo[obj_id] = cloned_tuple
        return cloned_tuple

    if isinstance(val, dict):
        cloned_dict: Dict[Any, Any] = {}
        memo[obj_id] = cloned_dict
        cloned_dict.update(
            {k: _clone_tensors_preserve_graph_with_memo(v, memo) for k, v in val.items()}
        )
        return cloned_dict

    memo[obj_id] = val
    return val


def _copy_containers_preserve_tensors(val: Any) -> Any:
    """Copy container structure while keeping tensor leaves untouched."""
    return _copy_containers_preserve_tensors_with_memo(val, {})


def _copy_containers_preserve_tensors_with_memo(val: Any, memo: Dict[int, Any]) -> Any:
    """Copy lists/tuples/dicts without inserting extra tensor clone ops."""
    obj_id = id(val)
    if obj_id in memo:
        return memo[obj_id]

    if isinstance(val, torch.Tensor):
        memo[obj_id] = val
        return val

    if isinstance(val, list):
        copied_list: List[Any] = []
        memo[obj_id] = copied_list
        copied_list.extend(_copy_containers_preserve_tensors_with_memo(v, memo) for v in val)
        return copied_list

    if isinstance(val, tuple):
        placeholder: List[Any] = []
        memo[obj_id] = placeholder
        copied_tuple = type(val)(
            _copy_containers_preserve_tensors_with_memo(v, memo) for v in val
        )
        memo[obj_id] = copied_tuple
        return copied_tuple

    if isinstance(val, dict):
        copied_dict: Dict[Any, Any] = {}
        memo[obj_id] = copied_dict
        copied_dict.update(
            {k: _copy_containers_preserve_tensors_with_memo(v, memo) for k, v in val.items()}
        )
        return copied_dict

    memo[obj_id] = val
    return val


def _maybe_retry_getitem_with_safe_indexing(
    func: Any,
    args_list: List[Any],
    kwargs_dict: Dict[str, Any],
) -> Optional[Any]:
    """Retry tensor ``__getitem__`` with filtered integer indices after OOB errors.

    This is a narrow replay fallback for dynamic graphs that derive an index tensor
    from runtime data but only store the final integer values in ``captured_args``.
    When the replayed source tensor becomes shorter than in the logged run, using the
    stale index tensor can raise an out-of-bounds error. In that specific case, keep
    only the indices that are valid for the current source tensor.
    """
    if getattr(func, "__name__", None) != "__getitem__":
        return None
    if kwargs_dict or len(args_list) < 2:
        return None

    source = args_list[0]
    index = args_list[1]
    if not isinstance(source, torch.Tensor) or source.ndim == 0:
        return None
    if not isinstance(index, torch.Tensor):
        return None
    if index.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        return None

    dim0 = source.shape[0]
    valid_mask = (index >= -dim0) & (index < dim0)
    safe_index = index[valid_mask]
    return func(source, safe_index)


def _get_leading_length(val: Any) -> Optional[int]:
    """Return the leading length of a tensor or sequence, if defined."""
    if isinstance(val, torch.Tensor):
        if val.ndim == 0:
            return None
        return int(val.shape[0])
    if isinstance(val, (list, tuple)):
        return len(val)
    return None


def _infer_runtime_randperm_arg(
    model_log: "ModelLog",
    randperm_label: str,
    computed: Dict[str, Any],
) -> Optional[int]:
    """Infer a runtime ``randperm`` length from downstream indexing structure.

    This follows chains of ``__getitem__`` layers stemming from a ``randperm`` node.
    If one of those nodes is later used to index another currently-computed tensor,
    the source tensor's leading dimension is the most plausible runtime length.
    """
    visited = {randperm_label}
    frontier = [randperm_label]

    while frontier:
        next_frontier = []
        for current_label in frontier:
            for child_label in model_log[current_label].child_layers:
                if child_label in visited:
                    continue
                visited.add(child_label)
                child = model_log[child_label]
                if getattr(child.func_applied, "__name__", None) != "__getitem__":
                    continue

                other_parents = [label for label in child.parent_layers if label not in visited]
                for parent_label in other_parents:
                    if parent_label not in computed:
                        continue
                    inferred_length = _get_leading_length(computed[parent_label])
                    if inferred_length is not None:
                        return inferred_length

                next_frontier.append(child_label)
        frontier = next_frontier

    return None




def _apply_value_to_args(
    args_list: Union[List[Any], Dict[str, Any]],
    pos: Union[int, str, tuple],
    value: Any,
) -> None:
    """Apply a computed value to a positional slot or nested arg location."""
    if not isinstance(pos, tuple):
        args_list[pos] = value
    else:
        args_list[pos[0]] = assign_to_sequence_or_dict(args_list[pos[0]], pos[1], value)


def _index_nested(obj: Any, path: Tuple[Any, ...]) -> Any:
    """Index a nested tuple/list/dict by following ``path`` keys in order."""
    out = obj
    for key in path:
        out = out[key]
    return out


def _find_tensor_path(container: Any, target: torch.Tensor) -> Optional[Tuple[Any, ...]]:
    """Find the nested path of ``target`` tensor within ``container``."""
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


def _normalize_split_indices(
    split_layer_indices: Union[int, List[int]],
    total_layers: int,
) -> List[int]:
    """Normalize split indices, resolving negatives and validating range."""
    if isinstance(split_layer_indices, int):
        split_layer_indices = [split_layer_indices]

    normalized_indices = []
    for idx in split_layer_indices:
        if idx < 0:
            idx = total_layers + idx
        if idx < 0 or idx >= total_layers:
            raise ValueError(f"Split index {idx} out of range for {total_layers} layers")
        normalized_indices.append(idx)

    return normalized_indices


def _normalize_replay_positional_inputs(
    new_input: Union[torch.Tensor, List[Any], Tuple[Any, ...], None],
    expected_num_positional_args: int,
) -> List[Any]:
    """Normalize replay positional inputs using the expected input-root count."""
    if type(new_input) in (tuple, list):
        if expected_num_positional_args == 1 and len(new_input) != 1:
            return [new_input]
        return list(new_input)
    if new_input is None:
        return []
    return [new_input]


def _parse_input_io_role(io_role: str) -> Tuple[str, Tuple[Any, ...]]:
    """Parse an input-layer io_role like ``input.x.0`` into root name + nested path."""
    if not io_role.startswith("input."):
        raise ValueError(f"Expected input io_role, got {io_role}")

    parts = io_role.split(".")[1:]
    root_name = parts[0]
    nested_path = tuple(int(part) if part.isdigit() else part for part in parts[1:])
    return root_name, nested_path


def _build_replay_input_layer_values(
    model_log: "ModelLog",
    new_input: Union[torch.Tensor, List[Any], Tuple[Any, ...], None],
    new_input_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Map replay-time inputs onto logged input-layer labels via each layer's io_role."""
    if new_input_kwargs is None:
        new_input_kwargs = {}

    input_root_names = []
    seen_root_names = set()
    for input_layer_label in model_log.input_layers:
        input_layer = model_log[input_layer_label]
        if input_layer.io_role is None:
            continue
        root_name, _ = _parse_input_io_role(input_layer.io_role)
        if root_name not in seen_root_names:
            input_root_names.append(root_name)
            seen_root_names.add(root_name)

    kwarg_root_names = {name for name in input_root_names if name in new_input_kwargs}
    positional_root_names = [name for name in input_root_names if name not in kwarg_root_names]
    positional_inputs = _normalize_replay_positional_inputs(new_input, len(positional_root_names))

    if len(positional_inputs) < len(positional_root_names):
        raise ValueError(
            "Not enough positional replay inputs provided: "
            f"expected {len(positional_root_names)}, got {len(positional_inputs)}."
        )

    positional_input_by_name = {
        root_name: positional_inputs[idx] for idx, root_name in enumerate(positional_root_names)
    }

    input_layer_values: Dict[str, Any] = {}
    for input_layer_label in model_log.input_layers:
        input_layer = model_log[input_layer_label]
        if input_layer.io_role is None:
            continue

        root_name, nested_path = _parse_input_io_role(input_layer.io_role)
        if root_name in new_input_kwargs:
            root_value = new_input_kwargs[root_name]
        elif root_name in positional_input_by_name:
            root_value = positional_input_by_name[root_name]
        else:
            raise ValueError(
                "Missing replay input value for input root "
                f"'{root_name}' required by layer '{input_layer_label}'."
            )

        input_layer_values[input_layer_label] = (
            _index_nested(root_value, nested_path) if nested_path else root_value
        )

    return input_layer_values


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

            try:
                return dict_type(**rebuilt_dict)
            except Exception:
                pass

            try:
                return dict_type(rebuilt_dict)
            except Exception:
                pass

            try:
                instance = dict_type()
                instance.update(rebuilt_dict)
                return instance
            except Exception:
                pass

            try:
                return dict_type(*rebuilt_dict.values())
            except Exception:
                pass

            return rebuilt_dict

        return {
            k: _reconstruct_output_from_template(v, computed_activations, model_log)
            for k, v in template.items()
        }

    return template


def _unwrap_replay_model(model: nn.Module) -> nn.Module:
    """Unwrap DataParallel for differentiable replay helpers."""
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


def _build_live_state_maps(model: nn.Module) -> Tuple[Dict[str, torch.nn.Parameter], Dict[str, torch.Tensor]]:
    """Build parameter and buffer lookup tables keyed by logged addresses."""
    model = _unwrap_replay_model(model)
    param_map: Dict[str, torch.nn.Parameter] = {}
    for name, param in model.named_parameters():
        param_map[name] = param
        tl_addr = getattr(param, "tl_param_address", None)
        if tl_addr is not None:
            param_map[tl_addr] = param

    buffer_map: Dict[str, torch.Tensor] = {}
    for name, buffer in model.named_buffers():
        buffer_map[name] = buffer
    return param_map, buffer_map


def _swap_copied_params_for_live_params(
    val: Any,
    live_params_by_addr: Dict[str, torch.nn.Parameter],
) -> Any:
    """Replace copied parameter snapshots in captured args with live model parameters."""
    if isinstance(val, torch.nn.Parameter):
        param_addr = getattr(val, "tl_param_address", None)
        if param_addr is not None and param_addr in live_params_by_addr:
            return live_params_by_addr[param_addr]
        return val
    if isinstance(val, list):
        return [_swap_copied_params_for_live_params(v, live_params_by_addr) for v in val]
    if isinstance(val, tuple):
        return type(val)(_swap_copied_params_for_live_params(v, live_params_by_addr) for v in val)
    if isinstance(val, dict):
        return {k: _swap_copied_params_for_live_params(v, live_params_by_addr) for k, v in val.items()}
    return val


def _sync_runtime_training_flag(
    func: Any,
    args_list: List[Any],
    kwargs_dict: Dict[str, Any],
    model_training: bool,
) -> None:
    """Update a captured ``training`` argument to match the current model mode."""
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return

    param_names = list(signature.parameters.keys())
    for training_name in ("training", "train"):
        if training_name not in param_names:
            continue

        if training_name in kwargs_dict and isinstance(kwargs_dict[training_name], bool):
            kwargs_dict[training_name] = model_training
            return

        training_idx = param_names.index(training_name)
        if training_idx < len(args_list) and isinstance(args_list[training_idx], bool):
            args_list[training_idx] = model_training
            return


def _execute_replay(
    model_log: "ModelLog",
    layer_labels: List[str],
    seeded_values: Dict[str, Any],
) -> Dict[str, Any]:
    """Replay the requested layers in topological order with provided seed values."""
    replay_cache = _get_replay_cache(model_log)
    computed = {key: _deep_clone_tensors(val) for key, val in seeded_values.items()}
    label_to_layer = replay_cache["label_to_layer"]
    parent_child_route_cache = replay_cache["parent_child_route_cache"]

    for label in layer_labels:
        if label in computed:
            continue

        layer = label_to_layer[label]
        raw_args = getattr(layer, "captured_args", None)
        raw_kwargs = getattr(layer, "captured_kwargs", None)

        args_list = _deep_clone_tensors(list(raw_args) if raw_args is not None else [])
        kwargs_dict = _deep_clone_tensors(dict(raw_kwargs) if raw_kwargs is not None else {})

        parent_arg_locs = getattr(layer, "parent_layer_arg_locs", {})
        for arg_pos, parent_label in parent_arg_locs.get("args", {}).items():
            if parent_label in computed:
                parent_value = _resolve_parent_value_for_child(
                    model_log, parent_label, label, computed[parent_label], parent_child_route_cache
                )
                _apply_value_to_args(args_list, arg_pos, parent_value)

        for kwarg_name, parent_label in parent_arg_locs.get("kwargs", {}).items():
            if parent_label in computed:
                kwargs_dict[kwarg_name] = _resolve_parent_value_for_child(
                    model_log, parent_label, label, computed[parent_label], parent_child_route_cache
                )

        if getattr(layer.func_applied, "__name__", None) == "randperm" and len(args_list) >= 1:
            inferred_n = _infer_runtime_randperm_arg(model_log, label, computed)
            if inferred_n is not None:
                args_list[0] = inferred_n

        func = getattr(layer, "func_applied", None)
        if func is not None:
            try:
                output = func(*args_list, **kwargs_dict)
            except IndexError:
                output = _maybe_retry_getitem_with_safe_indexing(func, args_list, kwargs_dict)
                if output is None:
                    raise
            if output is None:
                if getattr(layer, "func_is_inplace", False) and len(args_list) > 0:
                    output = args_list[0]
                else:
                    output = _deep_clone_tensors(layer.activation)
            if layer.iterable_output_index is not None:
                output = index_nested(output, layer.iterable_output_index)
        else:
            output = _deep_clone_tensors(layer.activation)

        computed[label] = output

    return computed


def _execute_replay_differentiable(
    model_log: "ModelLog",
    model: nn.Module,
    layer_labels: List[str],
    seeded_values: Dict[str, Any],
) -> Dict[str, Any]:
    """Replay layers while preserving autograd links through inputs and live parameters."""
    replay_cache = _get_replay_cache(model_log)
    computed = {key: _copy_containers_preserve_tensors(val) for key, val in seeded_values.items()}
    label_to_layer = replay_cache["label_to_layer"]
    parent_child_route_cache = replay_cache["parent_child_route_cache"]
    live_params_by_addr, live_buffers_by_addr = _build_live_state_maps(model)

    for label in layer_labels:
        if label in computed:
            continue

        layer = label_to_layer[label]
        raw_args = getattr(layer, "captured_args", None)
        raw_kwargs = getattr(layer, "captured_kwargs", None)

        args_list = _clone_tensors_preserve_graph(list(raw_args) if raw_args is not None else [])
        kwargs_dict = _clone_tensors_preserve_graph(dict(raw_kwargs) if raw_kwargs is not None else {})
        args_list = _swap_copied_params_for_live_params(args_list, live_params_by_addr)
        kwargs_dict = _swap_copied_params_for_live_params(kwargs_dict, live_params_by_addr)

        parent_arg_locs = getattr(layer, "parent_layer_arg_locs", {})
        for arg_pos, parent_label in parent_arg_locs.get("args", {}).items():
            if parent_label in computed:
                parent_value = _resolve_parent_value_for_child(
                    model_log, parent_label, label, computed[parent_label], parent_child_route_cache
                )
                _apply_value_to_args(args_list, arg_pos, parent_value)

        for kwarg_name, parent_label in parent_arg_locs.get("kwargs", {}).items():
            if parent_label in computed:
                kwargs_dict[kwarg_name] = _resolve_parent_value_for_child(
                    model_log, parent_label, label, computed[parent_label], parent_child_route_cache
                )

        if getattr(layer.func_applied, "__name__", None) == "randperm" and len(args_list) >= 1:
            inferred_n = _infer_runtime_randperm_arg(model_log, label, computed)
            if inferred_n is not None:
                args_list[0] = inferred_n

        func = getattr(layer, "func_applied", None)
        if func is not None:
            _sync_runtime_training_flag(func, args_list, kwargs_dict, model.training)
            try:
                output = func(*args_list, **kwargs_dict)
            except IndexError:
                output = _maybe_retry_getitem_with_safe_indexing(func, args_list, kwargs_dict)
                if output is None:
                    raise
            if output is None:
                if getattr(layer, "func_is_inplace", False) and len(args_list) > 0:
                    output = args_list[0]
                else:
                    output = _clone_tensors_preserve_graph(layer.activation)
            if layer.iterable_output_index is not None:
                output = index_nested(output, layer.iterable_output_index)
        else:
            if layer.is_input_layer and label in seeded_values:
                output = computed[label]
            elif layer.is_buffer_layer and layer.buffer_address in live_buffers_by_addr:
                output = live_buffers_by_addr[layer.buffer_address]
            else:
                output = _clone_tensors_preserve_graph(layer.activation)

        computed[label] = output

    return computed


def replay_forward_pass(
    model_log: "ModelLog",
    new_input: Union[torch.Tensor, List[Any], Tuple[Any, ...]],
    new_input_kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    """Replay the full computation using the saved graph structure and new input.

    Notes
    -----
    This is a value-level replay utility, not an autograd-preserving execution path.
    Saved tensors and captured arguments are cloned from the logged pass, so the
    returned tensors are suitable for forward-value validation but are not connected
    to the original model's autograd graph for training-time backward passes.
    """
    input_layer_values = _build_replay_input_layer_values(model_log, new_input, new_input_kwargs)
    computed_activations = _execute_replay(
        model_log,
        [layer.layer_label for layer in model_log.layer_list],
        input_layer_values,
    )

    output_template = getattr(model_log, "_output_structure_template", None)
    if output_template is not None:
        return _reconstruct_output_from_template(output_template, computed_activations, model_log)

    if model_log.output_layers:
        if len(model_log.output_layers) == 1:
            return computed_activations[model_log.output_layers[0]]
        return [computed_activations[layer_label] for layer_label in model_log.output_layers]

    return None


def split_graph(
    model_log: "ModelLog",
    split_layer_indices: Union[int, List[int]],
) -> Tuple[List[str], List[str], List[str]]:
    """Split a DAG with a topological boundary induced by layer index."""
    total_layers = len(model_log.layer_list)
    split_indices = _normalize_split_indices(split_layer_indices, total_layers)
    max_split_idx = max(split_indices)
    replay_cache = _get_replay_cache(model_log)
    split_boundary_cache = replay_cache["split_boundary_cache"]

    if max_split_idx in split_boundary_cache:
        return split_boundary_cache[max_split_idx]

    subgraph1_labels = [layer.layer_label for layer in model_log.layer_list[: max_split_idx + 1]]
    subgraph2_labels = [layer.layer_label for layer in model_log.layer_list[max_split_idx + 1 :]]
    subgraph2_set = set(subgraph2_labels)

    split_point_labels = []
    for layer_label in subgraph1_labels:
        layer = model_log[layer_label]
        child_labels = getattr(layer, "child_layers", [])
        has_child_in_sg2 = any(child_label in subgraph2_set for child_label in child_labels)
        is_output = layer_label in model_log.output_layers
        if has_child_in_sg2 or is_output:
            split_point_labels.append(layer_label)

    result = (subgraph1_labels, subgraph2_labels, split_point_labels)
    split_boundary_cache[max_split_idx] = result
    return result


def replay_subgraph(
    model_log: "ModelLog",
    subgraph_labels: List[str],
    input_values: Dict[str, Any],
) -> Dict[str, Any]:
    """Replay a subgraph with provided input values.

    Like :func:`replay_forward_pass`, this reconstructs values only and does not
    preserve a training-time autograd graph across the replayed subgraph boundary.
    """
    return _execute_replay(model_log, subgraph_labels, input_values)


def split_and_replay_graph(
    model_log: "ModelLog",
    split_layer_indices: Union[int, List[int]],
    new_input: Union[torch.Tensor, List[Any], Tuple[Any, ...]],
    new_input_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Any]:
    """Split graph at specified layers and replay both subgraphs with new input.

    The intermediate features and final outputs reproduce forward values, but the
    split replay is not currently differentiable end-to-end with respect to the
    original model parameters or the pre-split inputs.
    """
    subgraph1_labels, subgraph2_labels, split_labels = split_graph(model_log, split_layer_indices)
    input_layer_values = _build_replay_input_layer_values(model_log, new_input, new_input_kwargs)

    subgraph1_outputs = replay_subgraph(model_log, subgraph1_labels, input_layer_values)
    intermediate_features = {label: subgraph1_outputs[label] for label in split_labels}
    subgraph2_outputs = replay_subgraph(model_log, subgraph2_labels, intermediate_features)

    combined_outputs = {**subgraph1_outputs, **subgraph2_outputs}
    output_template = getattr(model_log, "_output_structure_template", None)
    if output_template is not None:
        final_output = _reconstruct_output_from_template(output_template, combined_outputs, model_log)
    elif model_log.output_layers:
        if len(model_log.output_layers) == 1:
            final_output = combined_outputs[model_log.output_layers[0]]
        else:
            final_output = [combined_outputs[label] for label in model_log.output_layers]
    else:
        final_output = None

    return intermediate_features, final_output


def replay_forward_pass_differentiable(
    model_log: "ModelLog",
    model: nn.Module,
    new_input: Union[torch.Tensor, List[Any], Tuple[Any, ...]],
    new_input_kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    """Replay the full graph while preserving autograd through inputs and live params."""
    input_layer_values = _build_replay_input_layer_values(model_log, new_input, new_input_kwargs)
    computed_activations = _execute_replay_differentiable(
        model_log,
        model,
        [layer.layer_label for layer in model_log.layer_list],
        input_layer_values,
    )

    output_template = getattr(model_log, "_output_structure_template", None)
    if output_template is not None:
        return _reconstruct_output_from_template(output_template, computed_activations, model_log)

    if model_log.output_layers:
        if len(model_log.output_layers) == 1:
            return computed_activations[model_log.output_layers[0]]
        return [computed_activations[layer_label] for layer_label in model_log.output_layers]

    return None


def split_and_replay_graph_differentiable(
    model_log: "ModelLog",
    model: nn.Module,
    split_layer_indices: Union[int, List[int]],
    new_input: Union[torch.Tensor, List[Any], Tuple[Any, ...]],
    new_input_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Any]:
    """Split the graph and replay both halves while preserving autograd connectivity."""
    subgraph1_labels, subgraph2_labels, split_labels = split_graph(model_log, split_layer_indices)
    input_layer_values = _build_replay_input_layer_values(model_log, new_input, new_input_kwargs)

    subgraph1_outputs = _execute_replay_differentiable(
        model_log,
        model,
        subgraph1_labels,
        input_layer_values,
    )
    intermediate_features = {label: subgraph1_outputs[label] for label in split_labels}
    subgraph2_outputs = _execute_replay_differentiable(
        model_log,
        model,
        subgraph2_labels,
        intermediate_features,
    )

    combined_outputs = {**subgraph1_outputs, **subgraph2_outputs}
    output_template = getattr(model_log, "_output_structure_template", None)
    if output_template is not None:
        final_output = _reconstruct_output_from_template(output_template, combined_outputs, model_log)
    elif model_log.output_layers:
        if len(model_log.output_layers) == 1:
            final_output = combined_outputs[model_log.output_layers[0]]
        else:
            final_output = [combined_outputs[label] for label in model_log.output_layers]
    else:
        final_output = None

    return intermediate_features, final_output
