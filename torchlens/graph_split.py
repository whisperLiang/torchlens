"""Graph replay and split helpers.

Keeps replay/split implementation out of the public API module and uses a
single execution engine for both full-graph replay and subgraph replay.
"""

import copy
import inspect
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, cast

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .utils.collections import assign_to_sequence_or_dict, index_nested
from .utils.tensor_utils import tensor_nanequal

if TYPE_CHECKING:
    from .data_classes.model_log import ModelLog


_PARENT_TEMPLATE_SENTINEL = object()
_MODULE_INPUT_REF_MARKER = "__tl_module_input_ref__"
_DYNAMIC_UNSAFE_LAYER_TYPES = {
    "getitem",
    "setitem",
    "topk",
    "splitwithsizes",
    "randperm",
    "nonzero",
    "where",
    "unique",
    "unique2",
    "argsort",
    "sort",
    "maskedselect",
}


def _layer_in_module_pass_subtree(layer: Any, module_pass: str) -> bool:
    """Return whether a layer belongs to the given module-pass subtree."""
    containing_module = getattr(layer, "containing_module", None)
    if containing_module == module_pass:
        return True
    return module_pass in getattr(layer, "containing_modules", [])


def _collect_output_template_refs(template: Any) -> List[str]:
    """Collect all layer-label references stored in an output structure template."""
    if isinstance(template, tuple) and len(template) == 2 and template[0] == "__tl_output_ref__":
        return [template[1]]

    if isinstance(template, list):
        refs: List[str] = []
        for item in template:
            refs.extend(_collect_output_template_refs(item))
        return refs

    if isinstance(template, dict):
        refs = []
        if "__tl_tuple_type__" in template and "items" in template:
            for item in template["items"]:
                refs.extend(_collect_output_template_refs(item))
            return refs

        if "__tl_dict_type__" in template and "items" in template:
            for _, value in template["items"]:
                refs.extend(_collect_output_template_refs(value))
            return refs

        for value in template.values():
            refs.extend(_collect_output_template_refs(value))
        return refs

    if hasattr(template, "__dict__") and not isinstance(template, type):
        refs = []
        for value in vars(template).values():
            refs.extend(_collect_output_template_refs(value))
        return refs

    return []


def _template_has_structured_output(template: Any) -> bool:
    """Return whether an output template represents a non-scalar container output."""
    if isinstance(template, list):
        return True
    if isinstance(template, dict):
        return True
    if hasattr(template, "__dict__") and not isinstance(template, type):
        return True
    return False


def _collect_module_input_template_refs(
    template: Any,
    seen: Optional[set[int]] = None,
) -> List[str]:
    """Collect layer-label references stored in a module input template."""
    if seen is None:
        seen = set()

    obj_id = id(template)
    if obj_id in seen:
        return []
    seen.add(obj_id)

    if (
        isinstance(template, tuple)
        and len(template) == 2
        and template[0] == _MODULE_INPUT_REF_MARKER
    ):
        return [template[1]]

    if isinstance(template, (list, tuple)):
        refs: List[str] = []
        for item in template:
            refs.extend(_collect_module_input_template_refs(item, seen))
        return refs

    if isinstance(template, dict):
        refs = []
        for value in template.values():
            refs.extend(_collect_module_input_template_refs(value, seen))
        return refs

    if hasattr(template, "__dict__"):
        refs = []
        for value in vars(template).values():
            refs.extend(_collect_module_input_template_refs(value, seen))
        return refs

    return []


def _get_replay_target_output_labels(model_log: "ModelLog") -> List[str]:
    """Return the final labels required to rebuild the model's user-visible outputs."""
    output_template = getattr(model_log, "_output_structure_template", None)
    if output_template is None:
        return list(model_log.output_layers)

    raw_to_final = getattr(model_log, "_raw_to_final_layer_labels", {})
    template_labels = [
        raw_to_final.get(label, label) for label in _collect_output_template_refs(output_template)
    ]
    return list(dict.fromkeys(template_labels + list(model_log.output_layers)))


def _get_replay_cache(model_log: "ModelLog") -> Dict[str, Any]:
    """Lazily create and return replay-related caches stored on the ModelLog."""
    cache = getattr(model_log, "_graph_split_cache", None)
    if cache is None:
        cache = {
            "final_output_labels": _get_replay_target_output_labels(model_log),
            "label_to_layer": {layer.layer_label: layer for layer in model_log.layer_list},
            "output_ancestor_labels": [
                layer.layer_label
                for layer in model_log.layer_list
                if layer.is_output_layer or layer.is_output_ancestor
            ],
            "layer_positions": {
                layer.layer_label: idx for idx, layer in enumerate(model_log.layer_list)
            },
            "parent_child_route_cache": {},
            "replay_node_cache": {},
            "atomic_module_spans": None,
            "root_dynamic_spans": None,
            "atomic_module_plans": None,
            "split_boundary_cache": {},
        }
        setattr(model_log, "_graph_split_cache", cache)
    return cache


def _get_atomic_module_spans(model_log: "ModelLog") -> List[Tuple[int, int, str]]:
    """Return minimal module-pass spans that should not be split internally."""
    replay_cache = _get_replay_cache(model_log)
    cached_spans = replay_cache.get("atomic_module_spans")
    if cached_spans is not None:
        return cached_spans

    unsafe_modules = set()
    for layer in model_log.layer_list:
        containing_module = getattr(layer, "containing_module", None)
        if containing_module is None:
            continue
        layer_type = getattr(layer, "layer_type", "")
        if (
            layer_type in _DYNAMIC_UNSAFE_LAYER_TYPES
            or getattr(layer, "iterable_output_index", None) is not None
        ):
            unsafe_modules.add(containing_module)

    atomic_spans = []
    for module_pass in sorted(unsafe_modules):
        region_indices = [
            idx
            for idx, layer in enumerate(model_log.layer_list)
            if _layer_in_module_pass_subtree(layer, module_pass)
        ]
        if not region_indices:
            continue
        atomic_spans.append((region_indices[0], region_indices[-1], module_pass))

    atomic_spans.sort()
    replay_cache["atomic_module_spans"] = atomic_spans
    return atomic_spans


def _get_root_dynamic_spans(model_log: "ModelLog") -> List[Tuple[int, int]]:
    """Return root-level dynamic spans that should not be split internally."""
    replay_cache = _get_replay_cache(model_log)
    cached_spans = replay_cache.get("root_dynamic_spans")
    if cached_spans is not None:
        return cached_spans

    root_spans: List[Tuple[int, int]] = []
    total_layers = len(model_log.layer_list)
    for idx, layer in enumerate(model_log.layer_list):
        if getattr(layer, "containing_module", None) is not None:
            continue
        if not (
            getattr(layer, "layer_type", "") in _DYNAMIC_UNSAFE_LAYER_TYPES
            or getattr(layer, "iterable_output_index", None) is not None
        ):
            continue

        start_idx = idx
        while (
            start_idx > 0
            and getattr(model_log.layer_list[start_idx - 1], "containing_module", None) is None
        ):
            start_idx -= 1

        end_idx = idx
        while (
            end_idx + 1 < total_layers
            and getattr(model_log.layer_list[end_idx + 1], "containing_module", None) is None
        ):
            end_idx += 1

        if root_spans and start_idx <= root_spans[-1][1]:
            prev_start, prev_end = root_spans[-1]
            root_spans[-1] = (prev_start, max(prev_end, end_idx))
        else:
            root_spans.append((start_idx, end_idx))

    replay_cache["root_dynamic_spans"] = root_spans
    return root_spans


def _snap_split_index_to_atomic_boundary(model_log: "ModelLog", split_idx: int) -> int:
    """Move a split index outside any dynamic atomic module span that contains it."""
    total_layers = len(model_log.layer_list)
    if total_layers == 0:
        return split_idx

    current_idx = max(0, min(total_layers - 1, split_idx))
    atomic_spans = [(start_idx, end_idx) for start_idx, end_idx, _ in _get_atomic_module_spans(model_log)]
    atomic_spans.extend(_get_root_dynamic_spans(model_log))

    changed = True
    while changed:
        changed = False
        for start_idx, end_idx in atomic_spans:
            if current_idx < start_idx or current_idx >= end_idx:
                continue

            left_boundary = start_idx - 1
            right_boundary = end_idx
            candidate_boundaries = []
            if left_boundary >= 0:
                candidate_boundaries.append(left_boundary)
            if right_boundary < total_layers:
                candidate_boundaries.append(right_boundary)
            if not candidate_boundaries:
                return current_idx

            current_idx = min(
                candidate_boundaries,
                key=lambda candidate: (abs(candidate - split_idx), candidate < split_idx),
            )
            changed = True
            break

    return current_idx


def _get_live_replay_model(model_log: "ModelLog") -> Optional[nn.Module]:
    """Return the live model instance associated with a ModelLog, if it is still alive."""
    model_ref = getattr(model_log, "_source_model_ref", None)
    if model_ref is None:
        return None
    try:
        return model_ref()
    except TypeError:
        return cast(Optional[nn.Module], model_ref)


def _get_live_module_for_pass(model: nn.Module, module_pass: str) -> Optional[nn.Module]:
    """Resolve a ``module_pass`` label like ``foo.bar:1`` to a live submodule."""
    module_address, _, _ = module_pass.rpartition(":")
    live_model = _unwrap_replay_model(model)
    if module_address == "":
        return live_model
    return dict(live_model.named_modules()).get(module_address)


def _materialize_module_replay_template(
    val: Any,
    computed: Dict[str, Any],
    clone_fn: Any,
    device: Optional[Union[str, torch.device]] = None,
    memo: Optional[Dict[int, Any]] = None,
) -> Any:
    """Instantiate a cached module-call template with current computed values."""
    if memo is None:
        memo = {}

    obj_id = id(val)
    if obj_id in memo:
        return memo[obj_id]

    if isinstance(val, tuple) and len(val) == 2 and val[0] == _MODULE_INPUT_REF_MARKER:
        layer_label = val[1]
        if layer_label not in computed:
            raise KeyError(layer_label)
        return computed[layer_label]

    if isinstance(val, torch.Tensor):
        cloned = clone_fn(val, device=device)
        memo[obj_id] = cloned
        return cloned

    if isinstance(val, list):
        copied_list: List[Any] = []
        memo[obj_id] = copied_list
        copied_list.extend(
            _materialize_module_replay_template(v, computed, clone_fn, device, memo) for v in val
        )
        return copied_list

    if isinstance(val, tuple):
        placeholder: List[Any] = []
        memo[obj_id] = placeholder
        copied_tuple = type(val)(
            _materialize_module_replay_template(v, computed, clone_fn, device, memo) for v in val
        )
        memo[obj_id] = copied_tuple
        return copied_tuple

    if isinstance(val, dict):
        copied_dict: Dict[Any, Any] = type(val)()
        memo[obj_id] = copied_dict
        copied_dict.update(
            {
                key: _materialize_module_replay_template(value, computed, clone_fn, device, memo)
                for key, value in val.items()
            }
        )
        return copied_dict

    if hasattr(val, "__dict__"):
        copied_obj = copy.copy(val)
        memo[obj_id] = copied_obj
        for attr_name, attr_value in vars(val).items():
            setattr(
                copied_obj,
                attr_name,
                _materialize_module_replay_template(attr_value, computed, clone_fn, device, memo),
            )
        return copied_obj

    memo[obj_id] = val
    return val


def _collect_outputs_from_template(template: Any, actual: Any, collected: Dict[str, Any]) -> bool:
    """Collect tensor values addressed by a template from an actual module output."""
    if isinstance(template, tuple) and len(template) == 2 and template[0] == "__tl_output_ref__":
        collected[template[1]] = actual
        return True

    if isinstance(template, list):
        if not isinstance(actual, list) or len(template) != len(actual):
            return False
        return all(
            _collect_outputs_from_template(template_item, actual_item, collected)
            for template_item, actual_item in zip(template, actual)
        )

    if isinstance(template, dict):
        if "__tl_tuple_type__" in template and "items" in template:
            if not isinstance(actual, tuple) or len(template["items"]) != len(actual):
                return False
            return all(
                _collect_outputs_from_template(template_item, actual_item, collected)
                for template_item, actual_item in zip(template["items"], actual)
            )

        if "__tl_dict_type__" in template and "items" in template:
            if not isinstance(actual, dict):
                return False
            for key, template_value in template["items"]:
                if key not in actual:
                    return False
                if not _collect_outputs_from_template(template_value, actual[key], collected):
                    return False
            return True

    if hasattr(template, "__dict__") and not isinstance(template, type):
        if not hasattr(actual, "__dict__") or isinstance(actual, type):
            return False
        for attr_name, template_value in vars(template).items():
            if not hasattr(actual, attr_name):
                return False
            if not _collect_outputs_from_template(
                template_value,
                getattr(actual, attr_name),
                collected,
            ):
                return False
        return True

    return template == actual


def _get_atomic_module_plans(model_log: "ModelLog") -> List[Dict[str, Any]]:
    """Return cached execution plans for atomic module-pass regions."""
    replay_cache = _get_replay_cache(model_log)
    cached_plans = replay_cache.get("atomic_module_plans")
    if cached_plans is not None:
        return cached_plans

    module_templates = getattr(model_log, "_module_replay_templates", {})
    atomic_plans: List[Dict[str, Any]] = []
    planned_passes = set()

    def _append_module_plan(module_pass: str, start_idx: int, end_idx: int) -> None:
        if module_pass in planned_passes:
            return
        span_labels = [
            layer.layer_label for layer in model_log.layer_list[start_idx : end_idx + 1]
        ]
        if not span_labels:
            return

        region_labels = [
            layer.layer_label
            for layer in model_log.layer_list[start_idx : end_idx + 1]
            if _layer_in_module_pass_subtree(layer, module_pass)
        ]
        if not region_labels:
            return

        span_label_set = set(span_labels)
        template = module_templates.get(module_pass)
        output_labels = []
        input_labels = []
        if template is not None:
            output_labels = _collect_output_template_refs(template["output_template"])
            input_labels = list(
                dict.fromkeys(
                    _collect_module_input_template_refs(template["args_template"])
                    + _collect_module_input_template_refs(template["kwargs_template"])
                )
            )
        atomic_plans.append(
            {
                "module_pass": module_pass,
                "start_label": span_labels[0],
                "span_labels": span_labels,
                "span_label_set": span_label_set,
                "plan_label_set": span_label_set.union(output_labels),
                "region_labels": region_labels,
                "region_label_set": set(region_labels),
                "input_labels": [
                    label for label in input_labels if label not in span_label_set
                ],
                "output_labels": output_labels,
                "template": template,
            }
        )
        planned_passes.add(module_pass)

    for start_idx, end_idx, module_pass in _get_atomic_module_spans(model_log):
        _append_module_plan(module_pass, start_idx, end_idx)

    atomic_input_labels = {
        input_label
        for plan in atomic_plans
        for input_label in plan["input_labels"]
    }
    if atomic_input_labels:
        for module_pass, template in module_templates.items():
            if module_pass in planned_passes or template is None:
                continue
            module_address, _, _ = module_pass.rpartition(":")
            if module_address == "":
                continue
            if not _template_has_structured_output(template["output_template"]):
                continue

            output_labels = _collect_output_template_refs(template["output_template"])
            if not atomic_input_labels.intersection(output_labels):
                continue

            region_indices = [
                idx
                for idx, layer in enumerate(model_log.layer_list)
                if _layer_in_module_pass_subtree(layer, module_pass)
            ]
            if not region_indices:
                continue
            _append_module_plan(module_pass, region_indices[0], region_indices[-1])

    replay_cache["atomic_module_plans"] = atomic_plans
    return atomic_plans


def _execute_atomic_module_region(
    model_log: "ModelLog",
    model: nn.Module,
    module_pass: str,
    computed: Dict[str, Any],
    preserve_graph: bool,
    device: Optional[Union[str, torch.device]] = None,
) -> Optional[Dict[str, Any]]:
    """Execute one atomic module region via the live submodule and map its outputs."""
    template = getattr(model_log, "_module_replay_templates", {}).get(module_pass)
    if template is None:
        return None

    live_module = _get_live_module_for_pass(model, module_pass)
    if live_module is None:
        return None

    clone_fn = _clone_tensors_preserve_graph if preserve_graph else _deep_clone_tensors
    try:
        args = _materialize_module_replay_template(
            template["args_template"], computed, clone_fn, device=device
        )
        kwargs = _materialize_module_replay_template(
            template["kwargs_template"], computed, clone_fn, device=device
        )
    except KeyError:
        return None

    if not isinstance(args, tuple):
        args = tuple(args)

    output = live_module(*args, **kwargs)
    collected_outputs: Dict[str, Any] = {}
    if not _collect_outputs_from_template(template["output_template"], output, collected_outputs):
        return None
    return collected_outputs


def _deep_clone_tensors(val: Any, device: Optional[Union[str, torch.device]] = None) -> Any:
    """Recursively clone all tensors in a nested structure of lists/tuples/dicts."""
    if isinstance(val, torch.Tensor):
        t = val.detach().clone()
        return t if device is None else t.to(device)
    if isinstance(val, (list, tuple)):
        cloned = [_deep_clone_tensors(v, device) for v in val]
        return type(val)(cloned)
    if isinstance(val, dict):
        return {k: _deep_clone_tensors(v, device) for k, v in val.items()}
    return val


def _clone_tensors_preserve_graph(
    val: Any, device: Optional[Union[str, torch.device]] = None
) -> Any:
    """Recursively clone tensors while preserving their autograd connections."""
    return _clone_tensors_preserve_graph_with_memo(val, {}, device=device)


def _clone_tensors_preserve_graph_with_memo(
    val: Any, memo: Dict[int, Any], device: Optional[Union[str, torch.device]] = None
) -> Any:
    """Clone tensors while preserving graph links and shared-reference topology."""
    obj_id = id(val)
    if obj_id in memo:
        return memo[obj_id]

    if isinstance(val, torch.Tensor):
        if isinstance(val, torch.nn.Parameter):
            memo[obj_id] = val
            return val
        cloned_tensor = val.clone()
        if device is not None:
            cloned_tensor = cloned_tensor.to(device)
        memo[obj_id] = cloned_tensor
        return cloned_tensor

    if isinstance(val, list):
        cloned_list: List[Any] = []
        memo[obj_id] = cloned_list
        cloned_list.extend(
            _clone_tensors_preserve_graph_with_memo(v, memo, device=device) for v in val
        )
        return cloned_list

    if isinstance(val, tuple):
        placeholder: List[Any] = []
        memo[obj_id] = placeholder
        cloned_tuple = type(val)(
            _clone_tensors_preserve_graph_with_memo(v, memo, device=device) for v in val
        )
        memo[obj_id] = cloned_tuple
        return cloned_tuple

    if isinstance(val, dict):
        cloned_dict: Dict[Any, Any] = {}
        memo[obj_id] = cloned_dict
        cloned_dict.update(
            {
                k: _clone_tensors_preserve_graph_with_memo(v, memo, device=device)
                for k, v in val.items()
            }
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
        copied_tuple = type(val)(_copy_containers_preserve_tensors_with_memo(v, memo) for v in val)
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


def _maybe_retry_setitem_with_safe_indexing(
    func: Any,
    args_list: List[Any],
    kwargs_dict: Dict[str, Any],
) -> Optional[Any]:
    """Retry tensor ``__setitem__`` after filtering invalid integer indices."""
    if getattr(func, "__name__", None) != "__setitem__":
        return None
    if kwargs_dict or len(args_list) < 3:
        return None

    source = args_list[0]
    index = args_list[1]
    value = args_list[2]
    if not isinstance(source, torch.Tensor) or source.ndim == 0:
        return None
    if not isinstance(index, torch.Tensor):
        return None
    if index.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        return None

    dim0 = source.shape[0]
    valid_mask = (index >= -dim0) & (index < dim0)
    safe_index = index[valid_mask]
    if isinstance(value, torch.Tensor) and value.shape[:1] == index.shape[:1]:
        value = value[valid_mask]
    func(source, safe_index, value)
    return source


def _maybe_retry_topk_with_runtime_k(
    func: Any,
    args_list: List[Any],
    kwargs_dict: Dict[str, Any],
) -> Optional[Any]:
    """Retry ``topk`` after clamping ``k`` to the current runtime dimension size."""
    if getattr(func, "__name__", None) != "topk":
        return None
    if len(args_list) < 2:
        return None

    source = args_list[0]
    k = args_list[1]
    dim = kwargs_dict.get("dim", args_list[2] if len(args_list) >= 3 else -1)
    if not isinstance(source, torch.Tensor):
        return None
    if not isinstance(k, int) or not isinstance(dim, int):
        return None

    runtime_dim_size = int(source.shape[dim])
    if k <= runtime_dim_size:
        return None

    adjusted_args = list(args_list)
    adjusted_args[1] = runtime_dim_size
    return func(*adjusted_args, **kwargs_dict)


def _maybe_retry_split_with_runtime_sizes(
    func: Any,
    args_list: List[Any],
    kwargs_dict: Dict[str, Any],
) -> Optional[Any]:
    """Retry ``split_with_sizes`` after adapting the final chunk to runtime length."""
    if getattr(func, "__name__", None) != "split_with_sizes":
        return None
    if kwargs_dict or len(args_list) < 2:
        return None

    source = args_list[0]
    split_sizes = args_list[1]
    dim = args_list[2] if len(args_list) >= 3 else 0
    if not isinstance(source, torch.Tensor):
        return None
    if not isinstance(split_sizes, (list, tuple)) or len(split_sizes) == 0:
        return None
    if not all(isinstance(size, int) for size in split_sizes):
        return None
    if not isinstance(dim, int):
        return None

    runtime_dim_size = int(source.shape[dim])
    if sum(split_sizes) == runtime_dim_size:
        return None

    adjusted_split_sizes = list(split_sizes)
    fixed_prefix = sum(adjusted_split_sizes[:-1])
    adjusted_final_size = runtime_dim_size - fixed_prefix
    if adjusted_final_size < 0:
        return None

    adjusted_split_sizes[-1] = adjusted_final_size
    return func(source, adjusted_split_sizes, dim)


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
    if isinstance(args_list, list):
        if not isinstance(pos, tuple):
            if not isinstance(pos, int):
                raise TypeError("List replay args require integer positions.")
            args_list[pos] = value
            return

        outer_key, nested_key = pos
        if not isinstance(outer_key, int):
            raise TypeError("Nested list replay args require an integer outer position.")
        args_list[outer_key] = assign_to_sequence_or_dict(args_list[outer_key], nested_key, value)
        return

    if not isinstance(pos, tuple):
        if not isinstance(pos, str):
            raise TypeError("Dict replay args require string positions.")
        args_list[pos] = value
        return

    outer_key, nested_key = pos
    if not isinstance(outer_key, str):
        raise TypeError("Nested dict replay args require a string outer position.")
    args_list[outer_key] = assign_to_sequence_or_dict(args_list[outer_key], nested_key, value)


def _build_replay_node(layer: Any) -> Dict[str, Any]:
    """Build the cached replay template for a single layer."""
    raw_args = getattr(layer, "captured_args", None)
    raw_kwargs = getattr(layer, "captured_kwargs", None)
    args_template = _copy_containers_preserve_tensors(
        list(raw_args) if raw_args is not None else []
    )
    kwargs_template = _copy_containers_preserve_tensors(
        dict(raw_kwargs) if raw_kwargs is not None else {}
    )

    parent_arg_locs = getattr(layer, "parent_layer_arg_locs", {})
    for arg_pos in parent_arg_locs.get("args", {}):
        _apply_value_to_args(args_template, arg_pos, _PARENT_TEMPLATE_SENTINEL)
    for kwarg_name in parent_arg_locs.get("kwargs", {}):
        _apply_value_to_args(kwargs_template, kwarg_name, _PARENT_TEMPLATE_SENTINEL)

    return {
        "args_template": args_template,
        "kwargs_template": kwargs_template,
        "parent_arg_locs": parent_arg_locs,
        "func": getattr(layer, "func_applied", None),
        "func_is_inplace": getattr(layer, "func_is_inplace", False),
        "iterable_output_index": getattr(layer, "iterable_output_index", None),
    }


def _get_replay_node(model_log: "ModelLog", layer_label: str) -> Dict[str, Any]:
    """Fetch or lazily build the cached replay node for ``layer_label``."""
    replay_cache = _get_replay_cache(model_log)
    replay_node_cache = replay_cache["replay_node_cache"]
    if layer_label not in replay_node_cache:
        replay_node_cache[layer_label] = _build_replay_node(
            replay_cache["label_to_layer"][layer_label]
        )
    return replay_node_cache[layer_label]


def _select_execution_labels(
    model_log: "ModelLog",
    allowed_labels: List[str],
    target_labels: List[str],
) -> List[str]:
    """Return the target labels plus all of their ancestors within ``allowed_labels``."""
    replay_cache = _get_replay_cache(model_log)
    label_to_layer = replay_cache["label_to_layer"]
    allowed_set = set(allowed_labels)
    required_labels = {label for label in target_labels if label in allowed_set}
    frontier = list(required_labels)

    while frontier:
        current_label = frontier.pop()
        for parent_label in label_to_layer[current_label].parent_layers:
            if parent_label not in allowed_set or parent_label in required_labels:
                continue
            required_labels.add(parent_label)
            frontier.append(parent_label)

    return [label for label in allowed_labels if label in required_labels]


def _build_remaining_child_use_counts(
    model_log: "ModelLog",
    execution_labels: List[str],
    seeded_labels: List[str],
) -> Dict[str, int]:
    """Count how many in-subgraph children still need each computed value."""
    replay_cache = _get_replay_cache(model_log)
    label_to_layer = replay_cache["label_to_layer"]
    remaining_child_uses = {label: 0 for label in execution_labels}
    for seeded_label in seeded_labels:
        if seeded_label not in remaining_child_uses:
            remaining_child_uses[seeded_label] = 0

    tracked_labels = set(remaining_child_uses)
    for child_label in execution_labels:
        for parent_label in label_to_layer[child_label].parent_layers:
            if parent_label in tracked_labels:
                remaining_child_uses[parent_label] += 1

    return remaining_child_uses


def _partition_seed_values(
    seeded_values: Dict[str, Any],
) -> Tuple[List[str], List[torch.Tensor], Dict[str, Any]]:
    """Split seed values into tensor inputs and constant values."""
    tensor_seed_labels: List[str] = []
    tensor_seed_values: List[torch.Tensor] = []
    constant_seed_values: Dict[str, Any] = {}

    for label, value in seeded_values.items():
        if isinstance(value, torch.Tensor):
            tensor_seed_labels.append(label)
            tensor_seed_values.append(value)
        else:
            constant_seed_values[label] = value

    return tensor_seed_labels, tensor_seed_values, constant_seed_values


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
    if isinstance(new_input, (tuple, list)):
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


def _assign_value_to_nested_path(
    container: Any,
    path: Tuple[Any, ...],
    value: Any,
) -> Any:
    """Assign ``value`` into a nested list/dict structure described by ``path``."""
    if not path:
        return value

    key = path[0]
    if isinstance(key, int):
        if container is None or not isinstance(container, list):
            container = []
        while len(container) <= key:
            container.append(None)
        container[key] = _assign_value_to_nested_path(container[key], path[1:], value)
        return container

    if container is None or not isinstance(container, dict):
        container = {}
    container[key] = _assign_value_to_nested_path(container.get(key), path[1:], value)
    return container


def _build_live_model_replay_inputs(
    model_log: "ModelLog",
    new_input: Union[torch.Tensor, List[Any], Tuple[Any, ...], None],
    new_input_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Any], Dict[str, Any]]:
    """Reconstruct live-model call args/kwargs from replay-style inputs."""
    if new_input_kwargs is None:
        new_input_kwargs = {}

    input_root_names = []
    root_specs: Dict[str, List[Tuple[Tuple[Any, ...], str]]] = {}
    seen_root_names = set()
    for input_layer_label in model_log.input_layers:
        input_layer = model_log[input_layer_label]
        if input_layer.io_role is None:
            continue
        root_name, nested_path = _parse_input_io_role(input_layer.io_role)
        if root_name not in seen_root_names:
            input_root_names.append(root_name)
            seen_root_names.add(root_name)
        root_specs.setdefault(root_name, []).append((nested_path, input_layer_label))

    kwarg_root_names = {name for name in input_root_names if name in new_input_kwargs}
    positional_root_names = [name for name in input_root_names if name not in kwarg_root_names]
    positional_inputs = _normalize_replay_positional_inputs(new_input, len(positional_root_names))
    if len(positional_inputs) < len(positional_root_names):
        raise ValueError(
            "Not enough replay inputs to rebuild live-model arguments: "
            f"expected {len(positional_root_names)}, got {len(positional_inputs)}."
        )

    root_values: Dict[str, Any] = {}
    for idx, root_name in enumerate(positional_root_names):
        candidate = positional_inputs[idx]
        specs = root_specs.get(root_name, [])
        nonempty_paths = [path for path, _ in specs if path]
        if len(nonempty_paths) == 1 and not isinstance(candidate, (list, tuple, dict)):
            root_values[root_name] = _assign_value_to_nested_path(None, nonempty_paths[0], candidate)
        else:
            root_values[root_name] = candidate

    for root_name in kwarg_root_names:
        root_values[root_name] = new_input_kwargs[root_name]

    positional_args = [root_values[root_name] for root_name in positional_root_names]
    keyword_args = {root_name: root_values[root_name] for root_name in kwarg_root_names}
    return positional_args, keyword_args


def _final_outputs_overlap_root_dynamic_span(model_log: "ModelLog") -> bool:
    """Return whether any final replay output falls inside a root dynamic span."""
    replay_cache = _get_replay_cache(model_log)
    final_output_labels = set(replay_cache["final_output_labels"])
    if not final_output_labels:
        return False

    for start_idx, end_idx in _get_root_dynamic_spans(model_log):
        span_labels = {
            layer.layer_label for layer in model_log.layer_list[start_idx : end_idx + 1]
        }
        if final_output_labels.intersection(span_labels):
            return True
    return False


def _has_dynamic_replay_regions(model_log: "ModelLog") -> bool:
    """Return whether the graph contains dynamic module or root-level regions."""
    return bool(_get_atomic_module_plans(model_log) or _get_root_dynamic_spans(model_log))


def _call_live_model_with_replay_inputs(
    model_log: "ModelLog",
    new_input: Union[torch.Tensor, List[Any], Tuple[Any, ...], None],
    new_input_kwargs: Optional[Dict[str, Any]] = None,
    model: Optional[nn.Module] = None,
    preserve_graph: bool = False,
) -> Optional[Any]:
    """Execute the live model using replay-style inputs if reconstruction succeeds."""
    live_model = _unwrap_replay_model(model) if model is not None else _get_live_replay_model(model_log)
    if live_model is None:
        return None

    try:
        positional_args, keyword_args = _build_live_model_replay_inputs(
            model_log,
            new_input,
            new_input_kwargs,
        )
    except (KeyError, TypeError, ValueError, IndexError):
        return None

    if preserve_graph:
        return live_model(*positional_args, **keyword_args)

    with torch.no_grad():
        return live_model(*positional_args, **keyword_args)


def _maybe_call_live_model_for_outputs(
    model_log: "ModelLog",
    new_input: Union[torch.Tensor, List[Any], Tuple[Any, ...], None],
    new_input_kwargs: Optional[Dict[str, Any]] = None,
    model: Optional[nn.Module] = None,
    preserve_graph: bool = False,
) -> Optional[Any]:
    """Use the live model when final outputs land in unsupported root dynamic spans."""
    if not _final_outputs_overlap_root_dynamic_span(model_log):
        return None

    return _call_live_model_with_replay_inputs(
        model_log,
        new_input,
        new_input_kwargs,
        model=model,
        preserve_graph=preserve_graph,
    )


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
        final_label = cast(str, raw_to_final.get(layer_label, layer_label))
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


def _build_live_state_maps(
    model: nn.Module,
) -> Tuple[Dict[str, torch.nn.Parameter], Dict[str, torch.Tensor]]:
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
        return {
            k: _swap_copied_params_for_live_params(v, live_params_by_addr) for k, v in val.items()
        }
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
    target_labels: Optional[List[str]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Replay only the ancestor closure needed to produce ``target_labels``."""
    replay_cache = _get_replay_cache(model_log)
    label_to_layer = replay_cache["label_to_layer"]
    parent_child_route_cache = replay_cache["parent_child_route_cache"]
    requested_labels = layer_labels if target_labels is None else target_labels
    requested_labels_ordered = list(dict.fromkeys(requested_labels))
    requested_label_set = set(requested_labels_ordered)
    execution_labels = _select_execution_labels(model_log, layer_labels, requested_labels_ordered)
    computed = {key: _deep_clone_tensors(val, device) for key, val in seeded_values.items()}
    requested_outputs = {
        label: computed[label] for label in requested_labels_ordered if label in computed
    }
    remaining_child_uses = _build_remaining_child_use_counts(
        model_log,
        execution_labels,
        list(seeded_values.keys()),
    )
    live_model = _get_live_replay_model(model_log)
    atomic_plan_by_first_label: Dict[str, Dict[str, Any]] = {}
    if live_model is not None:
        for plan in _get_atomic_module_plans(model_log):
            if plan["template"] is None:
                continue
            plan_execution_labels = [
                label for label in execution_labels if label in plan["plan_label_set"]
            ]
            if not plan_execution_labels:
                continue
            if any(
                label in requested_label_set and label not in plan["output_labels"]
                for label in plan_execution_labels
            ):
                continue
            first_label = plan_execution_labels[0]
            candidate_plan = {
                **plan,
                "execution_labels": plan_execution_labels,
            }
            existing_plan = atomic_plan_by_first_label.get(first_label)
            if existing_plan is None or len(candidate_plan["execution_labels"]) > len(
                existing_plan["execution_labels"]
            ):
                atomic_plan_by_first_label[first_label] = candidate_plan
        for plan in atomic_plan_by_first_label.values():
            for input_label in plan["input_labels"]:
                remaining_child_uses[input_label] = remaining_child_uses.get(input_label, 0) + 1

    execution_idx = 0
    while execution_idx < len(execution_labels):
        label = execution_labels[execution_idx]
        atomic_plan = atomic_plan_by_first_label.get(label)
        if atomic_plan is not None and live_model is not None:
            module_outputs = _execute_atomic_module_region(
                model_log,
                live_model,
                atomic_plan["module_pass"],
                computed,
                preserve_graph=False,
                device=device,
            )
            if module_outputs is not None:
                computed.update(module_outputs)
                for input_label in atomic_plan["input_labels"]:
                    if input_label not in remaining_child_uses:
                        continue
                    remaining_child_uses[input_label] -= 1
                    if (
                        remaining_child_uses[input_label] == 0
                        and input_label not in requested_label_set
                    ):
                        computed.pop(input_label, None)
                for region_label in atomic_plan["execution_labels"]:
                    if region_label in requested_label_set and region_label in computed:
                        requested_outputs[region_label] = computed[region_label]

                    layer = label_to_layer[region_label]
                    for parent_label in layer.parent_layers:
                        if parent_label not in remaining_child_uses:
                            continue
                        remaining_child_uses[parent_label] -= 1
                        if (
                            remaining_child_uses[parent_label] == 0
                            and parent_label not in requested_label_set
                        ):
                            computed.pop(parent_label, None)

                    if (
                        remaining_child_uses.get(region_label, 0) == 0
                        and region_label not in requested_label_set
                    ):
                        computed.pop(region_label, None)

                execution_idx += len(atomic_plan["execution_labels"])
                continue

        layer = label_to_layer[label]
        if label not in computed:
            replay_node = _get_replay_node(model_log, label)
            args_list = _deep_clone_tensors(replay_node["args_template"], device)
            kwargs_dict = _deep_clone_tensors(replay_node["kwargs_template"], device)

            parent_arg_locs = replay_node["parent_arg_locs"]
            for arg_pos, parent_label in parent_arg_locs.get("args", {}).items():
                if parent_label in computed:
                    parent_value = _resolve_parent_value_for_child(
                        model_log,
                        parent_label,
                        label,
                        computed[parent_label],
                        parent_child_route_cache,
                    )
                    _apply_value_to_args(args_list, arg_pos, parent_value)

            for kwarg_name, parent_label in parent_arg_locs.get("kwargs", {}).items():
                if parent_label in computed:
                    _apply_value_to_args(
                        kwargs_dict,
                        kwarg_name,
                        _resolve_parent_value_for_child(
                            model_log,
                            parent_label,
                            label,
                            computed[parent_label],
                            parent_child_route_cache,
                        ),
                    )

            func = replay_node["func"]
            if getattr(func, "__name__", None) == "randperm" and len(args_list) >= 1:
                inferred_n = _infer_runtime_randperm_arg(model_log, label, computed)
                if inferred_n is not None:
                    args_list[0] = inferred_n

            if func is not None:
                try:
                    output = func(*args_list, **kwargs_dict)
                except IndexError:
                    output = _maybe_retry_getitem_with_safe_indexing(func, args_list, kwargs_dict)
                    if output is None:
                        output = _maybe_retry_setitem_with_safe_indexing(
                            func, args_list, kwargs_dict
                        )
                    if output is None:
                        raise
                except RuntimeError:
                    output = _maybe_retry_split_with_runtime_sizes(func, args_list, kwargs_dict)
                    if output is None:
                        output = _maybe_retry_topk_with_runtime_k(func, args_list, kwargs_dict)
                    if output is None:
                        raise
                if output is None:
                    if getattr(layer, "func_is_inplace", False) and len(args_list) > 0:
                        output = args_list[0]
                    else:
                        output = _deep_clone_tensors(layer.activation, device)
                if layer.iterable_output_index is not None:
                    output = index_nested(output, layer.iterable_output_index)
            else:
                output = _deep_clone_tensors(layer.activation, device)

            computed[label] = output
        if label in requested_label_set:
            requested_outputs[label] = computed[label]

        for parent_label in layer.parent_layers:
            if parent_label not in remaining_child_uses:
                continue
            remaining_child_uses[parent_label] -= 1
            if remaining_child_uses[parent_label] == 0 and parent_label not in requested_label_set:
                computed.pop(parent_label, None)

        if remaining_child_uses.get(label, 0) == 0 and label not in requested_label_set:
            computed.pop(label, None)
        execution_idx += 1

    return {
        label: requested_outputs[label]
        for label in requested_labels_ordered
        if label in requested_outputs
    }


def _execute_replay_differentiable(
    model_log: "ModelLog",
    model: nn.Module,
    layer_labels: List[str],
    seeded_values: Dict[str, Any],
    target_labels: Optional[List[str]] = None,
    prune_to_targets: bool = True,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Replay only the ancestor closure needed to produce ``target_labels`` with autograd."""
    replay_cache = _get_replay_cache(model_log)
    computed = {key: _copy_containers_preserve_tensors(val) for key, val in seeded_values.items()}
    label_to_layer = replay_cache["label_to_layer"]
    parent_child_route_cache = replay_cache["parent_child_route_cache"]
    requested_labels = layer_labels if target_labels is None else target_labels
    requested_labels_ordered = list(dict.fromkeys(requested_labels))
    requested_label_set = set(requested_labels_ordered)
    if prune_to_targets:
        execution_labels = _select_execution_labels(
            model_log, layer_labels, requested_labels_ordered
        )
    else:
        execution_labels = list(layer_labels)
    requested_outputs = {
        label: computed[label] for label in requested_labels_ordered if label in computed
    }
    remaining_child_uses = _build_remaining_child_use_counts(
        model_log,
        execution_labels,
        list(seeded_values.keys()),
    )
    live_params_by_addr, live_buffers_by_addr = _build_live_state_maps(model)
    atomic_plan_by_first_label: Dict[str, Dict[str, Any]] = {}
    for plan in _get_atomic_module_plans(model_log):
        if plan["template"] is None:
            continue
        plan_execution_labels = [
            label for label in execution_labels if label in plan["plan_label_set"]
        ]
        if not plan_execution_labels:
            continue
        if any(
            label in requested_label_set and label not in plan["output_labels"]
            for label in plan_execution_labels
        ):
            continue
        first_label = plan_execution_labels[0]
        candidate_plan = {
            **plan,
            "execution_labels": plan_execution_labels,
        }
        existing_plan = atomic_plan_by_first_label.get(first_label)
        if existing_plan is None or len(candidate_plan["execution_labels"]) > len(
            existing_plan["execution_labels"]
        ):
            atomic_plan_by_first_label[first_label] = candidate_plan
    for plan in atomic_plan_by_first_label.values():
        for input_label in plan["input_labels"]:
            remaining_child_uses[input_label] = remaining_child_uses.get(input_label, 0) + 1

    execution_idx = 0
    while execution_idx < len(execution_labels):
        label = execution_labels[execution_idx]
        atomic_plan = atomic_plan_by_first_label.get(label)
        if atomic_plan is not None:
            module_outputs = _execute_atomic_module_region(
                model_log,
                model,
                atomic_plan["module_pass"],
                computed,
                preserve_graph=True,
                device=device,
            )
            if module_outputs is not None:
                computed.update(module_outputs)
                for input_label in atomic_plan["input_labels"]:
                    if input_label not in remaining_child_uses:
                        continue
                    remaining_child_uses[input_label] -= 1
                    if (
                        remaining_child_uses[input_label] == 0
                        and input_label not in requested_label_set
                    ):
                        computed.pop(input_label, None)
                for region_label in atomic_plan["execution_labels"]:
                    if region_label in requested_label_set and region_label in computed:
                        requested_outputs[region_label] = computed[region_label]

                    layer = label_to_layer[region_label]
                    for parent_label in layer.parent_layers:
                        if parent_label not in remaining_child_uses:
                            continue
                        remaining_child_uses[parent_label] -= 1
                        if (
                            remaining_child_uses[parent_label] == 0
                            and parent_label not in requested_label_set
                        ):
                            computed.pop(parent_label, None)

                    if (
                        remaining_child_uses.get(region_label, 0) == 0
                        and region_label not in requested_label_set
                    ):
                        computed.pop(region_label, None)

                execution_idx += len(atomic_plan["execution_labels"])
                continue

        layer = label_to_layer[label]
        if label not in computed:
            replay_node = _get_replay_node(model_log, label)
            args_list = _clone_tensors_preserve_graph(replay_node["args_template"], device=device)
            kwargs_dict = _clone_tensors_preserve_graph(
                replay_node["kwargs_template"], device=device
            )
            args_list = _swap_copied_params_for_live_params(args_list, live_params_by_addr)
            kwargs_dict = _swap_copied_params_for_live_params(kwargs_dict, live_params_by_addr)

            parent_arg_locs = replay_node["parent_arg_locs"]
            for arg_pos, parent_label in parent_arg_locs.get("args", {}).items():
                if parent_label in computed:
                    parent_value = _resolve_parent_value_for_child(
                        model_log,
                        parent_label,
                        label,
                        computed[parent_label],
                        parent_child_route_cache,
                    )
                    _apply_value_to_args(args_list, arg_pos, parent_value)

            for kwarg_name, parent_label in parent_arg_locs.get("kwargs", {}).items():
                if parent_label in computed:
                    _apply_value_to_args(
                        kwargs_dict,
                        kwarg_name,
                        _resolve_parent_value_for_child(
                            model_log,
                            parent_label,
                            label,
                            computed[parent_label],
                            parent_child_route_cache,
                        ),
                    )

            func = replay_node["func"]
            if getattr(func, "__name__", None) == "randperm" and len(args_list) >= 1:
                inferred_n = _infer_runtime_randperm_arg(model_log, label, computed)
                if inferred_n is not None:
                    args_list[0] = inferred_n

            if func is not None:
                _sync_runtime_training_flag(func, args_list, kwargs_dict, model.training)
                try:
                    output = func(*args_list, **kwargs_dict)
                except IndexError:
                    output = _maybe_retry_getitem_with_safe_indexing(func, args_list, kwargs_dict)
                    if output is None:
                        output = _maybe_retry_setitem_with_safe_indexing(
                            func, args_list, kwargs_dict
                        )
                    if output is None:
                        raise
                except RuntimeError:
                    output = _maybe_retry_split_with_runtime_sizes(func, args_list, kwargs_dict)
                    if output is None:
                        output = _maybe_retry_topk_with_runtime_k(func, args_list, kwargs_dict)
                    if output is None:
                        raise
                if output is None:
                    if getattr(layer, "func_is_inplace", False) and len(args_list) > 0:
                        output = args_list[0]
                    else:
                        output = _clone_tensors_preserve_graph(layer.activation, device=device)
                if layer.iterable_output_index is not None:
                    output = index_nested(output, layer.iterable_output_index)
            else:
                if layer.is_input_layer and label in seeded_values:
                    output = computed[label]
                elif layer.is_buffer_layer and layer.buffer_address in live_buffers_by_addr:
                    output = live_buffers_by_addr[layer.buffer_address]
                else:
                    output = _clone_tensors_preserve_graph(layer.activation, device=device)

            computed[label] = output
        if label in requested_label_set:
            requested_outputs[label] = computed[label]

        for parent_label in layer.parent_layers:
            if parent_label not in remaining_child_uses:
                continue
            remaining_child_uses[parent_label] -= 1
            if remaining_child_uses[parent_label] == 0 and parent_label not in requested_label_set:
                computed.pop(parent_label, None)

        if remaining_child_uses.get(label, 0) == 0 and label not in requested_label_set:
            computed.pop(label, None)
        execution_idx += 1

    return {
        label: requested_outputs[label]
        for label in requested_labels_ordered
        if label in requested_outputs
    }


def _ensure_save_function_args_enabled(model_log: "ModelLog") -> None:
    if getattr(model_log, "save_function_args", False) is False:
        raise ValueError(
            "Graph replay requires model_log to be created with save_function_args=True.\n"
            "Please explicitly set `save_function_args=True` when calling `log_forward_pass`."
        )


def _checkpoint_replay_subgraph_differentiable(
    model_log: "ModelLog",
    model: nn.Module,
    subgraph_labels: List[str],
    seeded_values: Dict[str, Any],
    target_labels: List[str],
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Replay a subgraph under activation checkpointing to reduce forward memory."""
    if not target_labels:
        return {}

    tensor_seed_labels, tensor_seed_values, constant_seed_values = _partition_seed_values(
        seeded_values
    )

    def _forward_fn(*tensor_inputs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        replay_seeded_values = dict(constant_seed_values)
        replay_seeded_values.update(
            {label: value for label, value in zip(tensor_seed_labels, tensor_inputs)}
        )
        replayed_outputs = _execute_replay_differentiable(
            model_log,
            model,
            subgraph_labels,
            replay_seeded_values,
            target_labels=target_labels,
            prune_to_targets=False,
            device=device,
        )
        return tuple(cast(torch.Tensor, replayed_outputs[label]) for label in target_labels)

    checkpoint_outputs = checkpoint(
        _forward_fn,
        *tensor_seed_values,
        use_reentrant=False,
    )
    if isinstance(checkpoint_outputs, torch.Tensor):
        checkpoint_outputs_tuple: Tuple[torch.Tensor, ...] = (checkpoint_outputs,)
    else:
        checkpoint_outputs_tuple = tuple(checkpoint_outputs)

    return {label: output for label, output in zip(target_labels, checkpoint_outputs_tuple)}


def replay_forward_pass(
    model_log: "ModelLog",
    new_input: Union[torch.Tensor, List[Any], Tuple[Any, ...]],
    new_input_kwargs: Optional[Dict[str, Any]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Any:
    """Replay the full computation using the saved graph structure and new input.

    Notes
    -----
    This is a value-level replay utility, not an autograd-preserving execution path.
    Saved tensors and captured arguments are cloned from the logged pass, so the
    returned tensors are suitable for forward-value validation but are not connected
    to the original model's autograd graph for training-time backward passes.
    """
    _ensure_save_function_args_enabled(model_log)
    live_output = _maybe_call_live_model_for_outputs(
        model_log,
        new_input,
        new_input_kwargs,
        preserve_graph=False,
    )
    if live_output is not None:
        return live_output

    replay_cache = _get_replay_cache(model_log)
    input_layer_values = _build_replay_input_layer_values(model_log, new_input, new_input_kwargs)
    computed_activations = _execute_replay(
        model_log,
        replay_cache["output_ancestor_labels"],
        input_layer_values,
        target_labels=replay_cache["final_output_labels"],
        device=device,
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
    """Split a DAG with a topological boundary induced by layer index.

    Requested split points that land inside dynamic atomic module regions are
    snapped to the nearest module boundary so that those regions remain intact.
    """
    total_layers = len(model_log.layer_list)
    split_indices = _normalize_split_indices(split_layer_indices, total_layers)
    max_split_idx = _snap_split_index_to_atomic_boundary(model_log, max(split_indices))
    replay_cache = _get_replay_cache(model_log)
    split_boundary_cache = replay_cache["split_boundary_cache"]

    if max_split_idx in split_boundary_cache:
        return split_boundary_cache[max_split_idx]

    subgraph1_labels = [layer.layer_label for layer in model_log.layer_list[: max_split_idx + 1]]
    subgraph2_labels = [layer.layer_label for layer in model_log.layer_list[max_split_idx + 1 :]]
    subgraph2_set = set(subgraph2_labels)

    split_point_labels = []
    split_point_label_set = set()
    for layer_label in subgraph1_labels:
        layer = model_log[layer_label]
        child_labels = getattr(layer, "child_layers", [])
        has_child_in_sg2 = any(child_label in subgraph2_set for child_label in child_labels)
        is_output = layer_label in model_log.output_layers
        if has_child_in_sg2 or is_output:
            split_point_labels.append(layer_label)
            split_point_label_set.add(layer_label)

    subgraph1_set = set(subgraph1_labels)
    for plan in _get_atomic_module_plans(model_log):
        if not any(label in subgraph2_set for label in plan["span_labels"]):
            continue
        for input_label in plan["input_labels"]:
            if input_label in subgraph1_set:
                split_point_label_set.add(input_label)

    split_point_labels = [
        layer_label for layer_label in subgraph1_labels if layer_label in split_point_label_set
    ]

    result = (subgraph1_labels, subgraph2_labels, split_point_labels)
    split_boundary_cache[max_split_idx] = result
    return result


def replay_subgraph(
    model_log: "ModelLog",
    subgraph_labels: List[str],
    input_values: Dict[str, Any],
    target_labels: Optional[List[str]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Replay a subgraph with provided input values.

    Like :func:`replay_forward_pass`, this reconstructs values only and does not
    preserve a training-time autograd graph across the replayed subgraph boundary.
    """
    return _execute_replay(
        model_log,
        subgraph_labels,
        input_values,
        target_labels=target_labels,
        device=device,
    )


def split_and_replay_graph(
    model_log: "ModelLog",
    split_layer_indices: Union[int, List[int]],
    new_input: Union[torch.Tensor, List[Any], Tuple[Any, ...]],
    new_input_kwargs: Optional[Dict[str, Any]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[Dict[str, Any], Any]:
    """Split graph at specified layers and replay both subgraphs with new input.

    The intermediate features and final outputs reproduce forward values, but the
    split replay is not currently differentiable end-to-end with respect to the
    original model parameters or the pre-split inputs.
    """
    _ensure_save_function_args_enabled(model_log)
    subgraph1_labels, subgraph2_labels, split_labels = split_graph(model_log, split_layer_indices)
    replay_cache = _get_replay_cache(model_log)
    input_layer_values = _build_replay_input_layer_values(model_log, new_input, new_input_kwargs)
    subgraph1_set = set(subgraph1_labels)
    subgraph1_needed_labels = list(
        dict.fromkeys(
            split_labels
            + [
                layer_label
                for layer_label in replay_cache["final_output_labels"]
                if layer_label in subgraph1_set
            ]
        )
    )

    subgraph1_outputs = replay_subgraph(
        model_log,
        subgraph1_labels,
        input_layer_values,
        target_labels=subgraph1_needed_labels,
        device=device,
    )
    intermediate_features = {
        label: subgraph1_outputs[label] for label in split_labels if label in subgraph1_outputs
    }
    live_output = _maybe_call_live_model_for_outputs(
        model_log,
        new_input,
        new_input_kwargs,
        preserve_graph=False,
    )
    if live_output is not None:
        return intermediate_features, live_output

    subgraph2_set = set(subgraph2_labels)
    subgraph2_output_labels = [
        layer_label
        for layer_label in replay_cache["final_output_labels"]
        if layer_label in subgraph2_set
    ]
    subgraph2_outputs = replay_subgraph(
        model_log,
        subgraph2_labels,
        intermediate_features,
        target_labels=subgraph2_output_labels,
        device=device,
    )

    combined_outputs = {**subgraph1_outputs, **subgraph2_outputs}
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


def replay_forward_pass_differentiable(
    model_log: "ModelLog",
    model: nn.Module,
    new_input: Union[torch.Tensor, List[Any], Tuple[Any, ...]],
    new_input_kwargs: Optional[Dict[str, Any]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Any:
    """Replay the full graph while preserving autograd through inputs and live params."""
    _ensure_save_function_args_enabled(model_log)
    live_output = _maybe_call_live_model_for_outputs(
        model_log,
        new_input,
        new_input_kwargs,
        model=model,
        preserve_graph=True,
    )
    if live_output is not None:
        return live_output

    replay_cache = _get_replay_cache(model_log)
    input_layer_values = _build_replay_input_layer_values(model_log, new_input, new_input_kwargs)
    computed_activations = _execute_replay_differentiable(
        model_log,
        model,
        [layer.layer_label for layer in model_log.layer_list],
        input_layer_values,
        target_labels=replay_cache["final_output_labels"],
        prune_to_targets=False,
        device=device,
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
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[Dict[str, Any], Any]:
    """Split the graph and replay with checkpointed first-half recomputation."""
    _ensure_save_function_args_enabled(model_log)
    if model.training and _has_dynamic_replay_regions(model_log):
        live_output = _call_live_model_with_replay_inputs(
            model_log,
            new_input,
            new_input_kwargs,
            model=model,
            preserve_graph=True,
        )
        if live_output is not None:
            return {}, live_output

    subgraph1_labels, subgraph2_labels, split_labels = split_graph(model_log, split_layer_indices)
    replay_cache = _get_replay_cache(model_log)
    input_layer_values = _build_replay_input_layer_values(model_log, new_input, new_input_kwargs)
    subgraph1_set = set(subgraph1_labels)
    subgraph1_needed_labels = list(
        dict.fromkeys(
            split_labels
            + [
                layer_label
                for layer_label in replay_cache["final_output_labels"]
                if layer_label in subgraph1_set
            ]
        )
    )

    subgraph1_outputs = _checkpoint_replay_subgraph_differentiable(
        model_log,
        model,
        subgraph1_labels,
        input_layer_values,
        subgraph1_needed_labels,
        device=device,
    )
    intermediate_features = {label: subgraph1_outputs[label] for label in split_labels}
    live_output = _maybe_call_live_model_for_outputs(
        model_log,
        new_input,
        new_input_kwargs,
        model=model,
        preserve_graph=True,
    )
    if live_output is not None:
        return intermediate_features, live_output

    subgraph2_set = set(subgraph2_labels)
    subgraph2_output_labels = [
        layer_label
        for layer_label in replay_cache["final_output_labels"]
        if layer_label in subgraph2_set
    ]
    subgraph2_outputs = _execute_replay_differentiable(
        model_log,
        model,
        subgraph2_labels,
        intermediate_features,
        target_labels=subgraph2_output_labels,
        prune_to_targets=False,
        device=device,
    )

    combined_outputs = {**subgraph1_outputs, **subgraph2_outputs}
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
