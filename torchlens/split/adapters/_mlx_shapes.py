"""Rewrite only native MLX shape operands, preserving scalar values and axes."""

from __future__ import annotations

from typing import Any

_SHAPE_ARGUMENTS = {
    "reshape": (1, "shape"),
    "random_uniform": (2, "shape"),
    "random_normal": (0, "shape"),
    "random_bernoulli": (1, "shape"),
    "random_randint": (2, "shape"),
    "random_categorical": (2, "shape"),
    "random_truncated_normal": (2, "shape"),
    "random_gumbel": (0, "shape"),
    "random_laplace": (0, "shape"),
    "random_logistic": (0, "shape"),
    "random_multivariate_normal": (2, "shape"),
}


def mlx_shape_templates(capture: Any, node_id: str, program: Any, binding: Any) -> tuple[Any, Any]:
    """Evaluate a declared shape operand using the graph's symbolic batch recipes.

    Parameters
    ----------
    capture
        Live native call template.
    node_id
        Canonical value identifier owning this shape recipe.
    program, binding
        Optional compiled shape program and its runtime batch binding.
    """

    args, kwargs = tuple(capture.args), dict(capture.kwargs)
    shape_arg = _SHAPE_ARGUMENTS.get(capture.op_name)
    if program is None or binding is None or shape_arg is None:
        return args, kwargs
    index, name = shape_arg
    if len(args) > index:
        args = (*args[:index], program.rewrite(node_id, args[index], binding), *args[index + 1 :])
    elif name in kwargs:
        kwargs[name] = program.rewrite(node_id, kwargs[name], binding)
    return args, kwargs


__all__ = ["mlx_shape_templates"]
