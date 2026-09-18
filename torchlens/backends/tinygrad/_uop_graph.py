"""Traversal, operand addressing, and structural signatures for tinygrad UOp graphs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast


def _unique_uops(outputs: Sequence[Any]) -> tuple[Any, ...]:
    """Return UOps reachable from outputs in topological order.

    Parameters
    ----------
    outputs
        tinygrad output tensors.

    Returns
    -------
    tuple[Any, ...]
        Unique UOps in first-seen topological order.
    """

    seen: set[int] = set()
    ordered: list[Any] = []

    def callable_arguments(uop: Any) -> tuple[Any, ...]:
        """Traverse call arguments, never an unbound function/kernel body."""

        src = tuple(getattr(uop, "src", ()) or ())
        return src[1:] if _uop_name(uop) in {"FUNCTION", "CALL"} else src

    for output in outputs:
        stack = [(cast(Any, output).uop, False)]
        while stack:
            uop, expanded = stack.pop()
            if id(uop) in seen:
                continue
            if expanded:
                seen.add(id(uop))
                ordered.append(uop)
            else:
                stack.append((uop, True))
                stack.extend((source, False) for source in reversed(callable_arguments(uop)))
    return tuple(ordered)


def _uop_parent_paths(
    uop: Any, labels: Mapping[int, str]
) -> tuple[tuple[int | tuple[int, ...], str], ...]:
    """Find labeled tensor operands through non-tensor UOp containers.

    Parameters
    ----------
    uop:
        Value-producing UOp whose incoming dataflow edges are being recorded.
    labels:
        Already emitted tensor UOps, including public input sources.

    Returns
    -------
    tuple
        Direct integer positions or nested source-index paths and raw labels.
    """

    parents: list[tuple[int | tuple[int, ...], str]] = []

    def visit(node: Any, path: tuple[int, ...]) -> None:
        """Stop at a labeled tensor and bypass only non-tensor containers."""

        if (label := labels.get(id(node))) is not None:
            parents.append((path[0] if len(path) == 1 else path, label))
            return
        if _uop_name(node) not in {"FUNCTION", "CALL", "TUPLE"}:
            return
        for index, source in enumerate(getattr(node, "src", ()) or ()):
            if index == 0 and _uop_name(node) in {"FUNCTION", "CALL"}:
                continue
            visit(source, (*path, index))

    for index, source in enumerate(getattr(uop, "src", ()) or ()):
        visit(source, (index,))
    return tuple(parents)


def _uop_source_at_path(uop: Any, position: int | tuple[int, ...]) -> Any:
    """Resolve a checked direct or nested UOp source address.

    Parameters
    ----------
    uop:
        Captured or replay UOp root.
    position:
        Direct source index or non-empty nested source-index path.
    """

    path = (position,) if isinstance(position, int) else position
    if not isinstance(path, tuple) or not path:
        raise ValueError(f"tinygrad parent arg position {position!r} is invalid.")
    current = uop
    for index in path:
        src = tuple(getattr(current, "src", ()) or ())
        if not isinstance(index, int) or not 0 <= index < len(src):
            raise ValueError(f"tinygrad parent arg position {position!r} is invalid.")
        current = src[index]
    return current


def _replace_uop_source_at_path(uop: Any, position: int | tuple[int, ...], replacement: Any) -> Any:
    """Replace one checked UOp operand without mutating captured graph nodes.

    Parameters
    ----------
    uop:
        Replay UOp root.
    position:
        Direct source index or nested source-index path.
    replacement:
        Runtime tensor UOp to bind at that occurrence.
    """

    _uop_source_at_path(uop, position)
    path = (position,) if isinstance(position, int) else position
    ancestors = [uop]
    for index in path[:-1]:
        ancestors.append(ancestors[-1].src[index])
    result = replacement
    for parent, index in zip(reversed(ancestors), reversed(path)):
        src = list(parent.src)
        src[index] = result
        result = parent.replace(src=tuple(src))
    return result


def _is_materializable_uop(uop: Any) -> bool:
    """Return whether a UOp can be saved as a tinygrad Tensor payload.

    Parameters
    ----------
    uop
        Candidate tinygrad UOp.

    Returns
    -------
    bool
        True when tinygrad can expose shape and host payload for ``uop``.
    """

    try:
        from tinygrad import Tensor

        tensor = Tensor(uop)
        tuple(tensor.shape)
        if getattr(tensor.dtype.base, "fmt", None) is None:
            return False
        tensor.tolist()
    except Exception:
        return False
    return True


def _uop_name(uop: Any) -> str:
    """Return a stable tinygrad UOp name.

    Parameters
    ----------
    uop
        tinygrad UOp.

    Returns
    -------
    str
        Operation name without the ``Ops.`` prefix.
    """

    op = getattr(uop, "op", None)
    return str(getattr(op, "name", op)).removeprefix("Ops.")


def _uop_signature(uop: Any) -> str:
    """Return a structural UOp signature string.

    Parameters
    ----------
    uop
        tinygrad UOp.

    Returns
    -------
    str
        Recursive operation/dtype/arg signature.
    """

    src = getattr(uop, "src", ()) or ()
    children = ",".join(_uop_signature(child) for child in src)
    return f"{_uop_name(uop)}:{getattr(uop, 'dtype', None)}:{getattr(uop, 'arg', None)}[{children}]"


def _tinygrad_signature_key(
    *,
    uop_signature: str,
    ordinal: int,
    parent_signatures: tuple[str, ...],
    shape: tuple[int, ...],
    dtype: str,
) -> tuple[Any, ...]:
    """Return the conservative key used for tinygrad T1 matching.

    Parameters
    ----------
    uop_signature
        Recursive structural UOp signature.
    ordinal
        Legacy topological ordinal retained in the internal call signature for
        compatibility. tinygrad 0.13's graph normalization introduces
        unobserved UOps, so this value is intentionally excluded from matching.
    parent_signatures
        Direct parent structural signatures.
    shape
        Tensor shape.
    dtype
        Tensor dtype string.

    Returns
    -------
    tuple[Any, ...]
        Hashable conservative match key.
    """

    del ordinal
    return (uop_signature, parent_signatures, shape, dtype)
