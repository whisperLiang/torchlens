"""UOp-graph walks the tinygrad capture records ops from.

Extracted from ``backend`` so that module stays under its size ceiling.
:func:`_data_flow_uops` picks the UOps a capture records: those whose value
reaches an output as data.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Mapping, Sequence
from typing import Any, cast


def _uop_argument_sources(uop: Any) -> tuple[Any, ...]:
    """Return call operands, excluding an unbound function or kernel body."""

    sources = tuple(getattr(uop, "src", ()) or ())
    return sources[1:] if _uop_name(uop) in {"FUNCTION", "CALL"} else sources


def _uop_parent_paths(
    uop: Any, labels: Mapping[int, str]
) -> tuple[tuple[int | tuple[int, ...], str], ...]:
    """Find tensor operands through native function and tuple containers.

    Parameters
    ----------
    uop
        Value-producing UOp whose incoming data edges are being recorded.
    labels
        Already emitted tensor UOps, including capture inputs.

    Returns
    -------
    tuple
        Direct indices or nested operand paths, paired with raw labels.
    """

    parents: list[tuple[int | tuple[int, ...], str]] = []
    stack: list[tuple[tuple[int, ...], Any]] = [
        ((index,), source) for index, source in reversed(tuple(enumerate(uop.src)))
    ]
    while stack:
        path, source = stack.pop()
        if (label := labels.get(id(source))) is not None:
            parents.append((path[0] if len(path) == 1 else path, label))
        elif (name := _uop_name(source)) in {"FUNCTION", "CALL", "TUPLE"}:
            stack.extend(
                ((*path, index), child)
                for index, child in reversed(tuple(enumerate(source.src)))
                if index != 0 or name not in {"FUNCTION", "CALL"}
            )
    return tuple(parents)


def _uop_source_at_path(uop: Any, position: int | tuple[int, ...]) -> Any:
    """Resolve a native operand occurrence, including non-tensor call containers."""

    path = (position,) if isinstance(position, int) else position
    if not isinstance(path, tuple) or not path:
        raise ValueError(f"tinygrad parent arg position {position!r} is invalid.")
    current = uop
    for index in path:
        sources = tuple(getattr(current, "src", ()) or ())
        if not isinstance(index, int) or not 0 <= index < len(sources):
            raise ValueError(f"tinygrad parent arg position {position!r} is invalid.")
        current = sources[index]
    return current


def _replace_uop_source_at_path(uop: Any, position: int | tuple[int, ...], replacement: Any) -> Any:
    """Bind one native operand by rebuilding only its immutable ancestor path."""

    _uop_source_at_path(uop, position)
    path = (position,) if isinstance(position, int) else position
    ancestors = [uop]
    for index in path[:-1]:
        ancestors.append(ancestors[-1].src[index])
    result = replacement
    for parent, index in zip(reversed(ancestors), reversed(path), strict=True):
        sources = list(parent.src)
        sources[index] = result
        result = parent.replace(src=tuple(sources))
    return result


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
                stack.extend((source, False) for source in reversed(_uop_argument_sources(uop)))
    return tuple(ordered)


def _is_materializable_uop(uop: Any) -> bool:
    """Return whether a UOp can be saved as a tinygrad Tensor payload.

    A structural test: nothing is realized here (realize-once capture takes
    every payload from one later evaluation). The UOp must give a tinygrad
    Tensor with a shape and a dtype that has a host format. The host-only
    ``NPY`` device holds numpy-backed buffers and has no renderer: its buffers
    are readable, but a view over one (a ``RESHAPE`` before the copy to the
    compute device) has no value of its own until copied, so it is not a
    payload.

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
        device = str(tensor.device)
    except Exception:
        return False
    if getattr(tensor.dtype.base, "fmt", None) is None:
        return False
    return not device.startswith("NPY") or _uop_name(uop) == "BUFFER"


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


def _op_func_name(observed_names: Sequence[str], op_name: str) -> str:
    """The function name an op is recorded under.

    The observer reports the callable tinygrad passed to ``Tensor._apply_uop``.
    tinygrad routes most ops through internal helpers (``_mop`` for movement
    ops, ``_rop`` for reductions, a ``lambda`` for elementwise math), and those
    names do not say what the op computes. The last observed name that is not
    such a helper is used; otherwise the UOp's op name (``add``, ``reshape``,
    ``reduce``).

    Parameters
    ----------
    observed_names
        Names the observer recorded for the UOp, in call order.
    op_name
        The UOp's op name (:func:`_uop_name`).

    Returns
    -------
    str
        The recorded function name.
    """

    informative = [name for name in observed_names if not name.startswith(("_", "<"))]
    return informative[-1] if informative else op_name.lower()


def _data_flow_uops(
    uops: Sequence[Any],
    outputs: Sequence[Any],
    source_ids: Collection[int],
    *,
    boundary_ids: Collection[int] = frozenset(),
) -> tuple[Any, ...]:
    """Return the UOps whose values reach an output as data, in ``uops`` order.

    The walk starts at the outputs and follows ``src`` edges through
    materializable UOps only, stopping at the capture's sources (``source_ids``,
    the input UOps). A non-materializable UOp ends the walk: it carries no
    tensor value, so what feeds it is not data of the forward. That leaves out
    the scalar ``CONST`` elements of a movement op's shape vector and the
    storage ``BUFFER`` behind an input, which a plain topological sort reaches
    but which no recorded op consumes. The same rule decides which ops torch
    keeps: a value no output depends on and no input produces is an orphan.

    Parameters
    ----------
    uops
        UOps reachable from the outputs, in topological order (:func:`_unique_uops`).
    outputs
        tinygrad output tensors.
    source_ids
        ``id()`` of every source UOp (the capture's inputs).
    boundary_ids
        ``id()`` of UOps recorded whole: kept when reached, but the walk does
        not continue into what produced them (a pre-forward random weight's
        generator subgraph, see ``_realize_once``).

    Returns
    -------
    tuple[Any, ...]
        The data-flow UOps, sources excluded, each materializable.
    """

    materializable: dict[int, bool] = {}
    kept: set[int] = set()
    stack = [cast(Any, output).uop for output in outputs]
    while stack:
        uop = stack.pop()
        key = id(uop)
        if key in kept or key in source_ids:
            continue
        if key not in materializable:
            materializable[key] = _is_materializable_uop(uop)
        if not materializable[key]:
            if _uop_name(uop) in {"FUNCTION", "CALL", "TUPLE"}:
                stack.extend(_uop_argument_sources(uop))
            continue
        kept.add(key)
        if key not in boundary_ids:
            stack.extend(_uop_argument_sources(uop))
    return tuple(uop for uop in uops if id(uop) in kept)


def _uop_signature(uop: Any, memo: dict[int, str] | None = None) -> str:
    """Return a bounded structural UOp digest.

    Parameters
    ----------
    uop
        tinygrad UOp.
    memo
        Identity-keyed cache shared across one graph's signature calculations.

    Returns
    -------
    str
        Digest of the operation, dtype, argument, and ordered child digests.
    """

    signatures = {} if memo is None else memo
    stack = [(uop, False)]
    while stack:
        node, expanded = stack.pop()
        key = id(node)
        if key in signatures:
            continue
        src = tuple(getattr(node, "src", ()) or ())
        if not expanded:
            stack.append((node, True))
            stack.extend((child, False) for child in reversed(src))
            continue
        fields = (
            _uop_name(node),
            str(getattr(node, "dtype", None)),
            str(getattr(node, "arg", None)),
            tuple(signatures[id(child)] for child in src),
        )
        encoded = json.dumps(fields, ensure_ascii=True, separators=(",", ":")).encode("ascii")
        signatures[key] = "sha256:" + hashlib.sha256(encoded).hexdigest()
    return signatures[id(uop)]
