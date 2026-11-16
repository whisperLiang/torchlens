"""Public access to the capture FLOPs authority for external Split schedulers.

The formulas and custom rules belong to the upstream capture engine. These
exports keep callers independent of its private module layout. Unknown costs
remain ``None``; backward counts are estimates rather than measured execution.
"""

from .._capture import flops as _authority
from .._capture.flops import (
    compute_backward_flops,
    compute_forward_flops,
    register_op_rule,
)

ZERO_FLOPS_OPS = frozenset(_authority.ZERO_FLOPS_OPS)


def rule_revision() -> int:
    """Return the custom-rule generation used to invalidate static cost caches."""

    return _authority._rule_revision


__all__ = (
    "ZERO_FLOPS_OPS",
    "compute_backward_flops",
    "compute_forward_flops",
    "register_op_rule",
    "rule_revision",
)
