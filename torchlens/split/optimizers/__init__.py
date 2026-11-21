"""Device-independent optimizer state migration and native adapter registration."""

from .base import (
    OptimizerFactory,
    OptimizerState,
    OptimizerStateAdapter,
    OptimizerStateMigration,
    OptimizerStateMigrationPlan,
    OptimizerStateMigrationResult,
    OptimizerStateSnapshot,
    OptimizerStateTensor,
)
from .registry import get_optimizer_adapter, register_optimizer_adapter

__all__ = (
    "OptimizerFactory",
    "OptimizerState",
    "OptimizerStateAdapter",
    "OptimizerStateMigration",
    "OptimizerStateMigrationPlan",
    "OptimizerStateMigrationResult",
    "OptimizerStateSnapshot",
    "OptimizerStateTensor",
    "get_optimizer_adapter",
    "register_optimizer_adapter",
)
