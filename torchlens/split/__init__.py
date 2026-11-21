"""Backend-neutral split replay runtime."""

from __future__ import annotations

from .api import prepare, prepare_graph
from .boundary import ReplayBoundary
from .boundary_layout import BoundaryLayout, BoundaryTensorLayout
from .context import PartitionAnalysis, SplitGraph
from .graph import ComputeDAG
from .graph_profile import (
    GraphProfile,
    NodeCostFeatures,
    ParameterCostFeatures,
    PartitionMetrics,
    TensorCostFeatures,
)
from .ir import (
    BackendHandle,
    BoundarySchema,
    DevicePlacement,
    GraphPartition,
    OpIR,
    PlacementPlan,
    RegionIR,
    ShapeConstraint,
    SplitFeatures,
    SplitGraphIR,
    SplitModelProfile,
    SplitRequest,
    SplitVerificationStatus,
    StateIR,
    ValueIR,
)
from .optimizers import (
    OptimizerStateAdapter,
    OptimizerStateMigration,
    OptimizerStateMigrationPlan,
    OptimizerStateMigrationResult,
    OptimizerStateSnapshot,
    OptimizerStateTensor,
)
from .ownership import ParameterOwnership, StateMigration, StateMigrationPlan
from .profiles import (
    checkpoint_cache_path,
    get_model_profile,
    iter_model_profiles,
    model_cache_dir,
    profile_cache_dir,
    register_model_profile,
    resolve_model_profile,
)
from .profiling import LatencyStats, SplitProfile
from .program import (
    CapabilityEvidence,
    CapabilityStatus,
    ReplayOp,
    ReplayProgram,
    SplitCapabilityReport,
)
from .runtime import SplitRuntime
from .shape_program import DimExpr, ShapeBinding, ShapeProgram, ShapeRecipe, TensorShapeIR

__all__ = (
    "ComputeDAG",
    "BoundarySchema",
    "BoundaryLayout",
    "BoundaryTensorLayout",
    "LatencyStats",
    "SplitProfile",
    "BackendHandle",
    "CapabilityStatus",
    "CapabilityEvidence",
    "DevicePlacement",
    "GraphPartition",
    "GraphProfile",
    "NodeCostFeatures",
    "OptimizerStateAdapter",
    "OptimizerStateMigration",
    "OptimizerStateMigrationPlan",
    "OptimizerStateMigrationResult",
    "OptimizerStateSnapshot",
    "OptimizerStateTensor",
    "ParameterCostFeatures",
    "ParameterOwnership",
    "StateMigration",
    "StateMigrationPlan",
    "PartitionMetrics",
    "TensorCostFeatures",
    "OpIR",
    "PlacementPlan",
    "ReplayOp",
    "ReplayProgram",
    "ReplayBoundary",
    "RegionIR",
    "ShapeConstraint",
    "DimExpr",
    "ShapeBinding",
    "ShapeProgram",
    "ShapeRecipe",
    "TensorShapeIR",
    "SplitCapabilityReport",
    "SplitFeatures",
    "SplitGraphIR",
    "SplitModelProfile",
    "SplitRequest",
    "SplitRuntime",
    "SplitVerificationStatus",
    "StateIR",
    "ValueIR",
    "checkpoint_cache_path",
    "get_model_profile",
    "iter_model_profiles",
    "model_cache_dir",
    "prepare",
    "prepare_graph",
    "SplitGraph",
    "PartitionAnalysis",
    "profile_cache_dir",
    "register_model_profile",
    "resolve_model_profile",
)
