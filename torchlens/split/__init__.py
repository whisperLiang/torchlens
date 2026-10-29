"""Backend-neutral split replay runtime."""

from __future__ import annotations

from .api import prepare, prepare_graph
from .boundary import ReplayBoundary
from .boundary_layout import BoundaryLayout, BoundaryTensorLayout
from .candidates import PartitionAnalysis
from .context import SplitGraph
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
from .ownership import ParameterOwnership, StateMigration, StateMigrationPlan
from .pipeline import (
    analyze_split_capabilities,
    capture_canonical_model,
    capture_model,
    execute_split_runtime,
    lower_split_program,
    normalize_to_split_ir,
)
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

__all__ = [
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
    "analyze_split_capabilities",
    "capture_canonical_model",
    "capture_model",
    "checkpoint_cache_path",
    "get_model_profile",
    "iter_model_profiles",
    "model_cache_dir",
    "execute_split_runtime",
    "lower_split_program",
    "normalize_to_split_ir",
    "prepare",
    "prepare_graph",
    "SplitGraph",
    "PartitionAnalysis",
    "profile_cache_dir",
    "register_model_profile",
    "resolve_model_profile",
]
