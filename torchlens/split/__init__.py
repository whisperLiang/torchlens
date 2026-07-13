"""Backend-neutral split replay runtime."""

from __future__ import annotations

from .api import prepare
from .boundary import ReplayBoundary
from .ir import (
    BackendHandle,
    BoundarySchema,
    ModelProfile,
    OpIR,
    RegionIR,
    ShapeConstraint,
    SplitFeatures,
    SplitGraphIR,
    SplitModelProfile,
    SplitPoint,
    SplitRequest,
    SplitVerificationStatus,
    StateIR,
    ValueIR,
    after,
    before,
    percent,
)
from .shape_program import DimExpr, ShapeBinding, ShapeProgram, ShapeRecipe, TensorShapeIR
from .program import CapabilityStatus, ReplayOp, ReplayProgram, SplitCapabilityReport
from .profiles import (
    checkpoint_cache_path,
    get_model_profile,
    iter_model_profiles,
    model_cache_dir,
    profile_cache_dir,
    register_model_profile,
    resolve_model_profile,
)
from .pipeline import (
    analyze_split_capabilities,
    capture_model,
    execute_split_runtime,
    lower_split_program,
    normalize_to_split_ir,
)
from .runtime import SplitRuntime

__all__ = [
    "BoundarySchema",
    "BackendHandle",
    "CapabilityStatus",
    "ModelProfile",
    "OpIR",
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
    "SplitPoint",
    "SplitRequest",
    "SplitRuntime",
    "SplitVerificationStatus",
    "StateIR",
    "ValueIR",
    "after",
    "analyze_split_capabilities",
    "before",
    "capture_model",
    "checkpoint_cache_path",
    "get_model_profile",
    "iter_model_profiles",
    "model_cache_dir",
    "execute_split_runtime",
    "lower_split_program",
    "normalize_to_split_ir",
    "percent",
    "prepare",
    "profile_cache_dir",
    "register_model_profile",
    "resolve_model_profile",
]
