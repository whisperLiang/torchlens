"""Lifecycle stage builders for the public-surface oracle."""

from __future__ import annotations

import pickle
import random
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from capture_oracle._models import build_model_case

import torchlens as tl

from ._snapshot import snapshot_trace_surface

_SEED = 20260812

#: Model axes covered by the surface oracle. All are deterministic builders
#: shared with the capture-unification oracle.
MODEL_AXES = (
    "plain_cnn",
    "train_batchnorm",
    "recurrent",
    "conditional",
    "in_place",
    "tiny_transformer",
)

#: Lifecycle stages snapshotted per model axis.
STAGE_NAMES = ("live", "pickle", "tlspec", "fork", "run")


def _seed_everything() -> None:
    """Seed every RNG the fixtures consume."""

    torch.manual_seed(_SEED)
    random.seed(_SEED)
    np.random.seed(_SEED % (2**32 - 1))


def build_stage_snapshots(model_axis: str) -> dict[str, Any]:
    """Capture one model and snapshot every lifecycle stage.

    Parameters
    ----------
    model_axis:
        Deterministic model-axis identifier from ``MODEL_AXES``.

    Returns
    -------
    dict[str, Any]
        Mapping of stage name to canonical surface snapshot. A stage that
        raises is recorded as ``{"__stage_raises__": <exception type>}`` so
        stage-level refusals stay part of the frozen contract.
    """

    _seed_everything()
    model, model_input = build_model_case(model_axis)

    stages: dict[str, Any] = {}
    _seed_everything()
    trace = tl.trace(model, model_input)
    stages["live"] = snapshot_trace_surface(trace)

    def _stage(name: str, build: Callable[[], Any]) -> None:
        try:
            staged_trace = build()
        except Exception as error:  # noqa: BLE001 - refusals are contract
            stages[name] = {"__stage_raises__": type(error).__name__}
            return
        stages[name] = snapshot_trace_surface(staged_trace)

    _stage("pickle", lambda: pickle.loads(pickle.dumps(trace)))

    def _tlspec_round_trip() -> Any:
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_path = Path(tmp_dir) / "surface_oracle.tlspec"
            tl.save(trace, str(bundle_path))
            return tl.load(str(bundle_path))

    _stage("tlspec", _tlspec_round_trip)
    _stage("fork", lambda: trace.fork())

    def _run_round_trip() -> Any:
        _seed_everything()
        result = trace.run(inputs=model_input)
        return result.trace

    _stage("run", _run_round_trip)
    return stages
