"""Regression tests for r18a compat/_report.py detection-heuristic tightening.

Each detection row must anchor its structural marker on real module identity /
AST references / typed attributes, mirroring ``_is_quantized_module``. These tests
kill the false-positive and false-negative detection classes flagged in round 18
(A3-02, A3-03, A3-08, A3-09, A3-10, A3-11 + DDP/DeepSpeed siblings, A3-12, LOW-8).
"""

from __future__ import annotations

import torch
from torch import nn

from torchlens.compat import report


# ---------------------------------------------------------------------------
# A3-08 — HF Transformers detection must key on real transformers namespace,
# not on the mere presence of a ``.config`` attribute.
# ---------------------------------------------------------------------------


class OrdinaryConfiguredModel(nn.Module):
    """Plain module that happens to carry an application ``config``."""

    def __init__(self) -> None:
        """Attach a non-HF config object."""

        super().__init__()
        self.config = {"application": "not-huggingface"}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the input unchanged.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            The input tensor.
        """

        return x


class TransformersNamespaceModel(nn.Module):
    """Model advertising a real ``transformers`` module namespace."""

    __module__ = "transformers.modeling_utils"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the input unchanged.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            The input tensor.
        """

        return x


def test_hf_row_ignores_plain_config_attribute() -> None:
    """A generic ``.config`` attribute must not label a module as HF Transformers."""

    row = report(OrdinaryConfiguredModel(), torch.randn(1)).row("hf_transformers")

    assert row.detected is False
    assert row.status == "pass"
    assert row.severity == "ok"


def test_hf_row_detects_real_transformers_namespace() -> None:
    """A class defined under the ``transformers`` namespace stays detected."""

    row = report(TransformersNamespaceModel(), torch.randn(1)).row("hf_transformers")

    assert row.detected is True
    assert row.status == "pass"
    assert row.severity == "info"


def test_hf_row_detects_transformers_subclass_by_mro() -> None:
    """A user subclass of a transformers-namespace base is detected via its MRO."""

    class _FakePreTrainedBase(nn.Module):
        __module__ = "transformers.modeling_utils"

    class UserModel(_FakePreTrainedBase):
        __module__ = "my_project.models"

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return input unchanged."""

            return x

    row = report(UserModel(), torch.randn(1)).row("hf_transformers")

    assert row.detected is True


# ---------------------------------------------------------------------------
# A3-11 (+ DDP/DeepSpeed siblings) — distributed-wrapper detection must key on
# real module namespaces, not on a substring of a user class name.
# ---------------------------------------------------------------------------


class _Passthrough(nn.Module):
    """Minimal passthrough module used for namespace-detection fixtures."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the input unchanged.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            The input tensor.
        """

        return x


def _named(module_path: str, class_name: str) -> nn.Module:
    """Build a passthrough instance advertising a specific module/class identity.

    Parameters
    ----------
    module_path:
        Value for the synthetic type's ``__module__``.
    class_name:
        Name for the synthetic type.

    Returns
    -------
    nn.Module
        Instance of the synthesized module class.
    """

    klass = type(class_name, (_Passthrough,), {"__module__": module_path})
    return klass()


def test_fsdp_row_ignores_user_class_merely_named_fsdp() -> None:
    """A user class named ``FsdpExportHelper`` must not trip the FSDP row."""

    row = report(_named("my_project.helpers", "FsdpExportHelper"), torch.randn(1)).row("fsdp")

    assert row.detected is False
    assert row.status == "pass"


def test_fsdp_row_detects_real_fsdp_namespace() -> None:
    """A class defined under ``torch.distributed.fsdp`` stays detected."""

    model = _named("torch.distributed.fsdp.fully_sharded_data_parallel", "FullyShardedDataParallel")
    row = report(model, torch.randn(1)).row("fsdp")

    assert row.detected is True
    assert row.status == "scope"


def test_ddp_row_ignores_user_class_merely_named_distributed() -> None:
    """A user class named like DDP but in a user module must not be detected."""

    model = _named("my_project.net", "MyDistributedDataParallelHelper")
    row = report(model, torch.randn(1)).row("distributed_data_parallel")

    assert row.detected is False


def test_ddp_row_detects_real_torch_ddp_namespace() -> None:
    """A class under ``torch.nn.parallel.distributed`` stays detected."""

    model = _named("torch.nn.parallel.distributed", "DistributedDataParallel")
    row = report(model, torch.randn(1)).row("distributed_data_parallel")

    assert row.detected is True


def test_ddp_row_does_not_fire_on_data_parallel() -> None:
    """``nn.DataParallel`` (a different namespace) must not trip the DDP row."""

    row = report(nn.DataParallel(_Passthrough()), torch.randn(1)).row("distributed_data_parallel")

    assert row.detected is False


def test_deepspeed_row_ignores_user_class_merely_named_deepspeed() -> None:
    """A user class named ``DeepspeedConfigHelper`` must not trip the DeepSpeed row."""

    model = _named("my_project.cfg", "DeepspeedConfigHelper")
    row = report(model, torch.randn(1)).row("deepspeed")

    assert row.detected is False


def test_deepspeed_row_detects_real_deepspeed_namespace() -> None:
    """A class under the ``deepspeed`` namespace stays detected."""

    model = _named("deepspeed.runtime.engine", "DeepSpeedEngine")
    row = report(model, torch.randn(1)).row("deepspeed")

    assert row.detected is True
    assert row.status == "scope"
