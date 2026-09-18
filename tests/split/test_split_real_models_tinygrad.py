"""Opt-in tinygrad real-world split replay tests."""

from __future__ import annotations

import pytest
import torch
from real_model_helpers import (
    _run_backend_subprocess,
    _skip_if_module_missing,
    _skip_unless_enabled,
)

pytestmark = [pytest.mark.slow, pytest.mark.real_model]


def test_tinygrad_conv_and_transformerish_models_split_replay() -> None:
    """tinygrad Conv and embedding/LayerNorm models replay representative splits."""

    _skip_unless_enabled()
    _skip_if_module_missing("tinygrad")
    _run_backend_subprocess(
        """
        from tinygrad import Tensor, dtypes
        import tinygrad.nn as nn
        import os
        from pathlib import Path

        import torchlens as tl

        def flatten(value):
            if isinstance(value, list):
                out = []
                for item in value:
                    out.extend(flatten(item))
                return out
            return [float(value)]

        def max_abs_diff(left, right):
            pairs = zip(flatten(left.tolist()), flatten(right.tolist()), strict=True)
            return max(abs(a - b) for a, b in pairs)

        devices = ["CPU"]
        if os.environ.get("TORCHLENS_TINYGRAD_GPU") == "1":
            for candidate in ("CUDA", "GPU", "NV"):
                if candidate in ("CUDA", "NV") and not Path("/dev/nvidia0").exists():
                    continue
                if candidate == "GPU" and not any(Path("/dev/dri").glob("renderD*")):
                    continue
                try:
                    Tensor.ones(1, device=candidate).mul(2).realize()
                except Exception:
                    continue
                devices.append(candidate)
                break

        for device in devices:
            conv1 = nn.Conv2d(3, 4, 3, padding=1)
            conv2 = nn.Conv2d(4, 4, 3, padding=1)
            head = nn.Linear(4 * 8 * 8, 5)
            for obj in (conv1, conv2, head):
                for name in dir(obj):
                    param = getattr(obj, name)
                    if isinstance(param, Tensor):
                        param.to_(device).realize()

            def conv_model(x):
                hidden = conv1(x).relu()
                residual = hidden
                hidden = conv2(hidden).relu() + residual
                hidden = hidden.avg_pool2d(kernel_size=(2, 2))
                return head(hidden.flatten(1))

            x = Tensor.ones(2, 3, 16, 16, device=device).realize()
            for boundary in ("25%", "50%", "75%"):
                runtime = tl.split.prepare(
                    conv_model,
                    x,
                    split_request(boundary, backend="tinygrad"),
                )
                replay_x = Tensor.ones(2, 3, 16, 16, device=device).realize()
                diff = max_abs_diff(
                    runtime.replay(replay_x).realize(),
                    conv_model(replay_x).realize(),
                )
                assert diff < 1e-4

        embed = nn.Embedding(16, 8)
        norm = nn.LayerNorm(8)
        fc1 = nn.Linear(8, 16)
        fc2 = nn.Linear(16, 8)
        head = nn.Linear(8, 4)
        for obj in (embed, norm, fc1, fc2, head):
            for name in dir(obj):
                param = getattr(obj, name)
                if isinstance(param, Tensor):
                    param.realize()

        def token_model(tokens):
            embedded = embed(tokens)
            hidden = norm(embedded)
            hidden = fc2(fc1(hidden).relu()) + embedded
            return head(hidden).mean(axis=1)

        tokens = Tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=dtypes.int32).realize()
        for boundary in ("25%", "50%", "75%"):
            runtime = tl.split.prepare(
                token_model,
                tokens,
                split_request(boundary, backend="tinygrad"),
            )
            replay_tokens = Tensor(
                [[1, 2, 3, 4], [4, 3, 2, 1]],
                dtype=dtypes.int32,
            ).realize()
            diff = max_abs_diff(
                runtime.replay(replay_tokens).realize(),
                token_model(replay_tokens).realize(),
            )
            assert diff < 1e-4
        """,
        timeout=360,
        env_overrides={
            "DEV": "CPU",
            "TORCHLENS_TINYGRAD_GPU": "1" if torch.cuda.is_available() else "0",
        },
    )


def test_tinygrad_conv_split_replay_batch_symbolic_and_cpu_gpu() -> None:
    """tinygrad Conv replay covers dynamic batches and CPU-prefix/GPU-suffix."""

    _skip_unless_enabled()
    _skip_if_module_missing("tinygrad")
    _run_backend_subprocess(
        """
        from tinygrad import Tensor
        import tinygrad.nn as nn
        import os
        from pathlib import Path

        import torchlens as tl

        def flatten(value):
            if isinstance(value, list):
                out = []
                for item in value:
                    out.extend(flatten(item))
                return out
            return [float(value)]

        def max_abs_diff(left, right):
            pairs = zip(flatten(left.tolist()), flatten(right.tolist()), strict=True)
            return max(abs(a - b) for a, b in pairs)

        def move_parameters(model, device):
            for name in dir(model):
                value = getattr(model, name)
                if isinstance(value, Tensor):
                    value.to_(device).realize()

        def make_model(device="CPU"):
            class Model:
                def __init__(self):
                    self.conv1 = nn.Conv2d(3, 4, 3, padding=1)
                    self.conv2 = nn.Conv2d(4, 4, 3, padding=1)
                    self.head = nn.Linear(4 * 8 * 8, 5)

                def __call__(self, x):
                    hidden = self.conv1(x).relu()
                    residual = hidden
                    hidden = self.conv2(hidden).relu() + residual
                    hidden = hidden.avg_pool2d(kernel_size=(2, 2))
                    return self.head(hidden.flatten(1)).relu()

            model = Model()
            modules = (model.conv1, model.conv2, model.head)
            for module in modules:
                move_parameters(module, device)

            return model, modules

        model, _modules = make_model("CPU")
        trace_x = Tensor.ones(2, 3, 16, 16, device="CPU").realize()
        for boundary in ("25%", "50%", "75%"):
            runtime = tl.split.prepare(
                model,
                trace_x,
                split_request(boundary, backend="tinygrad"),
            )
            for batch in (1, 2, 4):
                replay_x = Tensor.ones(batch, 3, 16, 16, device="CPU").realize()
                diff = max_abs_diff(
                    runtime.replay(replay_x).realize(),
                    model(replay_x).realize(),
                )
                assert diff < 1e-4

        suffix_device = None
        for candidate in ("CUDA", "GPU", "NV"):
            if candidate in ("CUDA", "NV") and not Path("/dev/nvidia0").exists():
                continue
            if candidate == "GPU" and not any(Path("/dev/dri").glob("renderD*")):
                continue
            try:
                Tensor.ones(1, device=candidate).mul(2).realize()
            except Exception:
                continue
            suffix_device = candidate
            break
        if suffix_device is None:
            raise SystemExit(75)

        model, _modules = make_model("CPU")
        runtime = tl.split.prepare(
            model,
            trace_x,
            split_request("90%", backend="tinygrad"),
        )
        for batch in (1, 3):
            cpu_x = Tensor.ones(batch, 3, 16, 16, device="CPU").realize()
            reference = model(cpu_x).realize()
            boundary = runtime.run_prefix(cpu_x)
            gpu_boundary = boundary.to(suffix_device, adapter=runtime.adapter)
            for key, cpu_value in boundary.tensors.items():
                moved_diff = max_abs_diff(gpu_boundary.tensors[key], cpu_value)
                assert moved_diff < 1e-4, (key, moved_diff)
            gpu_output = runtime.run_suffix(gpu_boundary).realize()
            diff = max_abs_diff(gpu_output, reference)
            assert diff < 1e-4, (
                batch,
                suffix_device,
                diff,
                gpu_output.tolist(),
                reference.tolist(),
            )

        os._exit(0)
        """,
        timeout=240,
    )


def test_tinygrad_official_llm_transformer_block_split_replay() -> None:
    """tinygrad's packaged LLM TransformerBlock replays representative fixed-batch splits."""

    _skip_unless_enabled()
    _skip_if_module_missing("tinygrad")
    _run_backend_subprocess(
        """
        from tinygrad import Tensor
        from tinygrad.llm.model import TransformerBlock, TransformerConfig
        import tinygrad.nn.state as state

        import torchlens as tl

        def flatten(value):
            if isinstance(value, list):
                out = []
                for item in value:
                    out.extend(flatten(item))
                return out
            return [float(value)]

        def max_abs_diff(left, right):
            pairs = zip(flatten(left.tolist()), flatten(right.tolist()), strict=True)
            return max(abs(a - b) for a, b in pairs)

        config = TransformerConfig(
            num_blocks=1,
            dim=8,
            hidden_dim=16,
            n_heads=2,
            n_kv_heads=2,
            norm_eps=1e-5,
            vocab_size=32,
            head_dim=4,
            rope_theta=10000.0,
            rope_dim=4,
            v_head_dim=4,
            max_context=8,
        )
        block = TransformerBlock(config)
        for param in state.get_parameters(block):
            param.realize()

        def model(x):
            return block(x, 0)

        x = Tensor.ones(2, 4, 8).realize()
        for boundary in ("25%", "50%", "75%"):
            # This fixture exercises a fixed-batch, stateful KV-cache block.
            # Explicitly retain its B=2 input contract instead of requesting
            # automatic B=1 canonicalization and batch extrapolation.
            runtime = tl.split.prepare(
                model, x, split_request(boundary, backend="tinygrad", batch_axes={})
            )
            replay_x = (Tensor.arange(64).reshape(2, 4, 8).float() / 64).realize()
            diff = max_abs_diff(
                runtime.replay(replay_x).realize(),
                model(replay_x).realize(),
            )
            assert diff < 1e-4
        """,
        timeout=180,
        env_overrides={"DEV": "CPU"},
    )


def test_tinygrad_complete_multiscale_detector_all_nodes_cross_batch_and_device() -> None:
    """A complete tinygrad multiscale detector replays every UOp boundary."""

    _skip_unless_enabled()
    _skip_if_module_missing("tinygrad")
    _run_backend_subprocess(
        """
        import os
        from pathlib import Path

        from tinygrad import Tensor
        import tinygrad.nn as nn
        import torchlens as tl
        from torchlens.split import after, before

        class CompleteTinygradDetector:
            def __init__(self, device):
                self.stem = nn.Conv2d(3, 4, 1)
                self.modules = (self.stem,)
                self.to(device)

            def to(self, device):
                for module in self.modules:
                    for name in dir(module):
                        value = getattr(module, name)
                        if isinstance(value, Tensor):
                            value.to_(device).realize()

            def __call__(self, inputs):
                p1 = self.stem(inputs).relu()
                p2 = p1.avg_pool2d(kernel_size=(2, 2), stride=(2, 2)).relu()
                p3 = p2.avg_pool2d(kernel_size=(2, 2), stride=(2, 2)).relu()
                p1_down = p1.avg_pool2d(kernel_size=(2, 2), stride=(2, 2))
                n2 = (p2 + p1_down).relu()
                n2_down = n2.avg_pool2d(kernel_size=(2, 2), stride=(2, 2))
                n3 = p3.cat(n2_down, dim=1).relu()

                def flatten_scale(value):
                    return value.permute(0, 2, 3, 1).reshape(inputs.shape[0], -1, 8)

                head1 = p1.cat(p1, dim=1)
                head2 = n2
                head3 = n3
                predictions = flatten_scale(head1)
                predictions = predictions.cat(flatten_scale(head2), dim=1)
                predictions = predictions.cat(flatten_scale(head3), dim=1)
                return predictions[:, :, :4], predictions[:, :, 4:].sigmoid()

        def leaves(value):
            return value if isinstance(value, tuple) else (value,)

        def assert_same(actual, expected):
            for actual_leaf, expected_leaf in zip(leaves(actual), leaves(expected), strict=True):
                assert actual_leaf.shape == expected_leaf.shape
                actual_values = actual_leaf.realize().tolist()
                expected_values = expected_leaf.realize().tolist()

                def flatten(value):
                    if isinstance(value, list):
                        result = []
                        for item in value:
                            result.extend(flatten(item))
                        return result
                    return [float(value)]

                assert max(
                    abs(left - right)
                    for left, right in zip(flatten(actual_values), flatten(expected_values), strict=True)
                ) < 1e-4

        model = CompleteTinygradDetector("CPU")
        trace_x = Tensor.ones(1, 3, 4, 4, device="CPU").realize()
        seed_runtime = tl.split.prepare(
            model,
            trace_x,
            split_request("50%", backend="tinygrad"),
        )
        boundaries = [
            (kind, node.canonical_id)
            for node in seed_runtime.trace_graph.compute_nodes
            for kind in ("before", "after")
        ]
        assert boundaries
        expected_trace = model(trace_x)
        expected_trace = tuple(value.realize() for value in leaves(expected_trace))
        for kind, node_id in boundaries:
            point = after(node_id) if kind == "after" else before(node_id)
            runtime = seed_runtime.at(point)
            assert_same(runtime.replay(trace_x), expected_trace)

        for batch in (2, 3):
            replay_x = Tensor.ones(batch, 3, 4, 4, device="CPU").realize()
            runtime = seed_runtime.at(
                split_request("50%", backend="tinygrad").point
            )
            expected = model(replay_x)
            expected = tuple(value.realize() for value in leaves(expected))
            assert_same(runtime.replay(replay_x), expected)

        suffix_device = None
        for candidate in ("CUDA", "GPU", "NV"):
            if candidate in ("CUDA", "NV") and not Path("/dev/nvidia0").exists():
                continue
            if candidate == "GPU" and not any(Path("/dev/dri").glob("renderD*")):
                continue
            try:
                Tensor.ones(1, device=candidate).mul(2).realize()
            except Exception:
                continue
            suffix_device = candidate
            break
        if suffix_device is None:
            raise SystemExit(75)

        model = CompleteTinygradDetector("CPU")
        runtime = tl.split.prepare(
            model,
            trace_x,
            split_request("50%", backend="tinygrad"),
        )
        for batch in (1, 2):
            cpu_x = Tensor.ones(batch, 3, 4, 4, device="CPU").realize()
            boundary = runtime.run_prefix(cpu_x)
            model.to(suffix_device)
            boundary_device = boundary.to(suffix_device, adapter=runtime.adapter)
            device_x = Tensor.ones(batch, 3, 4, 4, device=suffix_device).realize()
            expected = model(device_x)
            expected = tuple(value.realize() for value in leaves(expected))
            assert_same(runtime.run_suffix(boundary_device), expected)
        os._exit(0)
        """,
        timeout=900,
        env_overrides={
            "DEV": "CPU",
            "TORCHLENS_TINYGRAD_GPU": "1" if torch.cuda.is_available() else "0",
        },
    )


def test_tinygrad_nn_mlp_split_replay_batch_symbolic_and_train() -> None:
    """tinygrad nn.Linear MLP exercises real layer params and split training."""

    _skip_unless_enabled()
    _skip_if_module_missing("tinygrad")
    _run_backend_subprocess(
        """
        from tinygrad import Tensor
        import tinygrad.nn as nn
        import os
        from pathlib import Path

        import torchlens as tl

        fc1 = nn.Linear(6, 8)
        fc2 = nn.Linear(8, 3)
        for param in (fc1.weight, fc1.bias, fc2.weight, fc2.bias):
            if param is not None:
                param.realize()

        def model(x):
            x = x.reshape(x.shape[0], -1)
            return fc2(fc1(x).relu())

        def flatten(value):
            if isinstance(value, list):
                out = []
                for item in value:
                    out.extend(flatten(item))
                return out
            return [float(value)]

        def max_abs_diff(left, right):
            pairs = zip(flatten(left.tolist()), flatten(right.tolist()), strict=True)
            return max(abs(a - b) for a, b in pairs)

        x = Tensor.ones(2, 2, 3).realize()
        runtime = tl.split.prepare(
            model,
            x,
            split_request("25%", backend="tinygrad"),
        )
        for batch in (1, 2, 4):
            replay_x = Tensor.ones(batch, 2, 3).realize()
            diff = max_abs_diff(runtime.replay(replay_x).realize(), model(replay_x).realize())
            assert diff < 1e-5

        train_x = Tensor.ones(2, 2, 3).realize()
        train_x.requires_grad = True
        train_runtime = tl.split.prepare(
            model,
            train_x,
            split_request("25%", backend="tinygrad", trainable=True),
        )
        boundary = train_runtime.run_training_prefix(train_x)
        loss, grads = train_runtime.train_suffix(boundary, Tensor.zeros(2, 3).realize())
        assert float(loss.realize().tolist()) >= 0.0
        assert grads
        train_runtime.backward_prefix(boundary, grads)
        assert train_x.grad is not None
        """,
        env_overrides={"DEV": "CPU"},
    )


def test_tinygrad_nn_mlp_split_train_cpu_prefix_gpu_suffix_cross_batch() -> None:
    """tinygrad split training runs CPU prefix and GPU suffix for multiple batches."""

    _skip_unless_enabled()
    _skip_if_module_missing("tinygrad")
    _run_backend_subprocess(
        """
        from tinygrad import Tensor
        import tinygrad.nn as nn
        import os
        from pathlib import Path

        import torchlens as tl

        suffix_device = None
        for candidate in ("CUDA", "GPU", "NV"):
            if candidate in ("CUDA", "NV") and not Path("/dev/nvidia0").exists():
                continue
            if candidate == "GPU" and not any(Path("/dev/dri").glob("renderD*")):
                continue
            try:
                Tensor.ones(1, device=candidate).mul(2).realize()
            except Exception:
                continue
            suffix_device = candidate
            break
        if suffix_device is None:
            print("No tinygrad GPU backend available for CPU-prefix/GPU-suffix split training.")
            raise SystemExit(75)

        class TinygradMlp:
            def __init__(self):
                self.fc1 = nn.Linear(4, 5)
                self.fc2 = nn.Linear(5, 3)
                self.fc1.weight = Tensor.ones(5, 4, device="CPU").realize()
                self.fc1.weight.requires_grad = True
                self.fc1.bias = Tensor.zeros(5, device="CPU").realize()
                self.fc1.bias.requires_grad = True
                self.fc2.weight = Tensor.ones(3, 5, device="CPU").realize()
                self.fc2.weight.requires_grad = True
                self.fc2.bias = Tensor.zeros(3, device="CPU").realize()
                self.fc2.bias.requires_grad = True

            def __call__(self, x):
                return self.fc2(self.fc1(x))

        def primary_tensors(boundary):
            return [
                value
                for key, value in boundary.tensors.items()
                if boundary.spec[key].role == "primary"
            ]

        for batch in (1, 3):
            model = TinygradMlp()
            trace_x = Tensor.ones(2, 4, device="CPU").realize()
            trace_x.requires_grad = True
            seed_runtime = tl.split.prepare(
                model,
                trace_x,
                split_request("50%", backend="tinygrad", trainable=True),
            )
            split_id = next(
                node.canonical_id
                for node in seed_runtime.trace_graph.compute_nodes
                if node.op_type == "add" and node.module_path == "fc1"
            )
            runtime = tl.split.prepare(
                model,
                trace_x,
                split_request(
                    f"after:{split_id}",
                    backend="tinygrad",
                    trainable=True,
                ),
            )

            train_x = Tensor.ones(batch, 4, device="CPU").realize()
            train_x.requires_grad = True
            boundary_cpu = runtime.run_training_prefix(train_x)
            for value in primary_tensors(boundary_cpu):
                assert value.device == "CPU"

            model.fc2.weight.to_(suffix_device).realize()
            model.fc2.bias.to_(suffix_device).realize()
            boundary_gpu = boundary_cpu.to(suffix_device, adapter=runtime.adapter)
            for value in primary_tensors(boundary_gpu):
                assert value.device == suffix_device

            targets = Tensor.zeros(batch, 3, device=suffix_device).realize()
            loss, grads = runtime.train_suffix(boundary_gpu, targets)
            assert loss.realize().device == suffix_device
            assert grads

            model.fc2.weight.to_("CPU").realize()
            model.fc2.bias.to_("CPU").realize()
            prefix_result = runtime.backward_prefix(boundary_cpu, grads)
            assert prefix_result
            assert train_x.grad is not None
            assert train_x.grad.device == "CPU"
        os._exit(0)
        """,
        timeout=240,
        env_overrides={"DEV": "CPU"},
    )


def test_tinygrad_nn_mlp_all_split_nodes_cross_batch_and_device() -> None:
    """Every compute-node before/after split replays for a tinygrad nn MLP."""

    _skip_unless_enabled()
    _skip_if_module_missing("tinygrad")
    _run_backend_subprocess(
        """
        from tinygrad import Tensor
        import tinygrad.nn as nn
        import os
        from pathlib import Path

        import torchlens as tl

        def flatten(value):
            if isinstance(value, list):
                out = []
                for item in value:
                    out.extend(flatten(item))
                return out
            return [float(value)]

        def max_abs_diff(left, right):
            pairs = zip(flatten(left.tolist()), flatten(right.tolist()), strict=True)
            return max(abs(a - b) for a, b in pairs)

        suffix_device = None
        for candidate in ("CUDA", "GPU", "NV"):
            if candidate in ("CUDA", "NV") and not Path("/dev/nvidia0").exists():
                continue
            if candidate == "GPU" and not any(Path("/dev/dri").glob("renderD*")):
                continue
            try:
                Tensor.ones(1, device=candidate).mul(2).realize()
            except Exception:
                continue
            suffix_device = candidate
            break
        if suffix_device is None:
            print("No tinygrad GPU backend available for CPU-prefix/GPU-suffix all-node replay.")
            raise SystemExit(75)

        def make_model():
            fc1 = nn.Linear(6, 8)
            fc2 = nn.Linear(8, 4)
            fc1.weight = Tensor.ones(8, 6, device="CPU").realize()
            fc1.bias = Tensor.zeros(8, device="CPU").realize()
            fc2.weight = Tensor.ones(4, 8, device="CPU").realize()
            fc2.bias = Tensor.zeros(4, device="CPU").realize()

            def move_params(device):
                fc1.weight = fc1.weight.to(device).realize()
                fc1.bias = fc1.bias.to(device).realize()
                fc2.weight = fc2.weight.to(device).realize()
                fc2.bias = fc2.bias.to(device).realize()

            def model(x):
                return fc2(fc1(x).relu())

            return model, move_params

        model, move_params = make_model()
        move_params("CPU")
        x = Tensor.ones(2, 6, device="CPU").realize()
        seed_runtime = tl.split.prepare(
            model,
            x,
            split_request("50%", backend="tinygrad"),
        )
        op_types = {"reduce", "add", "where"}
        candidate_count = len(
            [node for node in seed_runtime.trace_graph.compute_nodes if node.op_type in op_types]
        )
        for candidate_index in range(candidate_count):
            for kind in ("before", "after"):
                model, move_params = make_model()
                move_params("CPU")
                x = Tensor.ones(2, 6, device="CPU").realize()
                seed_runtime = tl.split.prepare(
                    model,
                    x,
                    split_request("50%", backend="tinygrad"),
                )
                candidates = [
                    node
                    for node in seed_runtime.trace_graph.compute_nodes
                    if node.op_type in op_types
                ]
                target = candidates[candidate_index]
                runtime = tl.split.prepare(
                    model,
                    x,
                    split_request(
                        f"{kind}:{target.canonical_id}",
                        backend="tinygrad",
                    ),
                )
                for batch in (1, 3):
                    move_params("CPU")
                    replay_x_cpu = Tensor.ones(batch, 6, device="CPU").realize()
                    boundary_cpu = runtime.run_prefix(replay_x_cpu)
                    for value in boundary_cpu.tensors.values():
                        assert value.device == "CPU"

                    move_params(suffix_device)
                    boundary_gpu = boundary_cpu.to(suffix_device, adapter=runtime.adapter)
                    for value in boundary_gpu.tensors.values():
                        assert value.device == suffix_device
                    replay_x_gpu = Tensor.ones(batch, 6, device=suffix_device).realize()
                    diff = max_abs_diff(
                        runtime.run_suffix(boundary_gpu).realize(),
                        model(replay_x_gpu).realize(),
                    )
                    assert diff < 1e-5
        os._exit(0)
        """,
        timeout=300,
        env_overrides={"DEV": "CPU"},
    )
