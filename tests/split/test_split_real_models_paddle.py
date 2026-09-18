"""Opt-in Paddle real-world split replay tests."""

from __future__ import annotations

import pytest
from real_model_helpers import (
    _run_backend_subprocess,
    _skip_if_module_missing,
    _skip_unless_enabled,
)

pytestmark = [pytest.mark.slow, pytest.mark.real_model]


def test_paddle_layer_model_split_replay_batch_symbolic_and_train() -> None:
    """Paddle Layer stack exercises preview split replay and training on real modules."""

    _skip_unless_enabled()
    _skip_if_module_missing("paddle")
    _run_backend_subprocess(
        """
        import paddle
        import torchlens as tl

        class PaddleMLP(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.flatten = paddle.nn.Flatten()
                self.fc1 = paddle.nn.Linear(6, 8)
                self.relu = paddle.nn.ReLU()
                self.fc2 = paddle.nn.Linear(8, 3)

            def forward(self, x):
                return self.fc2(self.relu(self.fc1(self.flatten(x))))

        paddle.seed(0)
        model = PaddleMLP()
        model.eval()
        x = paddle.ones([2, 2, 3], dtype="float32")
        runtime = tl.split.prepare(
            model,
            x,
            split_request("25%", backend="paddle"),
        )
        for batch in (1, 2, 4):
            replay_x = paddle.ones([batch, 2, 3], dtype="float32")
            diff = paddle.max(paddle.abs(runtime.replay(replay_x) - model(replay_x)))
            assert float(diff.item()) < 1e-5

        model.train()
        train_runtime = tl.split.prepare(
            model,
            x,
            split_request("25%", backend="paddle", trainable=True),
        )
        boundary = train_runtime.run_training_prefix(x)
        loss, grads = train_runtime.train_suffix(
            boundary,
            paddle.zeros([2, 3], dtype="float32"),
        )
        assert float(loss.item()) >= 0.0
        assert grads
        assert train_runtime.backward_prefix(boundary, grads)
        """,
    )


def test_paddle_split_train_cpu_prefix_gpu_suffix_cross_batch() -> None:
    """Paddle split training runs CPU prefix and GPU suffix for multiple batches."""

    _skip_unless_enabled()
    _skip_if_module_missing("paddle")
    _run_backend_subprocess(
        """
        import paddle
        import torchlens as tl

        if paddle.device.cuda.device_count() == 0:
            print("No Paddle GPU available for CPU-prefix/GPU-suffix split training.")
            raise SystemExit(75)

        class PaddleTrainMlp(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.fc1 = paddle.nn.Linear(4, 5)
                self.relu = paddle.nn.ReLU()
                self.fc2 = paddle.nn.Linear(5, 3)

            def forward(self, x):
                return self.fc2(self.relu(self.fc1(x)))

        def assert_place(tensor, token):
            assert token in str(tensor.place).lower(), tensor.place

        paddle.seed(123)
        paddle.set_device("cpu")
        model = PaddleTrainMlp()
        trace_x = paddle.randn([2, 4], dtype="float32")
        seed_runtime = tl.split.prepare(
            model,
            trace_x,
            split_request("50%", backend="paddle", trainable=True),
        )
        relu_id = next(
            node.canonical_id
            for node in seed_runtime.trace_graph.compute_nodes
            if "relu" in node.op_type
        )
        runtime = tl.split.prepare(
            model,
            trace_x,
            split_request(
                f"after:{relu_id}",
                backend="paddle",
                trainable=True,
            ),
        )

        for batch in (1, 3):
            paddle.set_device("cpu")
            train_x = paddle.randn([batch, 4], dtype="float32")
            train_x.stop_gradient = False
            boundary_cpu = runtime.run_training_prefix(train_x)
            for value in boundary_cpu.tensors.values():
                assert_place(value, "cpu")

            model.fc2.to("gpu:0")
            paddle.set_device("gpu:0")
            boundary_gpu = boundary_cpu.to("gpu:0", adapter=runtime.adapter)
            for value in boundary_gpu.tensors.values():
                assert_place(value, "gpu")

            targets = paddle.zeros([batch, 3], dtype="float32")
            loss, grads_gpu = runtime.train_suffix(boundary_gpu, targets)
            assert_place(loss, "gpu")
            assert grads_gpu
            for grad in grads_gpu.values():
                assert_place(grad, "gpu")

            grads_cpu = {key: grad.cpu() for key, grad in grads_gpu.items()}
            model.fc2.to("cpu")
            paddle.set_device("cpu")
            prefix_result = runtime.backward_prefix(boundary_cpu, grads_cpu)
            assert prefix_result
            assert train_x.grad is not None
        """,
        timeout=240,
    )


def test_paddle_mlp_all_split_nodes_cross_batch_and_device() -> None:
    """Every compute-node before/after split replays for a Paddle Layer MLP."""

    _skip_unless_enabled()
    _skip_if_module_missing("paddle")
    _run_backend_subprocess(
        """
        import paddle
        import torchlens as tl

        class PaddleMLP(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.fc1 = paddle.nn.Linear(6, 8)
                self.relu = paddle.nn.ReLU()
                self.fc2 = paddle.nn.Linear(8, 4)

            def forward(self, x):
                return self.fc2(self.relu(self.fc1(x)))

        def assert_place(tensor, token):
            assert token in str(tensor.place).lower(), tensor.place

        if paddle.device.cuda.device_count() == 0:
            print("No Paddle GPU available for CPU-prefix/GPU-suffix all-node replay.")
            raise SystemExit(75)

        paddle.seed(0)
        paddle.set_device("cpu")
        model = PaddleMLP()
        model.eval()
        x = paddle.ones([2, 6], dtype="float32")
        seed_runtime = tl.split.prepare(
            model,
            x,
            split_request("50%", backend="paddle"),
        )
        boundaries = [
            f"{kind}:{node.canonical_id}"
            for node in seed_runtime.trace_graph.compute_nodes
            for kind in ("before", "after")
        ]
        for boundary in boundaries:
            paddle.set_device("cpu")
            model.to("cpu")
            runtime = tl.split.prepare(
                model,
                x,
                split_request(boundary, backend="paddle"),
            )
            for batch in (1, 3):
                paddle.set_device("cpu")
                model.to("cpu")
                replay_x_cpu = paddle.ones([batch, 6], dtype="float32")
                boundary_cpu = runtime.run_prefix(replay_x_cpu)
                for value in boundary_cpu.tensors.values():
                    assert_place(value, "cpu")

                model.to("gpu:0")
                paddle.set_device("gpu:0")
                boundary_gpu = boundary_cpu.to("gpu:0", adapter=runtime.adapter)
                for value in boundary_gpu.tensors.values():
                    assert_place(value, "gpu")
                replay_x_gpu = paddle.ones([batch, 6], dtype="float32")
                split_output = runtime.run_suffix(boundary_gpu)
                full_output = model(replay_x_gpu)
                assert_place(split_output, "gpu")
                diff = paddle.max(paddle.abs(split_output - full_output))
                assert float(diff.item()) < 1e-5
        """,
        timeout=240,
    )


def test_paddle_transformer_block_split_replay_batch_symbolic_and_device() -> None:
    """A Paddle attention/FFN block replays representative splits across batch/device."""

    _skip_unless_enabled()
    _skip_if_module_missing("paddle")
    _run_backend_subprocess(
        """
        import paddle
        import paddle.nn.functional as F
        import torchlens as tl

        class PaddleTransformerBlock(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.q = paddle.nn.Linear(8, 8)
                self.k = paddle.nn.Linear(8, 8)
                self.v = paddle.nn.Linear(8, 8)
                self.o = paddle.nn.Linear(8, 8)
                self.ff1 = paddle.nn.Linear(8, 16)
                self.ff2 = paddle.nn.Linear(16, 8)

            def forward(self, x):
                query = self.q(x)
                key = self.k(x)
                value = self.v(x)
                scores = paddle.matmul(query, key, transpose_y=True) / (8 ** 0.5)
                attn = F.softmax(scores, axis=-1)
                hidden = self.o(paddle.matmul(attn, value)) + x
                return self.ff2(F.relu(self.ff1(hidden))) + hidden

        paddle.seed(0)
        devices = ["cpu"]
        if paddle.device.cuda.device_count() > 0:
            devices.append("gpu:0")

        for device in devices:
            paddle.set_device(device)
            model = PaddleTransformerBlock()
            model.eval()
            x = paddle.ones([2, 4, 8], dtype="float32")
            for boundary in ("25%", "50%", "75%"):
                runtime = tl.split.prepare(
                    model,
                    x,
                    split_request(boundary, backend="paddle"),
                )
                for batch in (1, 3):
                    replay_x = paddle.ones([batch, 4, 8], dtype="float32")
                    diff = paddle.max(paddle.abs(runtime.replay(replay_x) - model(replay_x)))
                    assert float(diff.item()) < 1e-5
        """,
        timeout=240,
    )


def test_paddle_official_vision_models_split_replay_batch_symbolic() -> None:
    """Official Paddle vision models replay representative splits across dynamic batch."""

    _skip_unless_enabled()
    _skip_if_module_missing("paddle")
    _run_backend_subprocess(
        """
        import paddle
        import torchlens as tl

        def assert_close(left, right, *, atol=1e-5, rtol=1e-4):
            assert bool(paddle.allclose(left, right, atol=atol, rtol=rtol).item())

        def assert_model_replays(name, factory, shape):
            paddle.seed(0)
            paddle.set_device("cpu")
            model = factory()
            model.eval()
            x = paddle.ones(shape, dtype="float32")
            for boundary in ("25%", "50%", "75%"):
                runtime = tl.split.prepare(
                    model,
                    x,
                    split_request(boundary, backend="paddle"),
                )
                for batch in (1, 3):
                    replay_x = paddle.ones([batch, *shape[1:]], dtype="float32")
                    assert_close(runtime.replay(replay_x), model(replay_x))

        assert_model_replays(
            "resnet18",
            lambda: paddle.vision.models.resnet18(pretrained=False, num_classes=10),
            [2, 3, 32, 32],
        )
        assert_model_replays(
            "lenet",
            lambda: paddle.vision.models.LeNet(num_classes=10),
            [2, 1, 28, 28],
        )
        assert_model_replays(
            "mobilenet_v2",
            lambda: paddle.vision.models.mobilenet_v2(
                pretrained=False,
                scale=0.25,
                num_classes=10,
            ),
            [2, 3, 32, 32],
        )
        """,
        timeout=240,
    )


def test_paddle_complete_multiscale_detector_all_nodes_cross_batch() -> None:
    """Paddle's complete multiscale detector graph replays at every boundary."""

    _skip_unless_enabled()
    _skip_if_module_missing("paddle")
    _run_backend_subprocess(
        """
        import paddle
        import paddle.nn.functional as F
        import torchlens as tl
        from torchlens.split import after, before

        class CompletePaddleDetector(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.stem = paddle.nn.Conv2D(3, 8, 3, padding=1)
                self.stage2 = paddle.nn.Conv2D(8, 16, 3, stride=2, padding=1)
                self.stage3 = paddle.nn.Conv2D(16, 24, 3, stride=2, padding=1)
                self.down2 = paddle.nn.Conv2D(8, 16, 3, stride=2, padding=1)
                self.neck2 = paddle.nn.Conv2D(32, 16, 1)
                self.down3 = paddle.nn.Conv2D(16, 24, 3, stride=2, padding=1)
                self.neck3 = paddle.nn.Conv2D(48, 24, 1)
                self.head1 = paddle.nn.Conv2D(8, 8, 1)
                self.head2 = paddle.nn.Conv2D(16, 8, 1)
                self.head3 = paddle.nn.Conv2D(24, 8, 1)

            def forward(self, inputs):
                p1 = F.relu(self.stem(inputs))
                p2 = F.relu(self.stage2(p1))
                p3 = F.relu(self.stage3(p2))
                n2 = F.relu(self.neck2(paddle.concat([self.down2(p1), p2], axis=1)))
                n3 = F.relu(self.neck3(paddle.concat([self.down3(n2), p3], axis=1)))

                def flatten_scale(value):
                    return paddle.reshape(value, [value.shape[0], -1, 8])

                predictions = paddle.concat(
                    [
                        flatten_scale(self.head1(p1)),
                        flatten_scale(self.head2(n2)),
                        flatten_scale(self.head3(n3)),
                    ],
                    axis=1,
                )
                return predictions[:, :, :4], F.sigmoid(predictions[:, :, 4:])

        paddle.set_device("cpu")
        paddle.seed(0)
        model = CompletePaddleDetector()
        model.eval()
        trace_x = paddle.ones([1, 3, 32, 32], dtype="float32")
        seed_runtime = tl.split.prepare(
            model,
            trace_x,
            split_request("50%", backend="paddle"),
        )
        boundaries = [
            (kind, node.canonical_id)
            for node in seed_runtime.trace_graph.compute_nodes
            for kind in ("before", "after")
        ]
        assert boundaries
        for kind, node_id in boundaries:
            point = after(node_id) if kind == "after" else before(node_id)
            runtime = seed_runtime.at(point)
            actual = runtime.replay(trace_x)
            expected = model(trace_x)
            for actual_leaf, expected_leaf in zip(actual, expected, strict=True):
                assert float(paddle.max(paddle.abs(actual_leaf - expected_leaf)).item()) < 1e-4

        for batch in (2, 3):
            runtime = seed_runtime.at(split_request("50%", backend="paddle").point)
            replay_x = paddle.ones([batch, 3, 32, 32], dtype="float32")
            actual = runtime.replay(replay_x)
            expected = model(replay_x)
            for actual_leaf, expected_leaf in zip(actual, expected, strict=True):
                assert actual_leaf.shape == expected_leaf.shape
                assert float(paddle.max(paddle.abs(actual_leaf - expected_leaf)).item()) < 1e-4
        """,
        timeout=600,
    )


def test_paddle_official_resnet_cross_batch_and_device() -> None:
    """Official Paddle ResNet18 replays CPU-prefix/GPU-suffix boundaries."""

    _skip_unless_enabled()
    _skip_if_module_missing("paddle")
    _run_backend_subprocess(
        """
        import paddle
        import torchlens as tl

        if paddle.device.cuda.device_count() == 0:
            print("No Paddle GPU available for official ResNet cross-device replay.")
            raise SystemExit(75)

        def assert_place(tensor, token):
            assert token in str(tensor.place).lower(), tensor.place

        paddle.seed(0)

        for boundary in ("25%", "50%", "75%"):
            paddle.set_device("cpu")
            model = paddle.vision.models.resnet18(pretrained=False, num_classes=10)
            model.eval()
            trace_x = paddle.ones([2, 3, 32, 32], dtype="float32")
            runtime = tl.split.prepare(
                model,
                trace_x,
                split_request(boundary, backend="paddle"),
            )
            for batch in (1, 3):
                paddle.set_device("cpu")
                model.to("cpu")
                replay_x_cpu = paddle.ones([batch, 3, 32, 32], dtype="float32")
                boundary_cpu = runtime.run_prefix(replay_x_cpu)
                for value in boundary_cpu.tensors.values():
                    assert_place(value, "cpu")

                model.to("gpu:0")
                paddle.set_device("gpu:0")
                boundary_gpu = boundary_cpu.to("gpu:0", adapter=runtime.adapter)
                for value in boundary_gpu.tensors.values():
                    assert_place(value, "gpu")
                replay_x_gpu = paddle.ones([batch, 3, 32, 32], dtype="float32")
                split_output = runtime.run_suffix(boundary_gpu)
                full_output = model(replay_x_gpu)
                assert_place(split_output, "gpu")
                assert float(paddle.max(paddle.abs(split_output - full_output)).item()) < 5e-4
        """,
        timeout=300,
    )


def test_paddle_official_lenet_all_split_nodes_cross_batch_and_device() -> None:
    """Every split before/after node replays for official Paddle LeNet across batch/device."""

    _skip_unless_enabled()
    _skip_if_module_missing("paddle")
    _run_backend_subprocess(
        """
        import paddle
        import torchlens as tl

        def assert_place(tensor, token):
            assert token in str(tensor.place).lower(), tensor.place

        if paddle.device.cuda.device_count() == 0:
            print("No Paddle GPU available for CPU-prefix/GPU-suffix LeNet all-node replay.")
            raise SystemExit(75)

        paddle.seed(0)
        paddle.set_device("cpu")
        model = paddle.vision.models.LeNet(num_classes=10)
        model.eval()
        x = paddle.ones([2, 1, 28, 28], dtype="float32")
        seed_runtime = tl.split.prepare(
            model,
            x,
            split_request("50%", backend="paddle"),
        )
        boundaries = [
            f"{kind}:{node.canonical_id}"
            for node in seed_runtime.trace_graph.compute_nodes
            if node.parents
            for kind in ("before", "after")
        ]
        for boundary in boundaries:
            paddle.set_device("cpu")
            model.to("cpu")
            runtime = tl.split.prepare(
                model,
                x,
                split_request(boundary, backend="paddle"),
            )
            for batch in (1, 3):
                paddle.set_device("cpu")
                model.to("cpu")
                replay_x_cpu = paddle.ones([batch, 1, 28, 28], dtype="float32")
                boundary_cpu = runtime.run_prefix(replay_x_cpu)
                for value in boundary_cpu.tensors.values():
                    assert_place(value, "cpu")

                model.to("gpu:0")
                paddle.set_device("gpu:0")
                boundary_gpu = boundary_cpu.to("gpu:0", adapter=runtime.adapter)
                for value in boundary_gpu.tensors.values():
                    assert_place(value, "gpu")
                replay_x_gpu = paddle.ones([batch, 1, 28, 28], dtype="float32")
                split_output = runtime.run_suffix(boundary_gpu)
                full_output = model(replay_x_gpu)
                assert_place(split_output, "gpu")
                diff = paddle.max(paddle.abs(split_output - full_output))
                assert float(diff.item()) < 1e-5
        """,
        timeout=240,
    )
