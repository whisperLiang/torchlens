"""Opt-in real-world split replay tests."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from importlib.util import find_spec
from pathlib import Path

import pytest
import torch

import torchlens as tl

pytestmark = [pytest.mark.slow, pytest.mark.real_model]

_SUBPROCESS_SKIP_EXIT_CODE = 75


def _enabled() -> bool:
    """Return whether real-model tests are enabled."""

    return os.environ.get("TORCHLENS_REAL_MODEL_TESTS") != "0"


def _skip_unless_enabled() -> None:
    """Skip real-model split tests only when explicitly disabled."""

    if not _enabled():
        pytest.skip("TORCHLENS_REAL_MODEL_TESTS=0 disables real-model split tests.")


def _skip_if_module_missing(module_name: str) -> None:
    """Skip when an optional backend/model dependency is unavailable."""

    if find_spec(module_name) is None:
        pytest.skip(f"{module_name!r} is not installed.")


def _run_backend_subprocess(
    code: str,
    *,
    timeout: int = 180,
    env_overrides: dict[str, str] | None = None,
) -> None:
    """Run a backend-isolated real-model split scenario."""

    env = os.environ.copy()
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if env_overrides:
        env.update(env_overrides)
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode == _SUBPROCESS_SKIP_EXIT_CODE:
        reason = "\n".join(
            part
            for part in (result.stdout.strip(), result.stderr.strip())
            if part
        )
        pytest.skip(reason or "backend subprocess skipped.")
    assert result.returncode == 0, (
        "real-model backend subprocess failed\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def _all_compute_split_boundaries(runtime: object) -> list[str]:
    """Return before/after split boundaries for every compute node in a runtime."""

    return [
        f"{kind}:{node.canonical_id}"
        for node in runtime.trace_graph.compute_nodes
        for kind in ("before", "after")
    ]


def _torch_test_devices() -> list[torch.device]:
    """Return the torch devices covered by cross-device split tests."""

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda:0"))
    return devices


def test_resnet18_representative_splits_skip_cleanly_when_disabled() -> None:
    """ResNet18 split replay validates a small number of representative boundaries."""

    _skip_unless_enabled()
    torchvision = pytest.importorskip("torchvision")
    model = torchvision.models.resnet18(weights=None).eval()
    x = torch.randn(1, 3, 64, 64)

    for spec in (tl.SplitSpec("25%"), tl.SplitSpec("50%"), tl.SplitSpec("before:fc")):
        runtime = tl.prepare_split(model, x, spec)
        boundary = runtime.run_prefix(x)
        runtime.validate_boundary(boundary)
        assert torch.allclose(runtime.run_suffix(boundary), model(x), atol=1e-4, rtol=1e-3)


def test_tiny_transformer_config_split_skip_cleanly_when_disabled() -> None:
    """A no-download transformer config exercises attention-style structure."""

    _skip_unless_enabled()
    transformers = pytest.importorskip("transformers")
    config = transformers.DistilBertConfig(
        vocab_size=128,
        n_layers=1,
        dim=32,
        hidden_dim=64,
        n_heads=4,
    )
    model = transformers.DistilBertModel(config).eval()
    x = torch.randint(0, 128, (2, 8))

    runtime = tl.prepare_split(model, x, tl.SplitSpec("50%"))
    boundary = runtime.run_prefix(x)
    runtime.validate_boundary(boundary)
    split_output = runtime.run_suffix(boundary).last_hidden_state
    full_output = model(x).last_hidden_state

    assert torch.allclose(split_output, full_output, atol=1e-4, rtol=1e-3)


def test_torchvision_extra_real_models_cross_batch_and_device() -> None:
    """Additional torchvision models replay representative splits across batch/device."""

    _skip_unless_enabled()
    torchvision = pytest.importorskip("torchvision")
    torch.manual_seed(0)
    model_cases = (
        (
            "mobilenet_v2",
            lambda: torchvision.models.mobilenet_v2(weights=None, num_classes=10),
            (2, 3, 64, 64),
            ("20%", "50%", "80%"),
        ),
        (
            "squeezenet1_0",
            lambda: torchvision.models.squeezenet1_0(weights=None, num_classes=10),
            (2, 3, 64, 64),
            ("20%", "50%", "80%"),
        ),
    )

    for device in _torch_test_devices():
        for _name, factory, shape, boundaries in model_cases:
            model = factory().to(device).eval()
            x = torch.ones(shape, device=device)
            for boundary in boundaries:
                runtime = tl.prepare_split(
                    model,
                    x,
                    tl.SplitSpec(boundary, dynamic_batch=(1, 3)),
                )
                for batch in (1, 3):
                    replay_x = torch.ones((batch, *shape[1:]), device=device)
                    with torch.no_grad():
                        split_output = runtime.replay(replay_x)
                        full_output = model(replay_x)
                    atol = 5e-4 if device.type == "cuda" else 1e-4
                    assert torch.allclose(split_output, full_output, atol=atol, rtol=1e-3)


def test_torchvision_ops_mlp_all_split_nodes_cross_batch_and_device() -> None:
    """Every compute-node before/after split replays for a torchvision module block."""

    _skip_unless_enabled()
    torchvision = pytest.importorskip("torchvision")
    torch.manual_seed(0)

    for device in _torch_test_devices():
        model = torchvision.ops.MLP(
            in_channels=6,
            hidden_channels=[8, 4],
            dropout=0.0,
        ).to(device)
        model.eval()
        x = torch.ones((2, 6), device=device)
        seed_runtime = tl.prepare_split(
            model,
            x,
            tl.SplitSpec("50%", dynamic_batch=(1, 3)),
        )

        for boundary in _all_compute_split_boundaries(seed_runtime):
            runtime = tl.prepare_split(
                model,
                x,
                tl.SplitSpec(boundary, dynamic_batch=(1, 3)),
            )
            for batch in (1, 3):
                replay_x = torch.ones((batch, 6), device=device)
                with torch.no_grad():
                    split_output = runtime.replay(replay_x)
                    full_output = model(replay_x)
                assert torch.allclose(split_output, full_output, atol=1e-5, rtol=1e-4)


def test_tf_vgg16_split_replay_dynamic_batch_and_train() -> None:
    """Keras VGG16 exercises TensorFlow real-model split replay and training."""

    _skip_unless_enabled()
    _skip_if_module_missing("tensorflow")
    _run_backend_subprocess(
        """
        import tensorflow as tf
        import torchlens as tl

        tf.keras.utils.set_random_seed(0)
        model = tf.keras.applications.VGG16(
            input_shape=(32, 32, 3),
            weights=None,
            classes=10,
        )
        x = tf.ones((2, 32, 32, 3), dtype=tf.float32)
        runtime = tl.prepare_split(
            model,
            x,
            tl.SplitSpec("25%", backend="tf", dynamic_batch=(1, 4)),
        )
        for batch in (1, 2, 4):
            replay_x = tf.ones((batch, 32, 32, 3), dtype=tf.float32)
            diff = tf.reduce_max(tf.abs(runtime.replay(replay_x) - model(replay_x)))
            assert float(diff.numpy()) < 1e-5

        train_runtime = tl.prepare_split(
            model,
            x,
            tl.SplitSpec("75%", backend="tf", dynamic_batch=(1, 4), trainable=True),
        )
        boundary = train_runtime.run_training_prefix(x)
        loss, grads = train_runtime.train_suffix(
            boundary,
            tf.zeros((2, 10), dtype=tf.float32),
        )
        assert float(loss.numpy()) >= 0.0
        assert grads
        assert train_runtime.backward_prefix(boundary, grads)
        """,
    )


def test_tf_keras_application_matrix_split_replay_dynamic_batch() -> None:
    """Additional Keras application models replay representative split boundaries."""

    _skip_unless_enabled()
    _skip_if_module_missing("tensorflow")
    _run_backend_subprocess(
        """
        import tensorflow as tf
        import torchlens as tl

        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass

        model_cases = (
            (
                "MobileNetV2",
                lambda: tf.keras.applications.MobileNetV2(
                    input_shape=(32, 32, 3),
                    alpha=0.35,
                    weights=None,
                    classes=10,
                ),
            ),
            (
                "ResNet50",
                lambda: tf.keras.applications.ResNet50(
                    input_shape=(32, 32, 3),
                    weights=None,
                    classes=10,
                ),
            ),
            (
                "DenseNet121",
                lambda: tf.keras.applications.DenseNet121(
                    input_shape=(32, 32, 3),
                    weights=None,
                    classes=10,
                ),
            ),
            (
                "EfficientNetB0",
                lambda: tf.keras.applications.EfficientNetB0(
                    input_shape=(32, 32, 3),
                    weights=None,
                    classes=10,
                ),
            ),
        )

        for _name, factory in model_cases:
            tf.keras.utils.set_random_seed(0)
            model = factory()
            x = tf.ones((2, 32, 32, 3), dtype=tf.float32)
            for boundary in ("25%", "50%", "75%"):
                runtime = tl.prepare_split(
                    model,
                    x,
                    tl.SplitSpec(boundary, backend="tf", dynamic_batch=(1, 3)),
                )
                for batch in (1, 3):
                    replay_x = tf.ones((batch, 32, 32, 3), dtype=tf.float32)
                    diff = tf.reduce_max(tf.abs(runtime.replay(replay_x) - model(replay_x)))
                    assert float(diff.numpy()) < 1e-4
        """,
        timeout=360,
    )


def test_tf_keras_cnn_all_split_nodes_cross_batch_and_device() -> None:
    """Every compute-node before/after split replays for a compact Keras CNN."""

    _skip_unless_enabled()
    _skip_if_module_missing("tensorflow")
    _run_backend_subprocess(
        """
        import tensorflow as tf
        import torchlens as tl

        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass

        def make_model():
            return tf.keras.Sequential(
                [
                    tf.keras.Input(shape=(8, 8, 3)),
                    tf.keras.layers.Conv2D(4, 3, padding="same"),
                    tf.keras.layers.ReLU(),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(5),
                ]
            )

        tf.keras.utils.set_random_seed(0)
        devices = ["/CPU:0"]
        gpus = tf.config.list_logical_devices("GPU")
        if gpus:
            devices.append(gpus[0].name)

        for device in devices:
            with tf.device(device):
                model = make_model()
                x = tf.ones((2, 8, 8, 3), dtype=tf.float32)
                seed_runtime = tl.prepare_split(
                    model,
                    x,
                    tl.SplitSpec("50%", backend="tf", dynamic_batch=(1, 3)),
                )
                boundaries = [
                    f"{kind}:{node.canonical_id}"
                    for node in seed_runtime.trace_graph.compute_nodes
                    for kind in ("before", "after")
                ]
                for boundary in boundaries:
                    runtime = tl.prepare_split(
                        model,
                        x,
                        tl.SplitSpec(boundary, backend="tf", dynamic_batch=(1, 3)),
                    )
                    for batch in (1, 3):
                        replay_x = tf.ones((batch, 8, 8, 3), dtype=tf.float32)
                        diff = tf.reduce_max(
                            tf.abs(runtime.replay(replay_x) - model(replay_x))
                        )
                        assert float(diff.numpy()) < 1e-5
        """,
        timeout=240,
    )


def test_tf_split_train_cpu_prefix_gpu_suffix_cross_batch() -> None:
    """TensorFlow split training runs CPU prefix and GPU suffix for multiple batches."""

    _skip_unless_enabled()
    _skip_if_module_missing("tensorflow")
    _run_backend_subprocess(
        """
        import tensorflow as tf
        import torchlens as tl

        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
        gpus = tf.config.list_logical_devices("GPU")
        if not gpus:
            print("No TensorFlow GPU available for CPU-prefix/GPU-suffix split training.")
            raise SystemExit(75)
        gpu_name = gpus[0].name

        class TrainModule(tf.Module):
            def __init__(self):
                super().__init__()
                with tf.device("/CPU:0"):
                    self.w1 = tf.Variable(
                        tf.reshape(tf.linspace(-0.3, 0.3, 20), (4, 5)),
                        name="w1",
                    )
                    self.b1 = tf.Variable(tf.zeros((5,), dtype=tf.float32), name="b1")
                    self.w2 = tf.Variable(
                        tf.reshape(tf.linspace(-0.2, 0.2, 15), (5, 3)),
                        name="w2",
                    )
                    self.b2 = tf.Variable(tf.zeros((3,), dtype=tf.float32), name="b2")

            def __call__(self, x):
                hidden = tf.nn.relu(tf.matmul(x, self.w1) + self.b1)
                return tf.matmul(hidden, self.w2) + self.b2

        def assert_on_device(tensor, token):
            assert token in tensor.device, tensor.device

        with tf.device("/CPU:0"):
            model = TrainModule()
            trace_x = tf.ones((2, 4), dtype=tf.float32)
            seed_runtime = tl.prepare_split(
                model,
                trace_x,
                tl.SplitSpec("50%", backend="tf", trainable=True, dynamic_batch=(1, 3)),
            )
            relu_id = next(
                node.canonical_id
                for node in seed_runtime.trace_graph.compute_nodes
                if node.op_type == "relu"
            )
            runtime = tl.prepare_split(
                model,
                trace_x,
                tl.SplitSpec(
                    f"after:{relu_id}",
                    backend="tf",
                    trainable=True,
                    dynamic_batch=(1, 3),
                ),
            )

        for batch in (1, 3):
            with tf.device("/CPU:0"):
                train_x = tf.ones((batch, 4), dtype=tf.float32)
                boundary_cpu = runtime.run_training_prefix(train_x)
            for value in boundary_cpu.tensors.values():
                assert_on_device(value, "CPU:0")

            boundary_gpu = boundary_cpu.to(gpu_name, adapter=runtime.adapter)
            for value in boundary_gpu.tensors.values():
                assert_on_device(value, "GPU:0")

            with tf.device(gpu_name):
                targets = tf.zeros((batch, 3), dtype=tf.float32)
                loss, grads_gpu = runtime.train_suffix(boundary_gpu, targets)
            assert_on_device(loss, "GPU:0")
            assert grads_gpu
            for grad in grads_gpu.values():
                assert_on_device(grad, "GPU:0")

            with tf.device("/CPU:0"):
                grads_cpu = {key: tf.identity(grad) for key, grad in grads_gpu.items()}
                prefix_grads = runtime.backward_prefix(boundary_cpu, grads_cpu)
            assert prefix_grads
        """,
        timeout=240,
    )


def test_paddle_layer_model_split_replay_dynamic_batch_and_train() -> None:
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
        runtime = tl.prepare_split(
            model,
            x,
            tl.SplitSpec("25%", backend="paddle", dynamic_batch=(1, 4)),
        )
        for batch in (1, 2, 4):
            replay_x = paddle.ones([batch, 2, 3], dtype="float32")
            diff = paddle.max(paddle.abs(runtime.replay(replay_x) - model(replay_x)))
            assert float(diff.item()) < 1e-5

        model.train()
        train_runtime = tl.prepare_split(
            model,
            x,
            tl.SplitSpec("25%", backend="paddle", dynamic_batch=(1, 4), trainable=True),
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
        seed_runtime = tl.prepare_split(
            model,
            trace_x,
            tl.SplitSpec("50%", backend="paddle", trainable=True, dynamic_batch=(1, 3)),
        )
        relu_id = next(
            node.canonical_id
            for node in seed_runtime.trace_graph.compute_nodes
            if "relu" in node.op_type
        )
        runtime = tl.prepare_split(
            model,
            trace_x,
            tl.SplitSpec(
                f"after:{relu_id}",
                backend="paddle",
                trainable=True,
                dynamic_batch=(1, 3),
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
        seed_runtime = tl.prepare_split(
            model,
            x,
            tl.SplitSpec("50%", backend="paddle", dynamic_batch=(1, 3)),
        )
        boundaries = [
            f"{kind}:{node.canonical_id}"
            for node in seed_runtime.trace_graph.compute_nodes
            for kind in ("before", "after")
        ]
        for boundary in boundaries:
            paddle.set_device("cpu")
            model.to("cpu")
            runtime = tl.prepare_split(
                model,
                x,
                tl.SplitSpec(boundary, backend="paddle", dynamic_batch=(1, 3)),
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


def test_paddle_transformer_block_split_replay_dynamic_batch_and_device() -> None:
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
                runtime = tl.prepare_split(
                    model,
                    x,
                    tl.SplitSpec(boundary, backend="paddle", dynamic_batch=(1, 3)),
                )
                for batch in (1, 3):
                    replay_x = paddle.ones([batch, 4, 8], dtype="float32")
                    diff = paddle.max(paddle.abs(runtime.replay(replay_x) - model(replay_x)))
                    assert float(diff.item()) < 1e-5
        """,
        timeout=240,
    )


def test_paddle_official_vision_models_split_replay_dynamic_batch() -> None:
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
                runtime = tl.prepare_split(
                    model,
                    x,
                    tl.SplitSpec(boundary, backend="paddle", dynamic_batch=(1, 3)),
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
        seed_runtime = tl.prepare_split(
            model,
            x,
            tl.SplitSpec("50%", backend="paddle", dynamic_batch=(1, 3)),
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
            runtime = tl.prepare_split(
                model,
                x,
                tl.SplitSpec(boundary, backend="paddle", dynamic_batch=(1, 3)),
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


def test_jax_stax_cnn_split_replay_dynamic_batch_and_train() -> None:
    """JAX stax CNN exercises library-model split replay and VJP handoff."""

    _skip_unless_enabled()
    _skip_if_module_missing("jax")
    _run_backend_subprocess(
        """
        import jax
        import jax.numpy as jnp
        from jax.example_libraries import stax

        import torchlens as tl

        init_fun, apply_fun = stax.serial(
            stax.Conv(4, (3, 3), padding="SAME"),
            stax.Relu,
            stax.MaxPool((2, 2), strides=(2, 2)),
            stax.Flatten,
            stax.Dense(10),
        )
        _shape, params = init_fun(jax.random.PRNGKey(0), (-1, 32, 32, 3))

        def model(params, x):
            return apply_fun(params, x)

        x = jnp.ones((2, 32, 32, 3), dtype=jnp.float32)
        runtime = tl.prepare_split(
            model,
            (params, x),
            tl.SplitSpec("25%", backend="jax", dynamic_batch=(1, 4)),
        )
        for batch in (1, 2, 4):
            replay_x = jnp.ones((batch, 32, 32, 3), dtype=jnp.float32)
            assert bool(jnp.allclose(runtime.replay(params, replay_x), model(params, replay_x)))

        train_runtime = tl.prepare_split(
            model,
            (params, x),
            tl.SplitSpec("25%", backend="jax", dynamic_batch=(1, 4), trainable=True),
        )
        boundary = train_runtime.run_training_prefix(params, x)
        loss, grads = train_runtime.train_suffix(
            boundary,
            jnp.zeros((2, 10), dtype=jnp.float32),
        )
        assert float(loss) >= 0.0
        assert grads
        prefix_grads = train_runtime.backward_prefix(boundary, grads)
        assert "inputs" in prefix_grads
        """,
    )


def test_jax_flax_haiku_equinox_all_split_nodes_cross_batch_and_device() -> None:
    """Flax, Haiku, and Equinox models replay every split across batch/device."""

    _skip_unless_enabled()
    _skip_if_module_missing("jax")
    _skip_if_module_missing("flax")
    _skip_if_module_missing("haiku")
    _skip_if_module_missing("equinox")
    _run_backend_subprocess(
        """
        import equinox as eqx
        import flax.linen as nn
        import haiku as hk
        import jax
        import jax.numpy as jnp

        import torchlens as tl

        def platform_devices(platform):
            try:
                return jax.devices(platform)
            except RuntimeError:
                return []

        cpu_devices = platform_devices("cpu")
        gpu_devices = platform_devices("gpu")
        if not cpu_devices or not gpu_devices:
            print("No JAX CPU/GPU pair available for CPU-prefix/GPU-suffix all-node replay.")
            raise SystemExit(75)
        cpu = cpu_devices[0]
        gpu = gpu_devices[0]

        def all_boundaries(runtime):
            return [
                f"{kind}:{node.canonical_id}"
                for node in runtime.trace_graph.compute_nodes
                for kind in ("before", "after")
            ]

        def assert_all_split_nodes(name, model, args_factory):
            with jax.default_device(cpu):
                traced_args = args_factory(2, cpu)
                seed_runtime = tl.prepare_split(
                    model,
                    traced_args,
                    tl.SplitSpec("50%", backend="jax", dynamic_batch=(1, 3)),
                )
            for boundary in all_boundaries(seed_runtime):
                with jax.default_device(cpu):
                    runtime = tl.prepare_split(
                        model,
                        traced_args,
                        tl.SplitSpec(boundary, backend="jax", dynamic_batch=(1, 3)),
                    )
                for batch in (1, 3):
                    with jax.default_device(cpu):
                        prefix_args = args_factory(batch, cpu)
                        boundary_cpu = runtime.run_prefix(*prefix_args)
                    for value in boundary_cpu.tensors.values():
                        assert value.device == cpu

                    boundary_gpu = boundary_cpu.to(gpu, adapter=runtime.adapter)
                    for value in boundary_gpu.tensors.values():
                        assert value.device == gpu

                    with jax.default_device(gpu):
                        split_output = runtime.run_suffix(boundary_gpu)
                        full_args = args_factory(batch, gpu)
                        full_output = model(*full_args)
                    assert split_output.device == gpu
                    assert bool(
                        jnp.allclose(split_output, full_output, atol=1e-4, rtol=1e-3)
                    ), (name, boundary, batch)

        class FlaxCNN(nn.Module):
            @nn.compact
            def __call__(self, x):
                x = nn.Conv(features=4, kernel_size=(3, 3), padding="SAME")(x)
                x = nn.relu(x)
                x = x.reshape((x.shape[0], -1))
                return nn.Dense(5)(x)

        flax_module = FlaxCNN()
        flax_params = flax_module.init(
            jax.random.PRNGKey(0),
            jnp.ones((2, 8, 8, 3), dtype=jnp.float32),
        )

        def flax_model(params, x):
            return flax_module.apply(params, x)

        def flax_args(batch, device):
            params = jax.tree_util.tree_map(lambda value: jax.device_put(value, device), flax_params)
            x = jax.device_put(jnp.ones((batch, 8, 8, 3), dtype=jnp.float32), device)
            return params, x

        assert_all_split_nodes("flax-cnn", flax_model, flax_args)

        def haiku_forward(x):
            hidden = hk.Linear(8)(x)
            query = hk.Linear(8)(hidden)
            key = hk.Linear(8)(hidden)
            value = hk.Linear(8)(hidden)
            scale = jnp.sqrt(jnp.array(8.0, dtype=jnp.float32))
            attn = jax.nn.softmax(
                (query @ jnp.swapaxes(key, -1, -2)) / scale,
                axis=-1,
            )
            hidden = hk.Linear(8)(attn @ value) + hidden
            return hk.Linear(4)(jax.nn.relu(hidden).mean(axis=1))

        haiku_transform = hk.without_apply_rng(hk.transform(haiku_forward))
        haiku_params = haiku_transform.init(
            jax.random.PRNGKey(1),
            jnp.ones((2, 4, 8), dtype=jnp.float32),
        )

        def haiku_model(params, x):
            return haiku_transform.apply(params, x)

        def haiku_args(batch, device):
            params = jax.tree_util.tree_map(lambda value: jax.device_put(value, device), haiku_params)
            x = jax.device_put(jnp.ones((batch, 4, 8), dtype=jnp.float32), device)
            return params, x

        assert_all_split_nodes("haiku-attention", haiku_model, haiku_args)

        class EquinoxBatchedMLP(eqx.Module):
            w1: jax.Array
            b1: jax.Array
            w2: jax.Array
            b2: jax.Array

            def __init__(self, key):
                key1, key2 = jax.random.split(key, 2)
                self.w1 = jax.random.normal(key1, (6, 8), dtype=jnp.float32) * 0.05
                self.b1 = jnp.zeros((8,), dtype=jnp.float32)
                self.w2 = jax.random.normal(key2, (8, 4), dtype=jnp.float32) * 0.05
                self.b2 = jnp.zeros((4,), dtype=jnp.float32)

            def __call__(self, x):
                return jax.nn.relu(x @ self.w1 + self.b1) @ self.w2 + self.b2

        equinox_module = EquinoxBatchedMLP(jax.random.PRNGKey(2))

        def equinox_model(module, x):
            return module(x)

        def equinox_args(batch, device):
            module = jax.tree_util.tree_map(
                lambda value: jax.device_put(value, device) if hasattr(value, "shape") else value,
                equinox_module,
            )
            x = jax.device_put(jnp.ones((batch, 6), dtype=jnp.float32), device)
            return module, x

        assert_all_split_nodes("equinox-mlp", equinox_model, equinox_args)
        """,
        timeout=360,
    )


def test_jax_flax_split_train_cpu_prefix_gpu_suffix_cross_batch() -> None:
    """Flax split training runs CPU prefix and GPU suffix for multiple batches."""

    _skip_unless_enabled()
    _skip_if_module_missing("jax")
    _skip_if_module_missing("flax")
    _run_backend_subprocess(
        """
        import flax.linen as nn
        import jax
        import jax.numpy as jnp
        import torchlens as tl

        try:
            cpu = jax.devices("cpu")[0]
            gpu = jax.devices("gpu")[0]
        except RuntimeError:
            print("No JAX CPU/GPU pair available for CPU-prefix/GPU-suffix split training.")
            raise SystemExit(75)

        class FlaxMlp(nn.Module):
            @nn.compact
            def __call__(self, x):
                hidden = nn.relu(nn.Dense(5)(x))
                return nn.Dense(3)(hidden)

        module = FlaxMlp()
        params_cpu = module.init(
            jax.random.PRNGKey(0),
            jax.device_put(jnp.ones((2, 4), dtype=jnp.float32), cpu),
        )
        params_cpu = jax.tree_util.tree_map(lambda value: jax.device_put(value, cpu), params_cpu)

        def model(params, x):
            return module.apply(params, x)

        def tree_allclose(left, right):
            checks = jax.tree_util.tree_leaves(
                jax.tree_util.tree_map(
                    lambda a, b: jnp.allclose(a, b, atol=1e-5, rtol=1e-4),
                    left,
                    right,
                )
            )
            return all(bool(check) for check in checks)

        trace_x = jax.device_put(jnp.ones((2, 4), dtype=jnp.float32), cpu)
        seed_runtime = tl.prepare_split(
            model,
            (params_cpu, trace_x),
            tl.SplitSpec("50%", backend="jax", trainable=True, dynamic_batch=(1, 3)),
        )
        relu_id = next(
            node.canonical_id
            for node in seed_runtime.trace_graph.compute_nodes
            if node.op_type == "max"
        )
        runtime = tl.prepare_split(
            model,
            (params_cpu, trace_x),
            tl.SplitSpec(
                f"after:{relu_id}",
                backend="jax",
                trainable=True,
                dynamic_batch=(1, 3),
            ),
        )

        for batch in (1, 3):
            x_cpu = jax.device_put(jnp.ones((batch, 4), dtype=jnp.float32), cpu)
            y_cpu = jax.device_put(jnp.zeros((batch, 3), dtype=jnp.float32), cpu)
            boundary_cpu = runtime.run_training_prefix(params_cpu, x_cpu)
            for value in boundary_cpu.tensors.values():
                assert value.device == cpu

            boundary_gpu = boundary_cpu.to(gpu, adapter=runtime.adapter)
            for value in boundary_gpu.tensors.values():
                assert value.device == gpu

            y_gpu = jax.device_put(y_cpu, gpu)
            loss, grads_gpu = runtime.train_suffix(boundary_gpu, y_gpu)
            assert loss.device == gpu
            assert grads_gpu
            for grad in grads_gpu.values():
                assert grad.device == gpu

            grads_cpu = {key: jax.device_put(grad, cpu) for key, grad in grads_gpu.items()}
            prefix_grads = runtime.backward_prefix(boundary_cpu, grads_cpu)

            def full_loss(params, x):
                return jnp.mean((model(params, x) - y_cpu) ** 2)

            expected_param_grads, expected_x_grad = jax.grad(full_loss, argnums=(0, 1))(
                params_cpu,
                x_cpu,
            )
            actual_param_grads, actual_x_grad = prefix_grads["inputs"]
            assert tree_allclose(actual_param_grads, expected_param_grads)
            assert bool(jnp.allclose(actual_x_grad, expected_x_grad, atol=1e-5, rtol=1e-4))
        """,
        timeout=240,
    )


def test_jax_reference_resnet_transformer_and_vit_split_replay_dynamic_batch() -> None:
    """JAX reference ResNet/Transformer/ViT-style architectures replay dynamic splits."""

    _skip_unless_enabled()
    _skip_if_module_missing("jax")
    _run_backend_subprocess(
        """
        import jax
        import jax.numpy as jnp
        from jax.example_libraries import stax

        import torchlens as tl

        residual_init, residual_apply = stax.serial(
            stax.FanOut(2),
            stax.parallel(
                stax.serial(stax.Dense(8), stax.Relu, stax.Dense(8)),
                stax.Identity,
            ),
            stax.FanInSum,
            stax.Relu,
            stax.Dense(4),
        )
        _shape, residual_params = residual_init(jax.random.PRNGKey(0), (-1, 8))

        def residual_model(params, x):
            return residual_apply(params, x)

        def init_transformer_params(key):
            keys = jax.random.split(key, 6)

            def weight(subkey, shape):
                return jax.random.normal(subkey, shape, dtype=jnp.float32) * 0.05

            return {
                "q": weight(keys[0], (8, 8)),
                "k": weight(keys[1], (8, 8)),
                "v": weight(keys[2], (8, 8)),
                "o": weight(keys[3], (8, 8)),
                "ff1": weight(keys[4], (8, 16)),
                "ff2": weight(keys[5], (16, 8)),
            }

        transformer_params = init_transformer_params(jax.random.PRNGKey(1))

        def transformer_model(params, x):
            query = x @ params["q"]
            key = x @ params["k"]
            value = x @ params["v"]
            scale = jnp.sqrt(jnp.array(8.0, dtype=jnp.float32))
            scores = (query @ jnp.swapaxes(key, -1, -2)) / scale
            attn = jax.nn.softmax(scores, axis=-1)
            hidden = (attn @ value) @ params["o"]
            residual = hidden + x
            ff = jax.nn.relu(residual @ params["ff1"]) @ params["ff2"]
            return ff + residual

        def init_vit_params(key):
            keys = jax.random.split(key, 8)

            def weight(subkey, shape):
                return jax.random.normal(subkey, shape, dtype=jnp.float32) * 0.05

            return {
                "patch": weight(keys[0], (4 * 4 * 3, 8)),
                "q": weight(keys[1], (8, 8)),
                "k": weight(keys[2], (8, 8)),
                "v": weight(keys[3], (8, 8)),
                "o": weight(keys[4], (8, 8)),
                "ff1": weight(keys[5], (8, 16)),
                "ff2": weight(keys[6], (16, 8)),
                "head": weight(keys[7], (8, 4)),
            }

        vit_params = init_vit_params(jax.random.PRNGKey(2))

        def vit_encoder_model(params, x):
            batch = x.shape[0]
            patches = (
                x.reshape(batch, 4, 4, 4, 4, 3)
                .transpose(0, 1, 3, 2, 4, 5)
                .reshape(batch, 16, 4 * 4 * 3)
            )
            tokens = patches @ params["patch"]
            query = tokens @ params["q"]
            key = tokens @ params["k"]
            value = tokens @ params["v"]
            scale = jnp.sqrt(jnp.array(8.0, dtype=jnp.float32))
            attn = jax.nn.softmax((query @ jnp.swapaxes(key, -1, -2)) / scale, axis=-1)
            hidden = (attn @ value) @ params["o"] + tokens
            mlp = jax.nn.relu(hidden @ params["ff1"]) @ params["ff2"]
            return (hidden + mlp).mean(axis=1) @ params["head"]

        model_cases = (
            (
                residual_model,
                residual_params,
                lambda batch: jnp.ones((batch, 8), dtype=jnp.float32),
            ),
            (
                transformer_model,
                transformer_params,
                lambda batch: jnp.ones((batch, 4, 8), dtype=jnp.float32),
            ),
            (
                vit_encoder_model,
                vit_params,
                lambda batch: jnp.ones((batch, 16, 16, 3), dtype=jnp.float32),
            ),
        )

        for model, params, input_factory in model_cases:
            x = input_factory(2)
            for boundary in ("25%", "50%", "75%"):
                runtime = tl.prepare_split(
                    model,
                    (params, x),
                    tl.SplitSpec(boundary, backend="jax", dynamic_batch=(1, 3)),
                )
                for batch in (1, 3):
                    replay_x = input_factory(batch)
                    assert bool(
                        jnp.allclose(
                            runtime.replay(params, replay_x),
                            model(params, replay_x),
                        )
                    )
        """,
        timeout=240,
    )


def test_jax_stax_mlp_all_split_nodes_cross_batch_and_device() -> None:
    """Every compute-node before/after split replays for a JAX stax MLP."""

    _skip_unless_enabled()
    _skip_if_module_missing("jax")
    _run_backend_subprocess(
        """
        import jax
        import jax.numpy as jnp
        from jax.example_libraries import stax

        import torchlens as tl

        def platform_devices(platform):
            try:
                return jax.devices(platform)
            except RuntimeError:
                return []

        init_fun, apply_fun = stax.serial(
            stax.Dense(8),
            stax.Relu,
            stax.Dense(4),
        )
        _shape, base_params = init_fun(jax.random.PRNGKey(0), (-1, 6))

        def model(params, x):
            return apply_fun(params, x)

        cpu_devices = platform_devices("cpu")
        gpu_devices = platform_devices("gpu")
        if not cpu_devices or not gpu_devices:
            print("No JAX CPU/GPU pair available for CPU-prefix/GPU-suffix stax replay.")
            raise SystemExit(75)
        cpu = cpu_devices[0]
        gpu = gpu_devices[0]
        params_cpu = jax.tree_util.tree_map(lambda value: jax.device_put(value, cpu), base_params)
        x_cpu = jax.device_put(jnp.ones((2, 6), dtype=jnp.float32), cpu)
        seed_runtime = tl.prepare_split(
            model,
            (params_cpu, x_cpu),
            tl.SplitSpec("50%", backend="jax", dynamic_batch=(1, 3)),
        )
        boundaries = [
            f"{kind}:{node.canonical_id}"
            for node in seed_runtime.trace_graph.compute_nodes
            for kind in ("before", "after")
        ]
        for boundary in boundaries:
            runtime = tl.prepare_split(
                model,
                (params_cpu, x_cpu),
                tl.SplitSpec(boundary, backend="jax", dynamic_batch=(1, 3)),
            )
            for batch in (1, 3):
                replay_x_cpu = jax.device_put(
                    jnp.ones((batch, 6), dtype=jnp.float32),
                    cpu,
                )
                boundary_cpu = runtime.run_prefix(params_cpu, replay_x_cpu)
                for value in boundary_cpu.tensors.values():
                    assert value.device == cpu

                boundary_gpu = boundary_cpu.to(gpu, adapter=runtime.adapter)
                for value in boundary_gpu.tensors.values():
                    assert value.device == gpu

                params_gpu = jax.tree_util.tree_map(
                    lambda value: jax.device_put(value, gpu),
                    base_params,
                )
                replay_x_gpu = jax.device_put(
                    jnp.ones((batch, 6), dtype=jnp.float32),
                    gpu,
                )
                split_output = runtime.run_suffix(boundary_gpu)
                assert split_output.device == gpu
                assert bool(
                    jnp.allclose(
                        split_output,
                        model(params_gpu, replay_x_gpu),
                    )
                )
        """,
        timeout=240,
    )


def test_tinygrad_conv_and_transformerish_models_split_replay() -> None:
    """tinygrad Conv and embedding/LayerNorm models replay representative splits."""

    _skip_unless_enabled()
    _skip_if_module_missing("tinygrad")
    _run_backend_subprocess(
        """
        from tinygrad import Tensor, dtypes
        import tinygrad.nn as nn

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
        for candidate in ("CUDA", "GPU", "NV"):
            try:
                Tensor.ones(1, device=candidate).realize()
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
                runtime = tl.prepare_split(
                    conv_model,
                    x,
                    tl.SplitSpec(boundary, backend="tinygrad"),
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
            runtime = tl.prepare_split(
                token_model,
                tokens,
                tl.SplitSpec(boundary, backend="tinygrad"),
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
        env_overrides={"DEV": "CPU"},
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
            runtime = tl.prepare_split(model, x, tl.SplitSpec(boundary, backend="tinygrad"))
            replay_x = Tensor.ones(2, 4, 8).realize()
            diff = max_abs_diff(
                runtime.replay(replay_x).realize(),
                model(replay_x).realize(),
            )
            assert diff < 1e-4
        """,
        timeout=180,
        env_overrides={"DEV": "CPU"},
    )


def test_tinygrad_nn_mlp_split_replay_dynamic_batch_and_train() -> None:
    """tinygrad nn.Linear MLP exercises real layer params and split training."""

    _skip_unless_enabled()
    _skip_if_module_missing("tinygrad")
    _run_backend_subprocess(
        """
        from tinygrad import Tensor
        import tinygrad.nn as nn

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
        runtime = tl.prepare_split(
            model,
            x,
            tl.SplitSpec("25%", backend="tinygrad", dynamic_batch=(1, 4)),
        )
        for batch in (1, 2, 4):
            replay_x = Tensor.ones(batch, 2, 3).realize()
            diff = max_abs_diff(runtime.replay(replay_x).realize(), model(replay_x).realize())
            assert diff < 1e-5

        train_x = Tensor.ones(2, 2, 3).realize()
        train_x.requires_grad = True
        train_runtime = tl.prepare_split(
            model,
            train_x,
            tl.SplitSpec("25%", backend="tinygrad", dynamic_batch=(1, 4), trainable=True),
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

        import torchlens as tl

        suffix_device = None
        for candidate in ("CUDA", "GPU", "NV"):
            try:
                Tensor.ones(1, device=candidate).realize()
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
            seed_runtime = tl.prepare_split(
                model,
                trace_x,
                tl.SplitSpec("50%", backend="tinygrad", trainable=True, dynamic_batch=(1, 3)),
            )
            split_id = next(
                node.canonical_id
                for node in seed_runtime.trace_graph.compute_nodes
                if node.op_type == "add" and node.module_path == "fc1"
            )
            runtime = tl.prepare_split(
                model,
                trace_x,
                tl.SplitSpec(
                    f"after:{split_id}",
                    backend="tinygrad",
                    trainable=True,
                    dynamic_batch=(1, 3),
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
            try:
                Tensor.ones(1, device=candidate).realize()
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
        seed_runtime = tl.prepare_split(
            model,
            x,
            tl.SplitSpec("50%", backend="tinygrad", dynamic_batch=(1, 3)),
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
                seed_runtime = tl.prepare_split(
                    model,
                    x,
                    tl.SplitSpec("50%", backend="tinygrad", dynamic_batch=(1, 3)),
                )
                candidates = [
                    node
                    for node in seed_runtime.trace_graph.compute_nodes
                    if node.op_type in op_types
                ]
                target = candidates[candidate_index]
                runtime = tl.prepare_split(
                    model,
                    x,
                    tl.SplitSpec(
                        f"{kind}:{target.canonical_id}",
                        backend="tinygrad",
                        dynamic_batch=(1, 3),
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
        """,
        timeout=300,
        env_overrides={"DEV": "CPU"},
    )
