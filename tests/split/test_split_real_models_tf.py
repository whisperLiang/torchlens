"""Opt-in TensorFlow real-world split replay tests."""

from __future__ import annotations

import pytest
from real_model_helpers import (
    _run_backend_subprocess,
    _skip_if_module_missing,
    _skip_unless_enabled,
)

pytestmark = [pytest.mark.slow, pytest.mark.real_model]


def test_tf_vgg16_split_replay_batch_symbolic_and_train() -> None:
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
        runtime = tl.split.prepare(
            model,
            x,
            split_request("25%", backend="tf"),
        )
        for batch in (1, 2, 4):
            replay_x = tf.ones((batch, 32, 32, 3), dtype=tf.float32)
            diff = tf.reduce_max(tf.abs(runtime.replay(replay_x) - model(replay_x)))
            assert float(diff.numpy()) < 1e-5

        train_runtime = tl.split.prepare(
            model,
            x,
            split_request("75%", backend="tf", trainable=True),
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
        # Each prepare now includes an isolated B=2 probe as well as the B=1
        # capture. VGG16's large dense weights are copied/serialized by Keras
        # in both inference and training preparations, exceeding the generic
        # 180-second budget on shared GPU hosts.
        timeout=600,
    )


@pytest.mark.parametrize(
    "model_name",
    ("MobileNetV2", "ResNet50", "DenseNet121"),
)
def test_tf_keras_application_matrix_split_replay_batch_symbolic(model_name: str) -> None:
    """Additional Keras application models replay representative split boundaries."""

    _skip_unless_enabled()
    _skip_if_module_missing("tensorflow")
    _run_backend_subprocess(
        """
        import os

        import tensorflow as tf
        import torchlens as tl

        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass

        model_cases = {
            "MobileNetV2": lambda: tf.keras.applications.MobileNetV2(
                    input_shape=(32, 32, 3),
                    alpha=0.35,
                    weights=None,
                    classes=10,
                ),
            "ResNet50": lambda: tf.keras.applications.ResNet50(
                    input_shape=(32, 32, 3),
                    weights=None,
                    classes=10,
                ),
            "DenseNet121": lambda: tf.keras.applications.DenseNet121(
                    input_shape=(32, 32, 3),
                    weights=None,
                    classes=10,
                ),
        }

        tf.keras.utils.set_random_seed(0)
        model = model_cases[os.environ["TORCHLENS_TF_APPLICATION"]]()
        x = tf.ones((2, 32, 32, 3), dtype=tf.float32)
        seed_runtime = tl.split.prepare(
            model,
            x,
            split_request("25%", backend="tf"),
        )
        for boundary in ("25%", "50%", "75%"):
            runtime = seed_runtime.at(split_request(boundary, backend="tf").point)
            for batch in (1, 3):
                replay_x = tf.ones((batch, 32, 32, 3), dtype=tf.float32)
                diff = tf.reduce_max(tf.abs(runtime.replay(replay_x) - model(replay_x)))
                assert float(diff.numpy()) < 1e-4
        """,
        timeout=600,
        env_overrides={"TORCHLENS_TF_APPLICATION": model_name},
    )


def test_tf_keras_cv_yolov8_complete_model_all_nodes_cross_batch_and_device() -> None:
    """Complete KerasCV YOLOv8 replay validates every node and full outputs.

    This intentionally captures the detector root, not a backbone or a
    hand-built detection head.  The KerasCV preset includes its CSP backbone,
    PAN/FPN, three prediction scales, box/class branches, and output dict.
    ``SplitRuntime.at`` reuses that one complete capture while changing the
    boundary, so every compute-node before/after point is exercised.
    """

    _skip_unless_enabled()
    _skip_if_module_missing("tensorflow")
    _skip_if_module_missing("keras_cv")
    _run_backend_subprocess(
        """
        import os
        import hashlib
        from pathlib import Path

        import tensorflow as tf

        os.environ.setdefault(
            "KAGGLEHUB_CACHE",
            str(Path.home() / ".cache" / "torchlens" / "models" / "kagglehub"),
        )
        import keras_cv

        import torchlens as tl
        from torchlens.split import after, before

        tf.keras.utils.set_random_seed(0)
        official_model = keras_cv.models.YOLOV8Detector.from_preset(
            "yolo_v8_m_pascalvoc",
        )
        weights_path = (
            Path(os.environ["KAGGLEHUB_CACHE"])
            / "models/keras/yolov8/keras/yolo_v8_m_pascalvoc/2/model.weights.h5"
        )
        digest = hashlib.sha256(weights_path.read_bytes()).hexdigest()
        assert digest == "6988dc0d736bc8dab04b82f9085a2314fb1ac6d90575cb47087b94df8cdf0741"  # pragma: allowlist secret
        official_model.trainable = False

        def leaves(value):
            if isinstance(value, tf.Tensor):
                return [value]
            if isinstance(value, dict):
                return [leaf for key in sorted(value) for leaf in leaves(value[key])]
            if isinstance(value, (list, tuple)):
                return [leaf for item in value for leaf in leaves(item)]
            return []

        def assert_same(actual, expected):
            actual_leaves = leaves(actual)
            expected_leaves = leaves(expected)
            assert len(actual_leaves) == len(expected_leaves)
            for index, (actual_leaf, expected_leaf) in enumerate(
                zip(actual_leaves, expected_leaves, strict=True)
            ):
                try:
                    tf.debugging.assert_near(actual_leaf, expected_leaf, atol=2e-4, rtol=2e-4)
                except tf.errors.InvalidArgumentError as exc:
                    raise AssertionError(
                        f"leaf {index} shape mismatch: "
                        f"actual={actual_leaf.shape}, expected={expected_leaf.shape}"
                    ) from exc

        trace_x = tf.ones((1, 32, 32, 3), dtype=tf.float32)

        # The pinned official detector covers real pretrained weights at
        # representative boundaries, dynamic batch, and device migration.
        official_runtime = tl.split.prepare(
            official_model,
            trace_x,
            split_request("50%", backend="tf"),
        )
        for boundary in ("25%", "50%", "75%"):
            runtime = official_runtime.at(split_request(boundary, backend="tf").point)
            replay_x = tf.ones((1, 32, 32, 3), dtype=tf.float32)
            try:
                assert_same(runtime.replay(replay_x), official_model(replay_x))
            except AssertionError as exc:
                raise AssertionError(f"official YOLOv8 mismatch at {boundary}") from exc

        # The B=2 probe clones Keras models to avoid mutating the caller's
        # variables. Supply Keras' normal reconstruction contract for this
        # local fixture, just as the official preset above does.
        @tf.keras.utils.register_keras_serializable(package="TorchLensTests")
        class CompleteYOLOV8Detector(tf.keras.Model):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.stem = tf.keras.layers.Conv2D(
                    8, 3, padding="same", activation="relu"
                )
                self.stage2 = tf.keras.layers.Conv2D(
                    16, 3, strides=2, padding="same", activation="relu"
                )
                self.stage3 = tf.keras.layers.Conv2D(
                    24, 3, strides=2, padding="same", activation="relu"
                )
                self.stage4 = tf.keras.layers.Conv2D(
                    32, 3, strides=2, padding="same", activation="relu"
                )
                self.up = tf.keras.layers.UpSampling2D(2)
                self.neck4 = tf.keras.layers.Conv2D(
                    16, 1, padding="same", activation="relu"
                )
                self.neck3 = tf.keras.layers.Conv2D(
                    8, 1, padding="same", activation="relu"
                )
                self.head3 = tf.keras.layers.Conv2D(8, 1)
                self.head4 = tf.keras.layers.Conv2D(8, 1)
                self.head5 = tf.keras.layers.Conv2D(8, 1)

            def call(self, inputs):
                p3 = self.stage2(self.stem(inputs))
                p4 = self.stage3(p3)
                p5 = self.stage4(p4)
                n4 = self.neck4(tf.concat([self.up(p5), p4], axis=-1))
                n3 = self.neck3(tf.concat([self.up(n4), p3], axis=-1))
                predictions = tf.concat(
                    [
                        tf.reshape(self.head3(n3), (tf.shape(n3)[0], -1, 8)),
                        tf.reshape(self.head4(n4), (tf.shape(n4)[0], -1, 8)),
                        tf.reshape(self.head5(p5), (tf.shape(p5)[0], -1, 8)),
                    ],
                    axis=1,
                )
                return {
                    "boxes": predictions[..., :4],
                    "classes": tf.sigmoid(predictions[..., 4:]),
                }

        # This is still a complete detector graph: stem, three-scale backbone,
        # PAN-style upsample/concat neck, three prediction heads, and a nested
        # box/class output.  It exists only to make every before/after node
        # contract affordable in CI; it is not a substitute for the official
        # weighted profile above.
        compact_model = CompleteYOLOV8Detector()
        compact_model.trainable = False
        compact_runtime = tl.split.prepare(
            compact_model,
            trace_x,
            split_request("50%", backend="tf"),
        )
        boundaries = [
            (kind, node.canonical_id)
            for node in compact_runtime.trace_graph.compute_nodes
            for kind in ("before", "after")
        ]
        assert boundaries

        # Full-model arbitrary-node contract: no boundary is reduced to a
        # backbone/head split, and both before/after forms are covered.
        for kind, node_id in boundaries:
            point = after(node_id) if kind == "after" else before(node_id)
            runtime = compact_runtime.at(point)
            assert runtime.capability_report is not None
            assert runtime.capability_report.preflight_ok
            replay_x = tf.ones((1, 32, 32, 3), dtype=tf.float32)
            try:
                assert_same(runtime.replay(replay_x), compact_model(replay_x))
            except (AssertionError, tf.errors.InvalidArgumentError) as exc:
                raise AssertionError(f"full-model mismatch at {kind}:{node_id}") from exc

        # The same complete detector is also batch-polymorphic at arbitrary
        # interior boundaries, not only at the default 50% point.
        for boundary in ("25%", "50%", "75%"):
            runtime = compact_runtime.at(split_request(boundary, backend="tf").point)
            for batch in (2, 3):
                replay_x = tf.ones((batch, 32, 32, 3), dtype=tf.float32)
                assert_same(runtime.replay(replay_x), compact_model(replay_x))

        gpus = tf.config.list_logical_devices("GPU")
        if gpus:
            runtime = compact_runtime.at(split_request("50%", backend="tf").point)
            cpu_x = tf.ones((2, 32, 32, 3), dtype=tf.float32)
            boundary_cpu = runtime.run_prefix(cpu_x)
            boundary_gpu = boundary_cpu.to(gpus[0].name, adapter=runtime.adapter)
            with tf.device(gpus[0].name):
                gpu_x = tf.ones((2, 32, 32, 3), dtype=tf.float32)
                assert_same(runtime.run_suffix(boundary_gpu), compact_model(gpu_x))
        """,
        timeout=1800,
    )


def test_tf_xception_split_replay_cross_batch_and_device() -> None:
    """Keras Xception replays representative boundaries on CPU and GPU."""

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

        devices = ["/CPU:0"]
        gpus = tf.config.list_logical_devices("GPU")
        if gpus:
            devices.append(gpus[0].name)

        for device in devices:
            with tf.device(device):
                tf.keras.utils.set_random_seed(0)
                model = tf.keras.applications.Xception(
                    input_shape=(75, 75, 3),
                    weights=None,
                    classes=10,
                )
                x = tf.ones((2, 75, 75, 3), dtype=tf.float32)
                for boundary in ("25%", "65%"):
                    runtime = tl.split.prepare(
                        model,
                        x,
                        split_request(boundary, backend="tf"),
                    )
                    for batch in (1, 3):
                        replay_x = tf.ones((batch, 75, 75, 3), dtype=tf.float32)
                        split_output = runtime.replay(replay_x)
                        full_output = model(replay_x)
                        diff = tf.reduce_max(tf.abs(split_output - full_output))
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
                seed_runtime = tl.split.prepare(
                    model,
                    x,
                    split_request("50%", backend="tf"),
                )
                boundaries = [
                    f"{kind}:{node.canonical_id}"
                    for node in seed_runtime.trace_graph.compute_nodes
                    for kind in ("before", "after")
                ]
                for boundary in boundaries:
                    runtime = tl.split.prepare(
                        model,
                        x,
                        split_request(boundary, backend="tf"),
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
            seed_runtime = tl.split.prepare(
                model,
                trace_x,
                split_request("50%", backend="tf", trainable=True),
            )
            relu_id = next(
                node.canonical_id
                for node in seed_runtime.trace_graph.compute_nodes
                if node.op_type == "relu"
            )
            runtime = tl.split.prepare(
                model,
                trace_x,
                split_request(
                    f"after:{relu_id}",
                    backend="tf",
                    trainable=True,
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
