"""TensorFlow real-model native DAG replay tests."""

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
        from dag_helpers import close_call_ids

        from torchlens.split import SplitFeatures, prepare_graph

        tf.keras.utils.set_random_seed(0)
        model = tf.keras.applications.VGG16(input_shape=(32, 32, 3), weights=None, classes=10)
        x = tf.ones((2, 32, 32, 3), dtype=tf.float32)
        graph = prepare_graph(model, x, backend="tf")
        dag = graph.compute_dag()
        partition = dag.to_partition(
            close_call_ids(
                dag, frozenset((call.call_id for call in dag.nodes[: (len(dag.nodes) - 1) * 25 // 100 + 1]))
            )
        )
        runtime = graph.materialize(graph.analyze(partition))
        for batch in (1, 2, 4):
            replay_x = tf.ones((batch, 32, 32, 3), dtype=tf.float32)
            diff = tf.reduce_max(tf.abs(runtime.replay(replay_x) - model(replay_x)))
            assert float(diff.numpy()) < 1e-05
        train_graph = prepare_graph(model, x, backend="tf", features=SplitFeatures(training=True))
        train_dag = train_graph.compute_dag()
        train_partition = train_dag.to_partition(
            close_call_ids(
                train_dag,
                frozenset(
                    (call.call_id for call in train_dag.nodes[: (len(train_dag.nodes) - 1) * 75 // 100 + 1])
                ),
            )
        )
        train_runtime = train_graph.materialize(train_graph.analyze(train_partition))
        boundary = train_runtime.run_training_prefix(x)
        loss, grads = train_runtime.train_suffix(boundary, tf.zeros((2, 10), dtype=tf.float32))
        assert float(loss.numpy()) >= 0.0
        assert grads
        assert train_runtime.backward_prefix(boundary, grads)

        """,
        timeout=600,
    )


@pytest.mark.parametrize("model_name", ("MobileNetV2", "ResNet50", "DenseNet121"))
def test_tf_keras_application_matrix_split_replay_batch_symbolic(model_name: str) -> None:
    """Additional Keras application models replay representative split boundaries."""
    _skip_unless_enabled()
    _skip_if_module_missing("tensorflow")
    _run_backend_subprocess(
        """
        import os

        import tensorflow as tf
        from dag_helpers import smallest_transfer_partitions

        from torchlens.split import prepare_graph

        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
        model_cases = {
            "MobileNetV2": lambda: tf.keras.applications.MobileNetV2(
                input_shape=(32, 32, 3), alpha=0.35, weights=None, classes=10
            ),
            "ResNet50": lambda: tf.keras.applications.ResNet50(
                input_shape=(32, 32, 3), weights=None, classes=10
            ),
            "DenseNet121": lambda: tf.keras.applications.DenseNet121(
                input_shape=(32, 32, 3), weights=None, classes=10
            ),
        }
        tf.keras.utils.set_random_seed(0)
        model = model_cases[os.environ["TORCHLENS_TF_APPLICATION"]]()
        x = tf.ones((2, 32, 32, 3), dtype=tf.float32)
        seed_graph = prepare_graph(model, x, backend="tf")
        seed_dag = seed_graph.compute_dag()
        seed_partition = seed_dag.to_partition(())
        seed_runtime = seed_graph.materialize(seed_graph.analyze(seed_partition))
        replay_inputs = {batch: tf.ones((batch, 32, 32, 3), dtype=tf.float32) for batch in (1, 3)}
        expected = {batch: model(value) for batch, value in replay_inputs.items()}
        partitions = smallest_transfer_partitions(seed_runtime)
        checked = 0
        for partition in partitions:
            runtime = seed_runtime.repartition(partition)
            for batch, replay_x in replay_inputs.items():
                diff = tf.reduce_max(tf.abs(runtime.replay(replay_x) - expected[batch]))
                assert float(diff.numpy()) < 0.0001
                checked += 1
        assert checked == len(partitions) * 2

        """,
        timeout=1800,
        env_overrides={"TORCHLENS_TF_APPLICATION": model_name},
    )


def test_tf_keras_cv_yolov8_complete_model_smallest_dag_partitions_cross_batch_and_device() -> None:
    """Complete KerasCV YOLOv8 replay validates native DAG partitions and full outputs.

    This intentionally captures the detector root, not a backbone or a
    hand-built detection head.  The KerasCV preset includes its CSP backbone,
    PAN/FPN, three prediction scales, box/class branches, and output dict.
    ``SplitRuntime.at`` reuses that one complete capture while changing the
    boundary, so the smallest native DAG partitions are exercised.
    """
    _skip_unless_enabled()
    _skip_if_module_missing("tensorflow")
    _skip_if_module_missing("keras_cv")
    _run_backend_subprocess(
        """
        import hashlib
        import os
        from pathlib import Path

        import tensorflow as tf
        from dag_helpers import smallest_transfer_partitions

        from torchlens.split import GraphPartition, prepare_graph

        os.environ.setdefault(
            "KAGGLEHUB_CACHE", str(Path.home() / ".cache" / "torchlens" / "models" / "kagglehub")
        )
        import keras_cv

        tf.keras.utils.set_random_seed(0)
        official_model = keras_cv.models.YOLOV8Detector.from_preset("yolo_v8_m_pascalvoc")
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
                    tf.debugging.assert_near(actual_leaf, expected_leaf, atol=0.0002, rtol=0.0002)
                except tf.errors.InvalidArgumentError as exc:
                    raise AssertionError(
                        f"leaf {index} shape mismatch: actual={actual_leaf.shape}, expected={expected_leaf.shape}"
                    ) from exc


        trace_x = tf.ones((1, 32, 32, 3), dtype=tf.float32)
        official_graph = prepare_graph(official_model, trace_x, backend="tf")
        official_dag = official_graph.compute_dag()
        official_partition = official_dag.to_partition(())
        official_runtime = official_graph.materialize(official_graph.analyze(official_partition))
        for boundary in smallest_transfer_partitions(official_runtime):
            runtime = official_runtime.repartition(boundary)
            replay_x = tf.ones((1, 32, 32, 3), dtype=tf.float32)
            try:
                assert_same(runtime.replay(replay_x), official_model(replay_x))
            except AssertionError as exc:
                raise AssertionError(f"official YOLOv8 mismatch at {boundary}") from exc


        @tf.keras.utils.register_keras_serializable(package="TorchLensTests")
        class CompleteYOLOV8Detector(tf.keras.Model):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.stem = tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu")
                self.stage2 = tf.keras.layers.Conv2D(16, 3, strides=2, padding="same", activation="relu")
                self.stage3 = tf.keras.layers.Conv2D(24, 3, strides=2, padding="same", activation="relu")
                self.stage4 = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")
                self.up = tf.keras.layers.UpSampling2D(2)
                self.neck4 = tf.keras.layers.Conv2D(16, 1, padding="same", activation="relu")
                self.neck3 = tf.keras.layers.Conv2D(8, 1, padding="same", activation="relu")
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
                return {"boxes": predictions[..., :4], "classes": tf.sigmoid(predictions[..., 4:])}


        compact_model = CompleteYOLOV8Detector()
        compact_model.trainable = False
        compact_graph = prepare_graph(compact_model, trace_x, backend="tf")
        compact_dag = compact_graph.compute_dag()
        compact_partition = compact_dag.to_partition(())
        compact_runtime = compact_graph.materialize(compact_graph.analyze(compact_partition))
        boundaries = smallest_transfer_partitions(compact_runtime)
        assert boundaries
        for point in boundaries:
            runtime = compact_runtime.repartition(point)
            assert runtime.capability_report is not None
            assert runtime.capability_report.preflight_ok
            replay_x = tf.ones((1, 32, 32, 3), dtype=tf.float32)
            try:
                assert_same(runtime.replay(replay_x), compact_model(replay_x))
            except (AssertionError, tf.errors.InvalidArgumentError) as exc:
                raise AssertionError(f"full-model mismatch at {point!r}") from exc
        for boundary in smallest_transfer_partitions(compact_runtime):
            runtime = compact_runtime.repartition(boundary)
            for batch in (2, 3):
                replay_x = tf.ones((batch, 32, 32, 3), dtype=tf.float32)
                assert_same(runtime.replay(replay_x), compact_model(replay_x))
        gpus = tf.config.list_logical_devices("GPU")
        if gpus:
            runtime = compact_runtime.repartition(GraphPartition(frozenset()))
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
        from dag_helpers import smallest_transfer_partitions

        from torchlens.split import prepare_graph

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
                model = tf.keras.applications.Xception(input_shape=(75, 75, 3), weights=None, classes=10)
                x = tf.ones((2, 75, 75, 3), dtype=tf.float32)
                dag_graph_1 = prepare_graph(model, x, backend="tf")
                replay_inputs = {batch: tf.ones((batch, 75, 75, 3), dtype=tf.float32) for batch in (1, 3)}
                expected = {batch: model(value) for batch, value in replay_inputs.items()}
                partitions = smallest_transfer_partitions(dag_graph_1)
                checked = 0
                for partition in partitions:
                    runtime = dag_graph_1.materialize(dag_graph_1.analyze(partition))
                    for batch, replay_x in replay_inputs.items():
                        diff = tf.reduce_max(tf.abs(runtime.replay(replay_x) - expected[batch]))
                        assert float(diff.numpy()) < 0.0001
                        checked += 1
                assert checked == len(partitions) * 2

        """,
        timeout=1800,
    )


def test_tf_keras_cnn_smallest_dag_partitions_cross_batch_and_device() -> None:
    """The smallest native DAG partitions replay for a compact Keras CNN."""
    _skip_unless_enabled()
    _skip_if_module_missing("tensorflow")
    _run_backend_subprocess(
        """
        import tensorflow as tf
        from dag_helpers import smallest_transfer_partitions

        from torchlens.split import prepare_graph

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
                dag_graph_1 = prepare_graph(model, x, backend="tf")
                for boundary in smallest_transfer_partitions(dag_graph_1):
                    runtime = dag_graph_1.materialize(dag_graph_1.analyze(boundary))
                    for batch in (1, 3):
                        replay_x = tf.ones((batch, 8, 8, 3), dtype=tf.float32)
                        diff = tf.reduce_max(tf.abs(runtime.replay(replay_x) - model(replay_x)))
                        assert float(diff.numpy()) < 1e-05

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
        from dag_helpers import close_call_ids

        from torchlens.split import SplitFeatures, prepare_graph

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
                    self.w1 = tf.Variable(tf.reshape(tf.linspace(-0.3, 0.3, 20), (4, 5)), name="w1")
                    self.b1 = tf.Variable(tf.zeros((5,), dtype=tf.float32), name="b1")
                    self.w2 = tf.Variable(tf.reshape(tf.linspace(-0.2, 0.2, 15), (5, 3)), name="w2")
                    self.b2 = tf.Variable(tf.zeros((3,), dtype=tf.float32), name="b2")

            def __call__(self, x):
                hidden = tf.nn.relu(tf.matmul(x, self.w1) + self.b1)
                return tf.matmul(hidden, self.w2) + self.b2


        def assert_on_device(tensor, token):
            assert token in tensor.device, tensor.device


        with tf.device("/CPU:0"):
            model = TrainModule()
            trace_x = tf.ones((2, 4), dtype=tf.float32)
            seed_graph = prepare_graph(model, trace_x, backend="tf", features=SplitFeatures(training=True))
            seed_dag = seed_graph.compute_dag()
            seed_partition = seed_dag.to_partition(
                close_call_ids(
                    seed_dag,
                    frozenset(
                        (
                            call.call_id
                            for call in seed_dag.nodes[: (len(seed_dag.nodes) - 1) * 50 // 100 + 1]
                        )
                    ),
                )
            )
            seed_runtime = seed_graph.materialize(seed_graph.analyze(seed_partition))
            relu_id = next(
                (
                    node.canonical_id
                    for node in seed_runtime.trace_graph.compute_nodes
                    if node.op_type == "relu"
                )
            )
            graph = prepare_graph(model, trace_x, backend="tf", features=SplitFeatures(training=True))
            dag = graph.compute_dag()
            partition = dag.to_partition(
                close_call_ids(
                    dag, frozenset({graph.trace_graph.replay_call_by_output_id[relu_id].call_id})
                )
            )
            runtime = graph.materialize(graph.analyze(partition))
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
