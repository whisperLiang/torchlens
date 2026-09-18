"""Opt-in JAX real-world split replay tests."""

from __future__ import annotations

import pytest
from real_model_helpers import (
    _run_backend_subprocess,
    _skip_if_module_missing,
    _skip_unless_enabled,
)

pytestmark = [pytest.mark.slow, pytest.mark.real_model]


def test_jax_stax_cnn_split_replay_batch_symbolic_and_train() -> None:
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
        runtime = tl.split.prepare(
            model,
            (params, x),
            split_request("25%", backend="jax"),
        )
        for batch in (1, 2, 4):
            replay_x = jnp.ones((batch, 32, 32, 3), dtype=jnp.float32)
            assert bool(jnp.allclose(runtime.replay(params, replay_x), model(params, replay_x)))

        train_runtime = tl.split.prepare(
            model,
            (params, x),
            split_request("25%", backend="jax", trainable=True),
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
                seed_runtime = tl.split.prepare(
                    model,
                    traced_args,
                    split_request("50%", backend="jax"),
                )
            for boundary in all_boundaries(seed_runtime):
                with jax.default_device(cpu):
                    runtime = tl.split.prepare(
                        model,
                        traced_args,
                        split_request(boundary, backend="jax"),
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
        seed_runtime = tl.split.prepare(
            model,
            (params_cpu, trace_x),
            split_request("50%", backend="jax", trainable=True),
        )
        activation_id = next(
            node.canonical_id
            for node in seed_runtime.trace_graph.compute_nodes
            if node.op_type in {"max", "jax_region"}
        )
        runtime = tl.split.prepare(
            model,
            (params_cpu, trace_x),
            split_request(
                f"after:{activation_id}",
                backend="jax",
                trainable=True,
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


def test_jax_reference_resnet_transformer_and_vit_split_replay_batch_symbolic() -> None:
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
                runtime = tl.split.prepare(
                    model,
                    (params, x),
                    split_request(boundary, backend="jax"),
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


def test_jax_complete_multiscale_detector_all_nodes_cross_batch() -> None:
    """A complete Flax multiscale detector replays every JAX split boundary."""

    _skip_unless_enabled()
    _skip_if_module_missing("jax")
    _skip_if_module_missing("flax")
    _run_backend_subprocess(
        """
        import flax.linen as nn
        import jax
        import jax.numpy as jnp
        import torchlens as tl
        from torchlens.split import after, before

        class CompleteJaxDetector(nn.Module):
            @nn.compact
            def __call__(self, inputs):
                p1 = nn.relu(nn.Conv(8, (3, 3), padding="SAME")(inputs))
                p2 = nn.relu(nn.Conv(16, (3, 3), strides=(2, 2), padding="SAME")(p1))
                p3 = nn.relu(nn.Conv(24, (3, 3), strides=(2, 2), padding="SAME")(p2))
                p1_down = nn.Conv(16, (3, 3), strides=(2, 2), padding="SAME")(p1)
                n2 = nn.relu(nn.Conv(16, (1, 1))(jnp.concatenate([p1_down, p2], axis=-1)))
                n2_down = nn.Conv(24, (3, 3), strides=(2, 2), padding="SAME")(n2)
                n3 = nn.relu(nn.Conv(24, (1, 1))(jnp.concatenate([n2_down, p3], axis=-1)))

                def flatten_scale(value):
                    return value.reshape((value.shape[0], -1, 8))

                predictions = jnp.concatenate(
                    [
                        flatten_scale(nn.Conv(8, (1, 1))(p1)),
                        flatten_scale(nn.Conv(8, (1, 1))(n2)),
                        flatten_scale(nn.Conv(8, (1, 1))(n3)),
                    ],
                    axis=1,
                )
                return {
                    "boxes": predictions[..., :4],
                    "classes": jax.nn.sigmoid(predictions[..., 4:]),
                }

        module = CompleteJaxDetector()
        trace_x = jnp.ones((1, 32, 32, 3), dtype=jnp.float32)
        params = module.init(jax.random.PRNGKey(0), trace_x)

        def model(params, inputs):
            return module.apply(params, inputs)

        def assert_same(actual, expected):
            for key in expected:
                if actual[key].shape != expected[key].shape:
                    raise AssertionError(
                        f"JAX detector shape mismatch {key=} {actual[key].shape=} "
                        f"{expected[key].shape=}"
                    )
                if not bool(jnp.allclose(actual[key], expected[key], atol=1e-4, rtol=1e-4)):
                    diff = float(jnp.max(jnp.abs(actual[key] - expected[key])))
                    raise AssertionError(f"JAX detector mismatch {key=} {diff=}")

        seed_runtime = tl.split.prepare(
            model,
            (params, trace_x),
            split_request("50%", backend="jax"),
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
            try:
                assert_same(runtime.replay(params, trace_x), model(params, trace_x))
            except AssertionError as exc:
                node = seed_runtime.trace_graph.node_by_id[node_id]
                raise AssertionError(
                    f"JAX detector boundary failed {kind=} {node_id=} {node.label=}: {exc}"
                ) from None

        for batch in (2, 3):
            replay_x = jnp.ones((batch, 32, 32, 3), dtype=jnp.float32)
            runtime = seed_runtime.at(
                split_request("50%", backend="jax").point
            )
            assert_same(runtime.replay(params, replay_x), model(params, replay_x))
        """,
        timeout=600,
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
        seed_runtime = tl.split.prepare(
            model,
            (params_cpu, x_cpu),
            split_request("50%", backend="jax"),
        )
        boundaries = ("25%", "50%", "75%")
        for boundary in boundaries:
            runtime = tl.split.prepare(
                model,
                (params_cpu, x_cpu),
                split_request(boundary, backend="jax"),
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
