"""Native oracles for identity-bound Paddle split parameters and buffers."""

from __future__ import annotations

import pytest
from _paddle_subprocess import run_paddle_subprocess

pytestmark = [pytest.mark.backend_paddle, pytest.mark.heavy]


@pytest.mark.parametrize("call_style", ["module", "operator", "positional", "keyword"])
def test_parameter_binding_matches_native_training_across_batches(call_style: str) -> None:
    """Bind actual operands despite unused state, equal shapes, and unentered child layers."""

    run_paddle_subprocess(
        """
        CALL_STYLE = __CALL_STYLE__

        import copy

        import paddle
        import torchlens as tl

        def assert_close(left, right):
            assert bool(paddle.allclose(left, right, atol=1e-5, rtol=1e-4).item())

        class Model(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.unused = paddle.nn.Linear(4, 4)
                self.first = paddle.nn.Linear(4, 4)
                self.second = paddle.nn.Linear(4, 4)

            def apply_layer(self, layer, x):
                if CALL_STYLE == 'operator':
                    return x @ layer.weight + layer.bias
                if CALL_STYLE == 'positional':
                    return paddle.nn.functional.linear(x, layer.weight, layer.bias)
                if CALL_STYLE == 'keyword':
                    return paddle.nn.functional.linear(
                        bias=layer.bias, x=x, weight=layer.weight
                    )
                return layer(x)

            def forward(self, x):
                hidden = paddle.nn.functional.relu(self.apply_layer(self.first, x))
                return self.apply_layer(self.second, hidden)

        devices = ['cpu']
        if paddle.device.cuda.device_count() > 0:
            devices.append('gpu:0')
        for device in devices:
            paddle.set_device(device)
            paddle.seed(123)
            model = Model()
            reference = copy.deepcopy(model)
            runtime = tl.split.prepare(
                model, paddle.ones([2, 4]),
                split_request('after:relu', backend='paddle', trainable=True),
            )
            assert runtime.trace_graph.shape_program.batch_probe.status == 'passed'
            for segment, expected in (
                (runtime.plan.prefix_node_ids, {'first.weight', 'first.bias'}),
                (runtime.plan.suffix_node_ids, {'second.weight', 'second.bias'}),
            ):
                consumed = {
                    param.address
                    for node in runtime.trace_graph.nodes
                    if node.canonical_id in segment
                    for param in node.param_refs
                }
                assert consumed == expected, (CALL_STYLE, consumed)
            prefix_opt = paddle.optimizer.SGD(0.02, parameters=model.first.parameters())
            suffix_opt = paddle.optimizer.SGD(0.02, parameters=model.second.parameters())
            native_opt = paddle.optimizer.SGD(0.02, parameters=reference.parameters())
            for batch in (1, 2, 4):
                x = paddle.arange(batch * 4, dtype='float32').reshape([batch, 4]) / 7 - 0.2
                native_x = x.clone()
                x.stop_gradient = native_x.stop_gradient = False
                targets = paddle.ones([batch, 4])
                assert_close(runtime.replay(x), reference(native_x))
                native_opt.clear_grad()
                native_loss = paddle.nn.functional.mse_loss(reference(native_x), targets)
                native_loss.backward()
                native_opt.step()
                boundary = runtime.run_training_prefix(x)
                split_loss, grads = runtime.train_suffix(boundary, targets, optimizer=suffix_opt)
                runtime.backward_prefix(boundary, grads, optimizer=prefix_opt)
                assert_close(split_loss, native_loss)
                assert_close(x.grad, native_x.grad)
                for name, value in model.state_dict().items():
                    assert_close(value, reference.state_dict()[name])
                assert all(value.grad is None for value in model.unused.parameters())
        """.replace("__CALL_STYLE__", repr(call_style))
    )


def test_repeated_and_reversed_operands_keep_exact_live_state() -> None:
    """Equal-shaped parameters and reused literals retain their actual identities after loads."""

    run_paddle_subprocess(
        """
        import paddle
        import torchlens as tl

        def assert_close(left, right):
            assert bool(paddle.allclose(left, right, atol=1e-5, rtol=1e-4).item())

        class Model(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.unused = self.create_parameter([4, 4])
                self.first = self.create_parameter([4, 4])
                self.second = self.create_parameter([4, 4])
                self.bias = self.create_parameter([4], is_bias=True)
                self.register_buffer('offset', paddle.arange(4, dtype='float32'))

            def forward(self, x):
                hidden = paddle.add(self.bias, x @ self.second)
                hidden = paddle.nn.functional.relu(hidden + self.offset)
                return (hidden @ self.first @ self.second) + paddle.add(self.bias, self.bias)

        devices = ['cpu']
        if paddle.device.cuda.device_count() > 0:
            devices.append('gpu:0')
        for device in devices:
            paddle.set_device(device)
            paddle.seed(12)
            model = Model()
            runtime = tl.split.prepare(
                model, paddle.ones([2, 4]), split_request('after:relu', backend='paddle'),
            )
            assert runtime.trace_graph.shape_program.batch_probe.status == 'passed'
            for shift in (0.0, 0.2):
                model.set_state_dict({name: value + shift for name, value in model.state_dict().items()})
                for updated in (runtime, runtime.at(tl.split.before('relu'))):
                    for batch in (1, 2, 4):
                        x = paddle.arange(batch * 4, dtype='float32').reshape([batch, 4]) / 7
                        assert_close(updated.replay(x), model(x))
        """
    )


def test_missing_state_provenance_refuses_instead_of_guessing_parameter() -> None:
    """A stripped live-state reference cannot be repaired by shape or module inventory."""

    run_paddle_subprocess(
        """
        import paddle
        import torchlens as tl
        from torchlens.split.errors import SplitUnsupportedError

        paddle.set_device('cpu')
        model = paddle.nn.Sequential(
            paddle.nn.Linear(4, 4), paddle.nn.ReLU(), paddle.nn.Linear(4, 4)
        )
        runtime = tl.split.prepare(
            model, paddle.ones([2, 4]), split_request('before:relu', backend='paddle'),
        )
        node = next(node for node in runtime.trace_graph.nodes if node.op_type == 'functional.linear')
        marker = node.kwargs_template['weight']
        assert marker['label'] is None
        marker.pop('value')
        try:
            runtime.run_prefix(paddle.ones([1, 4]))
        except SplitUnsupportedError as exc:
            assert exc.context.reason == 'unlabeled tensor template'
        else:
            raise AssertionError('missing tensor provenance was silently reconstructed')
        """
    )


def test_unwrapped_intermediate_never_becomes_a_historical_state_literal() -> None:
    """A capture gap cannot seed same-batch replay with a saved intermediate activation."""

    run_paddle_subprocess(
        """
        import paddle
        import torchlens as tl
        from torchlens.backends.paddle import wrappers
        from torchlens.split.errors import SplitUnsupportedError

        paddle.set_device('cpu')
        wrappers._TOP_LEVEL_CORE_OPS -= {'add'}
        wrappers._C_OPS_CORE_OPS -= {'add'}

        def model(x):
            hidden = paddle.add(x, paddle.ones_like(x))
            return paddle.nn.functional.relu(hidden) * 2

        runtime = tl.split.prepare(
            model, paddle.ones([2, 4]), split_request('after:relu', backend='paddle'),
        )
        try:
            runtime.replay(paddle.ones([1, 4]) * 5)
        except SplitUnsupportedError as exc:
            assert exc.context.reason == 'unlabeled tensor template'
        else:
            raise AssertionError('uncaptured intermediate became a stale activation literal')
        """
    )
