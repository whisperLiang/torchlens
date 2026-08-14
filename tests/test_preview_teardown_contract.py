"""Preview-backend teardown contract: guaranteed cleanup + settle-after-cleanup.

R17-1/R50-1 (b5 parity lane): MLX and Paddle previously stamped COMPLETE/HALTED
inside the capture ``try`` and ran fallible teardown in ``finally`` -- a
teardown raise propagated with the stamp intact, and a raising session cleanup
skipped ``unwrap``, leaking process-global wrappers. The contract pinned here
(mirroring torch's path-7 no-undemoted-claim guarantee, via the stronger
settle-after-cleanup form the path-20 stamp docstring prescribes):

- registry ``unwrap()`` restores EVERY slot even when one setattr raises
  (first failure re-raises after the sweep; failed slots stay registered);
- paddle hook cleanup removes EVERY handle even when one ``remove()`` raises;
- capture epilogues settle AFTER teardown, so a teardown raise escapes
  productless (no COMPLETE stamp) with the wrappers uninstalled.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from torchlens.backends.mlx.wrappers import _MLXWrapperRegistry
from torchlens.backends.paddle.wrappers import _PaddleWrapperRegistry


class _RefusingOwner:
    """Owner whose attribute restore always raises."""

    def __setattr__(self, name: str, value: object) -> None:
        raise RuntimeError("setattr refused")


@pytest.mark.smoke
@pytest.mark.parametrize("registry_cls", [_MLXWrapperRegistry, _PaddleWrapperRegistry])
def test_registry_unwrap_sweeps_past_failing_setattr(registry_cls: type) -> None:
    """One fallible setattr cannot strand the remaining wrapper restores."""

    class _Owner:
        fn: Any = staticmethod(lambda: "wrapped")

    registry = registry_cls()
    refusing = _RefusingOwner()
    good_a = _Owner()
    good_b = _Owner()
    original_a = lambda: "a"  # noqa: E731
    original_b = lambda: "b"  # noqa: E731
    registry._originals = {
        (good_a, "fn"): original_a,
        (refusing, "fn"): lambda: "refused",
        (good_b, "fn"): original_b,
    }
    registry._wrapped = True

    with pytest.raises(RuntimeError, match="setattr refused"):
        registry.unwrap()

    # Both healthy slots restored despite the mid-sweep failure.
    assert good_a.fn is original_a
    assert good_b.fn is original_b
    # The failed slot stays registered so a retry can restore it.
    assert list(registry._originals) == [(refusing, "fn")]
    assert registry.is_wrapped() is True


@pytest.mark.smoke
def test_paddle_hook_cleanup_sweeps_past_failing_remove() -> None:
    """One raising ``remove()`` cannot strand later hook handles."""

    from torchlens.backends.paddle.model_prep import cleanup_model_session

    removed: list[str] = []

    def _make_handle(tag: str) -> SimpleNamespace:
        return SimpleNamespace(remove=lambda tag=tag: removed.append(tag))

    def _raise() -> None:
        raise RuntimeError("hook remove refused")

    tree = SimpleNamespace(
        hook_handles=[
            _make_handle("first"),
            SimpleNamespace(remove=_raise),
            _make_handle("last"),
        ]
    )

    with pytest.raises(RuntimeError, match="hook remove refused"):
        cleanup_model_session(None, None, tree)  # type: ignore[arg-type]

    assert removed == ["first", "last"]
    assert tree.hook_handles == []


@pytest.mark.optional
@pytest.mark.backend_mlx
def test_mlx_teardown_failure_escapes_productless(monkeypatch: pytest.MonkeyPatch) -> None:
    """A raising MLX session cleanup propagates with NO settled stamp and no leak."""

    pytest.importorskip("mlx")
    import mlx.core as mx

    import torchlens as tl
    from torchlens.backends.mlx import backend as mlx_backend_mod
    from torchlens.backends.mlx.wrappers import is_mlx_wrapped
    from torchlens.capture.outcome import outcome_for

    seen: dict[str, Any] = {}

    def _raising_cleanup(self: Any, session: Any, prepared_model: Any) -> None:
        seen["trace"] = session
        raise RuntimeError("injected teardown failure")

    monkeypatch.setattr(mlx_backend_mod.MLXBackend, "cleanup_model_session", _raising_cleanup)

    with pytest.raises(RuntimeError, match="injected teardown failure"):
        tl.trace(lambda x: x * 2 + 1, mx.array([1.0, 2.0]), backend="mlx")

    trace = seen["trace"]
    # Settlement is the LAST act: the teardown raise preempted the stamp, so
    # the escaped object carries NO COMPLETE claim (derives UNATTESTED).
    assert outcome_for(trace) is None
    # And the process-global wrappers still came off (unwrap ran in finally).
    assert is_mlx_wrapped() is False


@pytest.mark.optional
@pytest.mark.backend_paddle
def test_paddle_teardown_failure_escapes_productless(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raising Paddle hook cleanup propagates with NO settled stamp and no leak."""

    paddle = pytest.importorskip("paddle")

    import torchlens as tl
    from torchlens.backends.paddle import backend as paddle_backend_mod
    from torchlens.backends.paddle.wrappers import is_paddle_wrapped
    from torchlens.capture.outcome import outcome_for

    seen: dict[str, Any] = {}

    def _raising_cleanup(session: Any, prepared_model: Any, tree: Any) -> None:
        seen["trace"] = session
        raise RuntimeError("injected teardown failure")

    monkeypatch.setattr(paddle_backend_mod, "cleanup_model_session", _raising_cleanup)

    with pytest.raises(RuntimeError, match="injected teardown failure"):
        tl.trace(
            lambda x: paddle.nn.functional.relu(x + 1),
            paddle.to_tensor([1.0, -2.0]),
            backend="paddle",
        )

    trace = seen["trace"]
    assert outcome_for(trace) is None
    assert is_paddle_wrapped() is False
