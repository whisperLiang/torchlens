"""GC and memory leak tests for TorchLens.

Verifies that Trace, Op, Param, and model parameters
are garbage-collectible after use / cleanup.

These are marked ``smoke``: a lifetime regression is invisible to call-count,
``tracemalloc``, and wall-clock gates, so this file is the only per-step gate
that can see one. It is fast (~5 s) and it has caught the same module-global
cache root twice now.
"""

import gc
import tracemalloc
import weakref

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import trace as trace_fn
from torchlens._io import FieldPolicy
from torchlens.data_classes._trace_accessors import _TRACE_MODULE_CALL_ACCESSOR_ATTR
from torchlens.data_classes.trace import Trace

pytestmark = pytest.mark.smoke


# ---------------------------------------------------------------------------
# Test models
# ---------------------------------------------------------------------------


class _SimpleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 3)

    def forward(self, x):
        return self.fc(x)


class _TwoLayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 4)
        self.fc2 = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTraceGC:
    def test_trace_gc_without_cleanup(self):
        """del trace; gc.collect() should release the Trace."""
        model = _SimpleLinear()
        trace = tl.trace(model, torch.randn(1, 5))
        ref = weakref.ref(trace)
        del trace
        gc.collect()
        assert ref() is None

    def test_trace_gc_with_cleanup(self):
        """cleanup() + del + gc.collect() should release the Trace."""
        model = _SimpleLinear()
        trace = tl.trace(model, torch.randn(1, 5))
        ref = weakref.ref(trace)
        trace.cleanup()
        del trace
        gc.collect()
        assert ref() is None

    def test_model_params_not_pinned_after_cleanup(self):
        """After cleanup + del trace, model params should be GC-able."""
        model = _SimpleLinear()
        param_ref = weakref.ref(list(model.parameters())[0])
        trace = tl.trace(model, torch.randn(1, 5))
        trace.cleanup()
        del trace
        del model
        gc.collect()
        assert param_ref() is None

    def test_fast_live_hooks_finalize_when_trace_is_deleted(self):
        """Deleting a fast Trace removes its hooks and leaves parameters collectible."""

        model = _TwoLayerNet().eval()
        trace = tl.trace(model, torch.randn(1, 5), save=tl.module("fc1"))
        trace.run(inputs=torch.randn(1, 5), fast=True)
        trace_ref = weakref.ref(trace)
        param_ref = weakref.ref(next(model.parameters()))
        assert model.fc1._forward_hooks

        del trace
        gc.collect()

        assert trace_ref() is None
        assert not model.fc1._forward_hooks
        del model
        gc.collect()
        assert param_ref() is None

    def test_model_gc_after_release_param_refs(self):
        """release_param_refs() then del model -> model GC'd while log alive."""
        model = _TwoLayerNet()
        model_ref = weakref.ref(model)
        trace = tl.trace(model, torch.randn(1, 5))
        trace.release_param_refs()
        del model
        gc.collect()
        assert model_ref() is None
        # trace is still usable
        assert len(trace) > 0
        trace.cleanup()

    def test_no_memory_growth_across_sessions(self):
        """5x trace + del should not leak memory."""
        model = _TwoLayerNet()
        x = torch.randn(1, 5)
        # Warm up
        ml = trace_fn(model, x)
        ml.cleanup()
        del ml
        gc.collect()

        tracemalloc.start()
        baseline = tracemalloc.take_snapshot()

        for _ in range(5):
            ml = trace_fn(model, x)
            ml.cleanup()
            del ml
            gc.collect()

        after = tracemalloc.take_snapshot()
        tracemalloc.stop()

        # Compare: filter to torchlens allocations
        stats = after.compare_to(baseline, "lineno")
        tl_growth = sum(s.size_diff for s in stats if "torchlens" in str(s.traceback))
        # Allow up to 256KB of noise (caches, interned strings, etc.)
        assert tl_growth < 256 * 1024, f"Memory grew by {tl_growth} bytes across 5 sessions"

    def test_save_new_outs_no_leak(self):
        """5x save_new_outs should not leak memory."""
        model = _TwoLayerNet()
        x = torch.randn(1, 5)
        # Two-pass path: exhaustive first, then fast via save_new_outs
        trace = tl.trace(model, x, layers_to_save=None)

        # Warm up
        trace.save_new_outs(model, torch.randn(1, 5), layers_to_save="all")
        gc.collect()

        tracemalloc.start()
        baseline = tracemalloc.take_snapshot()

        for _ in range(5):
            trace.save_new_outs(model, torch.randn(1, 5), layers_to_save="all")
            gc.collect()

        after = tracemalloc.take_snapshot()
        tracemalloc.stop()

        stats = after.compare_to(baseline, "lineno")
        tl_growth = sum(s.size_diff for s in stats if "torchlens" in str(s.traceback))
        assert tl_growth < 256 * 1024, f"Memory grew by {tl_growth} bytes across 5 save_new_outs"
        trace.cleanup()

    def test_cleanup_breaks_param_ref(self):
        """After cleanup, all Param._param_ref should be None."""
        model = _TwoLayerNet()
        trace = tl.trace(model, torch.randn(1, 5))
        param_logs = list(trace.param_logs)
        trace.cleanup()
        for pl in param_logs:
            assert pl._param_ref is None

    def test_release_param_refs_preserves_grad_metadata(self):
        """backward(), release_param_refs(), verify grad info is cached."""
        model = _TwoLayerNet()
        x = torch.randn(1, 5)
        trace = tl.trace(model, x, save_grads=True)
        # Run backward to populate grads
        out = model(x)
        out.sum().backward()
        # Access grad metadata to cache it
        for pl in trace.param_logs:
            pl._check_param_grad()
        # Now release
        trace.release_param_refs()
        # Grad metadata should still be accessible
        has_any_grad = False
        for pl in trace.param_logs:
            assert pl._param_ref is None
            if pl._has_grad:
                has_any_grad = True
                assert pl._grad_shape is not None
                assert pl._grad_dtype is not None
                assert pl._grad_memory > 0
        assert has_any_grad, "Expected at least one param to have grad metadata cached"
        trace.cleanup()

    def test_transient_data_cleared(self):
        """Verify module build scratch is removed after postprocess."""
        model = _TwoLayerNet()
        trace = tl.trace(model, torch.randn(1, 5))
        assert "_build_state" not in trace.__dict__
        assert "_raw_graph_ws" not in trace.__dict__
        assert "_module_capture_ws" not in trace.__dict__
        assert "_wrapper_runtime_ws" not in trace.__dict__
        assert not hasattr(trace, "_module_build_data")
        assert not hasattr(trace, "_module_metadata")
        assert not hasattr(trace, "_module_forward_args")
        trace.cleanup()

    def test_raw_layer_dict_cleared_after_cleanup(self):
        """Verify raw layer scratch is absent after postprocess and cleanup."""
        model = _TwoLayerNet()
        trace = tl.trace(model, torch.randn(1, 5))
        assert "_build_state" not in trace.__dict__
        assert "_raw_graph_ws" not in trace.__dict__
        assert "_module_capture_ws" not in trace.__dict__
        assert "_wrapper_runtime_ws" not in trace.__dict__
        assert not hasattr(trace, "_raw_layer_dict")
        trace.cleanup()
        assert not hasattr(trace, "_raw_layer_dict")

    def test_module_calls_accessor_is_cached_on_the_instance(self):
        """The flattened ModuleCall accessor memo lives on the Trace, not a global.

        A module-global cache keyed by the Trace cannot hold this value: the
        accessor holds ModuleCalls and ``ModuleCall._source_trace`` keeps a
        strong reference back to the Trace, so even a ``WeakKeyDictionary``
        value would reach its own key and pin every Trace forever.
        """

        model = _TwoLayerNet()
        trace = tl.trace(model, torch.randn(1, 5))
        accessor = trace.module_calls

        assert trace.__dict__[_TRACE_MODULE_CALL_ACCESSOR_ATTR] is accessor
        assert trace.module_calls is accessor
        assert Trace.PORTABLE_STATE_SPEC[_TRACE_MODULE_CALL_ACCESSOR_ATTR] is FieldPolicy.DROP

    def test_populated_module_call_accessor_does_not_pin_trace(self):
        """Reading ``module_calls`` must not make the Trace immortal.

        Every capture populates this accessor internally through the
        saved-summary refresh, so a root here leaks on the default
        ``tl.trace()`` path with no user API call at all.
        """

        model = _TwoLayerNet()
        refs = []
        for _ in range(3):
            trace = tl.trace(model, torch.randn(1, 5))
            assert len(trace.module_calls) > 0
            refs.append(weakref.ref(trace))
            del trace
            gc.collect()
        assert [ref() for ref in refs] == [None, None, None]

    def test_held_module_call_still_keeps_its_trace_alive(self):
        """The intentional ModuleCall -> Trace ownership edge survives the fix."""

        model = _TwoLayerNet()
        trace = tl.trace(model, torch.randn(1, 5))
        module_call = trace.module_calls[0]
        ref = weakref.ref(trace)

        del trace
        gc.collect()
        assert ref() is not None, "holding a ModuleCall must keep its Trace alive"
        assert module_call.trace is ref()

        del module_call
        gc.collect()
        assert ref() is None

    def test_last_captures_model_class_is_not_pinned_after_the_epilogue(self):
        """A dynamically created module class dies with its last capture.

        ``_module_class_metadata_cache`` is a plain dict keyed by the module
        CLASS (and stores the class again in its value), and it was only ever
        cleared at the START of the next capture. A process whose final capture
        used a generated / function-local class therefore kept that class, its
        code objects and its closure alive for the whole process lifetime.
        """

        def build_class():
            """Return a fresh module class defined in this call's scope."""

            class _Generated(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(4, 3)

                def forward(self, x):
                    return torch.relu(self.fc(x))

            return _Generated

        generated = build_class()
        class_ref = weakref.ref(generated)
        model = generated()
        trace = tl.trace(model, torch.randn(2, 4))

        del trace, model, generated
        gc.collect()

        assert class_ref() is None, (
            "the last capture's module class is still pinned after the capture "
            "epilogue (class-metadata cache not released)"
        )

    def test_failed_capture_registry_does_not_pin_a_dead_exception(self):
        """The partial-recovery fallback table holds its exception weakly.

        The table is keyed by ``id(exception)`` and was capped at 128 ENTRIES
        with no byte bound and no time eviction — but each retained exception
        keeps its ``__traceback__``, and that pins every frame local (the model,
        the inputs, the partial outputs). Recovery only ever reaches an entry
        through ``from_failed_capture(exc)``, which requires the caller to hold
        that exception, so once it dies the entry is unreachable garbage.
        """

        from torchlens import partial as partial_module

        class _FrameLocalMarker:
            """Stands in for the model/inputs a traceback frame keeps alive."""

        class _RejectsAttachment(Exception):
            """The realistic shape that reaches this fallback at all.

            Builtin exceptions and ``__slots__`` subclasses both ACCEPT
            ``exc.partial_log = ...`` (BaseException always carries a dict), so
            the registry is only ever reached by types that refuse attribute
            assignment outright — which are user-defined and weak-referenceable.
            """

            def __setattr__(self, name, value):
                raise AttributeError("read-only exception")

        def raise_holding_a_local():
            """Raise an exception whose traceback frame holds a marker object."""

            marker = _FrameLocalMarker()
            marker_ref = weakref.ref(marker)
            try:
                raise _RejectsAttachment("registry retention probe")
            except _RejectsAttachment as error:
                return error, marker_ref

        trace = tl.trace(_SimpleLinear(), torch.randn(1, 5))
        exception, marker_ref = raise_holding_a_local()
        partial_log = partial_module.PartialTrace(trace=trace, original_exception=exception)
        key = id(exception)
        partial_module._register_failed_capture(exception, partial_log)

        # Recovery works for as long as the caller holds the exception, and
        # repeated lookups hand back the same wrapper.
        recovered = partial_module.from_failed_capture(exception)
        assert recovered.original_exception is exception
        assert recovered.trace is trace
        assert partial_module.from_failed_capture(exception) is recovered
        assert key in partial_module._FAILED_CAPTURE_REGISTRY

        del exception, partial_log, recovered
        gc.collect()

        assert key not in partial_module._FAILED_CAPTURE_REGISTRY, (
            "the registry kept an entry for a collected exception"
        )
        assert marker_ref() is None, (
            "the registry pinned the dead exception's traceback frame locals"
        )

    def test_failed_capture_registry_falls_back_for_unweakrefable_exceptions(self):
        """A non-weak-referenceable exception still recovers, under the entry cap.

        Builtin exception instances cannot be weak-referenced. They never reach
        this table (they accept ``partial_log`` attachment), but a C-extension
        exception type could, so the fallback must keep working rather than
        losing recovery.
        """

        from torchlens import partial as partial_module

        trace = tl.trace(_SimpleLinear(), torch.randn(1, 5))
        exception = RuntimeError("unweakrefable probe")
        with pytest.raises(TypeError):
            weakref.ref(exception)  # the precondition this arm exists for
        partial_log = partial_module.PartialTrace(trace=trace, original_exception=exception)
        partial_module._register_failed_capture(exception, partial_log)
        try:
            assert partial_module.from_failed_capture(exception) is partial_log
            assert len(partial_module._FAILED_CAPTURE_REGISTRY) <= (
                partial_module._FAILED_CAPTURE_REGISTRY_LIMIT
            )
        finally:
            partial_module._FAILED_CAPTURE_REGISTRY.pop(id(exception), None)

    def test_transient_write_after_finish_does_not_recreate_build_state(self) -> None:
        """Finished traces reject writes after the build-state owner is dropped."""

        model = _TwoLayerNet()
        trace = tl.trace(model, torch.randn(1, 5))

        with pytest.raises(AttributeError):
            trace._module_capture_ws.mod_call_index = {"x": 1}

        assert "_build_state" not in trace.__dict__
        assert "_raw_graph_ws" not in trace.__dict__
        assert "_module_capture_ws" not in trace.__dict__
        assert "_wrapper_runtime_ws" not in trace.__dict__


def test_delattr_capture_events_releases_the_working_projection():
    """``del trace._capture_events`` clears a held stream's working lanes.

    Regression: ``Trace.__delattr__`` popped the attribute FIRST and then
    called ``forget_event_stream``, which looks up the attribute it just
    removed -- so ``release_working_projection()`` never ran and an outside
    holder kept every op event alive.
    """

    trace = tl.trace(_TwoLayerNet(), torch.randn(2, 5))
    stream = trace._capture_events
    assert stream.op_events

    del trace._capture_events

    assert trace.__dict__.get("_capture_events") is None
    assert not stream.op_events
    assert not stream.module_prep_events
