"""Trace intervention mixin."""

import copy
import copyreg
from dataclasses import fields
import os
from collections import OrderedDict
from functools import cached_property
from pathlib import Path
import time
import uuid
import weakref
import warnings
from typing import TYPE_CHECKING, Any, Set, cast

import torch
from torch import nn

if TYPE_CHECKING:
    from .trace import Trace

    _TraceMixinBase = Trace
else:
    _TraceMixinBase = object
from .. import _state
from .._deprecations import MISSING, MissingType
from .._trace_state import TraceState
from ..intervention.types import (
    ForkFieldPolicy,
    FrozenInterventionSpec,
    InterventionSpec,
    MODEL_LOG_FIELD_FORK_POLICY,
    TargetSpec,
)
from ..options import InterventionOptions, ReplayOptions, merge_intervention_options
from .layer import Layer, OpAccessor
from .op import Op
from ._state_adapter import state_items, state_new, state_restore

# Session-only backward projection guard/fold fields derived from the event
# stream. Any operation that replaces a trace's stream (pickle restore, fork)
# must drop these so the next projection access materializes from the new
# stream instead of trusting a guard computed over the old one.
_STREAM_DERIVED_GUARD_FIELDS = frozenset(
    {
        "_backward_projection_event_count",
        "_backward_projection_revision",
        "_backward_projection_fold_state",
        "_tl_materializing_backward_projection",
    }
)


class _ForkMemo(dict):
    """The single ``copy.deepcopy`` memo shared by every field copy of ONE fork.

    Forking used to give each field its own memo, which made the fork N
    independent structural copies instead of one. Two Trace fields carry a
    STRONG back-reference to their owning Trace -- ``ConditionalArm._trace`` and
    ``Module._source_trace`` -- so ``conditionals`` and ``_module_logs`` each
    re-cloned the entire object graph, on top of the separate clones made for
    ``layer_list``/``layer_dict_*``/``layer_logs``. On a GPT-2 capture that is
    roughly five redundant copies of the same ~120k-object graph per fork, and
    it also left the fork's own back-references pointing at orphan clones of the
    parent rather than at the fork.

    Sharing one memo (pre-seeded with parent-object -> fork-object identity)
    collapses that to one copy and makes the fork internally self-consistent. It
    can never introduce parent aliasing: every seeded mapping points at a
    fork-owned object, never at a parent-owned one.

    Rollback rides the dict's own insertion order: nothing ever deletes memo
    entries except ``rollback`` itself (which only pops the tail), so the keys
    inserted since a ``mark`` are exactly the keys past the marked length. That
    keeps the hot path -- one ``memo[id] = obj`` per object ``copy.deepcopy``
    visits -- a plain C-level ``dict.__setitem__`` with no Python-frame
    journaling hook, and a field copy that raises partway through can still
    roll its partially built entries back out. A half-constructed object must
    never survive in the memo to be reused by a later field.
    """

    __slots__ = ()

    def mark(self) -> int:
        """Return a rollback token for the current memo contents."""

        return len(self)

    def rollback(self, mark: int) -> None:
        """Drop every entry inserted since ``mark``.

        The parent-to-fork identity seeds are permanent: they are installed
        before the first field copy takes a mark, so their positions always
        precede every rollback token and no rollback can remove them.
        ``copy``'s keep-alive list entry is positioned like any other key, so
        rolling back never leaves a copied object whose source has been freed
        (and whose ``id`` could be recycled). Exception-only path (a deepcopy
        that raised), so the linear key scan is off the hot loop.
        """

        from itertools import islice

        for key in list(islice(self.keys(), mark, None)):
            del self[key]


def _seed_fork_memo(memo: _ForkMemo, mapping: dict[Any, Any]) -> None:
    """Install permanent identity seeds into ``memo``.

    Seeds are permanent because every seeding call precedes the first
    ``mark()`` a field copy takes, so no ``rollback`` position can reach them.
    """

    dict.update(memo, mapping)


def _deep_copy_fork_container(value: Any, memo: dict[Any, Any] | None = None) -> Any:
    """Deep-copy a container for a shallow fork, preserving tensor/callable identity.

    Recurses through dict/list/set/tuple structures so that mutable nested
    containers on the fork are independent of the parent, while tensors and
    callables (the large immutable payloads the shallow-fork path exists to avoid
    cloning) are still shared by reference.

    Parameters
    ----------
    value:
        Container (or leaf) value to copy.
    memo:
        Optional shared fork memo. Threading it through the generic-object
        fallback keeps a shallow-forked Op field from cloning the whole parent
        Trace when it happens to hold a back-reference to it.

    Returns
    -------
    Any
        A structurally independent copy sharing tensor/callable leaves.
    """

    if isinstance(value, torch.Tensor) or callable(value):
        return value
    if isinstance(value, dict):
        return {key: _deep_copy_fork_container(item, memo) for key, item in value.items()}
    if isinstance(value, list):
        return [_deep_copy_fork_container(item, memo) for item in value]
    if isinstance(value, set):
        return {_deep_copy_fork_container(item, memo) for item in value}
    if isinstance(value, tuple):
        return tuple(_deep_copy_fork_container(item, memo) for item in value)
    if isinstance(value, (str, bytes, int, float, bool, type(None))):
        return value
    return _memoized_deep_copy(value, memo, on_failure=copy.copy, fallback=value)


_PROPAGATE = object()


def _derive_deepcopy_atomic_types() -> frozenset:
    """Return the exact set of types ``copy.deepcopy`` treats as atomic.

    Derived from the running interpreter's own dispatch table, so the typed
    fast path below can never disagree with generic deepcopy about which types
    are returned uncopied. If the private table is unavailable, the fallback
    names only types that are atomic in every CPython release; a smaller set
    only shrinks the fast path, never changes copy semantics.
    """

    try:
        dispatch = copy._deepcopy_dispatch  # type: ignore[attr-defined]
        atomic_copier = copy._deepcopy_atomic  # type: ignore[attr-defined]
        derived = frozenset(cls for cls, fn in dispatch.items() if fn is atomic_copier)
        if str not in derived:
            derived = frozenset({type(None), int, float, bool, complex, bytes, str})
    except Exception:
        derived = frozenset({type(None), int, float, bool, complex, bytes, str})
    try:
        # ``torch.dtype`` instances are process-wide singletons that generic
        # deepcopy reduces back to themselves without memoizing, i.e. they
        # behave exactly like atomics. Verify that exhaustively against the
        # RUNNING torch before granting the fast path, so a future torch that
        # changes dtype pickling automatically loses it.
        dtypes = [d for d in vars(torch).values() if isinstance(d, torch.dtype)]
        if dtypes and all(copy.deepcopy(d) is d for d in dtypes):
            derived |= {torch.dtype}
    except Exception:
        pass
    return derived


_DEEPCOPY_ATOMIC_TYPES = _derive_deepcopy_atomic_types()
_NIL: Any = []
_OBJECT_REDUCE_EX = object.__reduce_ex__
_OBJECT_REDUCE = object.__reduce__
_OBJECT_GETSTATE = getattr(object, "__getstate__", None)
_PLAIN_COPY_CLASS_SAFE: dict[type, bool] = {}


def _probe_plain_object_protocol() -> bool:
    """Verify the interpreter's plain-object deepcopy contract once at import.

    The plain-object fast path in ``_typed_deep_copy`` replicates what generic
    deepcopy does for a hook-free instance: ``__reduce_ex__(4)`` returning
    ``(copyreg.__newobj__, (cls,), instance __dict__ or None, None, None)``,
    reconstructed as ``cls.__new__(cls)`` + memoize + deep-copied state dict.
    If a future interpreter changes any of that, this probe fails and the fast
    path disables itself rather than diverging.
    """

    class _Probe:
        pass

    try:
        inst = _Probe()
        inst.attr = [1]  # type: ignore[attr-defined]
        rv = inst.__reduce_ex__(4)
        if not (
            isinstance(rv, tuple)
            and len(rv) == 5
            and rv[0] is copyreg.__newobj__  # type: ignore[attr-defined]
            and rv[1] == (_Probe,)
            and rv[2] is inst.__dict__
            and rv[3] is None
            and rv[4] is None
        ):
            return False
        if _Probe().__reduce_ex__(4)[2] is not None:
            return False
        # Generic deepcopy must memoize the source instance AND its state dict
        # (the aliasing surface the fast path must reproduce).
        probe_memo: dict[Any, Any] = {}
        copy.deepcopy(inst, probe_memo)
        if id(inst) not in probe_memo or id(inst.__dict__) not in probe_memo:
            return False
    except Exception:
        return False
    return True


_PLAIN_COPY_ENABLED = _probe_plain_object_protocol()


def _plain_object_class_safe(cls: type) -> bool:
    """Structural scan gating the plain-object fast path for ``cls``.

    Anything that changes the shape of the default reduce protocol -- slots
    anywhere in the MRO, ``__getnewargs__``/``__getnewargs_ex__``, a custom
    ``__getstate__``/``__setstate__`` -- routes to generic deepcopy. The cheap
    per-call hooks (``__deepcopy__``, ``__reduce_ex__``, ``__reduce__``,
    ``copyreg.dispatch_table``) are re-checked on every copy, mirroring the
    lookups generic deepcopy itself performs.
    """

    try:
        if issubclass(
            cls,
            (list, tuple, dict, set, frozenset, str, bytes, bytearray, int, float, complex),
        ):
            # Builtin-base subclasses keep instance state outside __dict__
            # (list/dict subclasses additionally reduce with a list/dict
            # iterator); all of them stay on the generic path.
            return False
        if copyreg._slotnames(cls):  # type: ignore[attr-defined]
            return False
        for attr in ("__getnewargs_ex__", "__getnewargs__", "__setstate__"):
            if getattr(cls, attr, None) is not None:
                return False
        getstate = getattr(cls, "__getstate__", None)
        if getstate is not None and getstate is not _OBJECT_GETSTATE:
            return False
    except Exception:
        return False
    return True


def _memo_keep_alive(value: Any, memo: dict[Any, Any]) -> None:
    """Mirror ``copy._keep_alive``: pin ``value`` for the memo's lifetime.

    Without the pin, a memoized source object could be garbage collected while
    the memo is still in use and a new object recycling its ``id`` would
    wrongly hit its entry. The keep-alive list occupies a normal insertion
    position, so ``_ForkMemo.rollback`` handles it like any other key.
    """

    try:
        memo[id(memo)].append(value)
    except KeyError:
        memo[id(memo)] = [value]


def _typed_deep_copy(value: Any, memo: dict[Any, Any]) -> Any:
    """Deep-copy ``value`` with direct handling of the fork's common types.

    Most values the fork copies are atomics (or containers of atomics), and
    generic ``copy.deepcopy`` charges each one its full dispatch -- ``id()``,
    memo probe, type lookup -- just to return it unchanged. This copier answers
    atomics with one frozenset membership test and recurses through exact
    ``list``/``tuple``/``dict``/``set`` instances itself, replicating the memo
    and keep-alive protocol of CPython's ``_deepcopy_list``/``_deepcopy_tuple``
    /``_deepcopy_dict`` and the set reduce path byte-for-byte (including
    returning an all-atomic tuple by identity, exactly as ``_deepcopy_tuple``
    does). Instances of verified hook-free classes take the probed
    plain-object path below. Everything else -- container subclasses, objects
    with ``__deepcopy__``/``__reduce__``/state hooks/slots, tensors -- falls
    back to ``copy.deepcopy`` with the same shared memo, so aliasing across
    the typed/generic boundary is preserved.
    """

    cls = type(value)
    if cls in _DEEPCOPY_ATOMIC_TYPES:
        # deepcopy never memoizes atomics, so skipping its memo probe is safe.
        return value
    if cls is list:
        y = memo.get(id(value), _NIL)
        if y is not _NIL:
            return y
        copied_list: list[Any] = []
        memo[id(value)] = copied_list
        append = copied_list.append
        for item in value:
            append(_typed_deep_copy(item, memo))
        _memo_keep_alive(value, memo)
        return copied_list
    if cls is dict:
        y = memo.get(id(value), _NIL)
        if y is not _NIL:
            return y
        copied_dict: dict[Any, Any] = {}
        memo[id(value)] = copied_dict
        for key, item in value.items():
            copied_dict[_typed_deep_copy(key, memo)] = _typed_deep_copy(item, memo)
        _memo_keep_alive(value, memo)
        return copied_dict
    if cls is tuple:
        y = memo.get(id(value), _NIL)
        if y is not _NIL:
            return y
        copied_items = [_typed_deep_copy(item, memo) for item in value]
        # Copying the items may have reached this tuple again through a cycle.
        try:
            return memo[id(value)]
        except KeyError:
            pass
        for item, item_copy in zip(value, copied_items):
            if item is not item_copy:
                copied_tuple = tuple(copied_items)
                memo[id(value)] = copied_tuple
                _memo_keep_alive(value, memo)
                return copied_tuple
        # All items copied by identity: deepcopy returns the tuple itself,
        # unmemoized.
        return value
    if cls is set:
        y = memo.get(id(value), _NIL)
        if y is not _NIL:
            return y
        # Mirror the reduce path deepcopy takes for exact ``set``: elements are
        # copied BEFORE the new set exists or is memoized (a set cannot contain
        # itself, so pre-memoization is unreachable there), then the copy is
        # memoized and the source pinned. Only the reduce machinery's private
        # temp-list memo entry is skipped; nothing else can reference it.
        copied_set = {_typed_deep_copy(item, memo) for item in value}
        memo[id(value)] = copied_set
        _memo_keep_alive(value, memo)
        return copied_set
    y = memo.get(id(value), _NIL)
    if y is not _NIL:
        # Every non-atomic copy is memoized, so pre-seeded fork shells and
        # already-copied shared objects resolve here without paying a
        # ``copy.deepcopy`` frame.
        return y
    plain_dict: Any = _NIL
    if _PLAIN_COPY_ENABLED and not isinstance(value, type):
        try:
            instance_dict = value.__dict__
            if (
                type(instance_dict) is dict
                and "__deepcopy__" not in instance_dict
                and "__reduce_ex__" not in instance_dict
                and "__reduce__" not in instance_dict
                and getattr(cls, "__deepcopy__", None) is None
                and cls.__reduce_ex__ is _OBJECT_REDUCE_EX
                and cls.__reduce__ is _OBJECT_REDUCE
                and copyreg.dispatch_table.get(cls) is None
                and copy._deepcopy_dispatch.get(cls) is None  # type: ignore[attr-defined]
            ):
                safe = _PLAIN_COPY_CLASS_SAFE.get(cls)
                if safe is None:
                    safe = _plain_object_class_safe(cls)
                    _PLAIN_COPY_CLASS_SAFE[cls] = safe
                if safe:
                    plain_dict = instance_dict
        except Exception:
            plain_dict = _NIL
    if plain_dict is not _NIL:
        # Verified hook-free instance: replicate _reconstruct for the default
        # reduce -- construct, memoize, deep-copy the state dict, pin the
        # source. Exceptions from __new__ or the state copy propagate exactly
        # as they would from generic deepcopy.
        copied_obj = cls.__new__(cls)  # type: ignore[call-overload]
        memo[id(value)] = copied_obj
        if plain_dict:
            copied_obj.__dict__.update(_typed_deep_copy(plain_dict, memo))
        _memo_keep_alive(value, memo)
        return copied_obj
    # Verbatim port of ``copy.deepcopy``'s protocol branch (same lookups, same
    # order, same errors), except recursion routes back through this copier so
    # children of hooked/slotted objects keep the typed fast paths. A custom
    # ``__deepcopy__`` (e.g. torch.Tensor's) receives the same shared memo.
    copier = copy._deepcopy_dispatch.get(cls)  # type: ignore[attr-defined]
    if copier is not None:
        y = copier(value, memo)
    elif issubclass(cls, type):
        y = value
    else:
        deepcopy_hook = getattr(value, "__deepcopy__", None)
        if deepcopy_hook is not None:
            y = deepcopy_hook(memo)
        else:
            reductor = copyreg.dispatch_table.get(cls)
            if reductor:
                rv = reductor(value)
            else:
                reductor = getattr(value, "__reduce_ex__", None)
                if reductor is not None:
                    rv = reductor(4)
                else:
                    reductor = getattr(value, "__reduce__", None)
                    if reductor:
                        rv = reductor()
                    else:
                        raise copy.Error("un(deep)copyable object of type %s" % cls)
            if isinstance(rv, str):
                y = value
            else:
                y = _typed_reconstruct(value, memo, *rv)
    if y is not value:
        memo[id(value)] = y
        _memo_keep_alive(value, memo)
    return y


def _typed_reconstruct(
    x: Any,
    memo: dict[Any, Any],
    func: Any,
    args: Any,
    state: Any = None,
    listiter: Any = None,
    dictiter: Any = None,
) -> Any:
    """``copy._reconstruct`` specialized to deep copies through the typed copier.

    Line-for-line port of the stdlib reconstructor with ``deep = True`` (the
    fork memo always exists) and every recursive ``deepcopy`` call replaced by
    ``_typed_deep_copy``, which is equivalence-proven against it.
    """

    if args:
        args = (_typed_deep_copy(arg, memo) for arg in args)
    y = func(*args)
    memo[id(x)] = y

    if state is not None:
        state = _typed_deep_copy(state, memo)
        if hasattr(y, "__setstate__"):
            y.__setstate__(state)
        else:
            if isinstance(state, tuple) and len(state) == 2:
                state, slotstate = state
            else:
                slotstate = None
            if state is not None:
                y.__dict__.update(state)
            if slotstate is not None:
                for key, item in slotstate.items():
                    setattr(y, key, item)

    if listiter is not None:
        for item in listiter:
            y.append(_typed_deep_copy(item, memo))
    if dictiter is not None:
        for key, item in dictiter:
            y[_typed_deep_copy(key, memo)] = _typed_deep_copy(item, memo)
    return y


def _memoized_deep_copy(
    value: Any,
    memo: dict[Any, Any] | None,
    *,
    on_failure: Any,
    fallback: Any = _PROPAGATE,
) -> Any:
    """Deep-copy ``value`` under the shared fork memo, rolling back on failure.

    Parameters
    ----------
    value:
        Value to deep-copy.
    memo:
        Shared fork memo, or ``None`` for an isolated copy.
    on_failure:
        Callable applied to ``value`` when the deep copy raises.
    fallback:
        Value returned when ``on_failure`` also raises. Left at ``_PROPAGATE``
        the secondary failure is re-raised, matching the pre-memo behavior of
        each call site.

    Returns
    -------
    Any
        The deep copy, or the degraded fallback copy.
    """

    if memo is None:
        memo = _ForkMemo()
    mark = memo.mark() if isinstance(memo, _ForkMemo) else None
    try:
        return _typed_deep_copy(value, memo)
    except Exception:
        # A raised deepcopy can leave half-populated objects behind in the memo
        # (``copy._reconstruct`` memoizes before it restores state), so those
        # entries are discarded rather than handed to the next field.
        if mark is not None:
            cast(_ForkMemo, memo).rollback(mark)
        # The degradation to ``on_failure`` is intentional for opaque field
        # values, but it also silences genuine copy bugs (a field that MUST
        # fork independently silently becoming shared). The opt-in debug
        # channel re-raises so that failure class is never invisible.
        if os.environ.get("TORCHLENS_DEBUG_FORK_COPY"):
            raise
        if fallback is _PROPAGATE:
            return on_failure(value)
        try:
            return on_failure(value)
        except Exception:
            return fallback


class TraceInterventionMixin(_TraceMixinBase):
    def save_intervention(
        self: "Trace",
        path: str | Path,
        *,
        level: str = "executable_with_callables",
        allow_direct_writes: bool = False,
        overwrite: bool = False,
    ) -> None:
        """Save this log's intervention recipe to a ``.tlspec`` directory.

        Parameters
        ----------
        path:
            Destination ``.tlspec`` directory path.
        level:
            Save level: ``"audit"``, ``"executable_with_callables"``, or
            ``"portable"``.
        allow_direct_writes:
            Whether executable saves may proceed after direct out writes.
        overwrite:
            Whether an existing destination may be replaced.
        """

        from ..intervention.save import save_intervention
        from ..runnable import refuse_poisoned_trace

        refuse_poisoned_trace(self, "intervention export")
        save_intervention(
            self,
            path,
            level=level,
            allow_direct_writes=allow_direct_writes,
            overwrite=overwrite,
        )

    @cached_property
    def intervention_spec(self: "Trace") -> FrozenInterventionSpec:
        """Return an immutable snapshot of this log's intervention recipe.

        Returns
        -------
        FrozenInterventionSpec
            Frozen public view of the current mutable intervention spec.
        """

        return self._ensure_intervention_spec().freeze()

    def _history_site_payload(self: "Trace", site: Any) -> Any:
        """Return a stable site payload for ``state_history`` records.

        Parameters
        ----------
        site:
            Original selector-like site payload supplied to a mutator.

        Returns
        -------
        Any
            Plain layer labels for direct label targets, otherwise a stable
            repr-style string for human-readable history records.
        """

        del self
        if isinstance(site, str):
            return site
        selector_kind = getattr(site, "selector_kind", None)
        selector_value = getattr(site, "selector_value", None)
        if selector_kind == "label" and isinstance(selector_value, str):
            return selector_value
        return repr(site)

    def set(
        self: "Trace",
        site: Any,
        value: Any,
        *,
        direction: str = "forward",
        strict: bool = False,
        confirm_mutation: bool = False,
    ) -> "Trace":
        """Set a site out recipe without propagating it.

        Parameters
        ----------
        site:
            Selector-like target for the out to replace.
        value:
            Static replacement tensor or one-shot callable accepting the
            matched out and returning a replacement tensor.
        direction:
            Signal direction to replace: ``"forward"``, ``"backward"``, or
            ``"both"``.
        strict:
            Whether site resolution should reject non-portable selectors.
        confirm_mutation:
            Suppress the once-per-root mutate-in-place warning for callers that
            intentionally mutate this log.

        Returns
        -------
        Trace
            This model log, with a stale intervention recipe.
        """

        self._warn_if_root_mutation(confirm_mutation=confirm_mutation)
        if direction not in {"forward", "backward", "both"}:
            raise ValueError("set(..., direction=...) must be 'forward', 'backward', or 'both'.")
        if direction in {"backward", "both"}:

            def _backward_set_hook(grad: torch.Tensor, *, hook: Any) -> torch.Tensor:
                """Return a static or callable gradient replacement."""

                del hook
                if callable(value):
                    return cast(torch.Tensor, value(grad))
                return cast(torch.Tensor, value)

            self.attach_hooks(
                site,
                _backward_set_hook,
                direction="backward",
                strict=strict,
                confirm_mutation=True,
            )
            if direction == "backward":
                self._record_operation(
                    "set",
                    site=self._history_site_payload(site),
                    value_kind=type(value).__name__,
                    strict=strict,
                    callable=callable(value),
                    direction=direction,
                )
                return self
        from ..intervention.hooks import is_facet_target

        if is_facet_target(site):

            def _facet_replacement_hook(out: torch.Tensor, *, hook: Any) -> torch.Tensor:
                """Return a static or callable facet replacement slice.

                Parameters
                ----------
                out:
                    Facet slice supplied by the facet hook wrapper.
                hook:
                    Hook context supplied by TorchLens.

                Returns
                -------
                torch.Tensor
                    Replacement facet slice.
                """

                del hook
                if callable(value):
                    return cast(torch.Tensor, value(out))
                return cast(torch.Tensor, value)

            self.attach_hooks(
                site,
                _facet_replacement_hook,
                direction="forward",
                strict=strict,
                confirm_mutation=True,
            )
            self._record_operation(
                "set",
                site=self._history_site_payload(site),
                value_kind=type(value).__name__,
                strict=strict,
                callable=callable(value),
                facet_scatter=True,
                direction="forward",
            )
            return self
        self._validate_intervention_site(site, strict=strict)
        metadata = {"created_by": "set_callable_one_shot"} if callable(value) else {}
        self._ensure_intervention_spec().add_set(
            self._target_spec_from_site(site, strict=strict),
            value,
            metadata=metadata,
        )
        self._mark_intervention_spec_mutated()
        self._record_operation(
            "set",
            site=self._history_site_payload(site),
            value_kind=type(value).__name__,
            strict=strict,
            callable=callable(value),
            direction="forward",
        )
        return self

    def attach_hooks(
        self: "Trace",
        hooks_or_site: Any,
        hook: Any = None,
        *extra_hooks: Any,
        strict: bool = False,
        prepend: bool = False,
        confirm_mutation: bool = False,
        direction: str | None = None,
    ) -> Any:
        """Attach sticky hooks to the current intervention spec.

        Raw PyTorch ``register_forward_hook`` remains supported for users who
        need module-local replacement logic outside this API; during active
        TorchLens captures, returned replacement tensors are instrumented so the
        graph can continue through downstream ops.

        Parameters
        ----------
        hooks_or_site:
            Mapping/list batch input or selector-like site.
        hook:
            Optional hook for the ``(site, hook)`` input shape.
        *extra_hooks:
            Additional hooks to compose at ``hooks_or_site`` in left-to-right order.
        strict:
            Whether site resolution should reject non-portable selectors.
        prepend:
            Whether new sticky hooks should run before existing sticky hooks.
        confirm_mutation:
            Suppress the once-per-root mutate-in-place warning for callers that
            intentionally mutate this log.
        direction:
            Optional signal direction override: ``"forward"``, ``"backward"``,
            or ``"both"``.

        Returns
        -------
        Any
            Scoped removable hook handle.
        """

        self._warn_if_root_mutation(confirm_mutation=confirm_mutation)
        from ..intervention.errors import HookSignatureError
        from ..intervention.handles import HookHandle
        from ..intervention.hooks import normalize_hook_plan

        if direction is not None and direction not in {"forward", "backward", "both"}:
            raise ValueError(
                "attach_hooks(..., direction=...) must be 'forward', 'backward', or 'both'."
            )
        if extra_hooks:
            if hook is None:
                raise HookSignatureError("extra hooks require an initial hook argument.")
            entries = normalize_hook_plan(
                [(hooks_or_site, hook_like) for hook_like in (hook, *extra_hooks)],
                direction=cast(Any, direction),
            )
        else:
            entries = normalize_hook_plan(hooks_or_site, hook, direction=cast(Any, direction))
        from ..intervention.hooks import expand_facet_hook_entries

        entries = expand_facet_hook_entries(self, entries)
        for entry in entries:
            self._validate_intervention_site(entry.site_target, strict=strict)
        spec = self._ensure_intervention_spec()
        handle_ids: list[str] = []
        for entry in entries:
            handle_id = f"hook-{uuid.uuid4().hex}"
            handle_ids.append(handle_id)
            metadata = dict(entry.metadata)
            if metadata.get("facet_write"):
                # Facet-slice entries MUST store the scatter wrapper built by
                # expand_facet_hook_entries as the fire-time hook: storing the raw
                # helper would drop the wrapper, and rerun normalization would then
                # apply the helper to the whole home tensor instead of the selected
                # facet slice. The raw helper is kept in ``helper=`` as provenance.
                stored_hook: Any = entry.normalized_callable
            else:
                stored_hook = (
                    entry.helper_spec
                    if entry.helper_spec is not None
                    else entry.normalized_callable
                )
            spec.add_hook(
                self._target_spec_from_site(entry.site_target, strict=strict),
                stored_hook,
                helper=entry.helper_spec,
                handle=handle_id,
                metadata=metadata,
                prepend=prepend,
            )
        self._mark_intervention_spec_mutated()
        self._record_operation(
            "attach_hooks",
            hook_count=len(entries),
            sites=tuple(self._history_site_payload(entry.site_target) for entry in entries),
            strict=strict,
            prepend=prepend,
            handles=tuple(handle_ids),
            direction=direction,
        )
        self._last_hook_handle_ids = tuple(handle_ids)
        return HookHandle(self, tuple(handle_ids), confirm_mutation=confirm_mutation)

    def remove(self: "Trace") -> None:
        """Remove the most recent legacy-returned hook attachment.

        Returns
        -------
        None
            Hook specs attached by the last single-hook ``attach_hooks`` call
            are detached.
        """

        for handle_id in self._last_hook_handle_ids:
            self.detach_hooks(handle=handle_id, confirm_mutation=True)
        self._last_hook_handle_ids = ()

    def __enter__(self: "Trace") -> "Trace":
        """Enter a legacy scoped hook attachment.

        Returns
        -------
        Trace
            This log, acting as the most recent hook handle.
        """

        return self

    def __exit__(self: "Trace", exc_type: Any, exc: Any, traceback: Any) -> None:
        """Clean up a legacy scoped hook attachment.

        Parameters
        ----------
        exc_type:
            Exception type, if the body raised.
        exc:
            Exception value, if the body raised.
        traceback:
            Exception traceback, if the body raised.
        """

        self.remove()

    def detach_hooks(
        self: "Trace",
        site: Any = None,
        handle: Any = None,
        *,
        strict: bool = False,
        confirm_mutation: bool = False,
    ) -> "Trace":
        """Detach sticky hooks by site or handle.

        Parameters
        ----------
        site:
            Optional selector-like target. When provided, all sticky hooks for
            that target are removed.
        handle:
            Optional hook handle returned by ``attach_hooks``.
        strict:
            Whether no-op detach requests should raise.
        confirm_mutation:
            Suppress the once-per-root mutate-in-place warning for callers that
            intentionally mutate this log.

        Returns
        -------
        Trace
            This model log, with a stale recipe if hooks were removed.
        """

        self._warn_if_root_mutation(confirm_mutation=confirm_mutation)
        from ..intervention.errors import SpecMutationError

        if site is None and handle is None:
            if strict:
                raise SpecMutationError("detach_hooks requires a site or handle in strict mode.")
            return self

        target_spec = None
        if site is not None:
            self._validate_intervention_site(site, strict=strict)
            target_spec = self._target_spec_from_site(site, strict=strict)

        handle_values = (
            tuple(getattr(handle, "handle_ids", (str(handle),))) if handle is not None else (None,)
        )
        removed = 0
        for handle_value in handle_values:
            removed += self._ensure_intervention_spec().remove_hook(
                site_target=target_spec,
                handle=handle_value,
            )
        if removed == 0 and strict:
            raise SpecMutationError("detach_hooks did not match any sticky hooks.")
        if removed > 0:
            self._mark_intervention_spec_mutated()
        self._record_operation(
            "detach_hooks",
            site=self._history_site_payload(site) if site is not None else None,
            handle=str(handle) if handle is not None else None,
            removed=removed,
            strict=strict,
        )
        return self

    def clear_hooks(self: "Trace", *, confirm_mutation: bool = False) -> "Trace":
        """Clear all sticky hooks from the current intervention spec.

        Parameters
        ----------
        confirm_mutation:
            Suppress the once-per-root mutate-in-place warning for callers that
            intentionally mutate this log.

        Returns
        -------
        Trace
            This model log with hook specs cleared and marked stale.
        """

        self._warn_if_root_mutation(confirm_mutation=confirm_mutation)
        self._ensure_intervention_spec().clear()
        self._mark_intervention_spec_mutated()
        self._record_operation("clear_hooks")
        return self

    def do(
        self: "Trace",
        hooks_or_site: Any,
        value_or_hook: Any = None,
        *,
        model: nn.Module | None = None,
        x: Any = None,
        engine: str | MissingType = MISSING,
        confirm_mutation: bool | MissingType = MISSING,
        strict: bool | MissingType = MISSING,
        intervention: InterventionOptions | None = None,
        direction: str | None = None,
    ) -> "Trace":
        """Apply an intervention and dispatch to replay, rerun, or set-only.

        Parameters
        ----------
        hooks_or_site:
            Mapping/list batch input or selector-like site.
        value_or_hook:
            Optional hook for the ``(site, hook)`` input shape.
        model:
            Model required when ``engine="rerun"``.
        x:
            Input required when ``engine="rerun"``.
        engine:
            ``"auto"``, ``"replay"``, ``"rerun"``, or ``"set_only"``.
        confirm_mutation:
            Suppress the once-per-root mutate-in-place warning for callers that
            intentionally mutate this log.
        strict:
            Whether selector and propagation checks should raise.
        direction:
            Optional signal direction override for hook-style mutations.

        Returns
        -------
        Trace
            This model log after the selected propagation engine runs.
        """

        from ..intervention.errors import EngineDispatchError

        intervention_options = merge_intervention_options(
            intervention=intervention,
            engine=engine,
            confirm_mutation=confirm_mutation,
            strict=strict,
        )
        engine_value = intervention_options.engine
        confirm_mutation_value = intervention_options.confirm_mutation
        strict_value = intervention_options.strict

        if engine_value not in {"auto", "replay", "rerun", "set_only"}:
            raise ValueError(
                "do(..., engine=...) must be 'auto', 'replay', 'rerun', or 'set_only'."
            )

        selected_engine = self._select_do_engine(engine_value, model=model, x=x)
        if selected_engine == "rerun":
            if model is None:
                raise EngineDispatchError("do(..., engine='rerun') requires model= and x=.")
            self._validate_supplied_model_matches_capture(model)
        mutation_kind = self._apply_do_mutation(
            hooks_or_site,
            value_or_hook,
            engine=selected_engine,
            strict=strict_value,
            confirm_mutation=confirm_mutation_value,
            direction=direction,
        )
        self._record_operation(
            "do",
            mutation_kind=mutation_kind,
            engine=selected_engine,
            requested_engine=engine_value,
            model_supplied=model is not None,
            x_supplied=x is not None,
            strict=strict_value,
            direction=direction,
        )

        if selected_engine == "set_only":
            return self
        if selected_engine == "replay":
            return self.push(replay=ReplayOptions(strict=strict_value))
        assert model is not None
        return self.run(model, x, replay=ReplayOptions(strict=strict_value))

    def fork(self: "Trace", name: str | None = None) -> "Trace":
        """Create a copy-on-write intervention fork of this log.

        Parameters
        ----------
        name:
            Optional name for the forked log.

        Returns
        -------
        Trace
            Forked model log.
        """

        fork = self._fork_trace(name=name)
        self._record_operation("fork", source_id=id(self), name=fork.trace_label)
        return fork

    def _record_operation(self: "Trace", op: str, **payload: Any) -> None:
        """Append a structured operation record to ``state_history``.

        Parameters
        ----------
        op:
            Operation name.
        **payload:
            Operation-specific metadata.

        Returns
        -------
        None
            The history list is mutated in place.
        """

        self.state_history.append(
            {
                "op": op,
                "spec_revision": self._spec_revision,
                "timestamp": time.monotonic(),
                **payload,
            }
        )

    def _warn_if_root_mutation(self: "Trace", *, confirm_mutation: bool) -> None:
        """Emit the once-per-root mutate-in-place warning when appropriate.

        Parameters
        ----------
        confirm_mutation:
            Whether the caller explicitly accepted in-place mutation.
        """

        if confirm_mutation or self.parent_run is not None or self._warned_mutate_in_place:
            return
        from ..intervention.errors import MutateInPlaceWarning
        from ..options import suppress_mutate_warnings

        if suppress_mutate_warnings.is_suppressed:
            return
        warnings.warn(
            "MutateInPlaceWarning: Trace mutators modify root logs in place. "
            "Use log.fork(...) for isolated edits or pass confirm_mutation=True.",
            MutateInPlaceWarning,
            stacklevel=3,
        )
        self._warned_mutate_in_place = True

    def _select_do_engine(self: "Trace", engine: str, *, model: nn.Module | None, x: Any) -> str:
        """Resolve the concrete ``do`` engine from caller arguments.

        Parameters
        ----------
        engine:
            Requested engine name.
        model:
            Optional model supplied for rerun.
        x:
            Optional input supplied for rerun.

        Returns
        -------
        str
            Concrete engine name.

        Raises
        ------
        EngineDispatchError
            If the engine cannot be inferred from an incomplete model/input pair.
        """

        from ..intervention.errors import EngineDispatchError

        if engine != "auto":
            if engine == "rerun" and (model is None or x is None):
                raise EngineDispatchError(
                    "do(..., engine='rerun') requires both model= and x=. "
                    "Pass both, or use engine='replay' if full rerun is not intended."
                )
            return engine
        if (model is None) != (x is None):
            raise EngineDispatchError(
                "do(engine='auto') needs both model= and x= for rerun, or neither for "
                "replay. Pass both, or use engine='replay' if rerun is not intended."
            )
        return "rerun" if model is not None else "replay"

    def _apply_do_mutation(
        self: "Trace",
        hooks_or_site: Any,
        value_or_hook: Any,
        *,
        engine: str,
        strict: bool,
        confirm_mutation: bool,
        direction: str | None,
    ) -> str:
        """Apply the mutation part of ``do`` and report its kind.

        Parameters
        ----------
        hooks_or_site:
            Mapping/list batch input or selector-like site.
        value_or_hook:
            Optional value or hook.
        engine:
            Concrete engine selected by ``_select_do_engine``.
        strict:
            Whether selector checks should be strict.
        confirm_mutation:
            Whether root mutation warnings should be suppressed.
        direction:
            Optional signal direction override.

        Returns
        -------
        str
            ``"set"`` or ``"attach_hooks"``.
        """

        if engine == "set_only" and value_or_hook is not None:
            self.set(
                hooks_or_site,
                value_or_hook,
                direction=direction or "forward",
                strict=strict,
                confirm_mutation=confirm_mutation,
            )
            return "set"
        if value_or_hook is not None and not callable(value_or_hook):
            self.set(
                hooks_or_site,
                value_or_hook,
                direction=direction or "forward",
                strict=strict,
                confirm_mutation=confirm_mutation,
            )
            return "set"
        self.attach_hooks(
            hooks_or_site,
            value_or_hook,
            direction=direction,
            strict=strict,
            confirm_mutation=confirm_mutation,
        )
        return "attach_hooks"

    def _validate_supplied_model_matches_capture(self: "Trace", model: nn.Module) -> None:
        """Validate rerun model evidence against the captured source model.

        Parameters
        ----------
        model:
            Candidate model for rerun.

        Raises
        ------
        ModelMismatchError
            If available class or weight-fingerprint evidence differs.
        """

        from ..intervention.errors import ModelMismatchError
        from ..user_funcs import _fingerprint_model_weights, _qualname_for_model

        expected_class = getattr(self, "model_class_qualname", None)
        actual_class = _qualname_for_model(model)
        if expected_class is not None and actual_class != expected_class:
            raise ModelMismatchError(
                "Supplied model class does not match captured model class: "
                f"expected {expected_class!r}, got {actual_class!r}."
            )

        expected_fingerprint = getattr(self, "param_hash_quick", None)
        if expected_fingerprint is None:
            return
        actual_fingerprint = _fingerprint_model_weights(model)
        if actual_fingerprint != expected_fingerprint:
            raise ModelMismatchError(
                "Supplied model weight fingerprint does not match captured model weights."
            )

    def _fork_trace(
        self: "Trace",
        *,
        name: str | None,
        deep_copy_layer_labels: Set[str] | None = None,
    ) -> "Trace":
        """Build a forked Trace with policy-driven field handling.

        Parameters
        ----------
        name:
            Optional fork name.
        deep_copy_layer_labels:
            Optional layer labels that require full Op field copies. When
            provided, other Op shells are still forked but their fields use a
            shallow copy to avoid deep-copying untouched replay context.

        Returns
        -------
        Trace
            Forked log whose mutable containers are independent.
        """

        fork = state_new(type(self))
        # ONE structural copy per fork. Every child object the fork owns gets an
        # empty shell up front, and those shells plus parent-Trace -> fork
        # identity seed a single shared deepcopy memo (see ``_ForkMemo``). The
        # field passes below then walk the parent graph exactly once instead of
        # once per field, and every internal back-reference in the fork resolves
        # to the FORK's objects rather than to orphan clones of the parent's.
        # Shells are filled after the Trace-field pass; a memo hit only rebinds a
        # reference, so an as-yet-empty shell is safe to hand out.
        memo = _ForkMemo()
        _seed_fork_memo(memo, {id(self): fork})
        layer_map: dict[int, Op] = {
            id(parent_pass): state_new(Op) for parent_pass in self.layer_list
        }
        _seed_fork_memo(memo, layer_map)
        layer_log_map: dict[int, Layer] = {
            id(parent_layer): state_new(type(parent_layer))
            for parent_layer in self.layer_logs.values()
        }
        _seed_fork_memo(memo, layer_log_map)

        fork_state = {
            field_name: self._fork_model_field(field_name, value, memo)
            for field_name, value in state_items(self)
        }
        state_restore(fork, fork_state)
        # Stream-derived guard/fold state must be ABSENT (not None) on the
        # fork, mirroring pickle restore: the fork's detached stream starts a
        # new projection window and its first materialize must run cold.
        for stale_guard_field in _STREAM_DERIVED_GUARD_FIELDS:
            fork.__dict__.pop(stale_guard_field, None)
        # The fork gets exactly ONE fresh DETACHED stream — never the
        # parent's (shared lists let a fork backward corrupt the parent's
        # projection), and never none (a fork remains a supported
        # backward-capture target like a restored trace). Whether the parent
        # held its stream under ``capture_events`` (live capture),
        # ``_capture_events`` (restored), or only in the captured-run
        # registry, the fork's stream lives under ``_capture_events`` like a
        # restored trace's.
        from ..ir.capture_events import CaptureEvents

        fork.__dict__.pop("capture_events", None)
        fork.__dict__["_capture_events"] = CaptureEvents.detached_from(self)
        fork.parent_run = weakref.ref(self)
        fork.trace_label = name or self._next_fork_name()
        fork._intervention_spec = copy.deepcopy(self._ensure_intervention_spec(), memo)
        fork.state_history = copy.deepcopy(self.state_history, memo)
        fork.relationship_evidence = copy.deepcopy(self.relationship_evidence, memo)
        fork._out_recipe_revision = self._out_recipe_revision
        fork._spec_revision = self._spec_revision
        fork.state = self.state
        fork._warned_mutate_in_place = False
        fork._warned_direct_write = False
        fork.__dict__.pop("_validation_replay_status", None)

        fork._fork_layer_ops_from(
            self,
            deep_copy_layer_labels=deep_copy_layer_labels,
            layer_map=layer_map,
            memo=memo,
        )
        fork._rebuild_fork_layer_collections(
            self,
            layer_map,
            layer_log_map=layer_log_map,
            memo=memo,
        )
        fork._rebind_fork_owner_refs()
        _state._register_log(fork)
        return fork

    def _next_fork_name(self: "Trace") -> str:
        """Return a deterministic default fork name for this parent log."""

        base_name = self.trace_label or "trace"
        fork_count = sum(
            1
            for record in self.state_history
            if isinstance(record, dict) and record.get("op") == "fork"
        )
        return f"{base_name}_fork_{fork_count + 1}"

    def _fork_model_field(
        self: "Trace", field_name: str, value: Any, memo: dict[Any, Any] | None = None
    ) -> Any:
        """Apply the Trace fork policy to a single field.

        Parameters
        ----------
        field_name:
            Field being copied.
        value:
            Current field value.
        memo:
            Shared fork memo, so every field contributes to (and reuses) the one
            structural copy of the parent graph.

        Returns
        -------
        Any
            Field value for the fork.
        """

        if field_name == "_runnable":
            # Preserve the pre-container fork contract: only the three immutable
            # state bindings are shared, while every other runnable value follows
            # the ordinary fork-copy path. In particular, verdict-steering witness
            # dictionaries must never alias the parent.
            shared_fields = {"staged_user_state", "embedded_state", "capture_state"}
            forked = copy.copy(value)
            for runnable_field in fields(value):
                runnable_value = getattr(value, runnable_field.name)
                if runnable_field.name not in shared_fields:
                    runnable_value = self._copy_fork_value(runnable_value, memo)
                setattr(forked, runnable_field.name, runnable_value)
            return forked
        if field_name in ("capture_events", "_capture_events"):
            # Event streams never fork by copy: deep-copying one raises on the
            # frozen ``GradFnDiscovered.source`` proxies, which used to degrade
            # to a SHALLOW copy — a silently shared ``backward_events`` list
            # that let a fork's backward corrupt the parent's projection.
            # ``_fork_trace`` installs ONE fresh detached stream after the
            # field pass (live captures store the stream under
            # ``capture_events``, restored traces under ``_capture_events``,
            # and some parents hold it only in the captured-run registry, so
            # the single install site lives there).
            return None
        if field_name in _STREAM_DERIVED_GUARD_FIELDS:
            # Projection guard and fold state are derived from the stream that
            # was just replaced; carrying the parent's values would make the
            # fork's guard silently skip (or fold onto foreign state) its
            # first materialize. ``_fork_trace`` pops these after restore.
            return None
        policy = MODEL_LOG_FIELD_FORK_POLICY.get(field_name)
        if policy is None:
            policy = self._default_fork_policy(value)
        if policy is ForkFieldPolicy.FORK_SHARE:
            return value
        if policy is ForkFieldPolicy.FORK_RECONSTRUCT:
            return None
        return self._copy_fork_value(value, memo)

    def _fork_layer_ops_from(
        self: "Trace",
        parent: "Trace",
        *,
        deep_copy_layer_labels: Set[str] | None = None,
        layer_map: dict[int, Op] | None = None,
        memo: dict[Any, Any] | None = None,
    ) -> dict[int, Op]:
        """Fork every Op and return an old-object-id map.

        Parameters
        ----------
        parent:
            Parent log whose layer ops are being forked.
        deep_copy_layer_labels:
            Optional layer labels that require normal fork policy copies. Ops
            outside this set receive distinct shells with shallow-copied fields.
        layer_map:
            Optional pre-created ``id(parent_pass) -> fork Op`` shell map. The
            shells are created by ``_fork_trace`` before any field copy so they
            can seed the shared memo; this call only fills them.
        memo:
            Shared fork memo.

        Returns
        -------
        dict[int, Op]
            Mapping from ``id(parent_pass)`` to forked pass.
        """

        if memo is None:
            memo = _ForkMemo()
            _seed_fork_memo(cast(_ForkMemo, memo), {id(parent): self})
        if layer_map is None:
            layer_map = {id(parent_pass): state_new(Op) for parent_pass in parent.layer_list}
            _seed_fork_memo(cast(_ForkMemo, memo), layer_map)
        fork_equivalent_ops = self.op_equivalence_classes
        for parent_pass in parent.layer_list:
            fork_pass = layer_map[id(parent_pass)]
            deep_copy_fields = (
                deep_copy_layer_labels is None or parent_pass.layer_label in deep_copy_layer_labels
            )
            state_restore(
                fork_pass,
                {
                    field_name: self._fork_layer_pass_field(
                        field_name,
                        value,
                        deep_copy=deep_copy_fields,
                        memo=memo,
                    )
                    for field_name, value in state_items(parent_pass)
                },
            )
            fork_pass.source_trace = self
            eq_type = getattr(fork_pass, "equivalence_class", None)
            if eq_type in fork_equivalent_ops:
                fork_pass.equivalent_ops = fork_equivalent_ops[eq_type]
            object.__setattr__(fork_pass, "_construction_done", True)
        return layer_map

    def _fork_layer_pass_field(
        self: "Trace",
        field_name: str,
        value: Any,
        *,
        deep_copy: bool = True,
        memo: dict[Any, Any] | None = None,
    ) -> Any:
        """Apply the Op fork policy to a single field.

        Parameters
        ----------
        field_name:
            Op field being copied.
        value:
            Current field value.
        deep_copy:
            Whether to honor the normal copy policy. False shares field values
            for untouched differentiable-replay ops while still creating a
            distinct Op shell.
        memo:
            Shared fork memo.

        Returns
        -------
        Any
            Field value for the forked pass.
        """

        if field_name == "_source_trace_ref":
            return None
        if not deep_copy:
            return self._copy_shallow_fork_value(value, memo)
        policy = Op.FIELD_FORK_POLICY.get(field_name)
        if policy is None:
            policy = self._default_fork_policy(value)
        if policy is ForkFieldPolicy.FORK_SHARE:
            return value
        if policy is ForkFieldPolicy.FORK_RECONSTRUCT:
            return None
        return self._copy_fork_value(value, memo)

    def _rebuild_fork_layer_collections(
        self: "Trace",
        parent: "Trace",
        layer_map: dict[int, Op],
        *,
        layer_log_map: dict[int, Layer] | None = None,
        memo: dict[Any, Any] | None = None,
    ) -> None:
        """Rebuild layer lookup containers so they point at forked ops.

        Parameters
        ----------
        parent:
            Parent log whose containers are being mirrored.
        layer_map:
            Mapping from parent pass object id to forked pass.
        layer_log_map:
            Optional pre-created ``id(parent_layer) -> fork Layer`` shell map,
            seeded into the shared memo by ``_fork_trace``.
        memo:
            Shared fork memo.
        """

        def remap_pass(value: Any) -> Any:
            """Map a parent pass object to its forked counterpart.

            Parameters
            ----------
            value:
                Candidate parent-layer object or another value.

            Returns
            -------
            Any
                Forked layer pass when ``value`` is known, otherwise ``value``.
            """
            return layer_map.get(id(value), value)

        self.layer_list = [remap_pass(layer) for layer in parent.layer_list]
        self.layer_dict_main_keys = OrderedDict(
            (key, remap_pass(layer)) for key, layer in parent.layer_dict_main_keys.items()
        )
        self.layer_dict_all_keys = OrderedDict(
            (key, remap_pass(layer)) for key, layer in parent.layer_dict_all_keys.items()
        )
        fork_layer_logs: dict[str, Layer] = OrderedDict()
        # A pre-seeded shell is consumed at most once: were one parent Layer ever
        # registered under two labels, reusing its single shell would collapse
        # two fork Layers into one, so the repeat falls back to a fresh shell
        # (matching the pre-seeding behavior of one new object per label).
        consumed_shells: set[int] = set()
        for label, parent_layer in parent.layer_logs.items():
            parent_layer_id = id(parent_layer)
            if layer_log_map is None or parent_layer_id in consumed_shells:
                fork_layer_log = state_new(type(parent_layer))
            else:
                fork_layer_log = layer_log_map[parent_layer_id]
                consumed_shells.add(parent_layer_id)
            state_restore(
                fork_layer_log,
                {
                    key: self._copy_fork_value(value, memo)
                    for key, value in state_items(parent_layer)
                },
            )
            fork_layer_log.source_trace = self
            fork_layer_log.ops = OpAccessor(
                OrderedDict(
                    (call_index, remap_pass(layer_pass))
                    for call_index, layer_pass in parent_layer.ops.items()
                )
            )
            if getattr(fork_layer_log, "equivalence_class", None) in self.op_equivalence_classes:
                fork_layer_log.equivalent_ops = self.op_equivalence_classes[
                    fork_layer_log.equivalence_class
                ]
            fork_layer_logs[label] = fork_layer_log
        self.layer_logs = fork_layer_logs

    def _rebind_fork_owner_refs(self: "Trace") -> None:
        """Rebind weak owner references on forked child objects to this fork."""

        for layer_pass in self.layer_list:
            layer_pass.source_trace = self
            del layer_pass.facets
        for layer_log in self.layer_logs.values():
            layer_log.source_trace = self
        for module_log in self.modules:
            module_log._source_trace = self
            module_log.__dict__.pop("_facets_cache", None)
            for module_call in module_log.calls.values():
                module_call._source_trace = self

    @staticmethod
    def _copy_fork_value(value: Any, memo: dict[Any, Any] | None = None) -> Any:
        """Copy a fork field while preserving tensor and callable identity.

        Parameters
        ----------
        value:
            Value to copy.
        memo:
            Shared fork memo, so this field reuses (and contributes to) the one
            structural copy of the parent graph rather than cloning it again.

        Returns
        -------
        Any
            Fork-safe copy.
        """

        if isinstance(value, torch.Tensor) or callable(value):
            return value
        if isinstance(value, (str, bytes, int, float, bool, type(None))):
            # ``copy.deepcopy`` returns atomic immutables identically and never
            # memoizes them; skip its per-value dispatch on the fork hot path.
            return value
        return _memoized_deep_copy(value, memo, on_failure=copy.copy)

    @staticmethod
    def _copy_shallow_fork_value(value: Any, memo: dict[Any, Any] | None = None) -> Any:
        """Shallow-copy a fork field while preserving tensor and callable identity.

        Parameters
        ----------
        value:
            Value to copy.
        memo:
            Shared fork memo, threaded into the deep-copy fallbacks.

        Returns
        -------
        Any
            Lightweight fork-safe copy for replay-unaffected Op fields.
        """

        if isinstance(value, torch.Tensor) or callable(value):
            return value
        if isinstance(value, (str, bytes, int, float, bool, type(None))):
            return value
        # Mutable nested containers (dicts/lists/sets, or tuples that hold them --
        # e.g. an Op's `annotations` breadcrumb dict) must NOT be shared with the
        # parent: a post-fork mutation on the fork (`fork.annotate(...)`) would
        # otherwise land on the parent's Op too, silently corrupting it. `copy.copy`
        # is shallow, so it only rebinds the outermost container and still aliases
        # every nested value. Deep-copy the container structure instead, but keep
        # tensor/callable identity so the perf intent of shallow fork -- not cloning
        # large immutable payloads -- is preserved.
        if isinstance(value, (dict, list, set, tuple)):
            return _deep_copy_fork_container(value, memo)
        try:
            return copy.copy(value)
        except Exception:
            return value

    @staticmethod
    def _default_fork_policy(value: Any) -> ForkFieldPolicy:
        """Choose a conservative fork policy for fields outside policy tables.

        Parameters
        ----------
        value:
            Field value without an explicit policy.

        Returns
        -------
        ForkFieldPolicy
            Default share/copy decision.
        """

        if isinstance(value, (str, bytes, int, float, bool, type(None), tuple)):
            return ForkFieldPolicy.FORK_SHARE
        if isinstance(value, torch.Tensor) or callable(value):
            return ForkFieldPolicy.FORK_SHARE
        return ForkFieldPolicy.FORK_COPY

    def _recipe_is_clean(self: "Trace") -> bool:
        """Return whether propagated outs match the current spec revision.

        Returns
        -------
        bool
            ``True`` when the current out recipe revision equals the
            mutable intervention spec revision.
        """

        return self._spec_revision == self._out_recipe_revision

    def _ensure_intervention_spec(self: "Trace") -> InterventionSpec:
        """Return the mutable intervention spec, creating one if needed.

        Returns
        -------
        InterventionSpec
            Mutable intervention recipe owned by this log.
        """

        if self._intervention_spec is None:
            self._intervention_spec = InterventionSpec()
        return self._intervention_spec

    def _mark_intervention_spec_mutated(self: "Trace") -> None:
        """Invalidate cached frozen views and mark the spec stale.

        Returns
        -------
        None
            This model log is mutated in place.
        """

        self._spec_revision += 1
        self.__dict__.pop("intervention_spec", None)
        self.__dict__.pop("_frozen_intervention_spec", None)
        self.__dict__.pop("_cached_frozen_intervention_spec", None)
        self.state = TraceState.SPEC_STALE

    def _validate_intervention_site(self: "Trace", site: Any, *, strict: bool) -> None:
        """Validate that a mutator site resolves on this log.

        Parameters
        ----------
        site:
            Selector-like target to validate.
        strict:
            Whether selector resolution should be strict.

        Returns
        -------
        None
            Raises when the site cannot resolve.
        """

        from ..intervention.resolver import _selector_resolution_direction

        try:
            if _selector_resolution_direction(site) == "backward":
                return
        except Exception:
            pass
        max_fanout = max(1, len(self.layer_list))
        self.resolve_sites(site, strict=strict, max_fanout=max_fanout)

    def _target_spec_from_site(self: "Trace", site: Any, *, strict: bool) -> TargetSpec:
        """Convert a selector-like site to a mutable target spec.

        Parameters
        ----------
        site:
            Selector-like target, target spec, or layer pass.
        strict:
            Whether the resulting target should carry strict resolution.

        Returns
        -------
        TargetSpec
            Mutable target spec stored in the intervention recipe.
        """

        if isinstance(site, TargetSpec):
            target = copy.copy(site)
            target.strict = strict or target.strict
            return target
        if hasattr(site, "to_target_spec"):
            target = site.to_target_spec()
            target.strict = strict or target.strict
            return cast("TargetSpec", target)
        if hasattr(site, "layer_label"):
            return TargetSpec("label", str(site.layer_label), strict=strict)
        return TargetSpec("label", site, strict=strict)
