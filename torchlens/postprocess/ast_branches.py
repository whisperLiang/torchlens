"""AST indexing for boolean-context classification and branch attribution.

This module parses Python source files into a lightweight, cached index that can:

1. Classify where a captured scalar-bool operation occurred (``if`` test,
   ``elif`` test, ternary test, ``assert``, ``while``, comprehension filter,
   ``match`` guard, or standalone ``bool(...)`` cast).
2. Attribute an arbitrary operation to the enclosing conditional branch arms
   for each stack frame in ``FuncCallLocation``.

The index is structural and process-local. Dense conditional IDs are assigned
later by postprocess integration; this module works only with ``ConditionalKey``.

**Degraded mode without PEP 657 positions** (``HAS_CODE_POSITIONS`` is False,
i.e. CPython < 3.11): runtime frames carry no per-instruction column offsets,
so ``query_intervals`` falls back to line-only matching. Branch arms that
occupy distinct lines still attribute exactly; same-line arms -- notably
ternary / ``IfExp`` arms, and single-line ``if t: body`` forms -- are
DELIBERATELY dropped, because line-only evidence cannot prove which arm
executed and source-text heuristics (name matching) can lie under aliasing
(``r = torch.relu; torch.relu(x) if c else r(x)``). The posture is fail-closed:
un-attributed, never mis-attributed. Conditional events still materialize;
only the per-op arm attribution is withheld. Pinned by
``test_ternary_py310_fail_closed_model_drops_same_line_arm_attribution``.
"""

from __future__ import annotations

import ast
import os
import tokenize
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeAlias, cast

from torchlens.data_classes.func_call_location import FuncCallLocation

ConditionalKey: TypeAlias = Tuple[str, int, int, int]
SourceRange: TypeAlias = Tuple[int, int, int, int]
LineSpan: TypeAlias = Tuple[int, int]
FunctionNode: TypeAlias = ast.FunctionDef | ast.AsyncFunctionDef

_BRANCH_CONSUMER_KINDS = {"if_test", "elif_test", "ifexp"}
_FILE_CACHE_MAX_SIZE = 256
_file_cache: OrderedDict[str, "FileIndex"] = OrderedDict()


@dataclass(frozen=True)
class BoolClassification:
    """Classification result for a terminal boolean operation.

    Attributes
    ----------
    kind:
        The enclosing boolean consumer kind, or ``"unknown"``.
    wrapper_kind:
        Wrapper classification when a direct ``bool(...)`` cast is nested inside
        a branch-participating consumer. Otherwise ``None``.
    conditional_key:
        Structural key for the owning conditional when ``kind`` is
        branch-participating. Otherwise ``None``.
    branch_test_kind:
        Branch discriminator for branch-participating kinds. ``"then"`` is used
        for top-level ``if`` tests and ternary tests; flattened ``elif`` tests
        use ``"elif_N"``.
    """

    kind: str
    wrapper_kind: Optional[str]
    conditional_key: Optional[ConditionalKey]
    branch_test_kind: Optional[str]


@dataclass(frozen=True)
class ConditionalRecord:
    """Structural representation of a conditional found in source code.

    Attributes
    ----------
    key:
        Structural conditional key ``(file, func_firstlineno, if_lineno, if_col)``.
    kind:
        Conditional kind. Either ``"if_chain"`` or ``"ifexp"``.
    source_file:
        Source file that owns the conditional.
    function_span:
        Inclusive line span for the owning function scope.
    if_stmt_span:
        Inclusive line span for the full ``if`` chain or ternary node.
    test_span:
        Full source range for the conditional test expression.
    branch_ranges:
        Mapping from branch kind to full source range for that arm.
    branch_test_spans:
        Mapping from branch kind to the test expression range relevant for that
        arm. ``if`` chains use ``"then"`` and any flattened ``"elif_N"`` keys.
        Ternaries use ``"then"`` only.
    branch_test_structures:
        Mapping from branch kind to the bool-value structure of that arm's
        test expression: ``"bare"`` (a consumed bool's runtime value IS the
        test outcome), ``"negated"`` (outcome is the logical inverse: odd
        ``not`` parity over a single consumed expression), or ``"compound"``
        (``and``/``or`` aggregation: no single consumed bool determines the
        outcome). Keys mirror ``branch_test_spans``.
    call_depth:
        Lexical conditional nesting depth within the owning function scope.
    parent_conditional_key:
        Structural key of the lexically enclosing conditional, if any.
    parent_branch_kind:
        Branch kind inside the parent conditional that contains this record.
    """

    key: ConditionalKey
    kind: Literal["if_chain", "ifexp"]
    source_file: str
    function_span: LineSpan
    if_stmt_span: LineSpan
    test_span: SourceRange
    branch_ranges: Dict[str, SourceRange]
    branch_test_spans: Dict[str, SourceRange]
    branch_test_structures: Dict[str, str]
    call_depth: int
    parent_conditional_key: Optional[ConditionalKey]
    parent_branch_kind: Optional[str]


@dataclass(frozen=True)
class BoolConsumer:
    """AST node that consumes a truthy/falsy value.

    Attributes
    ----------
    kind:
        Consumer kind.
    span:
        Source range that contains the consumed boolean expression.
    depth:
        AST ancestor depth for innermost-first ordering.
    conditional_key:
        Structural conditional key when this consumer is tied to an ``if`` chain
        or ternary test. Otherwise ``None``.
    branch_test_kind:
        Branch discriminator for branch test consumers. Otherwise ``None``.
    """

    kind: str
    span: SourceRange
    depth: int
    conditional_key: Optional[ConditionalKey]
    branch_test_kind: Optional[str]


@dataclass(frozen=True)
class BranchInterval:
    """Branch-arm interval used by point queries inside a function scope.

    Attributes
    ----------
    conditional_key:
        Structural key of the owning conditional.
    branch_kind:
        Branch kind for this interval.
    call_depth:
        Lexical conditional depth of the owning conditional.
    span:
        Full source range for the branch arm.
    """

    conditional_key: ConditionalKey
    branch_kind: str
    call_depth: int
    span: SourceRange


@dataclass
class ScopeEntry:
    """Single function scope indexed within a source file.

    Attributes
    ----------
    code_firstlineno:
        First line number for the function code object.
    func_name:
        Simple function name.
    qualname:
        Qualified name matching ``code.co_qualname`` semantics where possible.
    node:
        Owning AST function node.
    span:
        Inclusive line span for the function body.
    decorated_firstlineno:
        Line number of the FIRST decorator when the function is decorated,
        else ``None``. Runtime code objects of decorated functions report
        ``co_firstlineno`` at the first decorator line while the AST ``def``
        line is ``code_firstlineno``; recording the decorator line makes
        scope resolution exact for any number of (multi-line) decorators.
    conditionals:
        Conditional records defined inside the scope.
    branch_intervals:
        Branch-arm intervals used for operation attribution.
    test_spans_by_key:
        Mapping from conditional key to every test-expression source range of
        that conditional (the top ``if`` test plus each flattened ``elif``
        test). Used by degraded line-only interval matching to fail closed
        when an arm-body interval shares a source line with its own test.
    """

    code_firstlineno: int
    func_name: str
    qualname: str
    node: FunctionNode
    span: LineSpan
    decorated_firstlineno: Optional[int] = None
    conditionals: List[ConditionalRecord] = field(default_factory=list)
    branch_intervals: List[BranchInterval] = field(default_factory=list)
    test_spans_by_key: Dict[ConditionalKey, List[SourceRange]] = field(default_factory=dict)

    def query_intervals(
        self, line: int, col: Optional[int]
    ) -> List[Tuple[ConditionalKey, str, int]]:
        """Return all branch arms containing a point in this scope.

        Parameters
        ----------
        line:
            Source line number for the query point.
        col:
            Source column for the query point. ``None`` activates degraded
            line-only matching.

        Returns
        -------
        List[Tuple[ConditionalKey, str, int]]
            Matching ``(conditional_key, branch_kind, call_depth)`` tuples
            sorted outermost-to-innermost. Ambiguous same-line matches for a
            single conditional are dropped in degraded mode.
        """

        matches: List[BranchInterval] = []
        for interval in self.branch_intervals:
            if col is None:
                if _range_contains_line(interval.span, line):
                    matches.append(interval)
            elif _range_contains_point(interval.span, line, col):
                matches.append(interval)

        if col is None:
            grouped: Dict[ConditionalKey, List[BranchInterval]] = {}
            for interval in matches:
                grouped.setdefault(interval.conditional_key, []).append(interval)

            filtered: List[BranchInterval] = []
            for intervals in grouped.values():
                if len(intervals) != 1:
                    continue
                interval = intervals[0]
                # NO-FALSE-FIRED guard: when a test expression of the SAME
                # conditional shares the query line (single-line ``if t: body``
                # or ``elif t: body``), line-only matching cannot tell a TEST
                # op from an arm-BODY op. Attributing a test op into the arm
                # records the arm as fired even though its body never ran
                # (round-24 condbranch seal, S1). The AST separates the test
                # from the body by column, so column-carrying runtimes still
                # attribute precisely; degraded mode must fail closed.
                test_spans = self.test_spans_by_key.get(interval.conditional_key, [])
                if any(_range_contains_line(test_span, line) for test_span in test_spans):
                    continue
                filtered.append(interval)
            matches = filtered

        matches.sort(key=lambda item: item.call_depth)
        return [
            (interval.conditional_key, interval.branch_kind, interval.call_depth)
            for interval in matches
        ]


class _ScopeCall(NamedTuple):
    """One ``ast.Call`` in a function scope with its precomputed source span.

    Attributes
    ----------
    node:
        The call node itself.
    span:
        ``_node_span(node)``, precomputed once instead of per query.
    visible_name:
        The call's visible callee name (``ast.Name.id`` or ``ast.Attribute.attr``),
        or ``None`` for any other callee form.
    """

    node: ast.Call
    span: SourceRange
    visible_name: Optional[str]


@dataclass
class FileIndex:
    """Parsed AST index for one source file.

    Attributes
    ----------
    filename:
        Source filename for the parsed module.
    mtime_ns:
        File modification timestamp used for cache invalidation.
    source:
        Source text the module was parsed from.
    module:
        Parsed AST module.
    scopes:
        All function scopes discovered in the file.
    conditionals:
        Flattened list of conditionals found across all scopes.
    bool_consumers:
        Flattened list of all boolean consumers found across all scopes.
    parent_map:
        Parent links for every node in ``module``.
    """

    filename: str
    mtime_ns: int
    source: str
    module: ast.Module
    scopes: List[ScopeEntry]
    conditionals: List[ConditionalRecord]
    bool_consumers: List[BoolConsumer]
    parent_map: Dict[ast.AST, ast.AST]
    _source_lines: Optional[List[str]] = field(default=None, repr=False, compare=False)
    _scope_calls: Optional[Dict[ast.AST, List[_ScopeCall]]] = field(
        default=None, repr=False, compare=False
    )

    def scope_calls(self, scope_node: ast.AST) -> List[_ScopeCall]:
        """Return every ``ast.Call`` under ``scope_node``, computed once per scope.

        Parameters
        ----------
        scope_node:
            Function scope node owned by this index's ``module``.

        Returns
        -------
        List[_ScopeCall]
            Call nodes in ``ast.walk`` order, each with its precomputed span and
            visible callee name.

        Notes
        -----
        Call-site resolution queries the same handful of scopes hundreds of times
        per capture, so the ``ast.walk`` is done once per touched scope instead of
        once per query. ``ast.walk`` order is preserved because the downstream
        width sort is stable and callers depend on the pre-sort order for ties.
        Entries stay valid for the index's lifetime: the AST is never mutated, and
        the existing ``mtime_ns`` check plus the file-cache LRU already govern
        invalidation, so this adds no new invalidation surface.
        """

        cache = self._scope_calls
        if cache is None:
            cache = {}
            self._scope_calls = cache
        entries = cache.get(scope_node)
        if entries is None:
            entries = [
                _ScopeCall(node, _node_span(node), _call_visible_name(node))
                for node in ast.walk(scope_node)
                if isinstance(node, ast.Call)
            ]
            cache[scope_node] = entries
        return entries

    def source_lines(self) -> List[str]:
        """Return ``source`` split into parser-style lines, computed once.

        Returns
        -------
        List[str]
            Lines with terminators kept, split exactly as the CPython parser
            splits source (``\\n``, ``\\r\\n``, ``\\r`` only), so byte-offset
            slicing against AST positions matches ``ast.get_source_segment``.
        """

        if self._source_lines is None:
            self._source_lines = _split_source_lines(self.source)
        return self._source_lines

    def resolve_scope(
        self, code_firstlineno: int, func_name: str, func_qualname: Optional[str]
    ) -> Optional[ScopeEntry]:
        """Resolve a runtime frame to a single indexed function scope.

        Parameters
        ----------
        code_firstlineno:
            ``co_firstlineno`` from the runtime frame.
        func_name:
            Simple function name from the runtime frame.
        func_qualname:
            Qualified function name from the runtime frame, when available.

        Returns
        -------
        Optional[ScopeEntry]
            Resolved scope entry, or ``None`` when the D14 fail-closed rules
            require the frame to be skipped.

        Notes
        -----
        Decorated functions report ``co_firstlineno`` at the FIRST decorator
        line, not the ``def`` line, so a frame lineno is accepted when it
        matches either the ``def`` line or the recorded first-decorator line.
        Multiple matches fail closed.
        """

        if func_qualname is not None:
            qualname_matches = [
                scope
                for scope in self.scopes
                if scope.qualname == func_qualname
                and _scope_accepts_firstlineno(scope, code_firstlineno)
            ]
            if len(qualname_matches) == 1:
                return qualname_matches[0]
            return None

        candidates = [
            scope
            for scope in self.scopes
            if scope.func_name == func_name and _scope_accepts_firstlineno(scope, code_firstlineno)
        ]
        if len(candidates) == 1:
            return candidates[0]
        return None


def _scope_accepts_firstlineno(scope: ScopeEntry, code_firstlineno: int) -> bool:
    """Return whether a runtime ``co_firstlineno`` can name this scope.

    Parameters
    ----------
    scope:
        Indexed function scope.
    code_firstlineno:
        ``co_firstlineno`` reported by the runtime frame.

    Returns
    -------
    bool
        ``True`` when the lineno matches the ``def`` line or, for decorated
        functions, the first decorator's line (which is where CPython points
        ``co_firstlineno`` for any number of stacked or multi-line decorators).
    """

    if scope.code_firstlineno == code_firstlineno:
        return True
    return (
        scope.decorated_firstlineno is not None and scope.decorated_firstlineno == code_firstlineno
    )


def get_file_index(filename: str) -> Optional[FileIndex]:
    """Return a cached AST index for ``filename``.

    Parameters
    ----------
    filename:
        Source file to parse and index.

    Returns
    -------
    Optional[FileIndex]
        Cached or newly parsed file index, or ``None`` when the file cannot be
        read, parsed, or stated.
    """

    try:
        mtime_ns = os.stat(filename).st_mtime_ns
    except OSError:
        return None

    cached = _get_cached_file_index(filename)
    if cached is not None and cached.mtime_ns == mtime_ns:
        return cached

    source = _read_source_file(filename)
    if source is None:
        return None

    try:
        module = ast.parse(source, filename=filename)
    except SyntaxError:
        return None

    parent_map = _build_parent_map(module)
    scopes = _collect_scopes(module)
    conditionals: List[ConditionalRecord] = []
    bool_consumers: List[BoolConsumer] = []

    for scope in scopes:
        indexer = _ScopeIndexer(
            filename=filename,
            scope=scope,
            parent_map=parent_map,
            all_conditionals=conditionals,
            all_bool_consumers=bool_consumers,
        )
        indexer.index_scope()

    file_index = FileIndex(
        filename=filename,
        mtime_ns=mtime_ns,
        source=source,
        module=module,
        scopes=scopes,
        conditionals=conditionals,
        bool_consumers=bool_consumers,
        parent_map=parent_map,
    )
    _set_cached_file_index(filename, file_index)
    return file_index


def _get_cached_file_index(filename: str) -> Optional[FileIndex]:
    """Return a cached file index and mark it as recently used.

    Parameters
    ----------
    filename:
        Source filename to look up.

    Returns
    -------
    Optional[FileIndex]
        Cached index for ``filename``, or ``None`` when absent.
    """

    cached = _file_cache.get(filename)
    if cached is not None:
        _file_cache.move_to_end(filename)
    return cached


def _set_cached_file_index(filename: str, file_index: FileIndex) -> None:
    """Store a file index and evict least-recently-used entries beyond the cap.

    Parameters
    ----------
    filename:
        Source filename for the index.
    file_index:
        Parsed index to cache.

    Returns
    -------
    None
        Mutates the module-level cache in place.
    """

    _file_cache[filename] = file_index
    _file_cache.move_to_end(filename)
    while len(_file_cache) > _FILE_CACHE_MAX_SIZE:
        _file_cache.popitem(last=False)


def classify_bool(filename: str, line: int, col: Optional[int] = None) -> BoolClassification:
    """Classify a scalar-bool operation by its enclosing AST consumer.

    Parameters
    ----------
    filename:
        Source file containing the operation.
    line:
        Source line number for the operation.
    col:
        Source column number for the operation. ``None`` activates degraded
        line-only matching.

    Returns
    -------
    BoolClassification
        Classification result following the D3/D18 rules.
    """

    file_index = get_file_index(filename)
    if file_index is None:
        return BoolClassification("unknown", None, None, None)

    consumers: List[BoolConsumer] = []
    for consumer in file_index.bool_consumers:
        if col is None:
            if _range_contains_line(consumer.span, line):
                consumers.append(consumer)
        elif _range_contains_point(consumer.span, line, col):
            consumers.append(consumer)

    if col is None:
        distinct_branch_keys = {
            consumer.conditional_key
            for consumer in consumers
            if consumer.kind in _BRANCH_CONSUMER_KINDS
        }
        if len(distinct_branch_keys) > 1:
            # Degraded line-only matching cannot tell WHICH branch test consumed
            # this bool when several distinct conditionals share the line (e.g.
            # a same-line nested ternary): the deepest-first pick would silently
            # cross-wire the outer bool into the inner conditional. Fail closed;
            # the column-carrying code-context fallback in phase 5b of
            # ``control_flow._classify_bool_layers`` disambiguates precisely.
            return BoolClassification("unknown", None, None, None)

    consumers.sort(
        key=lambda item: (item.depth, 1 if item.kind == "bool_cast" else 0),
        reverse=True,
    )

    saw_bool_cast = False
    for consumer in consumers:
        if consumer.kind == "bool_cast":
            saw_bool_cast = True
            continue
        if consumer.kind in _BRANCH_CONSUMER_KINDS:
            return BoolClassification(
                kind=consumer.kind,
                wrapper_kind="bool_cast" if saw_bool_cast else None,
                conditional_key=consumer.conditional_key,
                branch_test_kind=consumer.branch_test_kind,
            )
        return BoolClassification(consumer.kind, None, None, None)

    if saw_bool_cast:
        return BoolClassification("bool_cast", None, None, None)
    return BoolClassification("unknown", None, None, None)


def attribute_op(code_context: List[FuncCallLocation]) -> List[Tuple[ConditionalKey, str]]:
    """Attribute an operation to enclosing conditional branch arms.

    Parameters
    ----------
    code_context:
        Runtime call stack, ordered shallowest-to-deepest.

    Returns
    -------
    List[Tuple[ConditionalKey, str]]
        Concatenated branch stack across frames, with adjacent duplicate entries
        removed. Each tuple is ``(conditional_key, branch_kind)``.
    """

    branch_stack: List[Tuple[ConditionalKey, str]] = []
    for frame in code_context:
        file_index = get_file_index(frame.file)
        if file_index is None:
            continue

        scope = file_index.resolve_scope(
            code_firstlineno=frame.code_firstlineno,
            func_name=frame.func_name,
            func_qualname=frame.func_qualname,
        )
        if scope is None:
            continue

        for conditional_key, branch_kind, _depth in scope.query_intervals(
            frame.line_number, frame.col_offset
        ):
            entry = (conditional_key, branch_kind)
            if not branch_stack or branch_stack[-1] != entry:
                branch_stack.append(entry)

    return branch_stack


def resolve_var_names(code_context: List[FuncCallLocation], func_name: Optional[str]) -> list[str]:
    """Resolve assignment target names for the captured operation call site.

    Parameters
    ----------
    code_context:
        Runtime call stack, ordered shallowest-to-deepest.
    func_name:
        Captured function name used to disambiguate line-only matches when
        column offsets are unavailable.

    Returns
    -------
    list[str]
        Source variable names bound by the matched call, or an empty list when
        source, call matching, or target extraction is unavailable or ambiguous.
    """

    for frame in reversed(code_context):
        var_names = _resolve_frame_var_names(frame, func_name)
        if var_names:
            return var_names
    return []


def resolve_arg_expressions(
    code_context: List[FuncCallLocation], func_name: Optional[str]
) -> list[str]:
    """Resolve source expressions for a captured operation's call arguments.

    Parameters
    ----------
    code_context:
        Runtime call stack, ordered shallowest-to-deepest.
    func_name:
        Captured function name used to disambiguate line-only matches when
        column offsets are unavailable.

    Returns
    -------
    list[str]
        Source expressions for positional args and keyword args, or an empty
        list when source, call matching, or argument extraction is unavailable
        or ambiguous.
    """

    for frame in reversed(code_context):
        arg_expressions = _resolve_frame_arg_expressions(frame, func_name)
        if arg_expressions:
            return arg_expressions
    return []


def invalidate_cache(filename: Optional[str] = None) -> None:
    """Invalidate cached AST indexes.

    Parameters
    ----------
    filename:
        Specific filename to invalidate. When ``None``, the entire cache is
        cleared.
    """

    if filename is None:
        _file_cache.clear()
    else:
        _file_cache.pop(filename, None)


def _resolve_frame_var_names(frame: FuncCallLocation, func_name: Optional[str]) -> list[str]:
    """Resolve assignment target names for one captured frame.

    Parameters
    ----------
    frame:
        Runtime call-site metadata for the operation.
    func_name:
        Captured function name used to disambiguate line-only matches.

    Returns
    -------
    list[str]
        Assignment target names, or an empty list when resolution fails closed.
    """

    if not frame.source_loading_enabled:
        return []
    if frame.file.startswith("<") or frame.file.endswith(">"):
        return []

    file_index = get_file_index(frame.file)
    if file_index is None:
        return []

    scope = file_index.resolve_scope(
        code_firstlineno=frame.code_firstlineno,
        func_name=frame.func_name,
        func_qualname=frame.func_qualname,
    )
    if scope is None:
        return []

    candidates = _find_candidate_calls(
        file_index, scope.node, frame.line_number, frame.col_offset, func_name
    )
    resolved = [
        target_names
        for call_node in candidates
        if (target_names := _assignment_target_names(call_node, file_index.parent_map))
    ]
    if len(resolved) == 1:
        return resolved[0]
    return []


def _resolve_frame_arg_expressions(frame: FuncCallLocation, func_name: Optional[str]) -> list[str]:
    """Resolve call argument expressions for one captured frame.

    Parameters
    ----------
    frame:
        Runtime call-site metadata for the operation.
    func_name:
        Captured function name used to disambiguate line-only matches.

    Returns
    -------
    list[str]
        Argument expressions, or an empty list when resolution fails closed.
    """

    if frame.file.startswith("<") or frame.file.endswith(">"):
        return []

    file_index = get_file_index(frame.file)
    if file_index is None:
        return []

    scope = file_index.resolve_scope(
        code_firstlineno=frame.code_firstlineno,
        func_name=frame.func_name,
        func_qualname=frame.func_qualname,
    )
    if scope is None:
        return []

    candidates = _find_candidate_calls(
        file_index, scope.node, frame.line_number, frame.col_offset, func_name
    )
    if len(candidates) != 1:
        return []
    return _call_arg_expressions(candidates[0], file_index.source_lines())


def _call_arg_expressions(call_node: ast.Call, source_lines: List[str]) -> list[str]:
    """Return source expressions for a matched call's arguments.

    Parameters
    ----------
    call_node:
        AST call matched to the captured operation.
    source_lines:
        Parser-style split lines of the source containing ``call_node``
        (``FileIndex.source_lines()``).

    Returns
    -------
    list[str]
        Positional argument expressions followed by keyword expressions.
    """

    expressions: list[str] = []
    for arg_node in call_node.args:
        segment = _node_source_segment(source_lines, arg_node)
        if segment is None:
            return []
        expressions.append(segment.strip())
    for keyword in call_node.keywords:
        value_segment = _node_source_segment(source_lines, keyword.value)
        if value_segment is None:
            return []
        if keyword.arg is None:
            expressions.append(f"**{value_segment.strip()}")
        else:
            expressions.append(f"{keyword.arg}={value_segment.strip()}")
    return expressions


def _split_source_lines(source: str) -> List[str]:
    """Split source into lines exactly as the CPython parser does.

    Parameters
    ----------
    source:
        Source text to split.

    Returns
    -------
    List[str]
        Lines with terminators kept. Only ``\\n``, ``\\r\\n``, and ``\\r``
        terminate lines; characters ``str.splitlines`` also breaks on (form
        feed, ``\\x0b``, ``\\u2028``, ...) stay inside their line, matching
        ``ast._splitlines_no_ff`` so AST byte offsets index correctly.
    """

    lines: List[str] = []
    pending = ""
    for part in source.splitlines(keepends=True):
        pending += part
        if pending[-1] in "\r\n":
            lines.append(pending)
            pending = ""
    if pending:
        lines.append(pending)
    return lines


def _node_source_segment(source_lines: List[str], node: ast.AST) -> Optional[str]:
    """Return a node's source segment from pre-split lines.

    Byte-identical replica of ``ast.get_source_segment(source, node)``
    (``padded=False``) that reuses the per-file line split instead of
    re-splitting the whole source character by character on every call --
    the line split dominates ``to_pandas()``/``arg_expressions`` cost when
    resolved per argument (S1).

    Parameters
    ----------
    source_lines:
        Parser-style split lines of the node's source
        (``FileIndex.source_lines()``).
    node:
        AST node to extract.

    Returns
    -------
    Optional[str]
        Source segment, or ``None`` when end positions are missing. Column
        offsets are byte offsets into the UTF-8 encoding of each line, hence
        the encode/decode round-trips.
    """

    try:
        end_lineno = node.end_lineno  # type: ignore[attr-defined]
        end_col_offset = node.end_col_offset  # type: ignore[attr-defined]
        if end_lineno is None or end_col_offset is None:
            return None
        lineno = node.lineno - 1  # type: ignore[attr-defined]
        end_lineno -= 1
        col_offset = node.col_offset  # type: ignore[attr-defined]
    except AttributeError:
        return None

    if end_lineno == lineno:
        return source_lines[lineno].encode()[col_offset:end_col_offset].decode()

    first = source_lines[lineno].encode()[col_offset:].decode()
    last = source_lines[end_lineno].encode()[:end_col_offset].decode()
    return "".join([first, *source_lines[lineno + 1 : end_lineno], last])


def _find_candidate_calls(
    file_index: FileIndex,
    scope_node: FunctionNode,
    line: int,
    col: Optional[int],
    func_name: Optional[str],
) -> list[ast.Call]:
    """Find candidate calls matching a runtime source location.

    Parameters
    ----------
    file_index:
        Index owning ``scope_node``, which supplies the cached per-scope call list.
    scope_node:
        Function scope containing the runtime frame.
    line:
        Source line number for the operation.
    col:
        Source column offset for the operation, or ``None`` for line-only
        matching.
    func_name:
        Captured function name used to disambiguate line-only matches.

    Returns
    -------
    list[ast.Call]
        Candidate call nodes, sorted innermost first for point matches.
    """

    matches = [
        entry
        for entry in file_index.scope_calls(scope_node)
        if (func_name is None or entry.visible_name == func_name)
        and (
            _range_contains_line(entry.span, line)
            if col is None
            else _range_contains_point(entry.span, line, col)
        )
    ]
    if not matches:
        return []

    matches.sort(key=lambda entry: _source_range_width(entry.span))
    if len(matches) > 1 and matches[0].span == matches[1].span:
        return []
    return [entry.node for entry in matches]


def _call_visible_name(call_node: ast.Call) -> Optional[str]:
    """Return the visible callee name for an AST call.

    Parameters
    ----------
    call_node:
        Candidate call node.

    Returns
    -------
    Optional[str]
        ``ast.Name.id`` or ``ast.Attribute.attr`` for the callee, or ``None`` for
        any other callee form.
    """

    if isinstance(call_node.func, ast.Name):
        return call_node.func.id
    if isinstance(call_node.func, ast.Attribute):
        return call_node.func.attr
    return None


def _source_range_width(span: SourceRange) -> tuple[int, int]:
    """Return a sortable width for a source range.

    Parameters
    ----------
    span:
        Source range ``(start_line, start_col, end_line, end_col)``.

    Returns
    -------
    tuple[int, int]
        Line and column extent, with smaller ranges sorting first.
    """

    return (span[2] - span[0], span[3] - span[1])


def _assignment_target_names(call_node: ast.Call, parent_map: Dict[ast.AST, ast.AST]) -> list[str]:
    """Return assignment target names for a matched direct call expression.

    Parameters
    ----------
    call_node:
        AST call matched to the captured operation.
    parent_map:
        Parent links for the containing file.

    Returns
    -------
    list[str]
        Target names, or an empty list for inline, ambiguous, or unsupported
        assignment forms.
    """

    if _has_disallowed_call_ancestor(call_node, parent_map):
        return []

    parent = parent_map.get(call_node)
    if isinstance(parent, ast.Assign) and parent.value is call_node:
        return _flatten_assign_targets(parent.targets)
    if isinstance(parent, ast.AnnAssign) and parent.value is call_node:
        return _flatten_assign_targets([parent.target])
    if isinstance(parent, ast.NamedExpr) and parent.value is call_node:
        return _flatten_assign_targets([parent.target])
    return []


def _has_disallowed_call_ancestor(call_node: ast.Call, parent_map: Dict[ast.AST, ast.AST]) -> bool:
    """Return whether a call sits inside a source form that must fail closed.

    Parameters
    ----------
    call_node:
        AST call matched to the captured operation.
    parent_map:
        Parent links for the containing file.

    Returns
    -------
    bool
        ``True`` for comprehensions and augmented assignments.
    """

    current: ast.AST = call_node
    while current in parent_map:
        current = parent_map[current]
        if isinstance(
            current,
            (
                ast.ListComp,
                ast.SetComp,
                ast.DictComp,
                ast.GeneratorExp,
                ast.AugAssign,
            ),
        ):
            return True
    return False


def _flatten_assign_targets(targets: Sequence[ast.expr]) -> list[str]:
    """Flatten supported assignment targets into variable names.

    Parameters
    ----------
    targets:
        Assignment targets to inspect.

    Returns
    -------
    list[str]
        Name targets in source order, or an empty list when any target is not a
        bare name or a statically visible tuple/list of bare names.
    """

    names: list[str] = []
    for target in targets:
        target_names = _target_names(target)
        if not target_names:
            return []
        names.extend(target_names)
    return names


def _target_names(target: ast.expr) -> list[str]:
    """Return names for one supported assignment target.

    Parameters
    ----------
    target:
        Assignment target to inspect.

    Returns
    -------
    list[str]
        Names in this target, or an empty list for unsupported target forms.
    """

    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names: list[str] = []
        for element in target.elts:
            if not isinstance(element, ast.Name):
                return []
            names.append(element.id)
        return names
    return []


def _read_source_file(filename: str) -> Optional[str]:
    """Read a source file using its declared encoding.

    Parameters
    ----------
    filename:
        Source file to read.

    Returns
    -------
    Optional[str]
        File contents, or ``None`` if the file cannot be read.
    """

    try:
        with tokenize.open(filename) as handle:
            return handle.read()
    except OSError:
        return None


def _build_parent_map(module: ast.Module) -> Dict[ast.AST, ast.AST]:
    """Build a parent map for every AST node in a module.

    Parameters
    ----------
    module:
        Parsed module to index.

    Returns
    -------
    Dict[ast.AST, ast.AST]
        Mapping from child node to direct parent node.
    """

    parent_map: Dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(module):
        for child in ast.iter_child_nodes(parent):
            parent_map[child] = parent
    return parent_map


def _collect_scopes(module: ast.Module) -> List[ScopeEntry]:
    """Collect all function scopes in a module with runtime-style qualnames.

    Parameters
    ----------
    module:
        Parsed module to inspect.

    Returns
    -------
    List[ScopeEntry]
        Collected function scopes in source order.
    """

    scopes: List[ScopeEntry] = []
    _collect_scopes_from_node(module, None, "module", scopes)
    return scopes


def _collect_scopes_from_node(
    node: ast.AST,
    qualname_prefix: Optional[str],
    container_kind: Literal["module", "class", "function"],
    scopes: List[ScopeEntry],
) -> None:
    """Recursively collect function scopes from a node.

    Parameters
    ----------
    node:
        Current AST node.
    qualname_prefix:
        Qualname prefix for child definitions.
    container_kind:
        Container type for qualname composition.
    scopes:
        Output list to populate.
    """

    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.ClassDef):
            child_prefix = _compose_child_qualname(
                qualname_prefix=qualname_prefix,
                container_kind=container_kind,
                child_name=child.name,
            )
            _collect_scopes_from_node(child, child_prefix, "class", scopes)
            continue

        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            qualname = _compose_child_qualname(
                qualname_prefix=qualname_prefix,
                container_kind=container_kind,
                child_name=child.name,
            )
            scopes.append(
                ScopeEntry(
                    code_firstlineno=child.lineno,
                    func_name=child.name,
                    qualname=qualname,
                    node=child,
                    span=(child.lineno, _end_lineno(child)),
                    decorated_firstlineno=(
                        child.decorator_list[0].lineno if child.decorator_list else None
                    ),
                )
            )
            _collect_scopes_from_node(child, qualname, "function", scopes)
            continue

        _collect_scopes_from_node(child, qualname_prefix, container_kind, scopes)


def _compose_child_qualname(
    qualname_prefix: Optional[str],
    container_kind: Literal["module", "class", "function"],
    child_name: str,
) -> str:
    """Compose a child qualname using Python code-object conventions.

    Parameters
    ----------
    qualname_prefix:
        Existing qualname prefix, if any.
    container_kind:
        Type of container holding the child definition.
    child_name:
        Child function or class name.

    Returns
    -------
    str
        Composed qualified name.
    """

    if qualname_prefix is None:
        return child_name
    if container_kind == "function":
        return f"{qualname_prefix}.<locals>.{child_name}"
    return f"{qualname_prefix}.{child_name}"


def _range_contains_point(span: SourceRange, line: int, col: int) -> bool:
    """Return whether a source range contains a point.

    Parameters
    ----------
    span:
        Source range ``(start_line, start_col, end_line, end_col)``.
    line:
        Query line number.
    col:
        Query column number.

    Returns
    -------
    bool
        ``True`` when the point falls within the span.
    """

    return (span[0], span[1]) <= (line, col) <= (span[2], span[3])


def _range_contains_line(span: SourceRange, line: int) -> bool:
    """Return whether a source range contains a line in degraded mode.

    Parameters
    ----------
    span:
        Source range ``(start_line, start_col, end_line, end_col)``.
    line:
        Query line number.

    Returns
    -------
    bool
        ``True`` when the query line falls within the span's line bounds.
    """

    return span[0] <= line <= span[2]


def _node_span(node: ast.AST) -> SourceRange:
    """Return the source span for an AST node.

    Parameters
    ----------
    node:
        AST node with location information.

    Returns
    -------
    SourceRange
        Full source range for the node.
    """

    return (_lineno(node), _col_offset(node), _end_lineno(node), _end_col_offset(node))


def _statement_list_span(statements: List[ast.stmt]) -> SourceRange:
    """Return the bounding source range for a non-empty statement list.

    Parameters
    ----------
    statements:
        Statement list to span.

    Returns
    -------
    SourceRange
        Bounding range from the first statement start to the last statement end.
    """

    first = statements[0]
    last = statements[-1]
    return (_lineno(first), _col_offset(first), _end_lineno(last), _end_col_offset(last))


def _lineno(node: ast.AST) -> int:
    """Return a node's starting line number.

    Parameters
    ----------
    node:
        AST node with location information.

    Returns
    -------
    int
        Starting line number.
    """

    return cast(int, getattr(node, "lineno"))


def _col_offset(node: ast.AST) -> int:
    """Return a node's starting column offset.

    Parameters
    ----------
    node:
        AST node with location information.

    Returns
    -------
    int
        Starting column offset.
    """

    return cast(int, getattr(node, "col_offset"))


def _end_lineno(node: ast.AST) -> int:
    """Return a node's ending line number.

    Parameters
    ----------
    node:
        AST node with location information.

    Returns
    -------
    int
        Ending line number, falling back to ``lineno`` when absent.
    """

    end_lineno = getattr(node, "end_lineno", None)
    if end_lineno is None:
        return _lineno(node)
    return cast(int, end_lineno)


def _end_col_offset(node: ast.AST) -> int:
    """Return a node's ending column offset.

    Parameters
    ----------
    node:
        AST node with location information.

    Returns
    -------
    int
        Ending column offset, falling back to ``col_offset`` when absent.
    """

    end_col_offset = getattr(node, "end_col_offset", None)
    if end_col_offset is None:
        return _col_offset(node)
    return cast(int, end_col_offset)


def _ast_depth(node: ast.AST, parent_map: Dict[ast.AST, ast.AST]) -> int:
    """Return AST ancestor depth for ordering nested consumers.

    Parameters
    ----------
    node:
        Node to measure.
    parent_map:
        Full parent map for the parsed module.

    Returns
    -------
    int
        Number of ancestors between ``node`` and the module root.
    """

    depth = 0
    current = node
    while current in parent_map:
        current = parent_map[current]
        depth += 1
    return depth


def _test_value_structure(test: ast.expr) -> str:
    """Classify a branch test expression's bool-value semantics.

    Parameters
    ----------
    test:
        Full test expression node of one ``if``/``elif``/ternary arm.

    Returns
    -------
    str
        ``"bare"`` when the runtime value of a consumed bool IS the test
        outcome (possibly under an even number of ``not``\\ s), ``"negated"``
        when the outcome is the logical inverse (odd ``not`` parity over a
        single consumed expression), and ``"compound"`` when the test
        aggregates several truth values through ``and``/``or``, where no
        single consumed bool's raw value can honestly stand for the outcome.
    """

    negations = 0
    node: ast.expr = test
    while isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        negations += 1
        node = node.operand
    if any(isinstance(child, ast.BoolOp) for child in ast.walk(node)):
        return "compound"
    return "negated" if negations % 2 else "bare"


def _is_direct_bool_call(node: ast.AST) -> bool:
    """Return whether a node is a direct ``bool(...)`` cast call.

    Parameters
    ----------
    node:
        AST node to inspect.

    Returns
    -------
    bool
        ``True`` when the node is a direct call to the built-in ``bool``.
    """

    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "bool"
        and len(node.args) == 1
        and not node.keywords
    )


def _is_match_case_node(node: ast.AST) -> bool:
    """Return whether a node is a structural pattern-matching case node.

    Parameters
    ----------
    node:
        AST node to inspect.

    Returns
    -------
    bool
        ``True`` when the node is a ``match_case`` object.
    """

    return type(node).__name__ == "match_case"


class _ScopeIndexer:
    """Per-scope conditional and bool-consumer index builder."""

    def __init__(
        self,
        filename: str,
        scope: ScopeEntry,
        parent_map: Dict[ast.AST, ast.AST],
        all_conditionals: List[ConditionalRecord],
        all_bool_consumers: List[BoolConsumer],
    ) -> None:
        """Initialize the scope indexer.

        Parameters
        ----------
        filename:
            Source filename for the owning module.
        scope:
            Function scope being indexed.
        parent_map:
            Parent map for the whole parsed module.
        all_conditionals:
            Shared output list for flattened conditionals.
        all_bool_consumers:
            Shared output list for flattened bool consumers.
        """

        self.filename = filename
        self.scope = scope
        self.parent_map = parent_map
        self.all_conditionals = all_conditionals
        self.all_bool_consumers = all_bool_consumers

    def index_scope(self) -> None:
        """Index all conditionals and bool consumers for the scope."""

        self._walk_nodes(self.scope.node.body, None, None, 0)

    def _walk_nodes(
        self,
        nodes: Sequence[ast.AST],
        parent_conditional_key: Optional[ConditionalKey],
        parent_branch_kind: Optional[str],
        call_depth: int,
    ) -> None:
        """Walk a list of AST nodes under shared conditional context.

        Parameters
        ----------
        nodes:
            Nodes to traverse.
        parent_conditional_key:
            Structural key of the enclosing conditional, if any.
        parent_branch_kind:
            Enclosing branch kind, if any.
        call_depth:
            Current conditional nesting depth within the scope.
        """

        for node in nodes:
            self._walk_node(node, parent_conditional_key, parent_branch_kind, call_depth)

    def _walk_node(
        self,
        node: ast.AST,
        parent_conditional_key: Optional[ConditionalKey],
        parent_branch_kind: Optional[str],
        call_depth: int,
    ) -> None:
        """Walk one AST node under shared conditional context.

        Parameters
        ----------
        node:
            AST node to traverse.
        parent_conditional_key:
            Structural key of the enclosing conditional, if any.
        parent_branch_kind:
            Enclosing branch kind, if any.
        call_depth:
            Current conditional nesting depth within the scope.
        """

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return

        if isinstance(node, ast.If):
            self._handle_if(node, parent_conditional_key, parent_branch_kind, call_depth)
            return

        if isinstance(node, ast.IfExp):
            self._handle_ifexp(node, parent_conditional_key, parent_branch_kind, call_depth)
            return

        if isinstance(node, ast.While):
            self._add_bool_consumer("while", node.test, None, None)

        if isinstance(node, ast.Assert):
            self._add_bool_consumer("assert", node.test, None, None)

        if isinstance(node, ast.comprehension):
            for if_expr in node.ifs:
                self._add_bool_consumer("comprehension_filter", if_expr, None, None)

        if _is_match_case_node(node):
            guard = getattr(node, "guard", None)
            if isinstance(guard, ast.AST):
                self._add_bool_consumer("match_guard", guard, None, None)

        if _is_direct_bool_call(node):
            self._add_bool_consumer("bool_cast", node, None, None)

        for child in ast.iter_child_nodes(node):
            self._walk_node(child, parent_conditional_key, parent_branch_kind, call_depth)

    def _handle_if(
        self,
        node: ast.If,
        parent_conditional_key: Optional[ConditionalKey],
        parent_branch_kind: Optional[str],
        call_depth: int,
    ) -> None:
        """Index a flattened ``if``/``elif``/``else`` chain.

        Parameters
        ----------
        node:
            Top-level ``if`` node.
        parent_conditional_key:
            Structural key of the enclosing conditional, if any.
        parent_branch_kind:
            Enclosing branch kind, if any.
        call_depth:
            Current conditional nesting depth within the scope.
        """

        if self._is_synthetic_elif(node):
            self._walk_node(node.test, parent_conditional_key, parent_branch_kind, call_depth)
            self._walk_nodes(node.body, parent_conditional_key, parent_branch_kind, call_depth)
            self._walk_nodes(node.orelse, parent_conditional_key, parent_branch_kind, call_depth)
            return

        flattened_elifs, terminal_else = self._flatten_elif_chain(node)
        record = self._build_if_record(
            node=node,
            flattened_elifs=flattened_elifs,
            terminal_else=terminal_else,
            parent_conditional_key=parent_conditional_key,
            parent_branch_kind=parent_branch_kind,
            call_depth=call_depth,
        )
        self._register_conditional(record)

        self._add_bool_consumer("if_test", node.test, record.key, "then")
        self._walk_node(node.test, parent_conditional_key, parent_branch_kind, call_depth)
        self._walk_nodes(node.body, record.key, "then", call_depth + 1)

        for index, elif_node in enumerate(flattened_elifs, start=1):
            branch_kind = f"elif_{index}"
            self._add_bool_consumer("elif_test", elif_node.test, record.key, branch_kind)
            self._walk_node(elif_node.test, parent_conditional_key, parent_branch_kind, call_depth)
            self._walk_nodes(elif_node.body, record.key, branch_kind, call_depth + 1)

        if terminal_else:
            self._walk_nodes(terminal_else, record.key, "else", call_depth + 1)

    def _handle_ifexp(
        self,
        node: ast.IfExp,
        parent_conditional_key: Optional[ConditionalKey],
        parent_branch_kind: Optional[str],
        call_depth: int,
    ) -> None:
        """Index a ternary conditional expression.

        Parameters
        ----------
        node:
            Ternary expression node.
        parent_conditional_key:
            Structural key of the enclosing conditional, if any.
        parent_branch_kind:
            Enclosing branch kind, if any.
        call_depth:
            Current conditional nesting depth within the scope.
        """

        key: ConditionalKey = (
            self.filename,
            self.scope.code_firstlineno,
            node.lineno,
            node.col_offset,
        )
        record = ConditionalRecord(
            key=key,
            kind="ifexp",
            source_file=self.filename,
            function_span=self.scope.span,
            if_stmt_span=(node.lineno, _end_lineno(node)),
            test_span=_node_span(node.test),
            branch_ranges={
                "then": _node_span(node.body),
                "else": _node_span(node.orelse),
            },
            branch_test_spans={"then": _node_span(node.test)},
            branch_test_structures={"then": _test_value_structure(node.test)},
            call_depth=call_depth,
            parent_conditional_key=parent_conditional_key,
            parent_branch_kind=parent_branch_kind,
        )
        self._register_conditional(record)

        self._add_bool_consumer("ifexp", node.test, key, "then")
        self._walk_node(node.test, parent_conditional_key, parent_branch_kind, call_depth)
        self._walk_node(node.body, key, "then", call_depth + 1)
        self._walk_node(node.orelse, key, "else", call_depth + 1)

    def _flatten_elif_chain(self, node: ast.If) -> Tuple[List[ast.If], List[ast.stmt]]:
        """Flatten a synthetic ``elif`` chain rooted at ``node``.

        Parameters
        ----------
        node:
            Top-level ``if`` node.

        Returns
        -------
        Tuple[List[ast.If], List[ast.stmt]]
            Flattened synthetic ``elif`` nodes and terminal ``else`` statements.
        """

        flattened_elifs: List[ast.If] = []
        terminal_else: List[ast.stmt] = []
        current = node
        while current.orelse:
            if (
                len(current.orelse) == 1
                and isinstance(current.orelse[0], ast.If)
                and current.orelse[0].col_offset == current.col_offset
            ):
                next_if = current.orelse[0]
                flattened_elifs.append(next_if)
                current = next_if
                continue
            terminal_else = list(current.orelse)
            break
        return flattened_elifs, terminal_else

    def _build_if_record(
        self,
        node: ast.If,
        flattened_elifs: List[ast.If],
        terminal_else: List[ast.stmt],
        parent_conditional_key: Optional[ConditionalKey],
        parent_branch_kind: Optional[str],
        call_depth: int,
    ) -> ConditionalRecord:
        """Build the conditional record for a flattened ``if`` chain.

        Parameters
        ----------
        node:
            Top-level ``if`` node.
        flattened_elifs:
            Synthetic ``elif`` nodes flattened into the record.
        terminal_else:
            Terminal ``else`` statement list, if any.
        parent_conditional_key:
            Structural key of the enclosing conditional, if any.
        parent_branch_kind:
            Enclosing branch kind, if any.
        call_depth:
            Current conditional nesting depth within the scope.

        Returns
        -------
        ConditionalRecord
            Built conditional record.
        """

        key: ConditionalKey = (
            self.filename,
            self.scope.code_firstlineno,
            node.lineno,
            node.col_offset,
        )
        branch_ranges: Dict[str, SourceRange] = {"then": _statement_list_span(node.body)}
        branch_test_spans: Dict[str, SourceRange] = {"then": _node_span(node.test)}
        branch_test_structures: Dict[str, str] = {"then": _test_value_structure(node.test)}

        for index, elif_node in enumerate(flattened_elifs, start=1):
            branch_kind = f"elif_{index}"
            branch_ranges[branch_kind] = _statement_list_span(elif_node.body)
            branch_test_spans[branch_kind] = _node_span(elif_node.test)
            branch_test_structures[branch_kind] = _test_value_structure(elif_node.test)

        if terminal_else:
            branch_ranges["else"] = _statement_list_span(terminal_else)

        return ConditionalRecord(
            key=key,
            kind="if_chain",
            source_file=self.filename,
            function_span=self.scope.span,
            if_stmt_span=(node.lineno, _end_lineno(node)),
            test_span=_node_span(node.test),
            branch_ranges=branch_ranges,
            branch_test_spans=branch_test_spans,
            branch_test_structures=branch_test_structures,
            call_depth=call_depth,
            parent_conditional_key=parent_conditional_key,
            parent_branch_kind=parent_branch_kind,
        )

    def _register_conditional(self, record: ConditionalRecord) -> None:
        """Register a conditional record with scope-level interval data.

        Parameters
        ----------
        record:
            Conditional record to register.
        """

        self.scope.conditionals.append(record)
        self.all_conditionals.append(record)
        test_spans = [record.test_span]
        for test_span in record.branch_test_spans.values():
            if test_span not in test_spans:
                test_spans.append(test_span)
        self.scope.test_spans_by_key[record.key] = test_spans
        for branch_kind, span in record.branch_ranges.items():
            self.scope.branch_intervals.append(
                BranchInterval(
                    conditional_key=record.key,
                    branch_kind=branch_kind,
                    call_depth=record.call_depth,
                    span=span,
                )
            )

    def _add_bool_consumer(
        self,
        kind: str,
        node: ast.AST,
        conditional_key: Optional[ConditionalKey],
        branch_test_kind: Optional[str],
    ) -> None:
        """Add a bool consumer record to the flattened file index.

        Parameters
        ----------
        kind:
            Consumer kind.
        node:
            AST node spanning the consumed boolean expression.
        conditional_key:
            Structural conditional key for branch consumers.
        branch_test_kind:
            Branch discriminator for branch consumers.
        """

        consumer = BoolConsumer(
            kind=kind,
            span=_node_span(node),
            depth=_ast_depth(node, self.parent_map),
            conditional_key=conditional_key,
            branch_test_kind=branch_test_kind,
        )
        self.all_bool_consumers.append(consumer)

    def _is_synthetic_elif(self, node: ast.If) -> bool:
        """Return whether an ``ast.If`` is the synthetic child of an ``elif``.

        Parameters
        ----------
        node:
            ``ast.If`` node to inspect.

        Returns
        -------
        bool
            ``True`` when ``node`` is the sole statement in its parent's
            ``orelse`` list.
        """

        parent = self.parent_map.get(node)
        return (
            isinstance(parent, ast.If)
            and len(parent.orelse) == 1
            and parent.orelse[0] is node
            and parent.col_offset == node.col_offset
        )


__all__ = [
    "BoolClassification",
    "ConditionalKey",
    "ConditionalRecord",
    "FileIndex",
    "ScopeEntry",
    "attribute_op",
    "classify_bool",
    "get_file_index",
    "invalidate_cache",
    "resolve_arg_expressions",
]
