"""Shared TorchLens exception types."""

from __future__ import annotations

from typing import Any, cast

from .errors._base import CaptureError, ConfigurationError, TorchLensWarning


def _actionable_message(problem: str, remedy: str) -> str:
    """Combine a refusal description with its required user remedy.

    Parameters
    ----------
    problem:
        Description of the rejected object or operation and its cause.
    remedy:
        Concrete action the caller can take to resolve the refusal.

    Returns
    -------
    str
        Stable human-readable message containing both clauses.
    """

    problem_clause = problem.rstrip()
    if not problem_clause.endswith((".", "!", "?", ":", ";")):
        problem_clause = f"{problem_clause}."
    return f"{problem_clause} Remedy: {remedy.rstrip().rstrip('.')}."


def _restore_actionable_error(
    error_type: type[BaseException],
    args: tuple[object, ...],
    state: dict[str, object],
) -> BaseException:
    """Rebuild an actionable exception without replaying its strict constructor.

    Parameters
    ----------
    error_type:
        Concrete exception class stored by pickle.
    args:
        Already-formatted ``BaseException.args`` tuple.
    state:
        Instance dictionary containing structured fields and source context.

    Returns
    -------
    BaseException
        Restored exception with its exact message and structured payload.
    """

    error = error_type.__new__(error_type)
    BaseException.__init__(error, *args)
    error.__dict__.update(state)
    return error


class _ActionableErrorMixin:
    """Pickle support shared by strict actionable-error constructors."""

    def __reduce__(
        self,
    ) -> tuple[
        object,
        tuple[type[BaseException], tuple[object, ...], dict[str, object]],
    ]:
        """Return a pickle reconstruction recipe preserving structured fields."""

        if not isinstance(self, BaseException):  # pragma: no cover - MRO invariant.
            raise TypeError("_ActionableErrorMixin must be combined with BaseException")
        return (
            _restore_actionable_error,
            (type(self), self.args, dict(self.__dict__)),
        )


class InvalidArgumentError(_ActionableErrorMixin, ConfigurationError, ValueError):
    """Raised when a public argument value is outside its supported domain."""

    def __init__(
        self,
        problem: str,
        *,
        code: str,
        remedy: str,
        **context: object,
    ) -> None:
        """Initialize an actionable invalid-argument refusal.

        Parameters
        ----------
        problem:
            Description of the rejected argument and why it was rejected.
        code:
            Stable machine-readable refusal code.
        remedy:
            Concrete caller action that resolves the refusal.
        **context:
            Structured, non-authoritative diagnostic context.
        """

        super().__init__(
            _actionable_message(problem, remedy),
            code=code,
            remedy=remedy,
            **cast(dict[str, Any], context),
        )


class ArgumentTypeError(_ActionableErrorMixin, ConfigurationError, TypeError):
    """Raised when a public argument has an unsupported Python type."""

    def __init__(
        self,
        problem: str,
        *,
        code: str,
        remedy: str,
        **context: object,
    ) -> None:
        """Initialize an actionable argument-type refusal.

        Parameters
        ----------
        problem:
            Description of the rejected argument and its received type.
        code:
            Stable machine-readable refusal code.
        remedy:
            Concrete caller action that resolves the refusal.
        **context:
            Structured, non-authoritative diagnostic context.
        """

        super().__init__(
            _actionable_message(problem, remedy),
            code=code,
            remedy=remedy,
            **cast(dict[str, Any], context),
        )


class ArgumentConflictError(_ActionableErrorMixin, ConfigurationError, ValueError):
    """Raised when mutually exclusive public arguments are supplied together.

    Subclasses ``ValueError``, not ``TypeError``: every historical conflict
    refusal raised a raw ``ValueError`` (the arguments are well-typed; the
    combination is the problem), so existing ``except ValueError`` handlers
    keep catching it.
    """

    def __init__(
        self,
        problem: str,
        *,
        code: str,
        remedy: str,
        **context: object,
    ) -> None:
        """Initialize an actionable argument-conflict refusal.

        Parameters
        ----------
        problem:
            Description of the conflicting arguments.
        code:
            Stable machine-readable refusal code.
        remedy:
            Concrete caller action that resolves the refusal.
        **context:
            Structured, non-authoritative diagnostic context.
        """

        super().__init__(
            _actionable_message(problem, remedy),
            code=code,
            remedy=remedy,
            **cast(dict[str, Any], context),
        )


class CaptureContextError(_ActionableErrorMixin, CaptureError, RuntimeError):
    """Raised when a capture-only operation is called outside an active capture."""

    def __init__(
        self,
        problem: str,
        *,
        code: str,
        remedy: str,
        **context: object,
    ) -> None:
        """Initialize an actionable capture-context refusal.

        Parameters
        ----------
        problem:
            Description of the operation and unavailable capture state.
        code:
            Stable machine-readable refusal code.
        remedy:
            Concrete caller action that resolves the refusal.
        **context:
            Structured, non-authoritative diagnostic context.
        """

        super().__init__(
            _actionable_message(problem, remedy),
            code=code,
            remedy=remedy,
            **cast(dict[str, Any], context),
        )


class TorchLensCaptureGapError(CaptureError, RuntimeError):
    """Reserved enforcement error for an unrepresented torch invocation."""


class TorchLensCaptureGapWarning(TorchLensWarning):
    """Shadow-mode report for a possible unrepresented torch invocation."""


class OutputAttributionError(CaptureError, RuntimeError):
    """Raised when a model output tensor cannot be attributed to any traced op.

    The classic producer is a stale pre-wrap torch function reference in
    OUTPUT position: the escaped call is invisible to the wrappers, so its
    result reaches the output walk with no label. Typed so the capture entry
    can treat it as an escape signal (rescue re-run trigger) instead of
    string-matching ``RuntimeError`` text.
    """


class TorchLensPostfuncError(CaptureError, RuntimeError):
    """Raised when activation_transform or grad_transform raises."""


class BackwardStreamUnavailableError(CaptureError, RuntimeError):
    """Raised when backward capture needs an event stream the trace no longer owns.

    Historically a missing stream was silently replaced with a fresh empty
    buffer, so post-hoc backward capture appended into a container nothing
    read and reported success. A released or never-captured stream is now a
    typed refusal instead of a silent wrong answer.
    """


class MutatedReferenceError(CaptureError, RuntimeError):
    """Raised when a reference-mode saved tensor changed before it was read."""


class PostTraceParamUnavailable(CaptureError, RuntimeError):
    """Raised when a released Param cannot re-fetch its live model parameter."""


class AmbiguousOpLookupError(_ActionableErrorMixin, ConfigurationError, ValueError):
    """Raised when a bare Op lookup matches multiple pass-qualified Ops."""

    def __init__(self, message: str, **context: object) -> None:
        """Initialize an actionable ambiguous-accessor refusal.

        Parameters
        ----------
        message:
            Existing lookup-specific description of the ambiguous matches.
        **context:
            Structured, non-authoritative diagnostic context.
        """

        remedy = "use a pass-qualified label, full address, or explicit call index"
        super().__init__(
            _actionable_message(message, remedy),
            code="ambiguous_op_lookup",
            remedy=remedy,
            **cast(dict[str, Any], context),
        )


class ShapeInferenceError(ConfigurationError, RuntimeError):
    """Raised when debug input-shape inference cannot produce a valid input."""
