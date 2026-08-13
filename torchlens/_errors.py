"""Shared TorchLens exception types."""

from .errors._base import CaptureError
from .errors._base import ConfigurationError
from .errors._base import TorchLensWarning


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


class AmbiguousOpLookupError(ValueError):
    """Raised when a bare Op lookup matches multiple pass-qualified Ops."""


class ShapeInferenceError(ConfigurationError, RuntimeError):
    """Raised when debug input-shape inference cannot produce a valid input."""
