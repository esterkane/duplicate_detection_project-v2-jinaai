"""Structured tool errors for the MCP layer.

MCP tools must never leak an internal stack trace or a raw Elasticsearch / Jina
error into a result. Instead every failure is converted to a small, structured
payload::

    {
        "isError": True,
        "errorCategory": "validation" | "transient" | "permission" | "business",
        "isRetryable": bool,
        "message": "<safe, human-readable summary>",
        "details": { ... },   # optional, safe context only
    }

Handlers raise one of the typed errors below for expected failures; the
:func:`guard` decorator wraps every tool handler so that *any* unexpected
exception is logged server-side (with its traceback) and returned as a generic,
trace-free transient error. Elasticsearch / httpx connection and timeout errors
are treated as retryable transient failures.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Error categories.
VALIDATION = "validation"
TRANSIENT = "transient"
PERMISSION = "permission"
BUSINESS = "business"


class ToolError(Exception):
    """An expected, classified tool failure carrying a category and retryability."""

    category: str = TRANSIENT
    retryable: bool = False

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ToolValidationError(ToolError):
    """Bad or unsupported input (empty query, out-of-range threshold). Not retryable."""

    category = VALIDATION
    retryable = False


class ToolBusinessError(ToolError):
    """A valid request that cannot be satisfied (e.g. unknown chunk id, no
    candidates to compare). Retrying without changing inputs will not help."""

    category = BUSINESS
    retryable = False


class ToolPermissionError(ToolError):
    """The caller is not permitted to perform the request. Not retryable."""

    category = PERMISSION
    retryable = False


class ToolTransientError(ToolError):
    """A backend (Elasticsearch / Jina) was momentarily unavailable. Safe to retry."""

    category = TRANSIENT
    retryable = True


def error_result(
    category: str,
    message: str,
    *,
    retryable: bool,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the structured error payload returned in place of a result."""
    return {
        "isError": True,
        "errorCategory": category,
        "isRetryable": retryable,
        "message": message,
        "details": details or {},
    }


def _is_transient_backend_error(exc: BaseException) -> bool:
    """True for Elasticsearch / httpx connection or timeout errors.

    Imported lazily and matched by class name so the MCP package stays importable
    (and unit-testable with fakes) even when the heavy ``elasticsearch`` client is
    not installed in the environment running the tests.
    """
    transient_names = {
        "ConnectionError",
        "ConnectionTimeout",
        "TransportError",
        "ConnectTimeout",
        "ReadTimeout",
        "ConnectError",
        "TimeoutException",
    }
    for klass in type(exc).__mro__:
        module = getattr(klass, "__module__", "") or ""
        if klass.__name__ in transient_names and (
            module.startswith("elasticsearch")
            or module.startswith("elastic_transport")
            or module.startswith("httpx")
        ):
            return True
    return False


def guard(name: str) -> Callable[[Callable[..., dict]], Callable[..., dict]]:
    """Wrap a tool handler so no failure ever escapes as a stack trace.

    - :class:`ToolError` subclasses become their structured category payload.
    - Elasticsearch / httpx connection and timeout errors become a retryable
      transient error.
    - Anything else is logged with its traceback and returned as a generic,
      non-retryable transient error with no internal detail.
    """

    def decorator(fn: Callable[..., dict]) -> Callable[..., dict]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
            try:
                return fn(*args, **kwargs)
            except ToolError as exc:
                return error_result(
                    exc.category, exc.message, retryable=exc.retryable, details=exc.details
                )
            except Exception as exc:  # last-resort guard; traceback goes to logs only
                if _is_transient_backend_error(exc):
                    logger.warning(
                        "mcp tool %s: search backend unreachable (%s)",
                        name,
                        type(exc).__name__,
                    )
                    return error_result(
                        TRANSIENT,
                        "The search backend is currently unreachable. Please retry shortly.",
                        retryable=True,
                        details={"kind": type(exc).__name__},
                    )
                logger.exception("mcp tool %s failed unexpectedly", name)
                return error_result(
                    TRANSIENT,
                    "An unexpected internal error occurred while handling the request.",
                    retryable=False,
                )

        return wrapper

    return decorator
