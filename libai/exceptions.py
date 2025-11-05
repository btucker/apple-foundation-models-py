"""
Exception hierarchy for libai Python bindings.

All exceptions raised by the library inherit from LibAIError.
Each exception corresponds to a specific ai_result_t error code from the C library.
"""

from typing import Optional


class LibAIError(Exception):
    """Base exception for all libai errors."""

    def __init__(self, message: str, error_code: Optional[int] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code


class InitializationError(LibAIError):
    """Library initialization failed (AI_ERROR_INIT_FAILED)."""
    pass


class NotAvailableError(LibAIError):
    """Apple Intelligence not available on this device (AI_ERROR_NOT_AVAILABLE)."""
    pass


class InvalidParametersError(LibAIError):
    """Invalid parameters provided to function (AI_ERROR_INVALID_PARAMS)."""
    pass


class MemoryError(LibAIError):
    """Memory allocation error (AI_ERROR_MEMORY)."""
    pass


class JSONParseError(LibAIError):
    """JSON parsing or validation error (AI_ERROR_JSON_PARSE)."""
    pass


class GenerationError(LibAIError):
    """Text generation error (AI_ERROR_GENERATION)."""
    pass


class TimeoutError(LibAIError):
    """Operation timeout (AI_ERROR_TIMEOUT)."""
    pass


class SessionNotFoundError(LibAIError):
    """Session ID not found in context (AI_ERROR_SESSION_NOT_FOUND)."""
    pass


class StreamNotFoundError(LibAIError):
    """Stream ID not found or already completed (AI_ERROR_STREAM_NOT_FOUND)."""
    pass


class GuardrailViolationError(LibAIError):
    """Content blocked by safety filters (AI_ERROR_GUARDRAIL_VIOLATION)."""
    pass


class ToolNotFoundError(LibAIError):
    """Tool callback not registered for session (AI_ERROR_TOOL_NOT_FOUND)."""
    pass


class ToolExecutionError(LibAIError):
    """Tool execution failed or returned invalid result (AI_ERROR_TOOL_EXECUTION)."""
    pass


class UnknownError(LibAIError):
    """Unknown error occurred (AI_ERROR_UNKNOWN)."""
    pass


# Mapping from ai_result_t error codes to exception classes
ERROR_CODE_TO_EXCEPTION = {
    -1: InitializationError,
    -2: NotAvailableError,
    -3: InvalidParametersError,
    -4: MemoryError,
    -5: JSONParseError,
    -6: GenerationError,
    -7: TimeoutError,
    -8: SessionNotFoundError,
    -9: StreamNotFoundError,
    -10: GuardrailViolationError,
    -11: ToolNotFoundError,
    -12: ToolExecutionError,
    -99: UnknownError,
}


def raise_for_error_code(error_code: int, message: str) -> None:
    """
    Raise the appropriate exception for a given error code.

    Args:
        error_code: The ai_result_t error code from the C library
        message: Error message to include in the exception

    Raises:
        LibAIError: The appropriate exception subclass for the error code
    """
    exception_class = ERROR_CODE_TO_EXCEPTION.get(error_code, UnknownError)
    raise exception_class(message, error_code)
