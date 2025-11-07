"""
Exception hierarchy for applefoundationmodels Python bindings.

All exceptions raised by the library inherit from FoundationModelsError.
Each exception corresponds to a specific error code from the Swift API.
"""

from typing import Optional


class FoundationModelsError(Exception):
    """Base exception for all FoundationModels errors."""

    def __init__(self, message: str, error_code: Optional[int] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code


class InitializationError(FoundationModelsError):
    """Library initialization failed."""

    pass


class NotAvailableError(FoundationModelsError):
    """Apple Intelligence not available on this device."""

    pass


class InvalidParametersError(FoundationModelsError):
    """Invalid parameters provided to function."""

    pass


class MemoryError(FoundationModelsError):
    """Memory allocation error."""

    pass


class JSONParseError(FoundationModelsError):
    """JSON parsing or validation error."""

    pass


class GenerationError(FoundationModelsError):
    """Text generation error."""

    pass


class TimeoutError(FoundationModelsError):
    """Operation timeout."""

    pass


class SessionNotFoundError(FoundationModelsError):
    """Session ID not found."""

    pass


class StreamNotFoundError(FoundationModelsError):
    """Stream ID not found or already completed."""

    pass


class GuardrailViolationError(FoundationModelsError):
    """Content blocked by safety filters."""

    pass


class ToolNotFoundError(FoundationModelsError):
    """Tool callback not registered for session."""

    pass


class ToolExecutionError(FoundationModelsError):
    """Tool execution failed or returned invalid result."""

    pass


class ToolCallError(FoundationModelsError):
    """Tool call error (validation, schema, etc.)."""

    pass


class UnknownError(FoundationModelsError):
    """Unknown error occurred."""

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
        error_code: The error code from the Swift API
        message: Error message to include in the exception

    Raises:
        FoundationModelsError: The appropriate exception subclass for the error code
    """
    exception_class = ERROR_CODE_TO_EXCEPTION.get(error_code, UnknownError)
    raise exception_class(message, error_code)
