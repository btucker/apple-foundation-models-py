# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
"""
Cython implementation of Apple AI Python bindings.

This module provides low-level Python wrappers around the Swift FoundationModels
C API, handling memory management, error conversion, and callback marshalling.
"""

import json
from typing import Optional, Callable, Any, Dict
from .exceptions import raise_for_error_code
from .types import Result, Availability


# ============================================================================
# Helper functions
# ============================================================================

cdef bytes _encode_string(s):
    """Convert Python string to bytes for C."""
    if s is None:
        return None
    if isinstance(s, bytes):
        return s
    return s.encode('utf-8')


cdef str _decode_string(const char *s):
    """Convert C string to Python string."""
    if s == NULL:
        return None
    return s.decode('utf-8')


cdef void _check_result(int32_t result):
    """Check result code and raise exception if error."""
    if result != 0:  # AI_SUCCESS = 0
        raise_for_error_code(result, "Operation failed")


# ============================================================================
# Core library functions
# ============================================================================

def init() -> None:
    """
    Initialize the Apple AI library.

    Must be called before using any other library functions.

    Raises:
        InitializationError: If initialization fails
        NotAvailableError: If Apple Intelligence is not available
    """
    cdef int32_t result
    with nogil:
        result = apple_ai_init()
    _check_result(result)


def cleanup() -> None:
    """
    Cleanup and shutdown the Apple AI library.

    Should be called when the application is done using the library.
    """
    with nogil:
        apple_ai_cleanup()


def get_version() -> str:
    """
    Get library version string.

    Returns:
        Version string in format "major.minor.patch"
    """
    cdef const char *version
    with nogil:
        version = apple_ai_get_version()
    return _decode_string(version)


# ============================================================================
# Availability functions
# ============================================================================

def check_availability() -> int:
    """
    Check Apple Intelligence availability on this device.

    Returns:
        Availability status code from Availability enum
    """
    cdef int32_t status
    with nogil:
        status = apple_ai_check_availability()
    return status


def get_availability_reason() -> Optional[str]:
    """
    Get detailed availability status message.

    Returns:
        Detailed status description, or None if library not initialized
    """
    cdef char *reason
    with nogil:
        reason = apple_ai_get_availability_reason()
    if reason == NULL:
        return None
    try:
        return _decode_string(reason)
    finally:
        apple_ai_free_string(reason)


def is_ready() -> bool:
    """
    Check if Apple Intelligence is ready for immediate use.

    Returns:
        True if ready for use, False otherwise
    """
    cdef int32_t status
    with nogil:
        status = apple_ai_check_availability()
    return status == 1  # AI_AVAILABLE = 1


# ============================================================================
# Language support (stub for compatibility)
# ============================================================================

def get_supported_languages_count() -> int:
    """
    Get the number of languages supported by Apple Intelligence.

    Returns:
        Number of supported languages (stub returns 1)
    """
    return 1  # Stub


def get_supported_language(int index) -> Optional[str]:
    """
    Get the display name of a supported language by index.

    Args:
        index: Zero-based language index

    Returns:
        Localized language display name, or None if index is invalid
    """
    if index == 0:
        return "English"
    return None


# ============================================================================
# Session management
# ============================================================================

def create_session(config: Optional[Dict[str, Any]] = None) -> int:
    """
    Create a new AI session.

    Args:
        config: Optional configuration dictionary with 'instructions' key

    Returns:
        Session ID (always 0 in simplified API)

    Raises:
        InitializationError: If session creation fails
    """
    cdef bytes config_json_bytes = None
    cdef const char *config_json = NULL
    cdef int32_t result

    if config:
        config_json_str = json.dumps(config)
        config_json_bytes = _encode_string(config_json_str)
        config_json = config_json_bytes

    with nogil:
        result = apple_ai_create_session(config_json)

    _check_result(result)
    return 0  # Simplified API uses single global session


# ============================================================================
# Text generation
# ============================================================================

def generate(
    prompt: str,
    temperature: float = 1.0,
    max_tokens: int = 1024
) -> str:
    """
    Generate text response for a prompt.

    Args:
        prompt: Input text prompt
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate

    Returns:
        Generated text response

    Raises:
        GenerationError: If generation fails
        InvalidParametersError: If parameters are invalid
    """
    cdef bytes prompt_bytes = _encode_string(prompt)
    cdef const char *prompt_c = prompt_bytes
    cdef char *result_c
    cdef str result_str
    cdef double temp_c = temperature
    cdef int32_t tokens_c = max_tokens

    with nogil:
        result_c = apple_ai_generate(prompt_c, temp_c, tokens_c)

    if result_c == NULL:
        raise RuntimeError("Generation returned NULL")

    # Check if result is an error JSON
    result_str = _decode_string(result_c)
    apple_ai_free_string(result_c)

    # Check for error in response
    if result_str.startswith('{"error"'):
        try:
            error_data = json.loads(result_str)
            raise RuntimeError(error_data.get('error', 'Unknown error'))
        except json.JSONDecodeError:
            pass  # Not JSON, treat as normal response

    return result_str


# ============================================================================
# Streaming generation
# ============================================================================

# Global callback storage for streaming
cdef object _current_stream_callback = None


cdef void _stream_callback_wrapper(const char *chunk) noexcept with gil:
    """
    C callback wrapper that calls Python callback.

    This is called from Swift/C code during streaming generation.
    """
    global _current_stream_callback

    if chunk == NULL:
        # End of stream signal
        if _current_stream_callback:
            try:
                _current_stream_callback(None)
            except:
                pass  # Ignore exceptions in callback at end of stream
        return

    if _current_stream_callback:
        try:
            chunk_str = chunk.decode('utf-8')
            _current_stream_callback(chunk_str)
        except Exception as e:
            print(f"Error in stream callback: {e}")


def generate_stream(
    prompt: str,
    callback: Callable[[Optional[str]], None],
    temperature: float = 1.0,
    max_tokens: int = 1024
) -> None:
    """
    Generate text response with streaming chunks.

    Args:
        prompt: Input text prompt
        callback: Function called with each text chunk (None signals end)
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate

    Raises:
        GenerationError: If generation fails
        InvalidParametersError: If parameters are invalid
    """
    global _current_stream_callback

    cdef bytes prompt_bytes = _encode_string(prompt)
    cdef const char *prompt_c = prompt_bytes
    cdef int32_t result
    cdef double temp_c = temperature
    cdef int32_t tokens_c = max_tokens

    # Store callback globally (simplified for single-threaded use)
    _current_stream_callback = callback

    try:
        with nogil:
            result = apple_ai_generate_stream(
                prompt_c,
                temp_c,
                tokens_c,
                _stream_callback_wrapper
            )
        _check_result(result)
    finally:
        _current_stream_callback = None


# ============================================================================
# History management
# ============================================================================

def get_history() -> list:
    """
    Get conversation history.

    Returns:
        List of message dictionaries with 'role' and 'content' keys
    """
    cdef char *history_json

    with nogil:
        history_json = apple_ai_get_history()

    if history_json == NULL:
        return []

    try:
        history_str = _decode_string(history_json)
        return json.loads(history_str)
    finally:
        apple_ai_free_string(history_json)


def clear_history() -> None:
    """Clear conversation history."""
    with nogil:
        apple_ai_clear_history()


def add_message(role: str, content: str) -> None:
    """
    Add a message to conversation history.

    Args:
        role: Message role ('user', 'assistant', 'system')
        content: Message content

    Note: This is a stub for API compatibility. Use generate() instead.
    """
    # The Swift implementation doesn't support manual message addition
    # This is a stub for API compatibility
    pass


# ============================================================================
# Statistics
# ============================================================================

def get_stats() -> Dict[str, Any]:
    """
    Get usage statistics.

    Returns:
        Dictionary with statistics (stub implementation)
    """
    cdef char *stats_json

    with nogil:
        stats_json = apple_ai_get_stats()

    if stats_json == NULL:
        return {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens_generated": 0,
            "average_response_time": 0.0,
            "total_processing_time": 0.0
        }

    try:
        stats_str = _decode_string(stats_json)
        return json.loads(stats_str)
    finally:
        apple_ai_free_string(stats_json)


def reset_stats() -> None:
    """Reset usage statistics."""
    with nogil:
        apple_ai_reset_stats()
