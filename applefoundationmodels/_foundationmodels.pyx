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
from libc.string cimport memcpy


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
# Tool calling
# ============================================================================

# Global storage for tool functions
cdef object _registered_tools = {}


cdef int32_t _tool_callback_wrapper(
    const char *tool_name,
    const char *arguments_json,
    char *result_buffer,
    int32_t buffer_size
) noexcept with gil:
    """
    C callback wrapper for tool execution.

    Called from Swift when the model wants to execute a tool.
    """
    global _registered_tools

    if tool_name == NULL or arguments_json == NULL or result_buffer == NULL:
        return -3  # AI_ERROR_INVALID_PARAMS

    try:
        # Decode inputs
        name_str = tool_name.decode('utf-8')
        args_str = arguments_json.decode('utf-8')

        # Look up tool function
        if name_str not in _registered_tools:
            error_msg = f"Tool '{name_str}' not found"
            error_bytes = error_msg.encode('utf-8')
            if len(error_bytes) < buffer_size:
                for i, b in enumerate(error_bytes):
                    result_buffer[i] = b
                result_buffer[len(error_bytes)] = 0
            return -11  # AI_ERROR_TOOL_NOT_FOUND

        tool_func = _registered_tools[name_str]

        # Parse arguments
        args_dict = json.loads(args_str)

        # Execute tool
        try:
            result = tool_func(**args_dict)

            # Convert result to string
            if result is None:
                result_str = ""
            elif isinstance(result, str):
                result_str = result
            else:
                result_str = json.dumps(result)

            # Write to buffer using memcpy
            result_bytes = result_str.encode('utf-8')
            if len(result_bytes) >= buffer_size:
                # Buffer too small - signal retry needed
                error_msg = f"Result too large: {len(result_bytes)} bytes (buffer: {buffer_size} bytes)"
                error_bytes = error_msg.encode('utf-8')
                if len(error_bytes) >= buffer_size:
                    error_bytes = error_bytes[:buffer_size-1]
                memcpy(result_buffer, <char*>error_bytes, len(error_bytes))
                result_buffer[len(error_bytes)] = 0
                return -13  # AI_ERROR_BUFFER_TOO_SMALL

            memcpy(result_buffer, <char*>result_bytes, len(result_bytes))
            result_buffer[len(result_bytes)] = 0

            return 0  # AI_SUCCESS

        except Exception as e:
            error_msg = f"Tool execution failed: {str(e)}"
            error_bytes = error_msg.encode('utf-8')
            if len(error_bytes) >= buffer_size:
                error_bytes = error_bytes[:buffer_size-1]
            memcpy(result_buffer, <char*>error_bytes, len(error_bytes))
            result_buffer[len(error_bytes)] = 0
            return -12  # AI_ERROR_TOOL_EXECUTION

    except Exception as e:
        error_msg = f"Callback error: {str(e)}"
        error_bytes = error_msg.encode('utf-8')
        if len(error_bytes) >= buffer_size:
            error_bytes = error_bytes[:buffer_size-1]
        memcpy(result_buffer, <char*>error_bytes, len(error_bytes))
        result_buffer[len(error_bytes)] = 0
        return -99  # AI_ERROR_UNKNOWN


def register_tools(tools: Dict[str, Callable]) -> None:
    """
    Register tool functions for model to call.

    Args:
        tools: Dictionary mapping tool names to callable functions

    Raises:
        InvalidParametersError: If registration fails
    """
    global _registered_tools

    # Store tools globally
    _registered_tools = tools.copy()

    # Build tools JSON for Swift layer
    tools_list = []
    for name, func in tools.items():
        # Tool metadata will be added by Python layer
        # For now, just pass the names
        tools_list.append({
            "name": name,
            "description": getattr(func, "_tool_description", ""),
            "parameters": getattr(func, "_tool_parameters", {})
        })

    tools_json_str = json.dumps(tools_list)
    cdef bytes tools_json_bytes = _encode_string(tools_json_str)
    cdef const char *tools_json = tools_json_bytes
    cdef int32_t result

    with nogil:
        result = apple_ai_register_tools(tools_json, _tool_callback_wrapper)

    _check_result(result)


def get_transcript() -> list:
    """
    Get the session transcript including tool calls.

    Returns:
        List of transcript entries

    Raises:
        GenerationError: If transcript retrieval fails
    """
    cdef char *transcript_json

    with nogil:
        transcript_json = apple_ai_get_transcript()

    if transcript_json == NULL:
        raise_for_error_code(-6, "Failed to get transcript")

    try:
        transcript_str = _decode_string(transcript_json)
        return json.loads(transcript_str)
    finally:
        apple_ai_free_string(transcript_json)


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
# Structured generation
# ============================================================================

def generate_structured(
    prompt: str,
    schema: Dict[str, Any],
    temperature: float = 1.0,
    max_tokens: int = 1024
) -> Dict[str, Any]:
    """
    Generate structured JSON output matching a schema.

    Args:
        prompt: Input text prompt
        schema: JSON schema the output must conform to
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate

    Returns:
        Dictionary with 'object' key containing parsed JSON matching schema

    Raises:
        GenerationError: If generation fails
        InvalidParametersError: If parameters are invalid
        JSONParseError: If schema or response is invalid JSON
    """
    cdef bytes prompt_bytes = _encode_string(prompt)
    cdef const char *prompt_c = prompt_bytes

    # Convert schema dict to JSON string
    schema_json_str = json.dumps(schema)
    cdef bytes schema_bytes = _encode_string(schema_json_str)
    cdef const char *schema_c = schema_bytes

    cdef char *result_c
    cdef str result_str
    cdef double temp_c = temperature
    cdef int32_t tokens_c = max_tokens

    with nogil:
        result_c = apple_ai_generate_structured(prompt_c, schema_c, temp_c, tokens_c)

    if result_c == NULL:
        raise RuntimeError("Structured generation returned NULL")

    # Get result string
    result_str = _decode_string(result_c)
    apple_ai_free_string(result_c)

    # Parse JSON response
    try:
        result_data = json.loads(result_str)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse response JSON: {e}")

    # Check for error in response
    if "error" in result_data:
        raise RuntimeError(result_data["error"])

    return result_data


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
