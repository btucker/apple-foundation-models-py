# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
"""
Cython implementation of libai Python bindings.

This module provides low-level Python wrappers around the libai C API,
handling memory management, error conversion, and callback marshalling.
"""

import json
from typing import Optional, Callable, Any, Dict
from libc.stdint cimport uintptr_t
from . cimport _libai
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


cdef void _check_result(ai_result_t result, ai_context_t *context):
    """Check result code and raise exception if error."""
    cdef const char *error_msg
    if result != AI_SUCCESS:
        error_msg = ai_get_last_error(context)
        msg = _decode_string(error_msg) if error_msg != NULL else "Unknown error"
        raise_for_error_code(result, msg)


# ============================================================================
# Core library functions
# ============================================================================

def init() -> None:
    """
    Initialize the libai library.

    Must be called before using any other library functions.

    Raises:
        InitializationError: If initialization fails
        NotAvailableError: If Apple Intelligence is not available
    """
    cdef ai_result_t result
    with nogil:
        result = ai_init()
    _check_result(result, NULL)


def cleanup() -> None:
    """
    Cleanup and shutdown the libai library.

    Should be called when the application is done using the library.
    """
    with nogil:
        ai_cleanup()


def get_version() -> str:
    """
    Get library version string.

    Returns:
        Version string in format "major.minor.patch"
    """
    cdef const char *version
    with nogil:
        version = ai_get_version()
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
    cdef ai_availability_t status
    with nogil:
        status = ai_check_availability()
    return status


def get_availability_reason() -> Optional[str]:
    """
    Get detailed availability status message.

    Returns:
        Detailed status description, or None if library not initialized
    """
    cdef char *reason
    with nogil:
        reason = ai_get_availability_reason()
    if reason == NULL:
        return None
    try:
        return _decode_string(reason)
    finally:
        ai_free_string(reason)


def is_ready() -> bool:
    """
    Check if Apple Intelligence is ready for immediate use.

    Returns:
        True if ready for use, False otherwise
    """
    cdef bint ready
    with nogil:
        ready = ai_is_ready()
    return ready


# ============================================================================
# Language support
# ============================================================================

def get_supported_languages_count() -> int:
    """
    Get the number of languages supported by Apple Intelligence.

    Returns:
        Number of supported languages
    """
    cdef int32_t count
    with nogil:
        count = ai_get_supported_languages_count()
    return count


def get_supported_language(int index) -> Optional[str]:
    """
    Get the display name of a supported language by index.

    Args:
        index: Zero-based language index

    Returns:
        Localized language display name, or None if index is invalid
    """
    cdef char *lang
    with nogil:
        lang = ai_get_supported_language(index)
    if lang == NULL:
        return None
    try:
        return _decode_string(lang)
    finally:
        ai_free_string(lang)


# ============================================================================
# Context wrapper class
# ============================================================================

cdef class Context:
    """
    Wrapper for ai_context_t handle.

    Provides automatic memory management and Pythonic interface to context operations.
    """
    cdef ai_context_t *_context

    def __cinit__(self):
        """Create a new AI context."""
        with nogil:
            self._context = ai_context_create()
        if self._context == NULL:
            raise MemoryError("Failed to create context")

    def __dealloc__(self):
        """Free the context when Python object is destroyed."""
        if self._context != NULL:
            with nogil:
                ai_context_free(self._context)
            self._context = NULL

    @property
    def handle(self) -> int:
        """Get the underlying C pointer as an integer (for debugging)."""
        return <uintptr_t>self._context

    def get_last_error(self) -> str:
        """
        Get the last error message for this context.

        Returns:
            Human-readable error description
        """
        cdef const char *error
        with nogil:
            error = ai_get_last_error(self._context)
        return _decode_string(error)

    def create_session(self, config: Optional[Dict[str, Any]] = None) -> int:
        """
        Create a new AI session within this context.

        Args:
            config: Optional session configuration dictionary

        Returns:
            Session ID for use with other functions

        Raises:
            Various LibAIError subclasses on failure
        """
        cdef ai_session_config_t c_config
        cdef ai_session_id_t session_id
        cdef bytes instructions_bytes
        cdef bytes tools_json_bytes

        # Prepare config
        if config is None:
            with nogil:
                session_id = ai_create_session(self._context, NULL)
        else:
            # Set defaults
            c_config.instructions = NULL
            c_config.tools_json = NULL
            c_config.enable_guardrails = True
            c_config.prewarm = False

            # Apply user config
            if 'instructions' in config and config['instructions'] is not None:
                instructions_bytes = _encode_string(config['instructions'])
                c_config.instructions = instructions_bytes
            if 'tools_json' in config and config['tools_json'] is not None:
                tools_json_bytes = _encode_string(config['tools_json'])
                c_config.tools_json = tools_json_bytes
            if 'enable_guardrails' in config:
                c_config.enable_guardrails = config['enable_guardrails']
            if 'prewarm' in config:
                c_config.prewarm = config['prewarm']

            with nogil:
                session_id = ai_create_session(self._context, &c_config)

        if session_id == AI_INVALID_ID:
            error_msg = self.get_last_error()
            raise_for_error_code(-99, error_msg)

        return session_id

    def destroy_session(self, int session_id) -> None:
        """
        Destroy a session and release all associated resources.

        Args:
            session_id: Session identifier to destroy
        """
        with nogil:
            ai_destroy_session(self._context, session_id)

    def get_session_history(self, int session_id) -> Optional[str]:
        """
        Get the conversation history for a session as JSON.

        Args:
            session_id: Session identifier

        Returns:
            JSON array of message objects, or None on failure
        """
        cdef char *history
        with nogil:
            history = ai_get_session_history(self._context, session_id)
        if history == NULL:
            return None
        try:
            return _decode_string(history)
        finally:
            ai_free_string(history)

    def clear_session_history(self, int session_id) -> None:
        """
        Clear the conversation history for a session.

        Args:
            session_id: Session identifier

        Raises:
            Various LibAIError subclasses on failure
        """
        cdef ai_result_t result
        with nogil:
            result = ai_clear_session_history(self._context, session_id)
        _check_result(result, self._context)

    def add_message_to_history(self, int session_id, str role, str content) -> None:
        """
        Manually add a message to a session's conversation history.

        Args:
            session_id: Session identifier
            role: Message role ("user", "assistant", "system", or "tool")
            content: Message content text

        Raises:
            Various LibAIError subclasses on failure
        """
        cdef ai_result_t result
        cdef bytes role_bytes = _encode_string(role)
        cdef bytes content_bytes = _encode_string(content)
        cdef const char *role_ptr = role_bytes
        cdef const char *content_ptr = content_bytes
        with nogil:
            result = ai_add_message_to_history(
                self._context, session_id, role_ptr, content_ptr
            )
        _check_result(result, self._context)

    def generate_response(
        self,
        int session_id,
        str prompt,
        params: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Generate a text response from a prompt (synchronous).

        Args:
            session_id: Session identifier
            prompt: Input text prompt
            params: Optional generation parameters

        Returns:
            Generated response text, or None on error

        Raises:
            Various LibAIError subclasses on failure
        """
        cdef ai_generation_params_t c_params
        cdef ai_generation_params_t *params_ptr = NULL
        cdef bytes prompt_bytes = _encode_string(prompt)
        cdef const char *prompt_ptr = prompt_bytes
        cdef char *response

        # Prepare params
        if params is not None:
            c_params.temperature = params.get('temperature', 0.0)
            c_params.max_tokens = params.get('max_tokens', 0)
            c_params.include_reasoning = params.get('include_reasoning', False)
            c_params.seed = params.get('seed', 0)
            params_ptr = &c_params

        with nogil:
            response = ai_generate_response(
                self._context, session_id, prompt_ptr, params_ptr
            )

        if response == NULL:
            error_msg = self.get_last_error()
            raise_for_error_code(-6, error_msg)

        try:
            return _decode_string(response)
        finally:
            ai_free_string(response)

    def generate_structured_response(
        self,
        int session_id,
        str prompt,
        schema_json: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Generate a structured response conforming to a JSON schema.

        Args:
            session_id: Session identifier
            prompt: Input text prompt
            schema_json: JSON schema defining expected response structure
            params: Optional generation parameters

        Returns:
            JSON object with "text" and "object" fields, or None on error

        Raises:
            Various LibAIError subclasses on failure
        """
        cdef ai_generation_params_t c_params
        cdef ai_generation_params_t *params_ptr = NULL
        cdef bytes prompt_bytes = _encode_string(prompt)
        cdef const char *prompt_ptr = prompt_bytes
        cdef bytes schema_bytes
        cdef const char *schema_ptr = NULL
        cdef char *response

        # Prepare schema
        if schema_json is not None:
            schema_bytes = _encode_string(schema_json)
            schema_ptr = schema_bytes

        # Prepare params
        if params is not None:
            c_params.temperature = params.get('temperature', 0.0)
            c_params.max_tokens = params.get('max_tokens', 0)
            c_params.include_reasoning = params.get('include_reasoning', False)
            c_params.seed = params.get('seed', 0)
            params_ptr = &c_params

        with nogil:
            response = ai_generate_structured_response(
                self._context, session_id, prompt_ptr,
                schema_ptr, params_ptr
            )

        if response == NULL:
            error_msg = self.get_last_error()
            raise_for_error_code(-6, error_msg)

        try:
            return _decode_string(response)
        finally:
            ai_free_string(response)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get generation statistics for this context.

        Returns:
            Dictionary with statistics fields
        """
        cdef ai_stats_t stats
        cdef ai_result_t result

        with nogil:
            result = ai_get_stats(self._context, &stats)
        _check_result(result, self._context)

        return {
            'total_requests': stats.total_requests,
            'successful_requests': stats.successful_requests,
            'failed_requests': stats.failed_requests,
            'total_tokens_generated': stats.total_tokens_generated,
            'average_response_time': stats.average_response_time,
            'total_processing_time': stats.total_processing_time,
        }

    def reset_stats(self) -> None:
        """Reset generation statistics for this context."""
        with nogil:
            ai_reset_stats(self._context)

    def generate_response_stream(
        self,
        int session_id,
        str prompt,
        callback: Callable,
        params: Optional[Dict[str, Any]] = None
    ) -> StreamHandle:
        """
        Generate a response with incremental streaming callback.

        Args:
            session_id: Session identifier
            prompt: Input text prompt
            callback: Function called for each response chunk
            params: Optional generation parameters

        Returns:
            StreamHandle for cancellation

        Raises:
            Various LibAIError subclasses on failure
        """
        cdef ai_generation_params_t c_params
        cdef ai_generation_params_t *params_ptr = NULL
        cdef bytes prompt_bytes = _encode_string(prompt)
        cdef const char *prompt_ptr = prompt_bytes
        cdef ai_stream_id_t stream_id
        cdef int callback_id

        # Register callback
        callback_id = _register_stream_callback(callback)

        # Prepare params
        if params is not None:
            c_params.temperature = params.get('temperature', 0.0)
            c_params.max_tokens = params.get('max_tokens', 0)
            c_params.include_reasoning = params.get('include_reasoning', False)
            c_params.seed = params.get('seed', 0)
            params_ptr = &c_params

        with nogil:
            stream_id = ai_generate_response_stream(
                self._context, session_id, prompt_ptr, params_ptr,
                _stream_callback_wrapper, <void*><uintptr_t>callback_id
            )

        if stream_id == AI_INVALID_ID:
            _stream_callbacks.pop(callback_id, None)
            error_msg = self.get_last_error()
            raise_for_error_code(-6, error_msg)

        return StreamHandle(self, stream_id, callback_id)

    def generate_structured_response_stream(
        self,
        int session_id,
        str prompt,
        callback: Callable,
        schema_json: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> StreamHandle:
        """
        Generate a structured response using streaming delivery.

        Args:
            session_id: Session identifier
            prompt: Input text prompt
            callback: Function called with complete structured response
            schema_json: JSON schema defining expected response structure
            params: Optional generation parameters

        Returns:
            StreamHandle for cancellation

        Raises:
            Various LibAIError subclasses on failure
        """
        cdef ai_generation_params_t c_params
        cdef ai_generation_params_t *params_ptr = NULL
        cdef bytes prompt_bytes = _encode_string(prompt)
        cdef const char *prompt_ptr = prompt_bytes
        cdef bytes schema_bytes
        cdef const char *schema_ptr = NULL
        cdef ai_stream_id_t stream_id
        cdef int callback_id

        # Prepare schema
        if schema_json is not None:
            schema_bytes = _encode_string(schema_json)
            schema_ptr = schema_bytes

        # Register callback
        callback_id = _register_stream_callback(callback)

        # Prepare params
        if params is not None:
            c_params.temperature = params.get('temperature', 0.0)
            c_params.max_tokens = params.get('max_tokens', 0)
            c_params.include_reasoning = params.get('include_reasoning', False)
            c_params.seed = params.get('seed', 0)
            params_ptr = &c_params

        with nogil:
            stream_id = ai_generate_structured_response_stream(
                self._context, session_id, prompt_ptr,
                schema_ptr, params_ptr,
                _stream_callback_wrapper, <void*><uintptr_t>callback_id
            )

        if stream_id == AI_INVALID_ID:
            _stream_callbacks.pop(callback_id, None)
            error_msg = self.get_last_error()
            raise_for_error_code(-6, error_msg)

        return StreamHandle(self, stream_id, callback_id)


# ============================================================================
# Streaming support (callback marshalling)
# ============================================================================

# Global storage for stream callbacks
_stream_callbacks: Dict[int, Callable] = {}
_next_callback_id: int = 0


cdef void _stream_callback_wrapper(ai_context_t *context, const char *chunk, void *user_data) noexcept with gil:
    """C callback wrapper that calls Python callback."""
    cdef int callback_id = <int><uintptr_t>user_data
    callback = _stream_callbacks.get(callback_id)
    if callback is None:
        return

    # Convert chunk to Python string
    py_chunk = _decode_string(chunk) if chunk != NULL else None

    # Call Python callback
    try:
        callback(py_chunk)
    except Exception:
        # Silently ignore exceptions in callbacks to avoid crashing C code
        pass

    # Clean up callback if this is the final call (None chunk)
    if chunk == NULL:
        _stream_callbacks.pop(callback_id, None)


def _register_stream_callback(callback: Callable) -> int:
    """Register a Python callback and return its ID."""
    global _next_callback_id
    callback_id = _next_callback_id
    _next_callback_id += 1
    _stream_callbacks[callback_id] = callback
    return callback_id


cdef class StreamHandle:
    """Handle for managing streaming operations."""
    cdef ai_context_t *_context
    cdef ai_stream_id_t _stream_id
    cdef int _callback_id

    def __cinit__(self, Context context, int stream_id, int callback_id):
        self._context = context._context
        self._stream_id = stream_id
        self._callback_id = callback_id

    def cancel(self) -> None:
        """Cancel this stream."""
        cdef ai_result_t result
        with nogil:
            result = ai_cancel_stream(self._context, self._stream_id)
        # Clean up callback
        _stream_callbacks.pop(self._callback_id, None)

    def __dealloc__(self):
        """Clean up callback on destruction."""
        _stream_callbacks.pop(self._callback_id, None)
