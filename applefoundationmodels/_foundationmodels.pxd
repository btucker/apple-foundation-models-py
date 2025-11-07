# cython: language_level=3
"""
Cython declarations for Apple AI Swift C API.

This file declares the external C types and functions exported from
the Swift FoundationModels wrapper for use in the Cython implementation.
"""

from libc.stdint cimport int32_t
from libc.stdlib cimport malloc, free


# Error codes and status enums matching Swift enums
cdef extern from "../applefoundationmodels/swift/foundation_models.h":
    # Result codes
    ctypedef enum ai_result_t:
        AI_SUCCESS = 0
        AI_ERROR_INIT_FAILED = -1
        AI_ERROR_NOT_AVAILABLE = -2
        AI_ERROR_INVALID_PARAMS = -3
        AI_ERROR_MEMORY = -4
        AI_ERROR_JSON_PARSE = -5
        AI_ERROR_GENERATION = -6
        AI_ERROR_TIMEOUT = -7
        AI_ERROR_TOOL_NOT_FOUND = -11
        AI_ERROR_TOOL_EXECUTION = -12
        AI_ERROR_UNKNOWN = -99

    # Availability status
    ctypedef enum ai_availability_t:
        AI_AVAILABLE = 1
        AI_DEVICE_NOT_ELIGIBLE = -1
        AI_NOT_ENABLED = -2
        AI_MODEL_NOT_READY = -3
        AI_AVAILABILITY_UNKNOWN = -99

    # Callback type for streaming
    ctypedef void (*ai_stream_callback_t)(const char *chunk)

    # Callback type for tool execution
    ctypedef int32_t (*ai_tool_callback_t)(const char *tool_name,
                                           const char *arguments_json,
                                           char *result_buffer,
                                           int32_t buffer_size)

    # Core library functions
    int32_t apple_ai_init() nogil
    void apple_ai_cleanup() nogil
    const char *apple_ai_get_version() nogil

    # Availability functions
    int32_t apple_ai_check_availability() nogil
    char *apple_ai_get_availability_reason() nogil

    # Session management
    int32_t apple_ai_create_session(const char *instructions_json) nogil

    # Tool calling
    int32_t apple_ai_register_tools(const char *tools_json,
                                    ai_tool_callback_t callback) nogil
    char *apple_ai_get_transcript() nogil

    # Text generation
    char *apple_ai_generate(const char *prompt,
                           double temperature,
                           int32_t max_tokens) nogil

    # Streaming generation
    int32_t apple_ai_generate_stream(const char *prompt,
                                    double temperature,
                                    int32_t max_tokens,
                                    ai_stream_callback_t callback) nogil

    # Structured generation
    char *apple_ai_generate_structured(const char *prompt,
                                       const char *schema_json,
                                       double temperature,
                                       int32_t max_tokens) nogil

    # History management
    char *apple_ai_get_history() nogil
    void apple_ai_clear_history() nogil

    # Statistics (stub for compatibility)
    char *apple_ai_get_stats() nogil
    void apple_ai_reset_stats() nogil

    # Memory management
    void apple_ai_free_string(char *ptr) nogil
