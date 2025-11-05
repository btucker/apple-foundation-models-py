# cython: language_level=3
"""
Cython declarations for libai C API.

This file declares the external C types and functions from ai.h
for use in the Cython implementation.
"""

from libc.stdint cimport uint8_t, int32_t, uint32_t, uint64_t
from libc.stdlib cimport malloc, free


# Error codes and status enums
cdef extern from "ai.h":
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
        AI_ERROR_SESSION_NOT_FOUND = -8
        AI_ERROR_STREAM_NOT_FOUND = -9
        AI_ERROR_GUARDRAIL_VIOLATION = -10
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

    # Session configuration
    ctypedef struct ai_session_config_t:
        const char *instructions
        const char *tools_json
        bint enable_guardrails
        bint prewarm

    # Identifier types
    ctypedef uint8_t ai_session_id_t
    ctypedef uint8_t ai_stream_id_t
    const ai_session_id_t AI_INVALID_ID

    # Generation parameters
    ctypedef struct ai_generation_params_t:
        double temperature
        int32_t max_tokens
        bint include_reasoning
        uint32_t seed

    # Context handle (opaque)
    ctypedef struct ai_context:
        pass
    ctypedef ai_context ai_context_t

    # Callback types
    ctypedef void (*ai_stream_callback_t)(ai_context_t *context,
                                          const char *chunk,
                                          void *user_data)
    ctypedef char *(*ai_tool_callback_t)(const char *parameters_json,
                                         void *user_data)

    # Statistics
    ctypedef struct ai_stats_t:
        uint64_t total_requests
        uint64_t successful_requests
        uint64_t failed_requests
        uint64_t total_tokens_generated
        double average_response_time
        double total_processing_time

    # Core library functions
    ai_result_t ai_init() nogil
    void ai_cleanup() nogil
    const char *ai_get_version() nogil
    const char *ai_get_last_error(ai_context_t *context) nogil

    # Availability functions
    ai_availability_t ai_check_availability() nogil
    char *ai_get_availability_reason() nogil
    bint ai_is_ready() nogil

    # Language support
    int32_t ai_get_supported_languages_count() nogil
    char *ai_get_supported_language(int32_t index) nogil

    # Context management
    ai_context_t *ai_context_create() nogil
    void ai_context_free(ai_context_t *context) nogil

    # Session management
    ai_session_id_t ai_create_session(ai_context_t *context,
                                      const ai_session_config_t *config) nogil
    ai_result_t ai_register_tool(ai_context_t *context,
                                 ai_session_id_t session_id,
                                 const char *tool_name,
                                 ai_tool_callback_t callback,
                                 void *user_data) nogil
    void ai_destroy_session(ai_context_t *context,
                           ai_session_id_t session_id) nogil
    char *ai_get_session_history(ai_context_t *context,
                                 ai_session_id_t session_id) nogil
    ai_result_t ai_clear_session_history(ai_context_t *context,
                                        ai_session_id_t session_id) nogil
    ai_result_t ai_add_message_to_history(ai_context_t *context,
                                         ai_session_id_t session_id,
                                         const char *role,
                                         const char *content) nogil

    # Text generation (synchronous)
    char *ai_generate_response(ai_context_t *context,
                              ai_session_id_t session_id,
                              const char *prompt,
                              const ai_generation_params_t *params) nogil
    char *ai_generate_structured_response(ai_context_t *context,
                                         ai_session_id_t session_id,
                                         const char *prompt,
                                         const char *schema_json,
                                         const ai_generation_params_t *params) nogil

    # Streaming generation (asynchronous)
    ai_stream_id_t ai_generate_response_stream(ai_context_t *context,
                                              ai_session_id_t session_id,
                                              const char *prompt,
                                              const ai_generation_params_t *params,
                                              ai_stream_callback_t callback,
                                              void *user_data) nogil
    ai_stream_id_t ai_generate_structured_response_stream(ai_context_t *context,
                                                         ai_session_id_t session_id,
                                                         const char *prompt,
                                                         const char *schema_json,
                                                         const ai_generation_params_t *params,
                                                         ai_stream_callback_t callback,
                                                         void *user_data) nogil

    # Stream control
    ai_result_t ai_cancel_stream(ai_context_t *context,
                                ai_stream_id_t stream_id) nogil

    # Utilities
    bint ai_validate_messages_json(const char *messages_json) nogil

    # Memory management
    void ai_free_string(char *str) nogil

    # Error handling
    const char *ai_get_error_description(ai_result_t result) nogil
    void ai_set_error_handler(ai_context_t *context,
                             void (*handler)(ai_result_t result,
                                           const char *message)) nogil

    # Statistics
    ai_result_t ai_get_stats(ai_context_t *context, ai_stats_t *stats) nogil
    void ai_reset_stats(ai_context_t *context) nogil
