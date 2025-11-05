/**
 * @file ai.h
 * @brief Apple Intelligence C Library
 *
 * High-level C interface for Apple Intelligence functionality. This library
 * provides a comprehensive, thread-safe wrapper around the Apple Intelligence
 * Bridge, offering context management, session handling, and simplified error
 * handling.
 *
 * @section overview Overview
 * The ai library provides:
 * - Context-based session management with isolation between contexts
 * - Comprehensive error handling and logging
 * - Thread-safe operations with proper synchronization
 * - Statistics tracking and monitoring capabilities
 * - Simplified tool registration and callback management
 * - Both synchronous and asynchronous generation modes
 *
 * @section usage Basic Usage
 * @code
 * // Initialize library
 * ai_init();
 *
 * // Create context and session
 * ai_context_t *ctx = ai_context_create();
 * ai_session_id_t session = ai_create_session(ctx, NULL);
 *
 * // Generate response
 * char *response = ai_generate_response(ctx, session, "Hello", NULL);
 * ai_free_string(response);
 *
 * // Cleanup
 * ai_context_free(ctx);
 * ai_cleanup();
 * @endcode
 *
 * @section memory Memory Management
 * - All functions returning char* require freeing with ai_free_string()
 * - Context objects must be freed with ai_context_free()
 * - Tool callbacks must return malloc-allocated strings
 * - The library handles internal memory management automatically
 *
 * @section threading Thread Safety
 * - All public functions are thread-safe
 * - Multiple contexts can be used concurrently from different threads
 * - Callbacks may be invoked from background threads
 * - Session operations within a context are serialized
 *
 * @section errors Error Handling
 * - Functions return ai_result_t codes for operation status
 * - Detailed error messages available via ai_get_last_error()
 * - Custom error handlers can be registered per context
 * - Automatic error code translation from underlying bridge
 */

#ifndef AI_H
#define AI_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @} */

/**
 * @defgroup errors Error Codes and Status
 * @{
 */

/**
 * @brief Result codes for ai operations
 *
 * These codes indicate the success or failure state of library operations.
 * Use ai_get_error_description() for human-readable descriptions.
 */
typedef enum {
  AI_SUCCESS = 0,            /**< Operation completed successfully */
  AI_ERROR_INIT_FAILED = -1, /**< Library initialization failed */
  AI_ERROR_NOT_AVAILABLE =
      -2, /**< Apple Intelligence not available on this device */
  AI_ERROR_INVALID_PARAMS = -3, /**< Invalid parameters provided to function */
  AI_ERROR_MEMORY = -4,         /**< Memory allocation error */
  AI_ERROR_JSON_PARSE = -5,     /**< JSON parsing or validation error */
  AI_ERROR_GENERATION = -6,     /**< Text generation error */
  AI_ERROR_TIMEOUT = -7,        /**< Operation timeout */
  AI_ERROR_SESSION_NOT_FOUND = -8, /**< Session ID not found in context */
  AI_ERROR_STREAM_NOT_FOUND =
      -9, /**< Stream ID not found or already completed */
  AI_ERROR_GUARDRAIL_VIOLATION = -10, /**< Content blocked by safety filters */
  AI_ERROR_TOOL_NOT_FOUND =
      -11, /**< Tool callback not registered for session */
  AI_ERROR_TOOL_EXECUTION =
      -12, /**< Tool execution failed or returned invalid result */
  AI_ERROR_UNKNOWN = -99 /**< Unknown error occurred */
} ai_result_t;

/**
 * @brief Apple Intelligence availability status
 *
 * Indicates whether Apple Intelligence is available and ready for use
 * on the current device and system configuration.
 */
typedef enum {
  AI_AVAILABLE = 1,            /**< Available and ready for use */
  AI_DEVICE_NOT_ELIGIBLE = -1, /**< Device hardware not supported */
  AI_NOT_ENABLED = -2,         /**< Feature not enabled in system settings */
  AI_MODEL_NOT_READY = -3, /**< AI model still downloading or initializing */
  AI_AVAILABILITY_UNKNOWN = -99 /**< Unknown availability status */
} ai_availability_t;

/**
 * @brief Session configuration structure
 *
 * Configuration options for creating an ai session. Sessions
 * maintain conversation state and can be configured with tools and
 * instructions.
 */
typedef struct {
  const char *instructions; /**< Optional system instructions to guide AI
                               behavior (can be NULL) */
  const char *tools_json; /**< Optional JSON array of tool definitions in Claude
                             format (can be NULL) */
  bool enable_guardrails; /**< Whether to enable content safety filtering */
  bool prewarm; /**< Whether to preload session resources for faster first
                   response */
} ai_session_config_t;

/**
 * @brief Default session configuration with safe defaults
 *
 * Provides reasonable defaults: no custom instructions, no tools,
 * guardrails enabled, no prewarming.
 */
#define AI_DEFAULT_SESSION_CONFIG \
  {.instructions = NULL,          \
   .tools_json = NULL,            \
   .enable_guardrails = true,     \
   .prewarm = false}

/** @} */

/**
 * @defgroup identifiers Identifier Types
 * @{
 */

/**
 * @brief Session identifier type
 *
 * Unique identifier for sessions within a context. Valid IDs are non-zero.
 * Sessions are automatically destroyed when their context is freed.
 */
typedef uint8_t ai_session_id_t;

/**
 * @brief Stream identifier type
 *
 * Unique identifier for streaming operations. Valid IDs are non-zero.
 * Streams automatically clean up when generation completes or is cancelled.
 */
typedef uint8_t ai_stream_id_t;

/**
 * @brief Invalid session/stream identifier
 *
 * Returned by functions when session/stream creation fails.
 * Used to check for invalid operations.
 */
#define AI_INVALID_ID 0

/** @} */

/**
 * @defgroup params Generation Parameters
 * @{
 */

/**
 * @brief Text generation parameters
 *
 * Controls various aspects of AI text generation including randomness,
 * length limits, and future extensibility options.
 */
typedef struct {
  double temperature; /**< Generation randomness (0.0 = deterministic, 2.0 =
                         very random, 0 = use default) */
  int32_t
      max_tokens; /**< Maximum response tokens (0 = use system default limit) */
  bool include_reasoning; /**< Include reasoning in response (reserved for
                             future use) */
  uint32_t seed; /**< Random seed for reproducibility (0 = use random seed) */
} ai_generation_params_t;

/**
 * @brief Default generation parameters
 *
 * Uses system defaults for all settings: default temperature, default token
 * limit, no reasoning, random seed.
 */
#define AI_DEFAULT_PARAMS {0.0, 0, false, 0}

/** @} */

/**
 * @defgroup context Context Management
 * @{
 */

/**
 * @brief Opaque context handle for ai operations
 *
 * Contexts provide isolation between different parts of an application.
 * Each context maintains its own sessions, error state, and statistics.
 * All operations are thread-safe across different contexts.
 */
typedef struct ai_context ai_context_t;

/**
 * @brief Create a new ai context
 *
 * Creates an isolated context for AI operations. Each context can manage
 * multiple sessions and maintains separate error states and statistics.
 *
 * @return Context handle for use with other functions, or NULL if creation
 * failed
 *
 * @note Context creation requires successful library initialization.
 * @note Call ai_context_free() to release resources.
 */
ai_context_t *ai_context_create(void);

/**
 * @brief Free an ai context and all associated sessions
 *
 * Destroys all sessions within the context and releases all associated
 * resources. Active streams are automatically cancelled. This function is safe
 * to call multiple times and with NULL contexts.
 *
 * @param context Context to free. Passing NULL is safe and ignored.
 *
 * @note All session IDs from this context become invalid after this call.
 * @note This function blocks until all active operations complete.
 */
void ai_context_free(ai_context_t *context);

/** @} */

/**
 * @defgroup callbacks Callback Types
 * @{
 */

/**
 * @brief Callback function for streaming text generation
 *
 * Called incrementally during streaming generation for each token or chunk.
 * The callback should process the chunk quickly and avoid blocking operations.
 *
 * @param context Context handle for this operation
 * @param chunk Text chunk to process. NULL indicates completion or error.
 *              Check for "Error:" prefix to distinguish error messages.
 * @param user_data User-provided data pointer passed to the streaming function
 *
 * @note The chunk string is only valid during the callback invocation.
 * @note Callbacks may be invoked from background threads.
 * @note A NULL chunk indicates either successful completion or an error
 * condition.
 */
typedef void (*ai_stream_callback_t)(ai_context_t *context, const char *chunk,
                                     void *user_data);

/**
 * @brief Callback function for tool execution
 *
 * Called when the AI requests to execute a registered tool. The callback
 * should process the provided parameters and return a result that will be
 * sent back to the AI.
 *
 * @param parameters_json JSON string containing tool parameters as defined in
 * the tool schema
 * @param user_data User data pointer provided during tool registration
 * @return JSON result string. **Memory ownership**: Must be allocated with
 * malloc() or strdup(). The library will call free() on the returned pointer.
 * Return NULL to indicate error.
 *
 * @note The parameters_json string is only valid during the callback.
 * @note Return value should be valid JSON or a plain string.
 * @note Tool callbacks may be invoked from background threads.
 * @note Ensure thread safety if accessing shared data from tool callbacks.
 */
typedef char *(*ai_tool_callback_t)(const char *parameters_json,
                                    void *user_data);

/** @} */

/**
 * @defgroup core Core Library Functions
 * @{
 */

/**
 * @brief Initialize the ai library
 *
 * Must be called before using any other library functions. Initializes
 * the underlying Apple Intelligence bridge and prepares the library for use.
 * This function is idempotent and safe to call multiple times.
 *
 * @return AI_SUCCESS on successful initialization, error code on failure
 *
 * @note Initialization may fail if Apple Intelligence is not available.
 * @note This function is thread-safe but should typically be called once at
 * startup.
 */
ai_result_t ai_init(void);

/**
 * @brief Cleanup and shutdown the ai library
 *
 * Performs final cleanup of library resources. Should be called when the
 * application is done using the library. This function is idempotent.
 *
 * @note All contexts should be freed before calling this function.
 * @note Active operations may be cancelled during cleanup.
 */
void ai_cleanup(void);

/**
 * @brief Get library version string
 *
 * Returns the semantic version of the ai library.
 *
 * @return Version string in format "major.minor.patch". **Memory ownership**:
 * Do not free.
 */
const char *ai_get_version(void);

/**
 * @brief Get the last error message for a context
 *
 * Retrieves the most recent error message for the specified context.
 * Error messages provide detailed information about operation failures.
 *
 * @param context Context to get error message for. NULL retrieves global error.
 * @return Human-readable error description. **Memory ownership**: Do not free.
 *
 * @note Error messages are thread-local within each context.
 * @note The returned string remains valid until the next error in the same
 * context.
 */
const char *ai_get_last_error(ai_context_t *context);

/** @} */

/**
 * @defgroup availability Availability Functions
 * @{
 */

/**
 * @brief Check Apple Intelligence availability on this device
 *
 * Determines whether Apple Intelligence is available and ready for use.
 * This check is performed in real-time and may change based on system state.
 *
 * @return Availability status from ai_availability_t enum
 *
 * @note This function can be called before library initialization.
 * @note Availability may change if system settings are modified.
 */
ai_availability_t ai_check_availability(void);

/**
 * @brief Get detailed availability status message
 *
 * Provides a human-readable explanation of the current availability status,
 * including specific steps to resolve issues when Apple Intelligence is not
 * available.
 *
 * @return Detailed status description. **Memory ownership**: Caller must call
 * ai_free_string().
 *
 * @note Returns NULL if library is not initialized.
 * @note The message includes actionable guidance for resolving availability
 * issues.
 */
char *ai_get_availability_reason(void);

/**
 * @brief Check if Apple Intelligence is ready for immediate use
 *
 * Convenience function that returns true only if Apple Intelligence is
 * fully available and ready for generation requests.
 *
 * @return true if ready for use, false if unavailable or not ready
 */
bool ai_is_ready(void);

/** @} */

/**
 * @defgroup language Language Support
 * @{
 */

/**
 * @brief Get the number of languages supported by Apple Intelligence
 *
 * Returns the count of languages that the current Apple Intelligence model
 * can understand and generate text in.
 *
 * @return Number of supported languages (0 if library not initialized)
 */
int32_t ai_get_supported_languages_count(void);

/**
 * @brief Get the display name of a supported language by index
 *
 * Retrieves the localized display name for a language supported by
 * Apple Intelligence. Use ai_get_supported_languages_count()
 * to determine the valid index range.
 *
 * @param index Zero-based language index
 * @return Localized language display name, or NULL if index is invalid.
 *         **Memory ownership**: Caller must call ai_free_string().
 */
char *ai_get_supported_language(int32_t index);

/** @} */

/**
 * @defgroup sessions Session Management
 * @{
 */

/**
 * @brief Create a new AI session within a context
 *
 * Creates an AI session that maintains conversation state and tool
 * registrations. Sessions are isolated from each other and can have different
 * configurations. Each context can manage multiple sessions up to the system
 * limit.
 *
 * @param context Context to create the session in
 * @param config Session configuration options. NULL uses default configuration.
 * @return Session identifier for use with other functions, or AI_INVALID_ID on
 * failure
 *
 * @note The session automatically maintains conversation history.
 * @note Tools defined in config->tools_json must be registered with
 * ai_register_tool().
 * @note Session creation fails if Apple Intelligence is not available.
 */
ai_session_id_t ai_create_session(ai_context_t *context,
                                  const ai_session_config_t *config);

/**
 * @brief Register a tool callback function for a session
 *
 * Associates a C callback function with a tool name, enabling the AI to invoke
 * the tool during generation. The tool must be defined in the session's
 * tools_json configuration.
 *
 * @param context Context containing the session
 * @param session_id Session identifier returned by ai_create_session()
 * @param tool_name Name of the tool as defined in the session's tools_json
 * @param callback Function to call when the AI invokes this tool
 * @param user_data Optional pointer passed to the callback function
 * @return AI_SUCCESS on successful registration, error code on failure
 *
 * @note Multiple tools can be registered for the same session.
 * @note Tool callbacks may be invoked concurrently; ensure thread safety if
 * needed.
 * @note The tool_name must match exactly with a tool defined in the session
 * configuration.
 */
ai_result_t ai_register_tool(ai_context_t *context, ai_session_id_t session_id,
                             const char *tool_name, ai_tool_callback_t callback,
                             void *user_data);

/**
 * @brief Destroy a session and release all associated resources
 *
 * Invalidates the session ID and cancels any active streams associated with
 * the session. This function is idempotent; calling it multiple times with
 * the same session ID is safe.
 *
 * @param context Context containing the session
 * @param session_id Session identifier to destroy
 *
 * @note Active generation operations for this session are cancelled.
 * @note The session ID becomes invalid immediately after this call.
 */
void ai_destroy_session(ai_context_t *context, ai_session_id_t session_id);

/**
 * @brief Get the conversation history for a session as JSON
 *
 * Returns the complete conversation history including user prompts, AI
 * responses, and tool interactions in a structured JSON format compatible with
 * OpenAI's chat completion API.
 *
 * @param context Context containing the session
 * @param session_id Session identifier
 * @return JSON array of message objects with "role" and "content" fields, or
 * NULL on failure.
 *         **Memory ownership**: Caller must call ai_free_string().
 *
 * @note The returned JSON follows OpenAI chat completion message format.
 * @note History is automatically maintained; this function retrieves the
 * current state.
 */
char *ai_get_session_history(ai_context_t *context, ai_session_id_t session_id);

/**
 * @brief Clear the conversation history for a session
 *
 * Removes all messages from the session's conversation history while preserving
 * the session configuration, system instructions, and registered tools.
 *
 * @param context Context containing the session
 * @param session_id Session identifier
 * @return AI_SUCCESS if history was cleared, error code on failure
 *
 * @note This operation may be a no-op depending on the underlying
 * implementation.
 * @note System instructions remain active after clearing history.
 */
ai_result_t ai_clear_session_history(ai_context_t *context,
                                     ai_session_id_t session_id);

/**
 * @brief Manually add a message to a session's conversation history
 *
 * Adds a message to the session history without generating a response.
 * This function is provided for compatibility and advanced use cases, but
 * is typically unnecessary since the session automatically manages its history.
 *
 * @param context Context containing the session
 * @param session_id Session identifier
 * @param role Message role: "user", "assistant", "system", or "tool"
 * @param content Message content text
 * @return AI_SUCCESS if message was added, error code on failure
 *
 * @note The session automatically tracks all interactions, so manual history
 * management is rarely needed.
 * @note This function exists primarily for compatibility and advanced usage
 * patterns.
 */
ai_result_t ai_add_message_to_history(ai_context_t *context,
                                      ai_session_id_t session_id,
                                      const char *role, const char *content);

/** @} */

/**
 * @defgroup generation Text Generation
 * @{
 */

/**
 * @brief Generate a text response from a prompt (synchronous)
 *
 * Sends a prompt to the AI and waits for the complete response. This function
 * blocks until generation is complete, fails, or times out. The response is
 * automatically added to the session's conversation history.
 *
 * @param context Context for error reporting and statistics
 * @param session_id Session identifier
 * @param prompt Input text prompt to send to the AI
 * @param params Generation parameters controlling temperature, length, etc.
 * NULL uses defaults.
 * @return Generated response text, or NULL on error.
 *         **Memory ownership**: Caller must call ai_free_string().
 *
 * @note This function may take several seconds to complete for long responses.
 * @note Check ai_get_last_error() for detailed error information on failure.
 * @note The prompt and response are automatically added to session history.
 */
char *ai_generate_response(ai_context_t *context, ai_session_id_t session_id,
                           const char *prompt,
                           const ai_generation_params_t *params);

/**
 * @brief Generate a structured response conforming to a JSON schema
 * (synchronous)
 *
 * Generates a response that conforms to the specified JSON schema structure.
 * Useful for extracting structured data or ensuring consistent response
 * formats. The result includes both a text representation and the structured
 * object.
 *
 * @param context Context for error reporting and statistics
 * @param session_id Session identifier
 * @param prompt Input text prompt to send to the AI
 * @param schema_json JSON schema defining the expected response structure. NULL
 * uses session default.
 * @param params Generation parameters controlling temperature, length, etc.
 * NULL uses defaults.
 * @return JSON object containing "text" (string representation) and "object"
 * (structured data) fields, or NULL on error. **Memory ownership**: Caller must
 * call ai_free_string().
 *
 * @note Structured generation may be slower than regular text generation.
 * @note The schema_json must be valid JSON Schema format.
 * @note If schema_json is NULL, the session must have been created with a
 * default schema.
 */
char *ai_generate_structured_response(ai_context_t *context,
                                      ai_session_id_t session_id,
                                      const char *prompt,
                                      const char *schema_json,
                                      const ai_generation_params_t *params);

/** @} */

/**
 * @defgroup streaming Streaming Generation
 * @{
 */

/**
 * @brief Generate a response with incremental streaming callback
 *
 * Begins generating a response and calls the callback function for each token
 * or chunk as it becomes available. This function returns immediately while
 * generation continues asynchronously in the background.
 *
 * @param context Context for error reporting and statistics
 * @param session_id Session identifier
 * @param prompt Input text prompt to send to the AI
 * @param params Generation parameters controlling temperature, length, etc.
 * NULL uses defaults.
 * @param callback Function called for each response chunk and completion/error
 * @param user_data User data passed to the callback function
 * @return Stream identifier for cancellation, or AI_INVALID_ID if streaming
 * failed to start
 *
 * @note The callback may be invoked from a background thread.
 * @note Use ai_cancel_stream() to stop generation early.
 * @note The stream automatically cleans up when generation completes.
 * @note The complete response is automatically added to session history.
 */
ai_stream_id_t ai_generate_response_stream(ai_context_t *context,
                                           ai_session_id_t session_id,
                                           const char *prompt,
                                           const ai_generation_params_t *params,
                                           ai_stream_callback_t callback,
                                           void *user_data);

/**
 * @brief Generate a structured response using streaming delivery
 *
 * Generates a structured response conforming to the JSON schema and delivers
 * the complete result via callback when ready. Unlike text streaming, this
 * calls the callback once with the entire structured response.
 *
 * @param context Context for error reporting and statistics
 * @param session_id Session identifier
 * @param prompt Input text prompt to send to the AI
 * @param schema_json JSON schema defining expected response structure. NULL
 * uses session default.
 * @param params Generation parameters controlling temperature, length, etc.
 * NULL uses defaults.
 * @param callback Function called once with the complete structured response
 * @param user_data User data passed to the callback function
 * @return Stream identifier for cancellation, or AI_INVALID_ID if streaming
 * failed to start
 *
 * @note Unlike text streaming, this calls the callback only once with the
 * complete result.
 * @note The callback receives a JSON object with "text" and "object" fields.
 * @note The structured response is automatically added to session history.
 */
ai_stream_id_t ai_generate_structured_response_stream(
    ai_context_t *context, ai_session_id_t session_id, const char *prompt,
    const char *schema_json, const ai_generation_params_t *params,
    ai_stream_callback_t callback, void *user_data);

/** @} */

/**
 * @defgroup async Stream Control
 * @{
 */

/**
 * @brief Cancel an active streaming operation
 *
 * Attempts to cancel the specified stream. If successful, the stream's callback
 * will be called once more with a NULL chunk to indicate cancellation.
 * Cancellation may not be immediate.
 *
 * @param context Context containing the stream
 * @param stream_id Stream identifier returned by a streaming function
 * @return AI_SUCCESS if the stream was found and cancelled, error code
 * otherwise
 *
 * @note Cancellation may not be immediate; additional chunks may be delivered.
 * @note This function is safe to call multiple times with the same stream ID.
 * @note Completed streams return AI_ERROR_STREAM_NOT_FOUND.
 */
ai_result_t ai_cancel_stream(ai_context_t *context, ai_stream_id_t stream_id);

/** @} */

/**
 * @defgroup utilities Utility Functions
 * @{
 */

/**
 * @brief Validate JSON message format for compatibility
 *
 * Checks whether the provided JSON string represents a valid message array
 * format compatible with the library's expectations.
 *
 * @param messages_json JSON string to validate
 * @return true if the JSON appears to be a valid message array, false otherwise
 *
 * @note This performs basic format validation, not comprehensive JSON parsing.
 * @note Useful for validating externally-generated message histories.
 */
bool ai_validate_messages_json(const char *messages_json);

/** @} */

/**
 * @defgroup memory Memory Management
 * @{
 */

/**
 * @brief Free string memory allocated by library functions
 *
 * Releases memory allocated by any ai library function that returns
 * a char pointer. This function must be called for all strings returned by
 * the library to prevent memory leaks.
 *
 * @param str String pointer to free. Passing NULL is safe and ignored.
 *
 * @note Do not call this function on strings not allocated by the ai library.
 * @note This function is thread-safe and can be called from any thread.
 */
void ai_free_string(char *str);

/** @} */

/**
 * @defgroup errors Error Handling
 * @{
 */

/**
 * @brief Get a human-readable description for a result code
 *
 * Converts an ai_result_t error code into a descriptive string
 * suitable for display to users or logging.
 *
 * @param result Result code to describe
 * @return Human-readable error description. **Memory ownership**: Do not free.
 */
const char *ai_get_error_description(ai_result_t result);

/**
 * @brief Set a custom error handler for a context
 *
 * Registers a callback function that will be invoked whenever an error
 * occurs within the specified context. This allows applications to implement
 * custom error logging, reporting, or recovery mechanisms.
 *
 * @param context Context to set the error handler for
 * @param handler Error handler function. NULL removes the current handler.
 *
 * @note The error handler is called synchronously when errors occur.
 * @note Error handlers should not perform blocking operations or call back into
 * the library.
 * @note Only one error handler per context is supported; setting a new handler
 * replaces the previous one.
 */
void ai_set_error_handler(ai_context_t *context,
                          void (*handler)(ai_result_t result,
                                          const char *message));

/** @} */

/**
 * @defgroup stats Statistics and Monitoring
 * @{
 */

/**
 * @brief Generation statistics and performance metrics
 *
 * Provides insights into the usage and performance of generation operations
 * within a context. Useful for monitoring, debugging, and optimization.
 */
typedef struct {
  uint64_t total_requests; /**< Total number of generation requests initiated */
  uint64_t successful_requests; /**< Number of requests that completed
                                   successfully */
  uint64_t
      failed_requests; /**< Number of requests that failed or were cancelled */
  uint64_t total_tokens_generated; /**< Total tokens generated across all
                                      requests (may be 0 if not tracked) */
  double average_response_time;    /**< Average response time in seconds (may be
                                      0.0 if not tracked) */
  double total_processing_time;    /**< Total processing time in seconds (may be
                                      0.0 if not tracked) */
} ai_stats_t;

/**
 * @brief Get generation statistics for a context
 *
 * Retrieves current usage statistics for the specified context, including
 * request counts, success rates, and performance metrics where available.
 *
 * @param context Context to get statistics for
 * @param stats Pointer to statistics structure to populate
 * @return AI_SUCCESS on success, error code on failure
 *
 * @note Some statistics fields may be zero if not implemented or tracked.
 * @note Statistics are accumulated since context creation or last reset.
 */
ai_result_t ai_get_stats(ai_context_t *context, ai_stats_t *stats);

/**
 * @brief Reset generation statistics for a context
 *
 * Clears all accumulated statistics for the specified context, resetting
 * counters to zero. This is useful for measuring performance over specific
 * time periods or after configuration changes.
 *
 * @param context Context to reset statistics for
 *
 * @note This operation does not affect active generation operations.
 * @note Statistics reset is immediate and atomic.
 */
void ai_reset_stats(ai_context_t *context);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* AI_H */