/**
 * @file ai_bridge.h
 * @brief Apple Intelligence Bridge Library
 *
 * C interface for accessing Apple Intelligence functionality through
 * FoundationModels. Requires macOS 26.0 or later.
 *
 * This library provides a C interface to Apple's FoundationModels framework,
 * enabling integration of Apple Intelligence capabilities into C/C++
 * applications. All functions are thread-safe unless otherwise noted.
 *
 * @section memory Memory Management
 * Functions that return char* pointers allocate memory that must be freed using
 * ai_bridge_free_string(). Tool callbacks must return malloc-allocated strings.
 *
 * @section sessions Sessions
 * Sessions maintain conversation state and tool registrations. Each session has
 * a unique identifier that remains valid until ai_bridge_destroy_session() is
 * called.
 *
 * @section streaming Streaming
 * Streaming functions return immediately and call the provided callback for
 * each response chunk. Use ai_bridge_cancel_stream() to stop generation early.
 */

#ifndef AI_BRIDGE_H
#define AI_BRIDGE_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Apple Intelligence availability status codes
 */
typedef enum {
  AI_BRIDGE_AVAILABLE = 1, /**< Apple Intelligence is available and ready */
  AI_BRIDGE_DEVICE_NOT_ELIGIBLE =
      -1, /**< Device not eligible for Apple Intelligence */
  AI_BRIDGE_INTELLIGENCE_NOT_ENABLED =
      -2, /**< Apple Intelligence not enabled in settings */
  AI_BRIDGE_MODEL_NOT_READY = -3, /**< AI model not ready (still downloading) */
  AI_BRIDGE_UNKNOWN_ERROR = -99   /**< Unknown error occurred */
} ai_availability_status_t;

/**
 * @brief Session identifier type
 *
 * Unique identifier for AI sessions. Valid IDs are non-zero values.
 * Use AI_BRIDGE_INVALID_ID to check for invalid sessions.
 */
typedef uint8_t ai_bridge_session_id_t;

/**
 * @brief Stream identifier type
 *
 * Unique identifier for streaming operations. Valid IDs are non-zero values.
 * Use AI_BRIDGE_INVALID_ID to check for invalid streams.
 */
typedef uint8_t ai_bridge_stream_id_t;

/**
 * @brief Invalid session/stream identifier
 *
 * Returned by functions when session/stream creation fails.
 */
#define AI_BRIDGE_INVALID_ID 0

/**
 * @brief Callback function type for streaming response chunks
 *
 * Called for each token or chunk during streaming generation. The callback
 * receives incremental content and should not block for extended periods.
 *
 * @param context Context pointer passed from the calling function
 * @param chunk Response chunk string, or NULL to indicate completion/error
 * @param user_data User data pointer passed from the calling function
 *
 * @note The chunk string is only valid during the callback. Copy if needed.
 * @note A NULL chunk indicates either successful completion or an error.
 *       Check for "Error:" prefix to distinguish error messages.
 */
typedef void (*ai_bridge_stream_callback_t)(void *context, const char *chunk,
                                            void *user_data);

/**
 * @brief Callback function type for tool execution
 *
 * Called when the AI requests to execute a registered tool. The callback
 * should process the parameters and return a result.
 *
 * @param parameters_json JSON string containing tool parameters
 * @param user_data User data pointer passed during tool registration
 * @return JSON result string. **Memory ownership**: Must be allocated with
 * malloc() or strdup(). The bridge will call free() on the returned pointer.
 * Return NULL on error.
 *
 * @note The parameters_json string is only valid during the callback.
 * @note Return value should be valid JSON or a plain string that will be
 * wrapped in JSON.
 */
typedef char *(*ai_bridge_tool_callback_t)(const char *parameters_json,
                                           void *user_data);

/**
 * @brief Initialize the Apple Intelligence bridge library
 *
 * Must be called before using any other bridge functions. This function
 * is idempotent and can be called multiple times safely.
 *
 * @return true if initialization succeeded, false otherwise
 */
bool ai_bridge_init(void);

/**
 * @brief Check Apple Intelligence model availability
 *
 * Determines whether Apple Intelligence is available on the current device
 * and ready for use. This check is performed each time the function is called.
 *
 * @return Availability status code from ai_availability_status_t enum
 */
ai_availability_status_t ai_bridge_check_availability(void);

/**
 * @brief Get detailed availability status message
 *
 * Provides a human-readable explanation of the current availability status,
 * including specific steps to resolve issues when applicable.
 *
 * @return Human-readable string describing availability status.
 *         **Memory ownership**: Caller must call ai_bridge_free_string() to
 * release.
 */
char *ai_bridge_get_availability_reason(void);

/**
 * @brief Get count of supported languages
 *
 * Returns the number of languages supported by the current Apple Intelligence
 * model. Use with ai_bridge_get_supported_language() to enumerate all supported
 * languages.
 *
 * @return Number of supported languages (always >= 0)
 */
int32_t ai_bridge_get_supported_languages_count(void);

/**
 * @brief Get supported language display name at the specified index
 *
 * Retrieves the localized display name for a supported language.
 * Use ai_bridge_get_supported_languages_count() to determine valid index range.
 *
 * @param index Zero-based language index
 * @return Localized language display name, or NULL if index is out of bounds.
 *         **Memory ownership**: Caller must call ai_bridge_free_string() to
 * release.
 */
char *ai_bridge_get_supported_language(int32_t index);

/**
 * @brief Create a new AI session with the specified configuration
 *
 * Creates an AI session that maintains conversation state and tool
 * registrations. Sessions are isolated from each other and can have different
 * configurations.
 *
 * @param instructions Optional system instructions to guide AI behavior. Can be
 * NULL.
 * @param tools_json Optional JSON array defining available tools in Claude tool
 * format. Can be NULL. Example: [{"name": "calculator", "description": "...",
 * "input_schema": {...}}]
 * @param enable_guardrails Whether to enable content safety filtering and
 * guardrails
 * @param prewarm Whether to preload session resources for faster first response
 * @return Session identifier for use with other functions, or
 * AI_BRIDGE_INVALID_ID on failure
 *
 * @note The session automatically manages conversation history.
 * @note Tools defined in tools_json must be registered with
 * ai_bridge_register_tool().
 */
ai_bridge_session_id_t ai_bridge_create_session(const char *instructions,
                                                const char *tools_json,
                                                bool enable_guardrails,
                                                bool prewarm);

/**
 * @brief Register a tool callback function for the specified session
 *
 * Associates a C callback function with a tool name, enabling the AI to call
 * the tool during generation. The tool must be defined in the session's
 * tools_json.
 *
 * @param session_id Session identifier returned by ai_bridge_create_session()
 * @param tool_name Name of the tool as defined in the session's tools_json
 * @param callback Function to call when the AI invokes this tool
 * @param user_data Optional pointer passed to the callback function
 * @return true if registration succeeded, false if session not found or tool
 * not defined
 *
 * @note Multiple tools can be registered for the same session.
 * @note Tool callbacks may be called concurrently; ensure thread safety if
 * needed.
 */
bool ai_bridge_register_tool(ai_bridge_session_id_t session_id,
                             const char *tool_name,
                             ai_bridge_tool_callback_t callback,
                             void *user_data);

/**
 * @brief Destroy a session and release all associated resources
 *
 * Invalidates the session ID and cancels any active streams associated with the
 * session. This function is idempotent; calling it multiple times with the same
 * ID is safe.
 *
 * @param session_id Session identifier to destroy
 */
void ai_bridge_destroy_session(ai_bridge_session_id_t session_id);

/**
 * @brief Generate a text response from the given prompt (synchronous)
 *
 * Sends a prompt to the AI and waits for the complete response. This function
 * blocks until generation is complete or an error occurs.
 *
 * @param session_id Session identifier
 * @param prompt Input text prompt to send to the AI
 * @param temperature Controls randomness in generation (0.0 =
 * deterministic, 2.0 = very random, 0 = use default)
 * @param max_tokens Maximum number of tokens to generate (0 = use default
 * limit)
 * @return Generated response text, or error message prefixed with "Error:".
 *         **Memory ownership**: Caller must call ai_bridge_free_string() to
 * release.
 *
 * @note This function may take several seconds to complete for long responses.
 * @note The response is automatically added to the session's conversation
 * history.
 */
char *ai_bridge_generate_response(ai_bridge_session_id_t session_id,
                                  const char *prompt, double temperature,
                                  int32_t max_tokens);

/**
 * @brief Generate a structured response conforming to the provided JSON schema
 * (synchronous)
 *
 * Generates a response that conforms to the specified JSON schema structure.
 * Useful for extracting structured data or ensuring consistent response format.
 *
 * @param session_id Session identifier
 * @param prompt Input text prompt to send to the AI
 * @param schema_json JSON schema defining the expected response structure. Can
 * be NULL to use session default. Must be valid JSON Schema format.
 * @param temperature Controls randomness in generation (0.0 =
 * deterministic, 2.0 = very random, 0 = use default)
 * @param max_tokens Maximum number of tokens to generate (0 = use default
 * limit)
 * @return JSON object containing "text" (string representation) and "object"
 * (structured data) fields.
 *         **Memory ownership**: Caller must call ai_bridge_free_string() to
 * release.
 *
 * @note Structured generation may be slower than regular text generation.
 * @note If schema_json is NULL, a default schema must have been provided during
 * session creation.
 */
char *ai_bridge_generate_structured_response(ai_bridge_session_id_t session_id,
                                             const char *prompt,
                                             const char *schema_json,
                                             double temperature,
                                             int32_t max_tokens);

/**
 * @brief Start streaming text generation from the given prompt
 *
 * Begins generating a response and calls the callback function for each token
 * or chunk. This function returns immediately; generation continues
 * asynchronously.
 *
 * @param session_id Session identifier
 * @param prompt Input text prompt to send to the AI
 * @param temperature Controls randomness in generation (0.0 =
 * deterministic, 2.0 = very random, 0 = use default)
 * @param max_tokens Maximum number of tokens to generate (0 = use default
 * limit)
 * @param context Opaque pointer passed to each callback invocation
 * @param callback Function called for each response chunk or completion/error
 * @param user_data Additional user data passed to the callback
 * @return Stream identifier for cancellation, or AI_BRIDGE_INVALID_ID if
 * streaming failed to start
 *
 * @note The callback may be called from a background thread.
 * @note Call ai_bridge_cancel_stream() to stop generation early.
 * @note The stream automatically cleans up when generation completes.
 */
ai_bridge_stream_id_t ai_bridge_generate_response_stream(
    ai_bridge_session_id_t session_id, const char *prompt, double temperature,
    int32_t max_tokens, void *context, ai_bridge_stream_callback_t callback,
    void *user_data);

/**
 * @brief Start streaming structured response generation
 *
 * Generates a structured response conforming to the JSON schema and delivers
 * the complete result via callback when ready. Unlike text streaming, this
 * delivers the entire structured response at once.
 *
 * @param session_id Session identifier
 * @param prompt Input text prompt to send to the AI
 * @param schema_json JSON schema defining expected response structure. Can be
 * NULL to use session default.
 * @param temperature Controls randomness in generation (0.0 =
 * deterministic, 2.0 = very random, 0 = use default)
 * @param max_tokens Maximum number of tokens to generate (0 = use default
 * limit)
 * @param context Opaque pointer passed to the callback
 * @param callback Function called once with the complete structured response
 * @param user_data Additional user data passed to the callback
 * @return Stream identifier for cancellation, or AI_BRIDGE_INVALID_ID if
 * streaming failed to start
 *
 * @note Unlike text streaming, this calls the callback only once with the
 * complete result.
 * @note The callback receives a JSON object with "text" and "object" fields.
 */
ai_bridge_stream_id_t ai_bridge_generate_structured_response_stream(
    ai_bridge_session_id_t session_id, const char *prompt,
    const char *schema_json, double temperature, int32_t max_tokens,
    void *context, ai_bridge_stream_callback_t callback, void *user_data);

/**
 * @brief Cancel an active streaming operation
 *
 * Attempts to cancel the specified stream. If successful, the stream's callback
 * will be called once more with a NULL chunk to indicate cancellation.
 *
 * @param stream_id Stream identifier returned by a streaming function
 * @return true if the stream was found and cancelled, false if stream not found
 * or already completed
 *
 * @note Cancellation may not be immediate; the callback may receive additional
 * chunks.
 * @note This function is safe to call multiple times with the same stream ID.
 */
bool ai_bridge_cancel_stream(ai_bridge_stream_id_t stream_id);

/**
 * @brief Get the conversation history for the specified session as JSON
 *
 * Returns the complete conversation history including user prompts, AI
 * responses, and tool interactions in a structured JSON format compatible with
 * chat APIs.
 *
 * @param session_id Session identifier
 * @return JSON array of message objects with "role" and "content" fields, or
 * NULL if session not found.
 *         **Memory ownership**: Caller must call ai_bridge_free_string() to
 * release.
 *
 * @note The returned JSON follows OpenAI chat completion format.
 * @note History is automatically maintained; manual history management is
 * optional.
 */
char *ai_bridge_get_session_history(ai_bridge_session_id_t session_id);

/**
 * @brief Clear the conversation history for the specified session
 *
 * Removes all messages from the session's conversation history. This does not
 * affect the session's configuration or registered tools.
 *
 * @param session_id Session identifier
 * @return true if history was cleared, false if session not found
 *
 * @note This operation may be a no-op depending on the underlying
 * implementation.
 * @note Clearing history does not affect the session's system instructions.
 */
bool ai_bridge_clear_session_history(ai_bridge_session_id_t session_id);

/**
 * @brief Manually add a message to the session's conversation history
 *
 * Adds a message to the session history without generating a response.
 * This function is kept for API compatibility but may be a no-op since
 * the session automatically manages its conversation history.
 *
 * @param session_id Session identifier
 * @param role Message role: "user", "assistant", "system", or "tool"
 * @param content Message content text
 * @return true if message was added, false if session not found
 *
 * @note The session automatically tracks all interactions, so manual history
 * management is typically unnecessary.
 * @note This function exists for compatibility and advanced use cases.
 */
bool ai_bridge_add_message_to_history(ai_bridge_session_id_t session_id,
                                      const char *role, const char *content);

/**
 * @brief Free string memory allocated by bridge functions
 *
 * Releases memory allocated by any bridge function that returns a char pointer.
 * This function must be called for all strings returned by the bridge to
 * prevent memory leaks.
 *
 * @param ptr Pointer to string allocated by bridge functions. Can be NULL
 * (no-op).
 *
 * @note Do not call this function on strings not allocated by the bridge.
 * @note This function is safe to call with NULL pointers.
 */
void ai_bridge_free_string(char *ptr);

#ifdef __cplusplus
}
#endif

#endif /* AI_BRIDGE_H */