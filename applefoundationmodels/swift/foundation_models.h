/**
 * foundation_models.h
 *
 * C header for Swift FoundationModels bindings
 * Declares C-compatible functions exported from foundation_models.swift
 */

#ifndef FOUNDATION_MODELS_H
#define FOUNDATION_MODELS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Callback type for streaming
typedef void (*ai_stream_callback_t)(const char *chunk);

// Callback type for tool execution
typedef int32_t (*ai_tool_callback_t)(const char *tool_name,
                                       const char *arguments_json,
                                       char *result_buffer,
                                       int32_t buffer_size);

// Core library functions
int32_t apple_ai_init(void);
void apple_ai_cleanup(void);
const char *apple_ai_get_version(void);

// Availability functions
int32_t apple_ai_check_availability(void);
char *apple_ai_get_availability_reason(void);

// Session management
int32_t apple_ai_create_session(const char *instructions_json);

// Tool calling
int32_t apple_ai_register_tools(const char *tools_json,
                                ai_tool_callback_t callback);
char *apple_ai_get_transcript(void);

// Text generation
char *apple_ai_generate(const char *prompt,
                       double temperature,
                       int32_t max_tokens);

// Streaming generation
int32_t apple_ai_generate_stream(const char *prompt,
                                double temperature,
                                int32_t max_tokens,
                                ai_stream_callback_t callback);

// Structured generation
char *apple_ai_generate_structured(const char *prompt,
                                   const char *schema_json,
                                   double temperature,
                                   int32_t max_tokens);

// History management
char *apple_ai_get_history(void);
void apple_ai_clear_history(void);

// Statistics
char *apple_ai_get_stats(void);
void apple_ai_reset_stats(void);

// Memory management
void apple_ai_free_string(char *ptr);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDATION_MODELS_H */
