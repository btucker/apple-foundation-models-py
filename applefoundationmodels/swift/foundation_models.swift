/**
 * foundation_models.swift
 *
 * Swift bindings for FoundationModels framework
 * Exports C-compatible API for Python/Cython FFI
 */

import Foundation

#if canImport(FoundationModels)
import FoundationModels
#endif

// MARK: - Global State

private var isInitialized = false
private var currentSession: LanguageModelSession?
private var sessionInstructions: String?
private var registeredTools: [any Tool] = []

// MARK: - Error Codes

public enum AIResult: Int32 {
    case success = 0
    case errorInitFailed = -1
    case errorNotAvailable = -2
    case errorInvalidParams = -3
    case errorMemory = -4
    case errorJSONParse = -5
    case errorGeneration = -6
    case errorTimeout = -7
    case errorToolNotFound = -11
    case errorToolExecution = -12
    case errorUnknown = -99
}

public enum AIAvailability: Int32 {
    case available = 1
    case deviceNotEligible = -1
    case notEnabled = -2
    case modelNotReady = -3
    case unknown = -99
}

// MARK: - Tool Calling Infrastructure

/// C-compatible callback for Python tool execution
public typealias ToolCallback = @convention(c) (
    UnsafePointer<CChar>?,  // tool_name
    UnsafePointer<CChar>?,  // arguments_json
    UnsafeMutablePointer<CChar>?,  // result_buffer
    Int32  // buffer_size
) -> Int32

/// Store tool callback globally
private var toolCallback: ToolCallback?

/// Python tool wrapper that bridges Swift Tool protocol to Python callbacks
@available(macOS 26.0, *)
struct PythonToolWrapper: Tool, Sendable {
    let toolName: String
    let toolDescription: String
    let dynamicSchema: DynamicGenerationSchema

    var name: String { toolName }
    var description: String { toolDescription }

    // Use GeneratedContent as Arguments - it's already Generable and supports dynamic schemas!
    typealias Arguments = GeneratedContent
    typealias Output = String

    // Override parameters with our dynamic schema
    var parameters: GenerationSchema {
        // Must not throw - create schema or use empty one on error
        (try? GenerationSchema(root: dynamicSchema, dependencies: [])) ?? GenerationSchema(
            type: GeneratedContent.self,
            properties: []
        )
    }

    nonisolated func call(arguments: Arguments) async throws -> Output {
        // Extract JSON directly from GeneratedContent!
        let argsJson = arguments.jsonString

        // Allocate result buffer (16KB)
        let bufferSize: Int32 = 16384
        let resultBuffer = UnsafeMutablePointer<CChar>.allocate(capacity: Int(bufferSize))
        resultBuffer.initialize(repeating: 0, count: Int(bufferSize))
        defer { resultBuffer.deallocate() }

        // Call Python callback
        guard let callback = toolCallback else {
            throw NSError(domain: "ToolError", code: -1, userInfo: [NSLocalizedDescriptionKey: "Tool callback not set"])
        }

        let result = argsJson.withCString { argsPtr in
            toolName.withCString { namePtr in
                callback(namePtr, argsPtr, resultBuffer, bufferSize)
            }
        }

        guard result == 0 else {
            // Extract error message from buffer
            let errorMsg = String(cString: resultBuffer)
            // Throw simpler error to avoid buffer overflow in error messages
            throw NSError(domain: "ToolError", code: Int(result), userInfo: [
                NSLocalizedDescriptionKey: "Tool '\(toolName)' failed: \(errorMsg)"
            ])
        }

        return String(cString: resultBuffer)
    }
}

// MARK: - Helper Functions

/// Create an error response in JSON format using safe serialization
private func createErrorResponse(_ message: String) -> UnsafeMutablePointer<CChar>? {
    let errorDict: [String: String] = ["error": message]

    do {
        let jsonData = try JSONSerialization.data(withJSONObject: errorDict, options: [])
        if let jsonString = String(data: jsonData, encoding: .utf8) {
            return strdup(jsonString)
        }
    } catch {
        // Fallback to generic error message if serialization fails
        let fallback = "{\"error\":\"An error occurred\"}"
        return strdup(fallback)
    }

    // If UTF-8 encoding fails, return generic error
    let fallback = "{\"error\":\"An error occurred\"}"
    return strdup(fallback)
}

/// Get existing session or create a new one with stored instructions
@available(macOS 26.0, *)
private func getOrCreateSession() -> LanguageModelSession {
    if let session = currentSession {
        return session
    }

    let session: LanguageModelSession
    if let instructions = sessionInstructions {
        session = LanguageModelSession(
            model: SystemLanguageModel.default,
            instructions: { instructions }
        )
    } else {
        session = LanguageModelSession(
            model: SystemLanguageModel.default
        )
    }
    currentSession = session
    return session
}

/// Create a new session, replacing any existing one
@available(macOS 26.0, *)
private func createNewSession() -> LanguageModelSession {
    let session: LanguageModelSession
    if let instructions = sessionInstructions {
        session = LanguageModelSession(
            model: SystemLanguageModel.default,
            instructions: { instructions }
        )
    } else {
        session = LanguageModelSession(
            model: SystemLanguageModel.default
        )
    }
    currentSession = session
    return session
}

// MARK: - Initialization

@_cdecl("apple_ai_init")
public func appleAIInit() -> Int32 {
    guard !isInitialized else {
        return AIResult.success.rawValue
    }

    #if canImport(FoundationModels)
    if #available(macOS 26.0, *) {
        // Check if model is available
        let model = SystemLanguageModel.default
        switch model.availability {
        case .available:
            isInitialized = true
            return AIResult.success.rawValue
        case .unavailable:
            return AIResult.errorNotAvailable.rawValue
        }
    }
    #endif

    return AIResult.errorNotAvailable.rawValue
}

@_cdecl("apple_ai_cleanup")
public func appleAICleanup() {
    currentSession = nil
    sessionInstructions = nil
    isInitialized = false
}

// MARK: - Availability Check

@_cdecl("apple_ai_check_availability")
public func appleAICheckAvailability() -> Int32 {
    #if canImport(FoundationModels)
    if #available(macOS 26.0, *) {
        let model = SystemLanguageModel.default
        switch model.availability {
        case .available:
            return AIAvailability.available.rawValue
        case .unavailable(let reason):
            // Map unavailability reason to status code
            let description = String(describing: reason)
            if description.contains("not enabled") || description.contains("disabled") {
                return AIAvailability.notEnabled.rawValue
            } else if description.contains("downloading") || description.contains("not ready") {
                return AIAvailability.modelNotReady.rawValue
            } else {
                return AIAvailability.deviceNotEligible.rawValue
            }
        }
    } else {
        return AIAvailability.deviceNotEligible.rawValue
    }
    #else
    return AIAvailability.deviceNotEligible.rawValue
    #endif
}

@_cdecl("apple_ai_get_availability_reason")
public func appleAIGetAvailabilityReason() -> UnsafeMutablePointer<CChar>? {
    #if canImport(FoundationModels)
    if #available(macOS 26.0, *) {
        let model = SystemLanguageModel.default
        switch model.availability {
        case .available:
            return strdup("Apple Intelligence is available and ready")
        case .unavailable(let reason):
            return strdup("Apple Intelligence is unavailable: \(reason)")
        }
    } else {
        return strdup("Device does not support Apple Intelligence (requires macOS 26.0+)")
    }
    #else
    return strdup("FoundationModels framework not available")
    #endif
}

@_cdecl("apple_ai_get_version")
public func appleAIGetVersion() -> UnsafeMutablePointer<CChar>? {
    return strdup("1.0.0-foundationmodels")
}

// MARK: - Tool Management

@_cdecl("apple_ai_register_tools")
public func appleAIRegisterTools(
    toolsJson: UnsafePointer<CChar>?,
    callback: ToolCallback?
) -> Int32 {
    guard isInitialized else {
        return AIResult.errorInitFailed.rawValue
    }

    #if canImport(FoundationModels)
    if #available(macOS 26.0, *) {
        guard let jsonPtr = toolsJson,
              let callback = callback else {
            return AIResult.errorInvalidParams.rawValue
        }

        let jsonString = String(cString: jsonPtr)
        guard let jsonData = jsonString.data(using: .utf8),
              let toolsArray = try? JSONSerialization.jsonObject(with: jsonData, options: []) as? [[String: Any]] else {
            return AIResult.errorJSONParse.rawValue
        }

        // Store callback
        toolCallback = callback

        // Clear existing tools
        registeredTools.removeAll()

        // Create PythonToolWrapper for each tool - fail fast on any error
        for (index, toolDef) in toolsArray.enumerated() {
            // Validate required fields
            guard let name = toolDef["name"] as? String else {
                print("ERROR: Tool at index \(index) missing required 'name' field")
                registeredTools.removeAll()
                return AIResult.errorInvalidParams.rawValue
            }

            guard let description = toolDef["description"] as? String else {
                print("ERROR: Tool '\(name)' at index \(index) missing required 'description' field")
                registeredTools.removeAll()
                return AIResult.errorInvalidParams.rawValue
            }

            guard let parameters = toolDef["parameters"] as? [String: Any] else {
                print("ERROR: Tool '\(name)' at index \(index) missing required 'parameters' field")
                registeredTools.removeAll()
                return AIResult.errorInvalidParams.rawValue
            }

            // Convert JSON Schema to DynamicGenerationSchema - fail fast if conversion fails
            guard let dynamicSchema = convertJSONSchemaToDynamic(parameters, name: "\(name)_params") else {
                print("ERROR: Failed to convert JSON schema for tool '\(name)' at index \(index)")
                registeredTools.removeAll()
                return AIResult.errorJSONParse.rawValue
            }

            let tool = PythonToolWrapper(
                toolName: name,
                toolDescription: description,
                dynamicSchema: dynamicSchema
            )
            registeredTools.append(tool)
        }

        return AIResult.success.rawValue
    }
    #endif

    return AIResult.errorNotAvailable.rawValue
}

// MARK: - Session Management

@_cdecl("apple_ai_create_session")
public func appleAICreateSession(
    instructionsJson: UnsafePointer<CChar>?
) -> Int32 {
    guard isInitialized else {
        return AIResult.errorInitFailed.rawValue
    }

    #if canImport(FoundationModels)
    if #available(macOS 26.0, *) {
        // Parse instructions if provided
        if let jsonPtr = instructionsJson {
            let jsonString = String(cString: jsonPtr)
            if let jsonData = jsonString.data(using: .utf8),
               let config = try? JSONDecoder().decode([String: String].self, from: jsonData),
               let inst = config["instructions"] {
                sessionInstructions = inst
            }
        }

        // Create session with instructions and tools if provided
        if let instructions = sessionInstructions {
            if !registeredTools.isEmpty {
                currentSession = LanguageModelSession(
                    model: SystemLanguageModel.default,
                    tools: registeredTools,
                    instructions: { instructions }
                )
            } else {
                currentSession = LanguageModelSession(
                    model: SystemLanguageModel.default,
                    instructions: { instructions }
                )
            }
        } else {
            if !registeredTools.isEmpty {
                currentSession = LanguageModelSession(
                    model: SystemLanguageModel.default,
                    tools: registeredTools
                )
            } else {
                currentSession = LanguageModelSession(
                    model: SystemLanguageModel.default
                )
            }
        }

        return AIResult.success.rawValue
    }
    #endif

    return AIResult.errorNotAvailable.rawValue
}

// MARK: - Generation

/// Generate text response
/// - Parameters:
///   - prompt: User prompt as C string
///   - temperature: Sampling temperature (0.0 to 2.0)
///   - maxTokens: Maximum tokens to generate
/// - Returns: JSON response or error message
@_cdecl("apple_ai_generate")
public func appleAIGenerate(
    prompt: UnsafePointer<CChar>,
    temperature: Double,
    maxTokens: Int32
) -> UnsafeMutablePointer<CChar>? {
    guard isInitialized else {
        return createErrorResponse("Not initialized")
    }

    #if canImport(FoundationModels)
    if #available(macOS 26.0, *) {
        let promptString = String(cString: prompt)

        // Use semaphore for async coordination
        let semaphore = DispatchSemaphore(value: 0)
        var result: String = ""

        Task {
            do {
                // Get or create session
                let session = getOrCreateSession()

                // Configure generation options
                let options = GenerationOptions(
                    temperature: temperature,
                    maximumResponseTokens: Int(maxTokens)
                )

                // Generate response
                let response = try await session.respond(
                    to: promptString,
                    options: options
                )

                result = response.content
            } catch {
                // Use safe JSON serialization for error messages
                if let errorJson = createErrorResponse(error.localizedDescription) {
                    result = String(cString: errorJson)
                    free(errorJson)
                } else {
                    result = "{\"error\":\"An error occurred\"}"
                }
            }
            semaphore.signal()
        }

        semaphore.wait()
        return strdup(result)
    }
    #endif

    return createErrorResponse("FoundationModels not available")
}

// Streaming callback type
public typealias StreamCallback = @convention(c) (UnsafePointer<CChar>?) -> Void

/// Generate streaming text response
/// - Parameters:
///   - prompt: User prompt as C string
///   - temperature: Sampling temperature (0.0 to 2.0)
///   - maxTokens: Maximum tokens to generate
///   - callback: Callback function to receive text chunks (receives nil to signal end)
/// - Returns: Result code (0 = success, negative = error)
@_cdecl("apple_ai_generate_stream")
public func appleAIGenerateStream(
    prompt: UnsafePointer<CChar>,
    temperature: Double,
    maxTokens: Int32,
    callback: StreamCallback?
) -> Int32 {
    guard isInitialized, let cb = callback else {
        return AIResult.errorInvalidParams.rawValue
    }

    #if canImport(FoundationModels)
    if #available(macOS 26.0, *) {
        let promptString = String(cString: prompt)

        let semaphore = DispatchSemaphore(value: 0)
        var resultCode = AIResult.success

        Task {
            do {
                // Get or create session
                let session = getOrCreateSession()

                // Configure generation options
                let options = GenerationOptions(
                    temperature: temperature,
                    maximumResponseTokens: Int(maxTokens)
                )

                // Stream response
                let stream = try await session.streamResponse(
                    options: options
                ) {
                    promptString
                }

                var previousContent = ""
                for try await partial in stream {
                    let currentContent = partial.content

                    // Calculate delta from previous snapshot
                    if currentContent.count > previousContent.count {
                        let delta = String(currentContent.dropFirst(previousContent.count))
                        if !delta.isEmpty {
                            cb(strdup(delta))
                        }
                    }

                    previousContent = currentContent
                }

                // Signal end of stream
                cb(nil)

            } catch {
                // Use safe JSON serialization for error messages
                let errorMessage = "Error: \(error.localizedDescription)"
                if let errorJson = createErrorResponse(errorMessage) {
                    cb(errorJson)
                    // Note: callback takes ownership, will be freed by caller
                } else {
                    cb(strdup("{\"error\":\"An error occurred\"}"))
                }
                cb(nil)
                resultCode = .errorGeneration
            }
            semaphore.signal()
        }

        semaphore.wait()
        return resultCode.rawValue
    }
    #endif

    cb(strdup("FoundationModels not available"))
    cb(nil)
    return AIResult.errorNotAvailable.rawValue
}

// MARK: - Transcript Access

/// Get the session transcript
/// - Returns: JSON array of transcript entries or error message
@_cdecl("apple_ai_get_transcript")
public func appleAIGetTranscript() -> UnsafeMutablePointer<CChar>? {
    guard isInitialized else {
        return createErrorResponse("Not initialized")
    }

    #if canImport(FoundationModels)
    if #available(macOS 26.0, *) {
        guard let session = currentSession else {
            return createErrorResponse("No active session")
        }

        // Use semaphore for async coordination
        let semaphore = DispatchSemaphore(value: 0)
        var result: String = ""

        Task {
            do {
                let transcript = session.transcript
                var entries: [NSDictionary] = []

                for entry in transcript {
                    var entryDict: [String: Any] = [:]

                    switch entry {
                    case .instructions(let text):
                        entryDict["type"] = "instructions" as NSString
                        entryDict["content"] = String(describing: text) as NSString

                    case .prompt(let text):
                        entryDict["type"] = "prompt" as NSString
                        entryDict["content"] = String(describing: text) as NSString

                    case .response(let text):
                        entryDict["type"] = "response" as NSString
                        entryDict["content"] = String(describing: text) as NSString

                    case .toolCalls(let toolCalls):
                        entryDict["type"] = "tool_calls" as NSString
                        // Convert tool calls to JSON array
                        var callsArray: [[String: Any]] = []
                        for call in toolCalls {
                            callsArray.append([
                                "id": String(describing: call.id) as NSString
                            ])
                        }
                        entryDict["tool_calls"] = callsArray as NSArray

                    case .toolOutput(let output):
                        entryDict["type"] = "tool_output" as NSString
                        entryDict["tool_id"] = String(describing: output.id) as NSString
                        // Note: Additional properties may be available in final API
                        entryDict["content"] = "" as NSString

                    @unknown default:
                        entryDict["type"] = "unknown" as NSString
                    }

                    entries.append(entryDict as NSDictionary)
                }

                // Convert to JSON
                let jsonData = try JSONSerialization.data(withJSONObject: entries, options: .prettyPrinted)
                result = String(data: jsonData, encoding: .utf8) ?? "{\"error\":\"Failed to encode transcript\"}"
            } catch {
                if let errorJson = createErrorResponse(error.localizedDescription) {
                    result = String(cString: errorJson)
                    free(errorJson)
                } else {
                    result = "{\"error\":\"An error occurred\"}"
                }
            }
            semaphore.signal()
        }

        semaphore.wait()
        return strdup(result)
    }
    #endif

    return createErrorResponse("FoundationModels not available")
}

// MARK: - Structured Generation

// Helper to convert JSON Schema dictionary to DynamicGenerationSchema
@available(macOS 26.0, *)
private func convertJSONSchemaToDynamic(_ schema: [String: Any], name: String = "root") -> DynamicGenerationSchema? {
    guard let type = schema["type"] as? String else {
        return nil
    }

    switch type {
    case "object":
        guard let properties = schema["properties"] as? [String: [String: Any]] else {
            return nil
        }

        var dynamicProperties: [DynamicGenerationSchema.Property] = []

        for (propName, propSchema) in properties {
            guard let propDynamicSchema = convertJSONSchemaToDynamic(propSchema, name: propName) else {
                // Fail fast if property conversion fails - don't produce incomplete schemas
                print("ERROR: Failed to convert property '\(propName)' in schema '\(name)'")
                return nil
            }

            let description = propSchema["description"] as? String
            dynamicProperties.append(
                DynamicGenerationSchema.Property(
                    name: propName,
                    description: description,
                    schema: propDynamicSchema
                )
            )
        }

        return DynamicGenerationSchema(
            name: name,
            description: schema["description"] as? String,
            properties: dynamicProperties
        )

    case "array":
        guard let items = schema["items"] as? [String: Any],
              let itemSchema = convertJSONSchemaToDynamic(items, name: "\(name)Item") else {
            // Fail fast - don't silently fall back to String type for malformed arrays
            print("ERROR: Array schema missing or invalid 'items' specification for '\(name)'")
            return nil
        }

        // Extract min/max items if specified
        let minItems = schema["minItems"] as? Int
        let maxItems = schema["maxItems"] as? Int

        return DynamicGenerationSchema(
            arrayOf: itemSchema,
            minimumElements: minItems,
            maximumElements: maxItems
        )

    case "string":
        if let enumValues = schema["enum"] as? [String] {
            return DynamicGenerationSchema(name: name, anyOf: enumValues)
        }
        return DynamicGenerationSchema(type: String.self)

    case "integer", "number":
        return DynamicGenerationSchema(type: Double.self)

    case "boolean":
        return DynamicGenerationSchema(type: Bool.self)

    default:
        return nil
    }
}

// Helper to extract JSON from GeneratedContent
@available(macOS 26.0, *)
private func extractJSON(from content: GeneratedContent) throws -> [String: Any] {
    guard case let .structure(properties, _) = content.kind else {
        throw NSError(domain: "FoundationModels", code: -1, userInfo: [NSLocalizedDescriptionKey: "Expected structure content"])
    }

    var result: [String: Any] = [:]

    for (key, value) in properties {
        result[key] = try extractValue(from: value)
    }

    return result
}

@available(macOS 26.0, *)
private func extractValue(from content: GeneratedContent) throws -> Any {
    switch content.kind {
    case .string(let str):
        return str
    case .number(let num):
        return num
    case .bool(let bool):
        return bool
    case .null:
        return NSNull()
    case .structure(let properties, _):
        var result: [String: Any] = [:]
        for (key, value) in properties {
            result[key] = try extractValue(from: value)
        }
        return result
    case .array(let items):
        var result: [Any] = []
        for item in items {
            result.append(try extractValue(from: item))
        }
        return result
    @unknown default:
        throw NSError(domain: "FoundationModels", code: -1, userInfo: [NSLocalizedDescriptionKey: "Unsupported GeneratedContent kind"])
    }
}

/// Generate structured output conforming to JSON Schema
/// - Parameters:
///   - prompt: User prompt as C string
///   - schemaJson: JSON Schema as C string
///   - temperature: Sampling temperature (0.0 to 2.0)
///   - maxTokens: Maximum tokens to generate
/// - Returns: JSON object conforming to schema, or error message
@_cdecl("apple_ai_generate_structured")
public func appleAIGenerateStructured(
    prompt: UnsafePointer<CChar>,
    schemaJson: UnsafePointer<CChar>,
    temperature: Double,
    maxTokens: Int32
) -> UnsafeMutablePointer<CChar>? {
    guard isInitialized else {
        return createErrorResponse("Not initialized")
    }

    #if canImport(FoundationModels)
    if #available(macOS 26.0, *) {
        let promptString = String(cString: prompt)
        let schemaString = String(cString: schemaJson)

        // Parse schema JSON to dictionary
        guard let schemaData = schemaString.data(using: .utf8),
              let schemaDict = try? JSONSerialization.jsonObject(with: schemaData) as? [String: Any] else {
            return createErrorResponse("Invalid schema JSON")
        }

        // Convert JSON Schema to DynamicGenerationSchema
        guard let dynamicSchema = convertJSONSchemaToDynamic(schemaDict) else {
            return createErrorResponse("Failed to convert schema")
        }

        // Use semaphore for async coordination
        let semaphore = DispatchSemaphore(value: 0)
        var result: String = ""

        Task {
            do {
                // Get or create session
                let session = getOrCreateSession()

                // Configure generation options
                let options = GenerationOptions(
                    temperature: temperature,
                    maximumResponseTokens: Int(maxTokens)
                )

                // Create GenerationSchema from DynamicGenerationSchema
                let generationSchema = try GenerationSchema(root: dynamicSchema, dependencies: [])

                // Generate response with proper schema
                let response = try await session.respond(
                    to: promptString,
                    schema: generationSchema,
                    options: options
                )

                // Extract JSON from GeneratedContent
                let jsonObject = try extractJSON(from: response.content)
                let jsonData = try JSONSerialization.data(withJSONObject: jsonObject)
                if let jsonString = String(data: jsonData, encoding: .utf8) {
                    result = jsonString
                } else {
                    result = "{\"error\": \"Failed to encode JSON as string\"}"
                }
            } catch {
                result = "{\"error\": \"\(error.localizedDescription)\"}"
            }
            semaphore.signal()
        }

        semaphore.wait()
        return strdup(result)
    }
    #endif

    return createErrorResponse("FoundationModels not available")
}

// MARK: - Memory Management

@_cdecl("apple_ai_free_string")
public func appleAIFreeString(ptr: UnsafeMutablePointer<CChar>?) {
    guard let ptr = ptr else { return }
    free(ptr)
}

// MARK: - History Management

@_cdecl("apple_ai_get_history")
public func appleAIGetHistory() -> UnsafeMutablePointer<CChar>? {
    #if canImport(FoundationModels)
    if #available(macOS 26.0, *) {
        guard currentSession != nil else {
            return strdup("[]")
        }

        // The FoundationModels framework doesn't expose history directly
        // This is a limitation of the framework
        return strdup("[]")
    }
    #endif

    return strdup("[]")
}

@_cdecl("apple_ai_clear_history")
public func appleAIClearHistory() {
    // Clear by creating a new session
    #if canImport(FoundationModels)
    if #available(macOS 26.0, *) {
        currentSession = createNewSession()
    }
    #endif
}

// MARK: - Statistics (Stubs)

@_cdecl("apple_ai_get_stats")
public func appleAIGetStats() -> UnsafeMutablePointer<CChar>? {
    let stats = """
    {
        "total_requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "total_tokens_generated": 0,
        "average_response_time": 0.0,
        "total_processing_time": 0.0
    }
    """
    return strdup(stats)
}

@_cdecl("apple_ai_reset_stats")
public func appleAIResetStats() {
    // Stub for compatibility
}
