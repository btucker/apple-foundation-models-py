/**
 * apple_ai.swift
 *
 * Swift bindings for FoundationModels framework
 * Exports C-compatible API for Python/Cython FFI
 */

import Foundation
import FoundationModels

// MARK: - Global State

private var isInitialized = false
private var modelSession: LanguageModelSession?
private var currentTranscript: Transcript = Transcript()

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
    case errorUnknown = -99
}

public enum AIAvailability: Int32 {
    case available = 1
    case deviceNotEligible = -1
    case notEnabled = -2
    case modelNotReady = -3
    case unknown = -99
}

// MARK: - Initialization

@_cdecl("apple_ai_init")
public func appleAIInit() -> Int32 {
    guard !isInitialized else {
        return AIResult.success.rawValue
    }

    do {
        // Initialize with default empty transcript
        currentTranscript = Transcript()
        isInitialized = true
        return AIResult.success.rawValue
    } catch {
        print("Initialization error: \(error)")
        return AIResult.errorInitFailed.rawValue
    }
}

@_cdecl("apple_ai_cleanup")
public func appleAICleanup() {
    modelSession = nil
    currentTranscript = Transcript()
    isInitialized = false
}

// MARK: - Availability Check

@_cdecl("apple_ai_check_availability")
public func appleAICheckAvailability() -> Int32 {
    // Check if FoundationModels is available
    if #available(macOS 26.0, *) {
        // Try to access SystemLanguageModel
        do {
            let _ = SystemLanguageModel.shared
            return AIAvailability.available.rawValue
        } catch {
            return AIAvailability.modelNotReady.rawValue
        }
    } else {
        return AIAvailability.deviceNotEligible.rawValue
    }
}

@_cdecl("apple_ai_get_availability_reason")
public func appleAIGetAvailabilityReason() -> UnsafePointer<CChar>? {
    let status = appleAICheckAvailability()

    let message: String
    switch AIAvailability(rawValue: status) {
    case .available:
        message = "Apple Intelligence is available and ready"
    case .deviceNotEligible:
        message = "Device does not support Apple Intelligence (requires macOS 26.0+)"
    case .notEnabled:
        message = "Apple Intelligence is not enabled in system settings"
    case .modelNotReady:
        message = "AI model is downloading or not ready"
    default:
        message = "Unknown availability status"
    }

    return strdup(message)
}

@_cdecl("apple_ai_get_version")
public func appleAIGetVersion() -> UnsafePointer<CChar>? {
    return strdup("1.0.0-swift")
}

// MARK: - Session Management

@_cdecl("apple_ai_create_session")
public func appleAICreateSession(
    instructionsJson: UnsafePointer<CChar>?
) -> Int32 {
    guard isInitialized else {
        return AIResult.errorInitFailed.rawValue
    }

    do {
        // Parse instructions if provided
        var instructions: String? = nil
        if let jsonPtr = instructionsJson {
            let jsonString = String(cString: jsonPtr)
            if let jsonData = jsonString.data(using: .utf8),
               let config = try? JSONDecoder().decode([String: String].self, from: jsonData),
               let inst = config["instructions"] {
                instructions = inst
            }
        }

        // Create new transcript
        currentTranscript = Transcript()

        // Add instructions if provided
        if let inst = instructions, !inst.isEmpty {
            currentTranscript.entries.append(.instructions(Transcript.Instructions(segments: [
                .text(Transcript.TextSegment(content: inst))
            ])))
        }

        // Create session
        modelSession = LanguageModelSession(
            model: SystemLanguageModel.shared,
            transcript: currentTranscript
        )

        return AIResult.success.rawValue
    } catch {
        print("Session creation error: \(error)")
        return AIResult.errorGeneration.rawValue
    }
}

// MARK: - Generation

@_cdecl("apple_ai_generate")
public func appleAIGenerate(
    prompt: UnsafePointer<CChar>,
    temperature: Double,
    maxTokens: Int32
) -> UnsafePointer<CChar>? {
    guard isInitialized else {
        return strdup("{\"error\": \"Not initialized\"}")
    }

    let promptString = String(cString: prompt)

    // Use semaphore for async coordination
    let semaphore = DispatchSemaphore(value: 0)
    var result: String = ""
    var error: Error? = nil

    Task {
        do {
            // Add prompt to transcript
            currentTranscript.entries.append(.prompt(Transcript.Prompt(segments: [
                .text(Transcript.TextSegment(content: promptString))
            ])))

            // Create or update session
            let session = LanguageModelSession(
                model: SystemLanguageModel.shared,
                transcript: currentTranscript
            )

            // Generate response
            let response = try await session.respond(to: currentTranscript)

            // Extract text from response
            var responseText = ""
            for entry in response.entries {
                if case .response(let resp) = entry {
                    for segment in resp.segments {
                        if case .text(let textSeg) = segment {
                            responseText += textSeg.content
                        }
                    }
                }
            }

            // Add response to transcript
            currentTranscript = response

            result = responseText
        } catch let err {
            error = err
        }
        semaphore.signal()
    }

    semaphore.wait()

    if let err = error {
        return strdup("{\"error\": \"\(err.localizedDescription)\"}")
    }

    return strdup(result)
}

// Streaming callback type
public typealias StreamCallback = @convention(c) (UnsafePointer<CChar>?) -> Void

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

    let promptString = String(cString: prompt)

    let semaphore = DispatchSemaphore(value: 0)
    var resultCode = AIResult.success

    Task {
        do {
            // Add prompt to transcript
            currentTranscript.entries.append(.prompt(Transcript.Prompt(segments: [
                .text(Transcript.TextSegment(content: promptString))
            ])))

            let session = LanguageModelSession(
                model: SystemLanguageModel.shared,
                transcript: currentTranscript
            )

            // Stream response
            var fullResponse = ""
            for try await delta in session.streamResponse(to: currentTranscript) {
                // Extract text from delta
                var deltaText = ""
                for entry in delta.entries {
                    if case .response(let resp) = entry {
                        for segment in resp.segments {
                            if case .text(let textSeg) = segment {
                                deltaText += textSeg.content
                            }
                        }
                    }
                }

                if !deltaText.isEmpty {
                    fullResponse += deltaText
                    cb(strdup(deltaText))
                }

                currentTranscript = delta
            }

            // Signal end of stream
            cb(nil)

        } catch {
            print("Streaming error: \(error)")
            resultCode = .errorGeneration
            cb(nil)
        }
        semaphore.signal()
    }

    semaphore.wait()
    return resultCode.rawValue
}

// MARK: - Memory Management

@_cdecl("apple_ai_free_string")
public func appleAIFreeString(ptr: UnsafeMutablePointer<CChar>?) {
    guard let ptr = ptr else { return }
    free(ptr)
}

// MARK: - History Management

@_cdecl("apple_ai_get_history")
public func appleAIGetHistory() -> UnsafePointer<CChar>? {
    var messages: [[String: String]] = []

    for entry in currentTranscript.entries {
        switch entry {
        case .prompt(let prompt):
            var content = ""
            for segment in prompt.segments {
                if case .text(let textSeg) = segment {
                    content += textSeg.content
                }
            }
            messages.append(["role": "user", "content": content])

        case .response(let response):
            var content = ""
            for segment in response.segments {
                if case .text(let textSeg) = segment {
                    content += textSeg.content
                }
            }
            messages.append(["role": "assistant", "content": content])

        case .instructions(let instructions):
            var content = ""
            for segment in instructions.segments {
                if case .text(let textSeg) = segment {
                    content += textSeg.content
                }
            }
            messages.append(["role": "system", "content": content])

        default:
            continue
        }
    }

    if let jsonData = try? JSONEncoder().encode(messages),
       let jsonString = String(data: jsonData, encoding: .utf8) {
        return strdup(jsonString)
    }

    return strdup("[]")
}

@_cdecl("apple_ai_clear_history")
public func appleAIClearHistory() {
    currentTranscript = Transcript()
}

// MARK: - Statistics (Stub for compatibility)

@_cdecl("apple_ai_get_stats")
public func appleAIGetStats() -> UnsafePointer<CChar>? {
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
