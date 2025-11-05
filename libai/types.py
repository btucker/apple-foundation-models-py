"""
Type definitions for libai Python bindings.

This module provides TypedDicts, enums, and type aliases for type-safe
interaction with the library.
"""

from enum import IntEnum
from typing import TypedDict, Optional, Callable, Any
from typing_extensions import NotRequired


class Result(IntEnum):
    """
    Result codes for AI operations.

    These codes indicate the success or failure state of library operations.
    """
    SUCCESS = 0
    INIT_FAILED = -1
    NOT_AVAILABLE = -2
    INVALID_PARAMS = -3
    MEMORY = -4
    JSON_PARSE = -5
    GENERATION = -6
    TIMEOUT = -7
    SESSION_NOT_FOUND = -8
    STREAM_NOT_FOUND = -9
    GUARDRAIL_VIOLATION = -10
    TOOL_NOT_FOUND = -11
    TOOL_EXECUTION = -12
    UNKNOWN = -99


class Availability(IntEnum):
    """
    Apple Intelligence availability status.

    Indicates whether Apple Intelligence is available and ready for use
    on the current device and system configuration.
    """
    AVAILABLE = 1
    DEVICE_NOT_ELIGIBLE = -1
    NOT_ENABLED = -2
    MODEL_NOT_READY = -3
    AVAILABILITY_UNKNOWN = -99


class SessionConfig(TypedDict, total=False):
    """
    Session configuration options.

    Configuration for creating an AI session. Sessions maintain conversation
    state and can be configured with tools and instructions.

    Attributes:
        instructions: Optional system instructions to guide AI behavior
        tools_json: Optional JSON array of tool definitions in Claude format
        enable_guardrails: Whether to enable content safety filtering (default: True)
        prewarm: Whether to preload session resources for faster first response (default: False)
    """
    instructions: NotRequired[Optional[str]]
    tools_json: NotRequired[Optional[str]]
    enable_guardrails: NotRequired[bool]
    prewarm: NotRequired[bool]


class GenerationParams(TypedDict, total=False):
    """
    Text generation parameters.

    Controls various aspects of AI text generation including randomness,
    length limits, and reproducibility.

    Attributes:
        temperature: Generation randomness (0.0 = deterministic, 2.0 = very random)
        max_tokens: Maximum response tokens (0 = use system default)
        include_reasoning: Include reasoning in response (reserved for future use)
        seed: Random seed for reproducibility (0 = use random seed)
    """
    temperature: NotRequired[float]
    max_tokens: NotRequired[int]
    include_reasoning: NotRequired[bool]
    seed: NotRequired[int]


class Stats(TypedDict):
    """
    Generation statistics and performance metrics.

    Provides insights into the usage and performance of generation operations.

    Attributes:
        total_requests: Total number of generation requests initiated
        successful_requests: Number of requests that completed successfully
        failed_requests: Number of requests that failed or were cancelled
        total_tokens_generated: Total tokens generated across all requests
        average_response_time: Average response time in seconds
        total_processing_time: Total processing time in seconds
    """
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_tokens_generated: int
    average_response_time: float
    total_processing_time: float


# Callback type aliases
StreamCallback = Callable[[Optional[str]], None]
"""
Callback function for streaming text generation.

Called incrementally during streaming generation for each token or chunk.
None indicates completion or error.
"""

ToolCallback = Callable[[dict], Any]
"""
Callback function for tool execution.

Receives tool parameters as a dict and should return the tool result.
The result will be automatically JSON-serialized.
"""
