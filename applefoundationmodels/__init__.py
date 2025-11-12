"""
applefoundationmodels: Python bindings for Apple's FoundationModels framework

High-level Pythonic interface for accessing Apple Intelligence on-device
Foundation models.

Basic text generation:
    from applefoundationmodels import Session

    with Session() as session:
        # Check availability
        if not Session.is_ready():
            print("Apple Intelligence not available")
            return

        # Generate response
        response = session.generate("Hello, how are you?")
        print(response.text)  # Access text via .text property

Structured output:
    from applefoundationmodels import Session

    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        }
    }

    with Session() as session:
        response = session.generate("Extract: Alice is 28", schema=schema)
        print(response.parsed)  # {"name": "Alice", "age": 28}

Sync streaming:
    from applefoundationmodels import Session

    with Session() as session:
        for chunk in session.generate("Tell me a story", stream=True):
            print(chunk.content, end='', flush=True)

Async streaming:
    import asyncio
    from applefoundationmodels import AsyncSession

    async def main():
        async with AsyncSession() as session:
            async for chunk in session.generate("Tell me a story", stream=True):
                print(chunk.content, end='', flush=True)

    asyncio.run(main())
"""

__version__ = "0.1.0"

# Public API exports
from .session import Session
from .async_session import AsyncSession
from .constants import (
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    MIN_TEMPERATURE,
    MAX_TEMPERATURE,
    TemperaturePreset,
)
from .types import (
    Result,
    Availability,
    SessionConfig,
    GenerationParams,
    GenerationResponse,
    StreamChunk,
    StreamCallback,
    ToolCallback,
)
from .exceptions import (
    FoundationModelsError,
    InitializationError,
    NotAvailableError,
    InvalidParametersError,
    MemoryError,
    JSONParseError,
    GenerationError,
    TimeoutError,
    SessionNotFoundError,
    StreamNotFoundError,
    GuardrailViolationError,
    ToolNotFoundError,
    ToolExecutionError,
    UnknownError,
)

__all__ = [
    # Version
    "__version__",
    # Main classes
    "Session",
    "AsyncSession",
    # Constants
    "DEFAULT_TEMPERATURE",
    "DEFAULT_MAX_TOKENS",
    "MIN_TEMPERATURE",
    "MAX_TEMPERATURE",
    "TemperaturePreset",
    # Type definitions
    "Result",
    "Availability",
    "SessionConfig",
    "GenerationParams",
    "GenerationResponse",
    "StreamChunk",
    "StreamCallback",
    "ToolCallback",
    # Exceptions
    "FoundationModelsError",
    "InitializationError",
    "NotAvailableError",
    "InvalidParametersError",
    "MemoryError",
    "JSONParseError",
    "GenerationError",
    "TimeoutError",
    "SessionNotFoundError",
    "StreamNotFoundError",
    "GuardrailViolationError",
    "ToolNotFoundError",
    "ToolExecutionError",
    "UnknownError",
]
