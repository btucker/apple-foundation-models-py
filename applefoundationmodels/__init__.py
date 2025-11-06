"""
applefoundationmodels: Python bindings for Apple's FoundationModels framework

High-level Pythonic interface for accessing Apple Intelligence on-device
Foundation models.

Basic usage:
    from applefoundationmodels import Client

    with Client() as client:
        # Check availability
        if not client.is_ready():
            print("Apple Intelligence not available")
            return

        # Create a session and generate response
        session = client.create_session()
        response = session.generate("Hello, how are you?")
        print(response)

Async streaming:
    import asyncio
    from applefoundationmodels import Client

    async def main():
        with Client() as client:
            session = client.create_session()
            async for chunk in session.generate_stream("Tell me a story"):
                print(chunk, end='', flush=True)

    asyncio.run(main())
"""

__version__ = "0.1.0"

# Public API exports
from .client import Client, client
from .session import Session
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
    Stats,
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
    "Client",
    "Session",
    "client",
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
    "Stats",
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
