"""
Session API for applefoundationmodels Python bindings.

Provides session management, text generation, and async streaming support.
"""

import asyncio
import json
from typing import (
    Optional,
    Dict,
    Any,
    AsyncIterator,
    Callable,
    Union,
    TYPE_CHECKING,
    List,
    cast,
)
from queue import Queue, Empty
import threading

from . import _foundationmodels
from .base import ContextManagedResource
from .types import GenerationParams, NormalizedGenerationParams
from .pydantic_compat import normalize_schema
from .tools import extract_function_schema, attach_tool_metadata

if TYPE_CHECKING:
    from pydantic import BaseModel


class Session(ContextManagedResource):
    """
    AI session for maintaining conversation state.

    Sessions maintain conversation history and can be configured with tools
    and instructions. Use as a context manager for automatic cleanup.

    Usage:
        with client.create_session() as session:
            response = session.generate("Hello!")
            print(response)
    """

    def __init__(self, session_id: int, config: Optional[Dict[str, Any]] = None):
        """
        Create a Session instance.

        Note: Users should create sessions via Client.create_session()
        rather than calling this constructor directly.

        Args:
            session_id: The session ID (always 0 in simplified API)
            config: Optional session configuration
        """
        self._session_id = session_id
        self._closed = False
        self._tools: Dict[str, Callable] = {}
        self._tools_registered = False
        self._config = config
        self._last_transcript_length = 0

    def close(self) -> None:
        """
        Close the session and cleanup resources.

        This is a no-op in the simplified API.
        """
        self._closed = False

    def _check_closed(self) -> None:
        """Raise error if session is closed."""
        if self._closed:
            raise RuntimeError("Session is closed")

    def _normalize_generation_params(
        self, temperature: Optional[float], max_tokens: Optional[int]
    ) -> NormalizedGenerationParams:
        """
        Normalize generation parameters with defaults.

        Args:
            temperature: Optional temperature value
            max_tokens: Optional max tokens value

        Returns:
            NormalizedGenerationParams with defaults applied
        """
        return NormalizedGenerationParams.from_optional(temperature, max_tokens)

    def _begin_generation(self) -> int:
        """
        Mark the beginning of a generation call.

        Returns:
            The current transcript length (boundary marker for this generation)
        """
        return len(self.transcript)

    def _end_generation(self, start_length: int) -> None:
        """
        Mark the end of a generation call.

        Args:
            start_length: The transcript length captured at generation start
        """
        self._last_transcript_length = start_length

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        include_reasoning: Optional[bool] = None,
        seed: Optional[int] = None,
    ) -> str:
        """
        Generate text response for a prompt.

        Args:
            prompt: Input text prompt
            temperature: Sampling temperature (0.0-2.0, default: DEFAULT_TEMPERATURE)
            max_tokens: Maximum tokens to generate (default: DEFAULT_MAX_TOKENS)
            include_reasoning: Include reasoning steps (not supported)
            seed: Random seed for reproducibility (not supported)

        Returns:
            Generated text response

        Raises:
            RuntimeError: If session is closed
            GenerationError: If generation fails

        Example:
            >>> response = session.generate("What is Python?")
            >>> print(response)
        """
        self._check_closed()
        params = self._normalize_generation_params(temperature, max_tokens)
        start_length = self._begin_generation()

        try:
            return _foundationmodels.generate(
                prompt, params.temperature, params.max_tokens
            )
        finally:
            self._end_generation(start_length)

    def generate_structured(
        self,
        prompt: str,
        schema: Union[Dict[str, Any], "BaseModel"],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate structured JSON output matching a schema.

        Args:
            prompt: Input text prompt
            schema: JSON schema dict or Pydantic BaseModel class
            temperature: Sampling temperature (0.0-2.0, default: DEFAULT_TEMPERATURE)
            max_tokens: Maximum tokens to generate (default: DEFAULT_MAX_TOKENS)

        Returns:
            Dictionary containing the parsed JSON matching the schema

        Raises:
            RuntimeError: If session is closed
            GenerationError: If generation fails
            JSONParseError: If schema or response is invalid JSON
            TypeError: If schema is neither dict nor Pydantic model
            ImportError: If Pydantic model provided but Pydantic not installed

        Example (JSON Schema):
            >>> schema = {
            ...     "type": "object",
            ...     "properties": {
            ...         "name": {"type": "string"},
            ...         "age": {"type": "integer"}
            ...     },
            ...     "required": ["name", "age"]
            ... }
            >>> result = session.generate_structured(
            ...     "Extract: Alice is 28",
            ...     schema=schema
            ... )
            >>> print(result)
            {'name': 'Alice', 'age': 28}

        Example (Pydantic):
            >>> from pydantic import BaseModel
            >>> class Person(BaseModel):
            ...     name: str
            ...     age: int
            >>> result = session.generate_structured(
            ...     "Extract: Alice is 28",
            ...     schema=Person
            ... )
            >>> person = Person(**result)  # Parse directly into Pydantic model
            >>> print(person.name, person.age)
            Alice 28
        """
        self._check_closed()
        params = self._normalize_generation_params(temperature, max_tokens)
        json_schema = normalize_schema(schema)
        start_length = self._begin_generation()

        try:
            return _foundationmodels.generate_structured(
                prompt, json_schema, params.temperature, params.max_tokens
            )
        finally:
            self._end_generation(start_length)

    async def generate_stream(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        include_reasoning: Optional[bool] = None,
        seed: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """
        Generate text response with async streaming.

        Args:
            prompt: Input text prompt
            temperature: Sampling temperature (0.0-2.0, default: DEFAULT_TEMPERATURE)
            max_tokens: Maximum tokens to generate (default: DEFAULT_MAX_TOKENS)
            include_reasoning: Include reasoning steps (not supported)
            seed: Random seed (not supported)

        Yields:
            Text chunks as they are generated

        Example:
            >>> async for chunk in session.generate_stream("Tell me a story"):
            ...     print(chunk, end='', flush=True)
        """
        self._check_closed()
        params = self._normalize_generation_params(temperature, max_tokens)
        start_length = self._begin_generation()

        try:
            # Use a queue to bridge the sync callback and async iterator
            queue: Queue = Queue()

            def callback(chunk: Optional[str]) -> None:
                queue.put(chunk)

            # Run streaming in a background thread
            def run_stream():
                try:
                    _foundationmodels.generate_stream(
                        prompt, callback, params.temperature, params.max_tokens
                    )
                except Exception as e:
                    queue.put(e)

            thread = threading.Thread(target=run_stream, daemon=True)
            thread.start()

            # Yield chunks from queue
            while True:
                # Use asyncio.sleep to yield control
                await asyncio.sleep(0)

                try:
                    chunk = queue.get(timeout=0.1)
                except Empty:
                    continue

                if isinstance(chunk, Exception):
                    raise chunk

                if chunk is None:  # End of stream
                    break

                yield chunk

            thread.join(timeout=1.0)
        finally:
            self._end_generation(start_length)

    def get_history(self) -> list:
        """
        Get conversation history.

        Returns:
            List of message dictionaries with 'role' and 'content' keys

        Example:
            >>> history = session.get_history()
            >>> for msg in history:
            ...     print(f"{msg['role']}: {msg['content']}")
        """
        self._check_closed()
        return _foundationmodels.get_history()

    def clear_history(self) -> None:
        """
        Clear conversation history.

        Removes all messages from the session while keeping the session active.
        """
        self._check_closed()
        _foundationmodels.clear_history()
        self._last_transcript_length = 0

    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to conversation history.

        Note: This is a stub in the simplified API.

        Args:
            role: Message role ('user', 'assistant', 'system')
            content: Message content
        """
        self._check_closed()
        _foundationmodels.add_message(role, content)

    def tool(
        self,
        description: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Callable[[Callable], Callable]:
        """
        Decorator to register a function as a tool for this session.

        The function's signature and docstring are used to automatically
        generate a JSON schema for the tool's parameters.

        Args:
            description: Optional tool description (uses docstring if not provided)
            name: Optional tool name (uses function name if not provided)

        Returns:
            Decorator function

        Note:
            Tool output size limits:
            - Initial buffer: 16KB
            - Maximum size: 1MB (automatically retried with larger buffers)
            - Tools returning outputs larger than 1MB will raise an error
            - For large outputs, consider returning references or summaries

        Example:
            @session.tool(description="Get current weather")
            def get_weather(location: str, units: str = "celsius") -> str:
                '''Get weather for a location.'''
                return f"Weather in {location}: 20°{units[0].upper()}"

            response = session.generate("What's the weather in Paris?")
        """

        def decorator(func: Callable) -> Callable:
            # Extract schema and attach metadata using shared helper
            schema = extract_function_schema(func)
            final_schema = attach_tool_metadata(func, schema, description, name)

            # Session-specific logic: store and register tool
            tool_name = final_schema["name"]
            self._tools[tool_name] = func
            self._register_tools()

            return func

        return decorator

    def _register_tools(self) -> None:
        """
        Register all tools with the FFI layer.

        Called automatically when tools are added via decorator.
        Recreates the session with tools enabled.
        """
        if not self._tools:
            return

        # Register tools with C FFI
        _foundationmodels.register_tools(self._tools)
        self._tools_registered = True

        # Recreate session with tools enabled
        # This is necessary because the session needs to be created with tools
        # for FoundationModels to know about them
        config = self._config or {}
        _foundationmodels.create_session(config)

    @property
    def transcript(self) -> List[Dict[str, Any]]:
        """
        Get the session transcript including tool calls.

        Returns a list of transcript entries showing the full conversation
        history including instructions, prompts, tool calls, tool outputs,
        and responses.

        Returns:
            List of transcript entry dictionaries with keys:
            - type: Entry type ('instructions', 'prompt', 'response', 'tool_call', 'tool_output')
            - content: Entry content (for text entries)
            - tool_name: Tool name (for tool_call entries)
            - tool_id: Tool call ID (for tool_call and tool_output entries)
            - arguments: Tool arguments as JSON string (for tool_call entries)

        Example:
            >>> transcript = session.transcript
            >>> for entry in transcript:
            ...     print(f"{entry['type']}: {entry.get('content', '')}")
        """
        self._check_closed()
        # Explicit cast to ensure type checkers see the correct return type
        return cast(List[Dict[str, Any]], _foundationmodels.get_transcript())

    @property
    def last_generation_transcript(self) -> List[Dict[str, Any]]:
        """
        Get transcript entries from the most recent generate() call only.

        Unlike the `transcript` property which returns the full accumulated history,
        this returns only the entries added during the last generation call
        (generate(), generate_structured(), or generate_stream()).

        This is useful when you need to inspect what happened during a specific
        generation without worrying about accumulated history from previous calls.

        Returns:
            List of transcript entries from the last generate() call.
            Returns empty list if no generation has been performed yet.

        Example:
            >>> # First generation
            >>> response1 = session.generate("What is 2 + 2?")
            >>> entries1 = session.last_generation_transcript
            >>> print(f"First call: {len(entries1)} entries")

            >>> # Second generation on same session
            >>> response2 = session.generate("What is 5 + 7?")
            >>> entries2 = session.last_generation_transcript
            >>> print(f"Second call: {len(entries2)} entries (only from second call)")
        """
        self._check_closed()
        full_transcript = self.transcript
        return full_transcript[self._last_transcript_length :]
