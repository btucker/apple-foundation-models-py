"""
Session API for applefoundationmodels Python bindings.

Provides session management, text generation, and async streaming support.
"""

import asyncio
import json
from typing import Optional, Dict, Any, AsyncIterator, Callable, Union, TYPE_CHECKING, List
from queue import Queue, Empty
import threading

from . import _foundationmodels
from .base import ContextManagedResource
from .constants import DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS
from .types import GenerationParams
from .pydantic_compat import normalize_schema
from .tools import extract_function_schema

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

    def __init__(self, session_id: int):
        """
        Create a Session instance.

        Note: Users should create sessions via Client.create_session()
        rather than calling this constructor directly.

        Args:
            session_id: The session ID (always 0 in simplified API)
        """
        self._session_id = session_id
        self._closed = False
        self._tools: Dict[str, Callable] = {}
        self._tools_registered = False

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
    ) -> tuple[float, int]:
        """
        Normalize generation parameters with defaults.

        Args:
            temperature: Optional temperature value
            max_tokens: Optional max tokens value

        Returns:
            Tuple of (temperature, max_tokens) with defaults applied
        """
        temp = temperature if temperature is not None else DEFAULT_TEMPERATURE
        tokens = max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS
        return temp, tokens

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
        temp, tokens = self._normalize_generation_params(temperature, max_tokens)
        return _foundationmodels.generate(prompt, temp, tokens)

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
        temp, tokens = self._normalize_generation_params(temperature, max_tokens)

        # Normalize schema to JSON Schema dict (handles Pydantic models)
        json_schema = normalize_schema(schema)

        return _foundationmodels.generate_structured(prompt, json_schema, temp, tokens)

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
        temp, tokens = self._normalize_generation_params(temperature, max_tokens)

        # Use a queue to bridge the sync callback and async iterator
        queue: Queue = Queue()

        def callback(chunk: Optional[str]) -> None:
            queue.put(chunk)

        # Run streaming in a background thread
        def run_stream():
            try:
                _foundationmodels.generate_stream(prompt, callback, temp, tokens)
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

        Example:
            @session.tool(description="Get current weather")
            def get_weather(location: str, units: str = "celsius") -> str:
                '''Get weather for a location.'''
                return f"Weather in {location}: 20°{units[0].upper()}"

            response = session.generate("What's the weather in Paris?")
        """

        def decorator(func: Callable) -> Callable:
            # Extract schema from function
            schema = extract_function_schema(func)

            # Override with provided values
            if description is not None:
                schema["description"] = description
            if name is not None:
                schema["name"] = name

            # Store tool function
            tool_name = schema["name"]
            self._tools[tool_name] = func

            # Attach metadata to function
            func._tool_name = schema["name"]
            func._tool_description = schema["description"]
            func._tool_parameters = schema["parameters"]

            # Register tools if not already done
            self._register_tools()

            return func

        return decorator

    def _register_tools(self) -> None:
        """
        Register all tools with the FFI layer.

        Called automatically when tools are added via decorator.
        """
        if not self._tools:
            return

        # Register tools with C FFI
        _foundationmodels.register_tools(self._tools)
        self._tools_registered = True

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
        return _foundationmodels.get_transcript()
