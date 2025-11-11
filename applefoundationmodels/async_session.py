"""
AsyncSession API for applefoundationmodels Python bindings.

Provides async session management, text generation, and async streaming support.
"""

import asyncio
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
    overload,
    Type,
)
from typing_extensions import Literal
from queue import Queue, Empty
import threading

from . import _foundationmodels
from .base import ContextManagedResource
from .types import (
    NormalizedGenerationParams,
    GenerationResponse,
    StreamChunk,
    ToolCall,
    Function,
)
from .pydantic_compat import normalize_schema
from .tools import extract_function_schema, attach_tool_metadata

if TYPE_CHECKING:
    from pydantic import BaseModel


class AsyncSession(ContextManagedResource):
    """
    Async AI session for maintaining conversation state.

    AsyncSession provides async/await support for all operations including
    streaming. Use this for async applications. Sessions maintain conversation
    history and can be configured with tools and instructions.

    Usage:
        async with client.create_async_session() as session:
            response = await session.generate("Hello!")
            print(response.text)

            # Async streaming
            async for chunk in session.generate("Story", stream=True):
                print(chunk.content, end='', flush=True)
    """

    def __init__(self, session_id: int, config: Optional[Dict[str, Any]] = None):
        """
        Create an AsyncSession instance.

        Note: Users should create sessions via Client.create_async_session()
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
        # Initialize to current transcript length to exclude any initial instructions
        self._last_transcript_length = len(self.transcript)

    async def close(self) -> None:
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

    def _extract_tool_calls_from_transcript(
        self, transcript_entries: List[Dict[str, Any]]
    ) -> Optional[List[ToolCall]]:
        """
        Extract tool calls from transcript entries.

        Args:
            transcript_entries: List of transcript entries to search

        Returns:
            List of ToolCall objects if any tool calls found, None otherwise
        """
        tool_calls = []
        for entry in transcript_entries:
            if entry.get("type") == "tool_call":
                tool_call = ToolCall(
                    id=entry.get("tool_id", ""),
                    type="function",
                    function=Function(
                        name=entry.get("tool_name", ""),
                        arguments=entry.get("arguments", "{}"),
                    ),
                )
                tool_calls.append(tool_call)

        return tool_calls if tool_calls else None

    # Type overloads for non-streaming text generation
    @overload
    async def generate(
        self,
        prompt: str,
        schema: None = None,
        stream: Literal[False] = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> GenerationResponse: ...

    # Type overload for non-streaming structured generation
    @overload
    async def generate(
        self,
        prompt: str,
        schema: Union[Dict[str, Any], Type["BaseModel"]],
        stream: Literal[False] = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> GenerationResponse: ...

    # Type overload for async streaming generation (text only)
    @overload
    async def generate(
        self,
        prompt: str,
        schema: None = None,
        stream: Literal[True] = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[StreamChunk]: ...

    async def generate(
        self,
        prompt: str,
        schema: Optional[Union[Dict[str, Any], Type["BaseModel"]]] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Union[GenerationResponse, AsyncIterator[StreamChunk]]:
        """
        Generate text or structured output asynchronously, with optional streaming.

        This unified async method supports three generation modes:
        1. Text generation (schema=None, stream=False) -> GenerationResponse
        2. Structured generation (schema=dict/model, stream=False) -> GenerationResponse
        3. Async streaming (schema=None, stream=True) -> AsyncIterator[StreamChunk]

        Args:
            prompt: Input text prompt
            schema: Optional JSON schema dict or Pydantic model for structured output
            stream: If True, return an async iterator of chunks
            temperature: Sampling temperature (0.0-2.0, default: DEFAULT_TEMPERATURE)
            max_tokens: Maximum tokens to generate (default: DEFAULT_MAX_TOKENS)

        Returns:
            - GenerationResponse with .text or .parsed property (if stream=False)
            - AsyncIterator[StreamChunk] yielding content deltas (if stream=True)

        Raises:
            RuntimeError: If session is closed
            GenerationError: If generation fails
            ValueError: If schema is provided with stream=True

        Examples:
            Text generation:
                >>> response = await session.generate("What is Python?")
                >>> print(response.text)

            Structured generation:
                >>> schema = {"type": "object", "properties": {"name": {"type": "string"}}}
                >>> response = await session.generate("Extract name: John", schema=schema)
                >>> print(response.parsed)

            Async streaming:
                >>> async for chunk in session.generate("Tell a story", stream=True):
                ...     print(chunk.content, end='', flush=True)
        """
        self._check_closed()
        params = self._normalize_generation_params(temperature, max_tokens)

        # Validate: streaming only supports text generation
        if stream and schema is not None:
            raise ValueError(
                "Streaming is not supported with structured output (schema parameter)"
            )

        if stream:
            # Return async iterator directly
            return self._generate_stream_impl(prompt, params)
        elif schema is not None:
            # Structured generation mode
            return await self._generate_structured_impl(prompt, schema, params)
        else:
            # Text generation mode
            return await self._generate_text_impl(prompt, params)

    async def _generate_text_impl(
        self, prompt: str, params: NormalizedGenerationParams
    ) -> GenerationResponse:
        """Internal implementation for async text generation."""
        start_length = self._begin_generation()
        try:
            # Run sync FFI call in thread pool
            text = await asyncio.to_thread(
                _foundationmodels.generate,
                prompt,
                params.temperature,
                params.max_tokens,
            )

            # Update the generation boundary marker
            self._end_generation(start_length)

            # Extract tool calls from the generation transcript
            tool_calls = self._extract_tool_calls_from_transcript(
                self.last_generation_transcript
            )

            # Set finish reason based on whether tools were called
            finish_reason = "tool_calls" if tool_calls else "stop"

            return GenerationResponse(
                content=text,
                is_structured=False,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
            )
        except Exception:
            # Still update boundary on error
            self._end_generation(start_length)
            raise

    async def _generate_structured_impl(
        self,
        prompt: str,
        schema: Union[Dict[str, Any], Type["BaseModel"]],
        params: NormalizedGenerationParams,
    ) -> GenerationResponse:
        """Internal implementation for async structured generation."""
        start_length = self._begin_generation()
        try:
            json_schema = normalize_schema(schema)
            # Run sync FFI call in thread pool
            result = await asyncio.to_thread(
                _foundationmodels.generate_structured,
                prompt,
                json_schema,
                params.temperature,
                params.max_tokens,
            )

            # Update the generation boundary marker
            self._end_generation(start_length)

            # Structured generation does not support tool calls
            return GenerationResponse(
                content=result,
                is_structured=True,
                tool_calls=None,
                finish_reason="stop",
            )
        except Exception:
            # Still update boundary on error
            self._end_generation(start_length)
            raise

    async def _generate_stream_impl(
        self, prompt: str, params: NormalizedGenerationParams
    ) -> AsyncIterator[StreamChunk]:
        """Internal implementation for async streaming generation."""
        start_length = self._begin_generation()
        try:
            # Use a queue to collect chunks from callback
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

            # Yield StreamChunk objects asynchronously
            chunk_index = 0
            while True:
                # Use asyncio.sleep to yield control
                await asyncio.sleep(0)

                try:
                    item = queue.get(timeout=0.1)
                except Empty:
                    continue

                if isinstance(item, Exception):
                    raise item

                if item is None:  # End of stream
                    # Yield final chunk with finish_reason
                    yield StreamChunk(
                        content="", finish_reason="stop", index=chunk_index
                    )
                    break

                # Yield chunk with content
                yield StreamChunk(content=item, finish_reason=None, index=chunk_index)
                chunk_index += 1

            thread.join(timeout=1.0)
        finally:
            self._end_generation(start_length)

    async def get_history(self) -> list:
        """
        Get conversation history asynchronously.

        Returns:
            List of message dictionaries with 'role' and 'content' keys

        Example:
            >>> history = await session.get_history()
            >>> for msg in history:
            ...     print(f"{msg['role']}: {msg['content']}")
        """
        self._check_closed()
        return await asyncio.to_thread(_foundationmodels.get_history)

    async def clear_history(self) -> None:
        """
        Clear conversation history asynchronously.

        Removes all messages from the session while keeping the session active.
        """
        self._check_closed()
        await asyncio.to_thread(_foundationmodels.clear_history)
        # Reset to current transcript length (may include persistent instructions)
        self._last_transcript_length = len(self.transcript)

    async def add_message(self, role: str, content: str) -> None:
        """
        Add a message to conversation history asynchronously.

        Note: This is a stub in the simplified API.

        Args:
            role: Message role ('user', 'assistant', 'system')
            content: Message content
        """
        self._check_closed()
        await asyncio.to_thread(_foundationmodels.add_message, role, content)

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

            response = await session.generate("What's the weather in Paris?")
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
        this returns only the entries added during the last generation call.

        This is useful when you need to inspect what happened during a specific
        generation without worrying about accumulated history from previous calls.

        Returns:
            List of transcript entries from the last generate() call.
            Returns empty list if no generation has been performed yet.

        Example:
            >>> # First generation
            >>> response1 = await session.generate("What is 2 + 2?")
            >>> entries1 = session.last_generation_transcript
            >>> print(f"First call: {len(entries1)} entries")

            >>> # Second generation on same session
            >>> response2 = await session.generate("What is 5 + 7?")
            >>> entries2 = session.last_generation_transcript
            >>> print(f"Second call: {len(entries2)} entries (only from second call)")
        """
        self._check_closed()
        full_transcript = self.transcript
        return full_transcript[self._last_transcript_length :]
