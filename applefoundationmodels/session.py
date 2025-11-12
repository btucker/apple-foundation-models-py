"""
Session API for applefoundationmodels Python bindings.

Provides session management, text generation, and async streaming support.
"""

import asyncio
import json
import logging
from typing import (
    Optional,
    Dict,
    Any,
    Callable,
    Union,
    TYPE_CHECKING,
    List,
    cast,
    Iterator,
    overload,
    Type,
)
from typing_extensions import Literal
from queue import Queue, Empty
import threading

from .base_session import BaseSession, StreamQueueItem
from .types import (
    GenerationResponse,
    StreamChunk,
)
from .pydantic_compat import normalize_schema

if TYPE_CHECKING:
    from pydantic import BaseModel

logger = logging.getLogger(__name__)


class Session(BaseSession):
    """
    AI session for maintaining conversation state.

    Sessions maintain conversation history and can be configured with tools
    and instructions. Use as a context manager for automatic cleanup.

    Usage:
        with Session() as session:
            response = session.generate("Hello!")
            print(response.text)

        # With configuration:
        def get_weather(location: str) -> str:
            '''Get current weather for a location.'''
            return f"Weather in {location}: 22°C"

        session = Session(
            instructions="You are a helpful assistant.",
            tools=[get_weather]
        )
        response = session.generate("What's the weather in Paris?")
    """

    def _call_ffi(self, func, *args, **kwargs):
        """Execute FFI call synchronously."""
        return func(*args, **kwargs)

    def _create_stream_queue_adapter(self) -> BaseSession._StreamQueueAdapter:
        """Return the queue adapter used by synchronous streaming."""
        queue: Queue[StreamQueueItem] = Queue()

        def push(item: StreamQueueItem) -> None:
            queue.put(item)

        def get_sync() -> StreamQueueItem:
            while True:
                try:
                    return queue.get(timeout=0.1)
                except Empty:
                    continue

        return BaseSession._StreamQueueAdapter(push=push, get_sync=get_sync)

    def close(self) -> None:
        """Close the session and cleanup resources."""
        self._mark_closed()

    # ========================================================================
    # Type overloads for generate() method
    #
    # IMPORTANT: These overloads must be kept in sync with AsyncSession.generate()
    # in async_session.py. The signatures are identical except for:
    # - async keyword (Session: def generate() vs AsyncSession: async def generate())
    # - Return type for streaming (Iterator vs AsyncIterator)
    #
    # When modifying these overloads, update both files to maintain consistency.
    # ========================================================================

    # Type overload for non-streaming text generation
    @overload
    def generate(
        self,
        prompt: str,
        schema: None = None,
        stream: Literal[False] = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> GenerationResponse: ...

    # Type overload for non-streaming structured generation
    @overload
    def generate(
        self,
        prompt: str,
        schema: Union[Dict[str, Any], Type["BaseModel"]],
        stream: Literal[False] = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> GenerationResponse: ...

    # Type overload for streaming generation (text only, no structured streaming)
    @overload
    def generate(
        self,
        prompt: str,
        schema: None = None,
        stream: Literal[True] = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Iterator[StreamChunk]: ...

    def generate(
        self,
        prompt: str,
        schema: Optional[Union[Dict[str, Any], Type["BaseModel"]]] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Union[GenerationResponse, Iterator[StreamChunk]]:
        """
        Generate text or structured output, with optional streaming.

        This unified method supports three generation modes:
        1. Text generation (schema=None, stream=False) -> GenerationResponse
        2. Structured generation (schema=dict/model, stream=False) -> GenerationResponse
        3. Streaming generation (schema=None, stream=True) -> Iterator[StreamChunk]

        Args:
            prompt: Input text prompt
            schema: Optional JSON schema dict or Pydantic model for structured output
            stream: If True, return an iterator of chunks instead of complete response
            temperature: Sampling temperature (0.0-2.0, default: DEFAULT_TEMPERATURE)
            max_tokens: Maximum tokens to generate (default: DEFAULT_MAX_TOKENS)

        Returns:
            - GenerationResponse with .text or .parsed property (if stream=False)
            - Iterator[StreamChunk] yielding content deltas (if stream=True)

        Raises:
            RuntimeError: If session is closed
            GenerationError: If generation fails
            ValueError: If schema is provided with stream=True

        Examples:
            Text generation:
                >>> response = session.generate("What is Python?")
                >>> print(response.text)

            Structured generation:
                >>> schema = {"type": "object", "properties": {"name": {"type": "string"}}}
                >>> response = session.generate("Extract name: John Doe", schema=schema)
                >>> print(response.parsed)

            Streaming generation:
                >>> for chunk in session.generate("Tell me a story", stream=True):
                ...     print(chunk.content, end='', flush=True)
        """
        plan = self._plan_generate_call(stream, schema, temperature, max_tokens)

        if plan.mode == "stream":
            # Streaming mode: return Iterator[StreamChunk]
            return self._generate_stream_impl(prompt, plan.temperature, plan.max_tokens)
        if plan.mode == "structured" and schema is not None:
            # Structured generation mode
            return self._generate_structured_impl(
                prompt, schema, plan.temperature, plan.max_tokens
            )

        # Text generation mode
        return self._generate_text_impl(prompt, plan.temperature, plan.max_tokens)

    def _generate_text_impl(
        self, prompt: str, temperature: float, max_tokens: int
    ) -> GenerationResponse:
        """Internal implementation for text generation."""
        with self._generation_context() as start_length:
            text = self._call_ffi(
                self._ffi.generate,
                prompt,
                temperature,
                max_tokens,
            )
            return self._build_generation_response(text, False, start_length)

    def _generate_structured_impl(
        self,
        prompt: str,
        schema: Union[Dict[str, Any], Type["BaseModel"]],
        temperature: float,
        max_tokens: int,
    ) -> GenerationResponse:
        """Internal implementation for structured generation."""
        with self._generation_context() as start_length:
            json_schema = normalize_schema(schema)
            result = self._call_ffi(
                self._ffi.generate_structured,
                prompt,
                json_schema,
                temperature,
                max_tokens,
            )
            return self._build_generation_response(result, True, start_length)

    def _generate_stream_impl(
        self, prompt: str, temperature: float, max_tokens: int
    ) -> Iterator[StreamChunk]:
        """Internal implementation for streaming generation."""
        start_length = self._begin_generation()
        adapter = self._create_stream_queue_adapter()
        try:
            # Use shared streaming implementation from base class
            yield from self._stream_chunks_impl(
                prompt, temperature, max_tokens, adapter
            )
        finally:
            self._end_generation(start_length)

    def get_history(self) -> List[Dict[str, Any]]:
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
        result = self._call_ffi(self._ffi.get_history)
        return cast(List[Dict[str, Any]], result)

    def clear_history(self) -> None:
        """
        Clear conversation history.

        Removes all messages from the session while keeping the session active.
        """
        self._check_closed()
        self._call_ffi(self._ffi.clear_history)
        # Reset to current transcript length (may include persistent instructions)
        self._last_transcript_length = len(self.transcript)

    # Properties inherited from BaseSession (transcript, last_generation_transcript)
