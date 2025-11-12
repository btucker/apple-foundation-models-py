"""
AsyncSession API for applefoundationmodels Python bindings.

Provides async session management, text generation, and async streaming support.
"""

import asyncio
import logging
from typing import (
    Optional,
    Dict,
    Any,
    List,
    AsyncIterator,
    Union,
    TYPE_CHECKING,
    overload,
    Type,
    Coroutine,
    cast,
)
from typing_extensions import Literal

from .base_session import BaseSession, StreamQueueItem
from .base import AsyncContextManagedResource
from .types import (
    GenerationResponse,
    StreamChunk,
)
from .pydantic_compat import normalize_schema

if TYPE_CHECKING:
    from pydantic import BaseModel

logger = logging.getLogger(__name__)


class AsyncSession(BaseSession, AsyncContextManagedResource):
    """
    Async AI session for maintaining conversation state.

    AsyncSession provides async/await support for all operations including
    streaming. Use this for async applications. Sessions maintain conversation
    history and can be configured with tools and instructions.

    Usage:
        async with AsyncSession() as session:
            response = await session.generate("Hello!")
            print(response.text)

            # Async streaming
            async for chunk in session.generate("Story", stream=True):
                print(chunk.content, end='', flush=True)

        # With configuration:
        def get_weather(location: str) -> str:
            '''Get current weather for a location.'''
            return f"Weather in {location}: 22°C"

        session = AsyncSession(
            instructions="You are a helpful assistant.",
            tools=[get_weather]
        )
        response = await session.generate("What's the weather in Paris?")
        await session.aclose()
    """

    async def _call_ffi(self, func, *args, **kwargs):
        """Execute FFI call asynchronously via a worker thread."""
        return await asyncio.to_thread(func, *args, **kwargs)

    def _create_stream_queue_adapter(self) -> BaseSession._StreamQueueAdapter:
        """Return the async-aware queue adapter used for streaming."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[StreamQueueItem] = asyncio.Queue()

        def push(item: StreamQueueItem) -> None:
            asyncio.run_coroutine_threadsafe(queue.put(item), loop)

        async def get_async() -> StreamQueueItem:
            return await queue.get()

        def get_sync() -> StreamQueueItem:
            raise RuntimeError(
                "Synchronous queue access is not supported for AsyncSession"
            )

        return BaseSession._StreamQueueAdapter(
            push=push,
            get_sync=get_sync,
            get_async=get_async,
        )

    def close(self) -> None:
        """Close the session synchronously.

        When no event loop is running, this method drives the async cleanup via
        asyncio.run(). If called while an event loop is active, a RuntimeError is
        raised to avoid nested event loops—the caller should instead await
        `aclose()` or use `async with`.
        """
        if self._closed:
            return

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.aclose())
            return

        raise RuntimeError(
            "AsyncSession.close() cannot be called while an event loop is running. "
            "Use await session.aclose() or 'async with AsyncSession()'."
        )

    async def aclose(self) -> None:
        """Asynchronously close the session and cleanup resources."""
        if self._closed:
            return
        self._mark_closed()

    # ========================================================================
    # Type overloads for generate() method
    #
    # IMPORTANT: These overloads must be kept in sync with Session.generate()
    # in session.py. The signatures are identical except for:
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
    ) -> Coroutine[Any, Any, GenerationResponse]: ...

    # Type overload for non-streaming structured generation
    @overload
    def generate(
        self,
        prompt: str,
        schema: Union[Dict[str, Any], Type["BaseModel"]],
        stream: Literal[False] = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Coroutine[Any, Any, GenerationResponse]: ...

    # Type overload for async streaming generation (text only)
    @overload
    def generate(
        self,
        prompt: str,
        schema: None = None,
        stream: Literal[True] = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[StreamChunk]: ...

    def generate(
        self,
        prompt: str,
        schema: Optional[Union[Dict[str, Any], Type["BaseModel"]]] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Union[Coroutine[Any, Any, GenerationResponse], AsyncIterator[StreamChunk]]:
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
        plan = self._plan_generate_call(stream, schema, temperature, max_tokens)

        if plan.mode == "stream":
            return self._generate_stream_impl(prompt, plan.temperature, plan.max_tokens)
        if plan.mode == "structured" and schema is not None:
            return self._generate_structured_impl(
                prompt, schema, plan.temperature, plan.max_tokens
            )

        return self._generate_text_impl(prompt, plan.temperature, plan.max_tokens)

    async def _generate_text_impl(
        self, prompt: str, temperature: float, max_tokens: int
    ) -> GenerationResponse:
        """Internal implementation for async text generation."""
        async with self._async_generation_context() as start_length:
            text = await self._call_ffi(
                self._ffi.generate,
                prompt,
                temperature,
                max_tokens,
            )
            return self._build_generation_response(text, False, start_length)

    async def _generate_structured_impl(
        self,
        prompt: str,
        schema: Union[Dict[str, Any], Type["BaseModel"]],
        temperature: float,
        max_tokens: int,
    ) -> GenerationResponse:
        """Internal implementation for async structured generation."""
        async with self._async_generation_context() as start_length:
            json_schema = normalize_schema(schema)
            result = await self._call_ffi(
                self._ffi.generate_structured,
                prompt,
                json_schema,
                temperature,
                max_tokens,
            )
            return self._build_generation_response(result, True, start_length)

    async def _generate_stream_impl(
        self, prompt: str, temperature: float, max_tokens: int
    ) -> AsyncIterator[StreamChunk]:
        """Internal implementation for async streaming generation."""
        start_length = self._begin_generation()
        adapter = self._create_stream_queue_adapter()
        try:
            async for chunk in self._stream_chunks_async_impl(
                prompt, temperature, max_tokens, adapter
            ):
                yield chunk
        finally:
            self._end_generation(start_length)

    async def get_history(self) -> List[Dict[str, Any]]:
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
        result = await self._call_ffi(self._ffi.get_history)
        return cast(List[Dict[str, Any]], result)

    async def clear_history(self) -> None:
        """
        Clear conversation history asynchronously.

        Removes all messages from the session while keeping the session active.
        """
        self._check_closed()
        await self._call_ffi(self._ffi.clear_history)
        # Reset to current transcript length (may include persistent instructions)
        self._last_transcript_length = len(self.transcript)

    # Properties inherited from BaseSession (transcript, last_generation_transcript)
