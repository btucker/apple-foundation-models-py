"""
AsyncSession API for applefoundationmodels Python bindings.

Provides async session management, text generation, and async streaming support.
"""

import asyncio
import time
import logging
from typing import (
    Optional,
    Dict,
    Any,
    AsyncIterator,
    Callable,
    Union,
    TYPE_CHECKING,
    overload,
    Type,
    Coroutine,
)
from typing_extensions import Literal

from .base_session import BaseSession
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
        await session.close()
    """

    async def _call_ffi(self, func, *args, **kwargs):
        """Execute FFI call asynchronously via a worker thread."""
        return await asyncio.to_thread(func, *args, **kwargs)

    def _create_stream_queue(self) -> asyncio.Queue:
        """Create an async queue for streaming."""
        return asyncio.Queue()

    def _create_stream_callback(
        self, queue: asyncio.Queue
    ) -> Callable[[Optional[str]], None]:
        """Create a callback that puts chunks into the async queue from background thread."""
        loop = asyncio.get_event_loop()
        return lambda chunk: asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)

    def _get_from_stream_queue(self, queue: asyncio.Queue) -> Optional[str]:
        """
        Get item from async queue synchronously by polling.

        Note: This is called from the sync generator _stream_chunks_impl.
        We poll the async queue using get_nowait() to make it work in a sync context.
        """
        while True:
            try:
                return queue.get_nowait()
            except asyncio.QueueEmpty:
                time.sleep(0.01)  # Small sleep to avoid busy-waiting
                continue

    async def close(self) -> None:
        """
        Close the session and cleanup resources.

        Example:
            >>> session = AsyncSession()
            >>> # ... use session ...
            >>> await session.close()
        """
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
        self._check_closed()
        self._validate_generate_params(stream, schema)

        # Apply defaults to parameters
        temp, max_tok = self._apply_defaults(temperature, max_tokens)

        if stream:
            # Return async iterator directly (no await needed - caller uses async for)
            return self._generate_stream_impl(prompt, temp, max_tok)
        elif schema is not None:
            # Structured generation mode - return coroutine (caller will await)
            return self._generate_structured_impl(prompt, schema, temp, max_tok)
        else:
            # Text generation mode - return coroutine (caller will await)
            return self._generate_text_impl(prompt, temp, max_tok)

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

        # Create an asyncio.Queue for inter-thread communication
        output_queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        # Create the queue and callback in the main thread (where event loop exists)
        inner_queue = self._create_stream_queue()
        callback = lambda chunk: asyncio.run_coroutine_threadsafe(
            inner_queue.put(chunk), loop
        )

        # Sentinel value to signal end of stream
        _DONE = object()

        def run_generator_in_thread():
            """Run the sync generator in a background thread and push chunks to async queue."""
            try:
                # Iterate over the sync generator and push chunks to the async queue
                for chunk in self._stream_chunks_impl(
                    prompt, temperature, max_tokens, inner_queue, callback
                ):
                    # Push chunk to async queue from background thread
                    asyncio.run_coroutine_threadsafe(
                        output_queue.put(chunk), loop
                    ).result()

                # Signal completion
                asyncio.run_coroutine_threadsafe(output_queue.put(_DONE), loop).result()
            except Exception as e:
                # Forward exceptions to the async queue
                asyncio.run_coroutine_threadsafe(output_queue.put(e), loop).result()

        # Spawn the generator in a background thread
        import concurrent.futures

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(run_generator_in_thread)

        try:
            # Yield chunks from the async queue (non-blocking, event-loop friendly)
            while True:
                item = await output_queue.get()

                # Check for end of stream sentinel
                if item is _DONE:
                    break

                # Check for exceptions
                if isinstance(item, Exception):
                    raise item

                # Yield the chunk
                yield item
        finally:
            # Cleanup: ensure the background thread completes
            try:
                # Wait for the background task to complete (with timeout)
                future.result(timeout=5.0)
            except concurrent.futures.TimeoutError:
                logger.warning(
                    "Background streaming task did not complete within 5 seconds. "
                    "Task will be cleaned up by the executor."
                )
            except Exception as e:
                logger.error(f"Error during streaming cleanup: {e}")
            finally:
                # Shutdown the executor
                executor.shutdown(wait=False)
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
        return await self._call_ffi(self._ffi.get_history)

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
