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
import threading

from . import _foundationmodels
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
        async with client.create_async_session() as session:
            response = await session.generate("Hello!")
            print(response.text)

            # Async streaming
            async for chunk in session.generate("Story", stream=True):
                print(chunk.content, end='', flush=True)
    """

    async def _call_ffi(self, func, *args, **kwargs):
        """
        Execute FFI call asynchronously.

        Streaming functions use callbacks and must run directly in threads,
        while other functions are safely wrapped in asyncio.to_thread.
        """
        # Streaming must run in thread with callback - don't wrap
        if func == _foundationmodels.generate_stream:
            return func(*args, **kwargs)
        # Other functions can be safely wrapped
        return await asyncio.to_thread(func, *args, **kwargs)

    def close(self) -> None:
        """
        Close the session and cleanup resources synchronously.

        This method runs the async cleanup on the event loop. For async
        contexts, prefer using aclose() or the async context manager.

        Note: This satisfies the ContextManagedResource.close() contract
        inherited from BaseSession.
        """
        try:
            # Check if we're in an async context with a running event loop
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop, safe to use asyncio.run()
            asyncio.run(self.aclose())
        else:
            # If we get here, there's a running loop - we can't use asyncio.run()
            # User should call aclose() instead in async contexts
            raise RuntimeError(
                "close() called from async context. Use 'await session.aclose()' instead."
            )

    async def aclose(self) -> None:
        """
        Close the session and cleanup resources asynchronously.

        This method should be used in async contexts or with the async
        context manager.

        Example:
            >>> session = await client.create_session()
            >>> # ... use session ...
            >>> await session.aclose()
        """
        self._closed = True

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
        self._validate_generate_params(stream, schema)

        # Apply defaults to parameters
        temp = self._get_temperature(temperature)
        max_tok = self._get_max_tokens(max_tokens)

        if stream:
            # Return async iterator directly
            return self._generate_stream_impl(prompt, temp, max_tok)
        elif schema is not None:
            # Structured generation mode
            return await self._generate_structured_impl(prompt, schema, temp, max_tok)
        else:
            # Text generation mode
            return await self._generate_text_impl(prompt, temp, max_tok)

    async def _generate_text_impl(
        self, prompt: str, temperature: float, max_tokens: int
    ) -> GenerationResponse:
        """Internal implementation for async text generation."""
        start_length = self._begin_generation()
        try:
            # Run sync FFI call in thread pool
            text = await asyncio.to_thread(
                _foundationmodels.generate,
                prompt,
                temperature,
                max_tokens,
            )
            return self._build_generation_response(text, False, start_length)
        except Exception:
            self._end_generation(start_length)
            raise

    async def _generate_structured_impl(
        self,
        prompt: str,
        schema: Union[Dict[str, Any], Type["BaseModel"]],
        temperature: float,
        max_tokens: int,
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
                temperature,
                max_tokens,
            )
            return self._build_generation_response(result, True, start_length)
        except Exception:
            self._end_generation(start_length)
            raise

    async def _generate_stream_impl(
        self, prompt: str, temperature: float, max_tokens: int
    ) -> AsyncIterator[StreamChunk]:
        """Internal implementation for async streaming generation."""
        start_length = self._begin_generation()
        try:
            # Use asyncio.Queue for truly async coordination
            queue: asyncio.Queue = asyncio.Queue()

            # Get the event loop for thread-safe queue operations
            loop = asyncio.get_event_loop()

            def callback(chunk: Optional[str]) -> None:
                # Use run_coroutine_threadsafe to put items from background thread
                asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)

            # Run streaming in a background thread
            def run_stream():
                try:
                    _foundationmodels.generate_stream(
                        prompt, callback, temperature, max_tokens
                    )
                except Exception as e:
                    asyncio.run_coroutine_threadsafe(queue.put(e), loop)

            thread = threading.Thread(target=run_stream, daemon=True)
            thread.start()

            # Yield StreamChunk objects asynchronously
            chunk_index = 0
            while True:
                # Truly async get - no polling needed
                item = await queue.get()

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

            # Wait for streaming thread to complete cleanup
            # By this point we've received the None sentinel, so the stream is done
            # and the thread should finish quickly. We wait up to 5 seconds for
            # clean shutdown.
            thread.join(timeout=5.0)

            if thread.is_alive():
                # Thread didn't finish in time - this shouldn't normally happen
                # since we've already received the end-of-stream signal
                logger.warning(
                    "Streaming thread did not complete within 5 seconds after "
                    "stream end. Thread will continue as daemon and be cleaned up "
                    "at process exit."
                )
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

    # Properties inherited from BaseSession (transcript, last_generation_transcript)
