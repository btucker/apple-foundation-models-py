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
from .base_session import BaseSession
from .types import (
    NormalizedGenerationParams,
    GenerationResponse,
    StreamChunk,
)
from .pydantic_compat import normalize_schema

if TYPE_CHECKING:
    from pydantic import BaseModel


class AsyncSession(BaseSession):
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

    async def close(self) -> None:
        """Close the session and cleanup resources."""
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
            return self._build_generation_response(text, False, start_length)
        except Exception:
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
            return self._build_generation_response(result, True, start_length)
        except Exception:
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

    # Properties inherited from BaseSession (transcript, last_generation_transcript)
