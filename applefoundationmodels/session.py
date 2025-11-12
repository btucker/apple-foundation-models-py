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
    AsyncIterator,
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

from .base_session import BaseSession
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
        with client.create_session() as session:
            response = session.generate("Hello!")
            print(response)
    """

    def _call_ffi(self, func, *args, **kwargs):
        """Execute FFI call synchronously."""
        return func(*args, **kwargs)

    def _create_stream_queue(self) -> Queue:
        """Create a synchronous queue for streaming."""
        return Queue()

    def _create_stream_callback(self, queue: Queue) -> Callable[[Optional[str]], None]:
        """Create a callback that puts chunks directly into the sync queue."""
        return lambda chunk: queue.put(chunk)

    def _get_from_stream_queue(self, queue: Queue) -> Optional[str]:
        """Get item from sync queue with polling."""
        while True:
            try:
                return queue.get(timeout=0.1)
            except Empty:
                continue

    def close(self) -> None:
        """Close the session and cleanup resources."""
        self._closed = True

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
        self._check_closed()
        self._validate_generate_params(stream, schema)

        # Apply defaults to parameters
        temp, max_tok = self._apply_defaults(temperature, max_tokens)

        if stream:
            # Streaming mode: return Iterator[StreamChunk]
            return self._generate_stream_impl(prompt, temp, max_tok)
        elif schema is not None:
            # Structured generation mode
            return self._generate_structured_impl(prompt, schema, temp, max_tok)
        else:
            # Text generation mode
            return self._generate_text_impl(prompt, temp, max_tok)

    def _generate_text_impl(
        self, prompt: str, temperature: float, max_tokens: int
    ) -> GenerationResponse:
        """Internal implementation for text generation."""
        from . import _foundationmodels

        with self._generation_context() as start_length:
            text = _foundationmodels.generate(prompt, temperature, max_tokens)
            return self._build_generation_response(text, False, start_length)

    def _generate_structured_impl(
        self,
        prompt: str,
        schema: Union[Dict[str, Any], Type["BaseModel"]],
        temperature: float,
        max_tokens: int,
    ) -> GenerationResponse:
        """Internal implementation for structured generation."""
        from . import _foundationmodels

        with self._generation_context() as start_length:
            json_schema = normalize_schema(schema)
            result = _foundationmodels.generate_structured(
                prompt, json_schema, temperature, max_tokens
            )
            return self._build_generation_response(result, True, start_length)

    def _generate_stream_impl(
        self, prompt: str, temperature: float, max_tokens: int
    ) -> Iterator[StreamChunk]:
        """Internal implementation for streaming generation."""
        start_length = self._begin_generation()
        try:
            # Create queue and callback using abstract methods
            queue = self._create_stream_queue()
            callback = self._create_stream_callback(queue)

            # Use shared streaming implementation from base class
            yield from self._stream_chunks_impl(
                prompt, temperature, max_tokens, queue, callback
            )
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
        from . import _foundationmodels

        self._check_closed()
        return _foundationmodels.get_history()

    def clear_history(self) -> None:
        """
        Clear conversation history.

        Removes all messages from the session while keeping the session active.
        """
        from . import _foundationmodels

        self._check_closed()
        _foundationmodels.clear_history()
        # Reset to current transcript length (may include persistent instructions)
        self._last_transcript_length = len(self.transcript)

    # Properties inherited from BaseSession (transcript, last_generation_transcript)
