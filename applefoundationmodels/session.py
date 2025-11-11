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

from . import _foundationmodels
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

    def close(self) -> None:
        """Close the session and cleanup resources."""
        self._closed = True

    # Type overloads for non-streaming text generation
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
        temp = self._get_temperature(temperature)
        max_tok = self._get_max_tokens(max_tokens)

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
        start_length = self._begin_generation()
        try:
            text = _foundationmodels.generate(prompt, temperature, max_tokens)
            return self._build_generation_response(text, False, start_length)
        except Exception:
            self._end_generation(start_length)
            raise

    def _generate_structured_impl(
        self,
        prompt: str,
        schema: Union[Dict[str, Any], Type["BaseModel"]],
        temperature: float,
        max_tokens: int,
    ) -> GenerationResponse:
        """Internal implementation for structured generation."""
        start_length = self._begin_generation()
        try:
            json_schema = normalize_schema(schema)
            result = _foundationmodels.generate_structured(
                prompt, json_schema, temperature, max_tokens
            )
            return self._build_generation_response(result, True, start_length)
        except Exception:
            self._end_generation(start_length)
            raise

    def _generate_stream_impl(
        self, prompt: str, temperature: float, max_tokens: int
    ) -> Iterator[StreamChunk]:
        """Internal implementation for streaming generation."""
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
                        prompt, callback, temperature, max_tokens
                    )
                except Exception as e:
                    queue.put(e)

            thread = threading.Thread(target=run_stream, daemon=True)
            thread.start()

            # Yield StreamChunk objects
            chunk_index = 0
            while True:
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
        # Reset to current transcript length (may include persistent instructions)
        self._last_transcript_length = len(self.transcript)

    # Properties inherited from BaseSession (transcript, last_generation_transcript)
