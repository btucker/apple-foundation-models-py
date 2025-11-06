"""
Session API for applefoundationmodels Python bindings.

Provides session management, text generation, and async streaming support.
"""

import asyncio
import json
from typing import Optional, Dict, Any, AsyncIterator, Callable
from queue import Queue, Empty
import threading

from . import _foundationmodels
from .base import ContextManagedResource
from .constants import DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS
from .types import GenerationParams


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
        self,
        temperature: Optional[float],
        max_tokens: Optional[int]
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
        seed: Optional[int] = None
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
        schema: Dict[str, Any],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate structured JSON output matching a schema.

        Note: Not yet implemented in simplified API.

        Args:
            prompt: Input text prompt
            schema: JSON schema the output must conform to
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate

        Returns:
            Dictionary with 'object' key containing parsed JSON

        Raises:
            NotImplementedError: Feature not yet implemented
        """
        raise NotImplementedError("Structured generation not yet implemented in Swift API")

    async def generate_stream(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        include_reasoning: Optional[bool] = None,
        seed: Optional[int] = None
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
