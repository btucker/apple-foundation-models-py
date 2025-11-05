"""
Session API for libai Python bindings.

Provides session management, text generation, and async streaming support.
"""

import asyncio
import json
from typing import Optional, Dict, Any, AsyncIterator, Callable
from queue import Queue, Empty
import threading

from . import _libai
from .types import GenerationParams


class Session:
    """
    AI session for maintaining conversation state.

    Sessions maintain conversation history and can be configured with tools
    and instructions. Use as a context manager for automatic cleanup.

    Usage:
        with client.create_session() as session:
            response = session.generate("Hello!")
            print(response)
    """

    def __init__(self, context: _libai.Context, session_id: int):
        """
        Create a Session instance.

        Note: Users should create sessions via Client.create_session()
        rather than calling this constructor directly.

        Args:
            context: The underlying Cython context
            session_id: The session ID from the C library
        """
        self._context = context
        self._session_id = session_id
        self._closed = False

    def __enter__(self) -> "Session":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit with automatic cleanup."""
        self.close()

    def close(self) -> None:
        """
        Close the session and release resources.

        Destroys the session and cancels any active streams.
        """
        if not self._closed:
            self._context.destroy_session(self._session_id)
            self._closed = True

    def _check_not_closed(self) -> None:
        """Raise exception if session is closed."""
        if self._closed:
            raise RuntimeError("Session is closed")

    @property
    def id(self) -> int:
        """Get the session ID."""
        return self._session_id

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None
    ) -> str:
        """
        Generate a text response from a prompt (synchronous).

        Sends a prompt to the AI and waits for the complete response.
        This function blocks until generation is complete.

        Args:
            prompt: Input text prompt
            temperature: Generation randomness (0.0-2.0, default: system default)
            max_tokens: Maximum response tokens (default: system default)
            seed: Random seed for reproducibility (default: random)

        Returns:
            Generated response text

        Raises:
            Various LibAIError subclasses on failure

        Example:
            >>> response = session.generate(
            ...     "What is the capital of France?",
            ...     temperature=0.7
            ... )
            >>> print(response)
        """
        self._check_not_closed()

        params = {}
        if temperature is not None:
            params['temperature'] = temperature
        if max_tokens is not None:
            params['max_tokens'] = max_tokens
        if seed is not None:
            params['seed'] = seed

        return self._context.generate_response(
            self._session_id, prompt, params if params else None
        )

    def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate a structured response conforming to a JSON schema.

        Args:
            prompt: Input text prompt
            schema: JSON schema dictionary defining expected response structure
            temperature: Generation randomness (0.0-2.0, default: system default)
            max_tokens: Maximum response tokens (default: system default)
            seed: Random seed for reproducibility (default: random)

        Returns:
            Dictionary with 'text' (string) and 'object' (structured data) fields

        Raises:
            Various LibAIError subclasses on failure

        Example:
            >>> schema = {
            ...     "type": "object",
            ...     "properties": {
            ...         "name": {"type": "string"},
            ...         "age": {"type": "integer"}
            ...     }
            ... }
            >>> result = session.generate_structured(
            ...     "Extract person info: John is 30 years old",
            ...     schema
            ... )
            >>> print(result['object'])  # {'name': 'John', 'age': 30}
        """
        self._check_not_closed()

        params = {}
        if temperature is not None:
            params['temperature'] = temperature
        if max_tokens is not None:
            params['max_tokens'] = max_tokens
        if seed is not None:
            params['seed'] = seed

        schema_json = json.dumps(schema)
        result_json = self._context.generate_structured_response(
            self._session_id, prompt, schema_json, params if params else None
        )

        return json.loads(result_json)

    async def generate_stream(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None
    ) -> AsyncIterator[str]:
        """
        Generate a response with async streaming.

        Yields chunks of text as they become available. Uses asyncio for
        async iteration.

        Args:
            prompt: Input text prompt
            temperature: Generation randomness (0.0-2.0, default: system default)
            max_tokens: Maximum response tokens (default: system default)
            seed: Random seed for reproducibility (default: random)

        Yields:
            Text chunks as they are generated

        Raises:
            Various LibAIError subclasses on failure

        Example:
            >>> async for chunk in session.generate_stream("Tell me a story"):
            ...     print(chunk, end='', flush=True)
        """
        self._check_not_closed()

        params = {}
        if temperature is not None:
            params['temperature'] = temperature
        if max_tokens is not None:
            params['max_tokens'] = max_tokens
        if seed is not None:
            params['seed'] = seed

        # Create a queue to bridge C callbacks to asyncio
        queue: Queue = Queue()
        done_event = threading.Event()
        error_container = [None]  # Container to hold exceptions

        def callback(chunk: Optional[str]) -> None:
            """Callback invoked from C thread."""
            if chunk is None:
                # End of stream
                done_event.set()
            elif chunk.startswith("Error:"):
                # Error occurred
                error_container[0] = Exception(chunk)
                done_event.set()
            else:
                # Normal chunk
                queue.put(chunk)

        # Start streaming
        params_dict = params if params else None
        stream_handle = self._context.generate_response_stream(
            self._session_id, prompt, callback, params_dict
        )

        try:
            # Yield chunks as they arrive
            while not done_event.is_set():
                try:
                    # Non-blocking check for chunks
                    chunk = queue.get(timeout=0.01)
                    yield chunk
                except Empty:
                    # Allow asyncio to switch tasks
                    await asyncio.sleep(0)

            # Drain any remaining chunks
            while not queue.empty():
                yield queue.get_nowait()

            # Check for errors
            if error_container[0] is not None:
                raise error_container[0]

        except Exception:
            # Cancel stream on exception
            stream_handle.cancel()
            raise

    def get_history(self) -> list:
        """
        Get the conversation history for this session.

        Returns:
            List of message dictionaries with 'role' and 'content' fields

        Example:
            >>> history = session.get_history()
            >>> for msg in history:
            ...     print(f"{msg['role']}: {msg['content']}")
        """
        self._check_not_closed()
        history_json = self._context.get_session_history(self._session_id)
        if history_json is None:
            return []
        return json.loads(history_json)

    def clear_history(self) -> None:
        """
        Clear the conversation history for this session.

        Removes all messages while preserving session configuration
        and registered tools.
        """
        self._check_not_closed()
        self._context.clear_session_history(self._session_id)

    def add_message(self, role: str, content: str) -> None:
        """
        Manually add a message to the conversation history.

        Args:
            role: Message role ("user", "assistant", "system", or "tool")
            content: Message content text

        Example:
            >>> session.add_message("system", "You are a helpful assistant")
            >>> session.add_message("user", "Hello!")
        """
        self._check_not_closed()
        self._context.add_message_to_history(self._session_id, role, content)
