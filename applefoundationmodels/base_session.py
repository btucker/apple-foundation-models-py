"""
Base Session implementation for applefoundationmodels Python bindings.

Provides shared logic for both sync and async sessions.
"""

from abc import ABC
from typing import Optional, Dict, Any, Callable, List, Union, cast

from . import _foundationmodels
from .base import ContextManagedResource
from .types import (
    NormalizedGenerationParams,
    GenerationResponse,
    ToolCall,
    Function,
)


class BaseSession(ContextManagedResource, ABC):
    """
    Base class for Session and AsyncSession with shared logic.

    This class contains all the common functionality between the sync
    and async session implementations to avoid duplication.
    """

    def __init__(self, session_id: int, config: Optional[Dict[str, Any]] = None):
        """
        Create a base session instance.

        Args:
            session_id: The session ID from the FFI layer
            config: Optional session configuration
        """
        self._session_id = session_id
        self._closed = False
        self._config = config
        # Initialize to current transcript length to exclude any initial instructions
        self._last_transcript_length = len(self.transcript)

    def _check_closed(self) -> None:
        """
        Raise error if session is closed.

        Raises:
            RuntimeError: If session has been closed
        """
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

    def _build_generation_response(
        self,
        content: Union[str, Dict[str, Any]],
        is_structured: bool,
        start_length: int,
    ) -> GenerationResponse:
        """
        Build a GenerationResponse with tool call extraction.

        This method centralizes the response building logic to avoid duplication
        between text and structured generation.

        Args:
            content: The generated content (str for text, dict for structured)
            is_structured: Whether this is structured output
            start_length: The transcript length at generation start

        Returns:
            GenerationResponse with tool_calls and finish_reason populated
        """
        # Update the generation boundary marker
        self._end_generation(start_length)

        # Structured generation does not support tool calls
        if is_structured:
            return GenerationResponse(
                content=content,
                is_structured=True,
                tool_calls=None,
                finish_reason="stop",
            )

        # Extract tool calls from the generation transcript
        tool_calls = self._extract_tool_calls_from_transcript(
            self.last_generation_transcript
        )

        # Set finish reason based on whether tools were called
        finish_reason = "tool_calls" if tool_calls else "stop"

        return GenerationResponse(
            content=content,
            is_structured=False,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )

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
        this returns only the entries added during the last generation call
        (generate(), generate_structured(), or generate_stream()).

        This is useful when you need to inspect what happened during a specific
        generation without worrying about accumulated history from previous calls.

        Returns:
            List of transcript entries from the last generate() call.
            Returns empty list if no generation has been performed yet.

        Example:
            >>> # First generation
            >>> response1 = session.generate("What is 2 + 2?")
            >>> entries1 = session.last_generation_transcript
            >>> print(f"First call: {len(entries1)} entries")

            >>> # Second generation on same session
            >>> response2 = session.generate("What is 5 + 7?")
            >>> entries2 = session.last_generation_transcript
            >>> print(f"Second call: {len(entries2)} entries (only from second call)")
        """
        self._check_closed()
        full_transcript = self.transcript
        return full_transcript[self._last_transcript_length :]
