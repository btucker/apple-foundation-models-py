"""
Base Session implementation for applefoundationmodels Python bindings.

Provides shared logic for both sync and async sessions.
"""

import platform
import threading
import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager, asynccontextmanager
from functools import lru_cache
from typing import (
    Optional,
    Dict,
    Any,
    Callable,
    List,
    Union,
    cast,
    Generator,
    AsyncGenerator,
    ClassVar,
)

from .base import ContextManagedResource
from .types import (
    GenerationResponse,
    StreamChunk,
    ToolCall,
    Function,
    Availability,
)
from .constants import DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS
from .exceptions import NotAvailableError

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_foundationmodels():
    """Return the cached _foundationmodels module."""
    from . import _foundationmodels

    return _foundationmodels


class BaseSession(ContextManagedResource, ABC):
    """
    Base class for Session and AsyncSession with shared logic.

    This class contains all the common functionality between the sync
    and async session implementations to avoid duplication.
    """

    # Class-level flag to track if library has been initialized
    _initialized: ClassVar[bool] = False

    def __init__(
        self,
        instructions: Optional[str] = None,
        tools: Optional[List[Callable]] = None,
    ):
        """
        Create a base session instance.

        Args:
            instructions: Optional system instructions to guide AI behavior
            tools: Optional list of tool functions to make available to the model

        Raises:
            InitializationError: If library initialization fails
            NotAvailableError: If Apple Intelligence is not available
            RuntimeError: If platform is not supported
        """
        # Validate platform and initialize library on first session creation
        self._validate_platform()
        self._initialize_library()

        self._ffi = get_foundationmodels()
        config = self._build_session_config(instructions, tools)
        self._session_id = self._ffi.create_session(config)
        self._closed = False
        self._config = config
        # Initialize to current transcript length to exclude any initial instructions
        self._last_transcript_length = len(self.transcript)

    @abstractmethod
    def _call_ffi(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute an FFI call (sync or async depending on implementation).

        This is the adapter method that subclasses must implement to handle
        sync vs async execution of FFI calls.

        Args:
            func: The FFI function to call
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            The result from the FFI call
        """
        pass

    def _check_closed(self) -> None:
        """
        Raise error if session is closed.

        Raises:
            RuntimeError: If session has been closed
        """
        if self._closed:
            raise RuntimeError("Session is closed")

    def _apply_defaults(
        self, temperature: Optional[float], max_tokens: Optional[int]
    ) -> tuple[float, int]:
        """
        Apply default values to generation parameters.

        Args:
            temperature: Temperature value or None to use default
            max_tokens: Max tokens value or None to use default

        Returns:
            Tuple of (temperature, max_tokens) with defaults applied
        """
        return (
            temperature if temperature is not None else DEFAULT_TEMPERATURE,
            max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS,
        )

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

    @contextmanager
    def _generation_context(self) -> Generator[int, None, None]:
        """
        Context manager for synchronous generation calls.

        Handles:
        - Marking generation start/end boundaries
        - Automatic cleanup on exception
        - Transcript length tracking

        Yields:
            start_length: Transcript length at generation start

        Example:
            >>> with self._generation_context() as start_length:
            ...     text = _foundationmodels.generate(prompt, temp, max_tok)
            ...     return self._build_generation_response(text, False, start_length)
        """
        start_length = self._begin_generation()
        try:
            yield start_length
        except Exception:
            self._end_generation(start_length)
            raise

    @asynccontextmanager
    async def _async_generation_context(self) -> AsyncGenerator[int, None]:
        """
        Context manager for asynchronous generation calls.

        Handles:
        - Marking generation start/end boundaries
        - Automatic cleanup on exception
        - Transcript length tracking

        Yields:
            start_length: Transcript length at generation start

        Example:
            >>> async with self._async_generation_context() as start_length:
            ...     text = await asyncio.to_thread(fm.generate, prompt, temp, max_tok)
            ...     return self._build_generation_response(text, False, start_length)
        """
        start_length = self._begin_generation()
        try:
            yield start_length
        except Exception:
            self._end_generation(start_length)
            raise

    @abstractmethod
    def _create_stream_queue(self) -> Any:
        """
        Create a queue for streaming coordination.

        Returns:
            Queue instance (Queue for sync, asyncio.Queue for async)
        """
        pass

    @abstractmethod
    def _create_stream_callback(self, queue: Any) -> Callable[[Optional[str]], None]:
        """
        Create a callback function for streaming that puts chunks in the queue.

        Args:
            queue: The queue to put chunks into

        Returns:
            Callback function that accepts optional string chunks
        """
        pass

    @abstractmethod
    def _get_from_stream_queue(self, queue: Any) -> Optional[str]:
        """
        Get an item from the streaming queue.

        For sync sessions: polls with timeout
        For async sessions: awaits the queue

        Args:
            queue: The queue to get from

        Returns:
            The chunk from the queue, None for end of stream, or Exception
        """
        pass

    def _stream_chunks_impl(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        queue: Any,
        callback: Callable,
    ) -> Generator[StreamChunk, None, None]:
        """
        Shared streaming implementation for both sync and async sessions.

        This method contains all the common streaming logic. Subclasses provide
        queue creation and access through abstract methods.

        Args:
            prompt: The text prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            queue: The queue instance (created by _create_stream_queue)
            callback: The callback function (created by _create_stream_callback)

        Yields:
            StreamChunk objects with content deltas

        Note:
            This is a generator that yields chunks synchronously. AsyncSession
            wraps this in an async generator.
        """

        # Run streaming in a background thread
        def run_stream():
            try:
                self._ffi.generate_stream(prompt, callback, temperature, max_tokens)
            except Exception as e:
                # Put exception in queue - subclass handles queue put
                try:
                    # For sync: direct put, for async: needs run_coroutine_threadsafe
                    # We'll just put it directly and let the exception propagate
                    if hasattr(queue, "put_nowait"):
                        # asyncio.Queue
                        queue.put_nowait(e)
                    else:
                        # queue.Queue
                        queue.put(e)
                except Exception:
                    # If we can't put the exception, log it
                    logger.error(
                        f"Failed to put exception in queue: {e}", exc_info=True
                    )

        thread = threading.Thread(target=run_stream, daemon=True)
        thread.start()

        # Yield StreamChunk objects
        chunk_index = 0
        while True:
            item = self._get_from_stream_queue(queue)

            if isinstance(item, Exception):
                raise item

            if item is None:  # End of stream
                # Yield final chunk with finish_reason
                yield StreamChunk(content="", finish_reason="stop", index=chunk_index)
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
        return cast(List[Dict[str, Any]], self._ffi.get_transcript())

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

    def _validate_generate_params(
        self,
        stream: bool,
        schema: Optional[Union[Dict[str, Any], type]],
    ) -> None:
        """
        Validate generation parameters.

        Args:
            stream: Whether streaming is requested
            schema: Schema if structured output is requested

        Raises:
            ValueError: If invalid parameter combination
        """
        if stream and schema is not None:
            raise ValueError(
                "Streaming is not supported with structured output (schema parameter)"
            )

    def _mark_closed(self) -> None:
        """
        Mark the session as closed.

        This is used by both Session.close() and AsyncSession.close() to
        set the closed flag.
        """
        self._closed = True

    @staticmethod
    def _validate_platform() -> None:
        """
        Validate platform requirements for Apple Intelligence.

        Raises:
            NotAvailableError: If platform is not supported or version is insufficient
        """
        # Check platform requirements
        if platform.system() != "Darwin":
            raise NotAvailableError(
                "Apple Intelligence is only available on macOS. "
                f"Current platform: {platform.system()}"
            )

        # Check macOS version
        mac_ver = platform.mac_ver()[0]
        if mac_ver:
            try:
                major_version = int(mac_ver.split(".")[0])
                if major_version < 26:
                    raise NotAvailableError(
                        f"Apple Intelligence requires macOS 26.0 or later. "
                        f"Current version: {mac_ver}"
                    )
            except (ValueError, IndexError):
                # If we can't parse the version, let it try anyway
                pass

    @staticmethod
    def _initialize_library() -> None:
        """
        Initialize the FoundationModels library if not already initialized.

        This is called automatically on first session creation.
        """
        if not BaseSession._initialized:
            get_foundationmodels().init()
            BaseSession._initialized = True

    @staticmethod
    def _build_session_config(
        instructions: Optional[str],
        tools: Optional[List[Callable]],
    ) -> Optional[Dict[str, Any]]:
        """
        Build session configuration dictionary and register tools.

        Args:
            instructions: Optional system instructions
            tools: Optional list of tool functions to register

        Returns:
            Configuration dictionary or None if empty
        """
        # Register tools if provided
        if tools:
            from .tools import register_tool_for_function

            # Build tool dictionary with function objects
            tool_dict = {}
            for func in tools:
                schema = register_tool_for_function(func)
                tool_name = schema["name"]
                tool_dict[tool_name] = func

            # Register with FFI
            get_foundationmodels().register_tools(tool_dict)

        config = {}
        if instructions is not None:
            config["instructions"] = instructions
        return config if config else None

    @staticmethod
    def check_availability() -> Availability:
        """
        Check Apple Intelligence availability on this device.

        This is a static method that can be called without creating a session.

        Returns:
            Availability status enum value

        Example:
            >>> from applefoundationmodels import Session, Availability
            >>> status = Session.check_availability()
            >>> if status == Availability.AVAILABLE:
            ...     print("Apple Intelligence is available!")
        """
        return Availability(get_foundationmodels().check_availability())

    @staticmethod
    def get_availability_reason() -> Optional[str]:
        """
        Get detailed availability status message.

        Returns:
            Detailed status description with actionable guidance,
            or None if library not initialized
        """
        return get_foundationmodels().get_availability_reason()

    @staticmethod
    def is_ready() -> bool:
        """
        Check if Apple Intelligence is ready for immediate use.

        Returns:
            True if ready for use, False otherwise
        """
        return get_foundationmodels().is_ready()

    @staticmethod
    def get_version() -> str:
        """
        Get library version string.

        Returns:
            Version string in format "major.minor.patch"
        """
        return get_foundationmodels().get_version()
