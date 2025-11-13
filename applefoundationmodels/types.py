"""
Type definitions for libai Python bindings.

This module provides TypedDicts, enums, and type aliases for type-safe
interaction with the library.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import (
    TypedDict,
    Optional,
    Callable,
    Any,
    Union,
    Dict,
    List,
    Type,
    TYPE_CHECKING,
    cast,
)
from typing_extensions import NotRequired

if TYPE_CHECKING:
    from pydantic import BaseModel


class Result(IntEnum):
    """
    Result codes for AI operations.

    These codes indicate the success or failure state of library operations.
    """

    SUCCESS = 0
    INIT_FAILED = -1
    NOT_AVAILABLE = -2
    INVALID_PARAMS = -3
    MEMORY = -4
    JSON_PARSE = -5
    GENERATION = -6
    TIMEOUT = -7
    SESSION_NOT_FOUND = -8
    STREAM_NOT_FOUND = -9
    GUARDRAIL_VIOLATION = -10
    TOOL_NOT_FOUND = -11
    TOOL_EXECUTION = -12
    BUFFER_TOO_SMALL = -13
    UNKNOWN = -99


class Availability(IntEnum):
    """
    Apple Intelligence availability status.

    Indicates whether Apple Intelligence is available and ready for use
    on the current device and system configuration.
    """

    AVAILABLE = 1
    DEVICE_NOT_ELIGIBLE = -1
    NOT_ENABLED = -2
    MODEL_NOT_READY = -3
    AVAILABILITY_UNKNOWN = -99


class SessionConfig(TypedDict, total=False):
    """
    Session configuration options.

    Configuration for creating an AI session. Sessions maintain conversation
    state and can be configured with tools and instructions.

    Attributes:
        instructions: Optional system instructions to guide AI behavior
    """

    instructions: NotRequired[Optional[str]]


class GenerationParams(TypedDict, total=False):
    """
    Text generation parameters.

    Controls various aspects of AI text generation including randomness
    and length limits.

    Attributes:
        temperature: Generation randomness (0.0 = deterministic, 2.0 = very random)
        max_tokens: Maximum response tokens (0 = use system default)
    """

    temperature: NotRequired[float]
    max_tokens: NotRequired[int]


@dataclass
class Function:
    """
    Function call information from a tool call.

    Represents the function that was called, including its name and arguments.
    Follows OpenAI's pattern for tool call representation.

    Attributes:
        name: The name of the function that was called
        arguments: JSON string containing the function arguments

    Example:
        >>> func = Function(name="get_weather", arguments='{"location": "Paris"}')
        >>> print(func.name)
        get_weather
    """

    name: str
    arguments: str  # JSON string of arguments


@dataclass
class ToolCall:
    """
    A tool call made during generation.

    Represents a single tool/function call that occurred during text generation.
    Follows OpenAI's pattern where tool calls are exposed directly on the response.

    Attributes:
        id: Unique identifier for this tool call
        type: Type of tool call (currently only "function" is supported)
        function: The function call details (name and arguments)

    Example:
        >>> tool_call = ToolCall(
        ...     id="call_123",
        ...     type="function",
        ...     function=Function(name="get_weather", arguments='{"location": "Paris"}')
        ... )
        >>> print(tool_call.function.name)
        get_weather
    """

    id: str
    type: str  # "function" - matches OpenAI's pattern
    function: Function


@dataclass
class GenerationResponse:
    """
    Response from non-streaming generation.

    Provides a unified interface for both text and structured generation results.
    Use the .text property for text responses and .parsed for structured outputs.

    Attributes:
        content: The generated content (str for text, dict for structured)
        is_structured: True if response is structured JSON, False for text
        tool_calls: List of tool calls made during generation (None if no tools called)
        finish_reason: Reason generation stopped ("stop", "tool_calls", "length", etc.)
        metadata: Optional metadata about the generation

    Example (text):
        >>> response = session.generate("Hello")
        >>> print(response.text)

    Example (structured):
        >>> response = session.generate("Extract name and age", schema={...})
        >>> data = response.parsed
        >>> person = Person(**data)  # Parse into Pydantic model

    Example (tool calls):
        >>> response = session.generate("What's the weather in Paris?")
        >>> if response.tool_calls:
        ...     for tool_call in response.tool_calls:
        ...         print(f"Called {tool_call.function.name}")
    """

    content: Union[str, Dict[str, Any]]
    is_structured: bool
    tool_calls: Optional[List[ToolCall]] = None
    finish_reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    @property
    def text(self) -> str:
        """
        Get response as text.

        Returns:
            The generated text

        Raises:
            ValueError: If response is structured (use .parsed instead)
        """
        if self.is_structured:
            raise ValueError(
                "Response is structured output. Use .parsed property instead of .text"
            )
        return cast(str, self.content)

    @property
    def parsed(self) -> Dict[str, Any]:
        """
        Get response as structured data.

        Returns:
            The parsed JSON dictionary

        Raises:
            ValueError: If response is text (use .text instead)
        """
        if not self.is_structured:
            raise ValueError(
                "Response is text output. Use .text property instead of .parsed"
            )
        return cast(Dict[str, Any], self.content)

    def parse_as(self, model: "Type[BaseModel]") -> "BaseModel":
        """
        Parse structured response into a Pydantic model.

        Args:
            model: Pydantic BaseModel class to parse into

        Returns:
            Instantiated Pydantic model

        Raises:
            ValueError: If response is not structured
            ImportError: If pydantic is not installed

        Example:
            >>> from pydantic import BaseModel
            >>> class Person(BaseModel):
            ...     name: str
            ...     age: int
            >>> response = session.generate("Extract: Alice is 28", schema=Person)
            >>> person = response.parse_as(Person)
            >>> print(person.name, person.age)
        """
        return model(**self.parsed)


@dataclass
class StreamChunk:
    """
    A chunk from streaming generation.

    Represents a single delta in the streaming response. Multiple chunks
    combine to form the complete response.

    Attributes:
        content: The text content delta for this chunk
        finish_reason: Reason streaming ended (None for intermediate chunks)
        index: Chunk sequence index (usually 0 for single-stream responses)

    Example:
        >>> for chunk in session.generate("Tell a story", stream=True):
        ...     print(chunk.content, end='', flush=True)
        ...     if chunk.finish_reason:
        ...         print(f"\\n[Finished: {chunk.finish_reason}]")
    """

    content: str
    finish_reason: Optional[str] = None
    index: int = 0


# Callback type aliases
StreamCallback = Callable[[Optional[str]], None]
"""
Callback function for streaming text generation.

Called incrementally during streaming generation for each token or chunk.
None indicates completion or error.
"""

ToolCallback = Callable[[dict], Any]
"""
Callback function for tool execution.

Receives tool parameters as a dict and should return the tool result.
The result will be automatically JSON-serialized.
"""
