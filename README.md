# apple-foundation-models

Unofficial Python bindings for Apple's [Foundation Models framework](https://developer.apple.com/documentation/FoundationModels) - Direct access to the on-device LLM available in macOS 26+.

## Features

- **Async Support**: Full async/await support with `AsyncSession`
- **Tool Calling**: Register Python functions as tools for the model to call
- **Structured Outputs**: JSON Schema and Pydantic model support
- **Streaming**: Real-time token-by-token response generation

## Limitations

- **Context Window**: 4,096 tokens per session (includes instructions, prompts, and outputs)

## Requirements

- macOS 26.0+ (macOS Sequoia or later)
- Python 3.9 or higher
- Apple Intelligence enabled on your device

## Installation

### From PyPI

```bash
pip install apple-foundation-models
```

**Optional dependencies:**

```bash
# For Pydantic model support in structured outputs
pip install apple-foundation-models[pydantic]
```

### From Source

```bash
# Clone the repository
git clone https://github.com/btucker/apple-foundation-models-py.git
cd apple-foundation-models-py

# Install (automatically builds Swift dylib and Cython extension)
pip install -e .
```

**Requirements:**

- macOS 26.0+ (Sequoia) with Apple Intelligence enabled
- Xcode command line tools (`xcode-select --install`)
- Python 3.9 or higher

**Note:** The Swift dylib is built automatically during installation.

## Quick Start

### Check Availability

```python
from applefoundationmodels import apple_intelligence_available

if apple_intelligence_available():
    print("Apple Intelligence is ready!")
else:
    print("Apple Intelligence is not available")
```

### Basic Usage

```python
from applefoundationmodels import Session

with Session(instructions="You are a helpful assistant.") as session:
    response = session.generate("What is the capital of France?")
    print(response.text)
```

### Streaming

```python
# Sync streaming
with Session() as session:
    for chunk in session.generate("Tell me a story", stream=True):
        print(chunk.content, end='', flush=True)
```

### Async/Await

```python
import asyncio
from applefoundationmodels import AsyncSession

async def main():
    async with AsyncSession() as session:
        # Use await for non-streaming
        response = await session.generate("What is 2 + 2?")
        print(response.text)

        # Use async for with streaming
        async for chunk in session.generate("Tell a story", stream=True):
            print(chunk.content, end='', flush=True)

asyncio.run(main())
```

### Structured Output

```python
from applefoundationmodels import Session

with Session() as session:
    # Define a JSON schema
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "city": {"type": "string"}
        },
        "required": ["name", "age", "city"]
    }

    # Generate structured response
    response = session.generate(
        "Extract person info: Alice is 28 and lives in Paris",
        schema=schema
    )

    print(response.parsed)  # {'name': 'Alice', 'age': 28, 'city': 'Paris'}
```

#### Using Pydantic Models

You can also use Pydantic models for structured outputs (requires `pip install pydantic>=2.0`):

```python
from applefoundationmodels import Session
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    city: str

with Session() as session:
    # Pass Pydantic model directly - no need for JSON schema!
    response = session.generate(
        "Extract person info: Alice is 28 and lives in Paris",
        schema=Person
    )

    print(response.parsed)  # {'name': 'Alice', 'age': 28, 'city': 'Paris'}

    # Parse directly into a Pydantic model for validation
    person = Person(**response.parsed)
    # Or use the convenience method:
    person = response.parse_as(Person)
    print(person.name, person.age, person.city)  # Alice 28 Paris
```

### Tool Calling

Tool calling allows the model to call your Python functions to access real-time data, perform actions, or integrate with external systems. Tools are registered when you create a session and remain available for all generate() calls on that session:

```python
from applefoundationmodels import Session

# Define tools as regular Python functions with docstrings
def get_weather(location: str, units: str = "celsius") -> str:
    """Get current weather for a location."""
    # Your implementation here
    return f"Weather in {location}: 22°{units[0].upper()}, sunny"

def calculate(expression: str) -> float:
    """Evaluate a mathematical expression safely."""
    # Your implementation here
    return eval(expression)  # Use safe_eval in production!

# Register tools at session creation - they'll be available for all generate() calls
with Session(tools=[get_weather, calculate]) as session:
    # The model will automatically call tools when needed
    response = session.generate("What's the weather in Paris and what's 15 times 23?")
    print(response.text)
    # "The weather in Paris is 22°C and sunny. 15 times 23 equals 345."

    # Check if tools were called using the tool_calls property
    if response.tool_calls:
        print(f"Tools called: {len(response.tool_calls)}")
        for tool_call in response.tool_calls:
            print(f"  - {tool_call.function.name}")
            print(f"    ID: {tool_call.id}")
            print(f"    Args: {tool_call.function.arguments}")

    # Check why generation stopped
    print(f"Finish reason: {response.finish_reason}")
    # "tool_calls" if tools were called, "stop" otherwise
```

See `examples/tool_calling_comprehensive.py` for complete examples of all supported patterns.

### Generation Parameters

```python
from applefoundationmodels import Session

with Session() as session:
    # Control generation with parameters
    response = session.generate(
        "Write a creative story",
        temperature=1.5,      # Higher = more creative (0.0-2.0)
        max_tokens=500        # Limit response length
    )
    print(response.text)
```

### Session Management

```python
from applefoundationmodels import Session

# Create multiple sessions
chat_session = Session(
    instructions="You are a friendly chatbot"
)
code_session = Session(
    instructions="You are a code review assistant"
)

# Each session maintains separate conversation history
chat_response = chat_session.generate("Hello!")
print(chat_response.text)

code_response = code_session.generate("Review this code: ...")
print(code_response.text)

# Clear history while keeping session
chat_session.clear_history()

# Clean up
chat_session.close()
code_session.close()
```

## API Reference

### Sync vs Async Pattern

We provide separate session classes for sync and async:

- **`Session`**: Synchronous operations with context manager support
- **`AsyncSession`**: Async operations with async context manager support

Both have identical method signatures - just use `await` with `AsyncSession`.

### Convenience Functions

```python
def apple_intelligence_available() -> bool:
    """Check if Apple Intelligence is available and ready for use."""
    ...
```

### Session

Manages conversation state and text generation (synchronous).

```python
class Session:
    def __init__(
        instructions: Optional[str] = None,
        tools: Optional[List[Callable]] = None
    ) -> None: ...

    def __enter__() -> Session: ...
    def __exit__(...) -> None: ...

    # Static utility methods
    @staticmethod
    def check_availability() -> Availability: ...
    @staticmethod
    def get_availability_reason() -> str: ...
    @staticmethod
    def is_ready() -> bool: ...
    @staticmethod
    def get_version() -> str: ...

    # Unified generation method
    def generate(
        prompt: str,
        schema: Optional[Union[dict, Type[BaseModel]]] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Union[GenerationResponse, Iterator[StreamChunk]]: ...

    @property
    def transcript() -> List[dict]: ...
    @property
    def last_generation_transcript() -> List[dict]: ...

    def get_history() -> List[dict]: ...
    def clear_history() -> None: ...
    def close() -> None: ...
```

### AsyncSession

Manages conversation state and text generation (asynchronous).

```python
class AsyncSession:
    def __init__(
        instructions: Optional[str] = None,
        tools: Optional[List[Callable]] = None
    ) -> None: ...

    async def __aenter__() -> AsyncSession: ...
    async def __aexit__(...) -> None: ...

    # Static utility methods (inherited from Session)
    @staticmethod
    def check_availability() -> Availability: ...
    @staticmethod
    def get_availability_reason() -> str: ...
    @staticmethod
    def is_ready() -> bool: ...
    @staticmethod
    def get_version() -> str: ...

    # Unified async generation method
    def generate(
        prompt: str,
        schema: Optional[Union[dict, Type[BaseModel]]] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Union[Coroutine[GenerationResponse], AsyncIterator[StreamChunk]]: ...

    @property
    def transcript() -> List[dict]: ...
    @property
    def last_generation_transcript() -> List[dict]: ...

    async def get_history() -> List[dict]: ...
    async def clear_history() -> None: ...
    async def close() -> None: ...
```

### Response Types

```python
@dataclass
class GenerationResponse:
    """Response from non-streaming generation."""
    content: Union[str, Dict[str, Any]]
    is_structured: bool

    @property
    def text() -> str: ...  # For text responses
    @property
    def parsed() -> Dict[str, Any]: ...  # For structured responses
    def parse_as(model: Type[BaseModel]) -> BaseModel: ...  # Parse into Pydantic

@dataclass
class StreamChunk:
    """Chunk from streaming generation."""
    content: str  # Text delta
    finish_reason: Optional[str] = None
    index: int = 0
```

### Types

```python
class Availability(IntEnum):
    AVAILABLE = 1
    DEVICE_NOT_ELIGIBLE = -1
    NOT_ENABLED = -2
    MODEL_NOT_READY = -3

class SessionConfig(TypedDict):
    instructions: Optional[str]

class GenerationParams(TypedDict):
    temperature: float
    max_tokens: int
```

### Exceptions

All exceptions inherit from `FoundationModelsError`:

**System Errors:**

- `InitializationError` - Library initialization failed
- `NotAvailableError` - Apple Intelligence not available
- `InvalidParametersError` - Invalid parameters
- `MemoryError` - Memory allocation failed
- `JSONParseError` - JSON parsing error
- `TimeoutError` - Operation timeout
- `SessionNotFoundError` - Session not found
- `StreamNotFoundError` - Stream not found
- `UnknownError` - Unknown error

**Generation Errors** (all inherit from `GenerationError`):

- `GenerationError` - Generic text generation error
- `ContextWindowExceededError` - Context window limit exceeded (4096 tokens)
- `AssetsUnavailableError` - Required model assets are unavailable
- `DecodingFailureError` - Failed to deserialize model output
- `GuardrailViolationError` - Content blocked by safety filters
- `RateLimitedError` - Session has been rate limited
- `RefusalError` - Session refused the request
- `ConcurrentRequestsError` - Multiple concurrent requests to same session
- `UnsupportedGuideError` - Unsupported generation guide pattern
- `UnsupportedLanguageError` - Unsupported language or locale

**Tool Errors:**

- `ToolNotFoundError` - Tool not registered
- `ToolExecutionError` - Tool execution failed
- `ToolCallError` - Tool call error (validation, schema, etc.)

## Examples

See the `examples/` directory for complete working examples:

- `basic_chat.py` - Simple synchronous conversation
- `basic_async_chat.py` - Async conversation with await (no streaming)
- `streaming_chat.py` - Async streaming responses
- `structured_output.py` - JSON schema validation
- `tool_calling_comprehensive.py` - Complete tool calling demonstration with all parameter types

## Development

### Building from Source

This project uses [uv](https://docs.astral.sh/uv/) for fast, reliable builds and dependency management:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install development dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Type checking
uv run mypy applefoundationmodels

# Format code
uv run black applefoundationmodels examples

# Build wheels
uv build --wheel
```

You can also use pip if preferred:

```bash
pip install -e ".[dev]"
pytest
mypy applefoundationmodels
black applefoundationmodels examples
```

### Project Structure

```
apple-foundation-models-py/
├── applefoundationmodels/   # Python package
│   ├── __init__.py     # Public API
│   ├── _foundationmodels.pyx  # Cython bindings
│   ├── _foundationmodels.pxd  # C declarations
│   ├── _foundationmodels.pyi  # Type stubs
│   ├── base.py         # Base context manager classes
│   ├── base_session.py # Shared session logic
│   ├── session.py      # Synchronous session
│   ├── async_session.py # Asynchronous session
│   ├── types.py        # Type definitions
│   ├── constants.py    # Constants and defaults
│   ├── tools.py        # Tool calling support
│   ├── pydantic_compat.py # Pydantic compatibility layer
│   ├── exceptions.py   # Exception classes
│   └── swift/          # Swift FoundationModels bindings
│       ├── foundation_models.swift  # Swift implementation
│       └── foundation_models.h      # C FFI header
├── lib/                # Swift dylib and modules (auto-generated)
│   ├── libfoundation_models.dylib    # Compiled Swift library
│   └── foundation_models.swiftmodule # Swift module
├── examples/           # Example scripts
│   └── utils.py        # Shared utilities
└── tests/              # Unit tests
```

## Architecture

apple-foundation-models-py uses a layered architecture for optimal performance:

```
Python API (session.py, async_session.py)
         ↓
    Cython FFI (_foundationmodels.pyx)
         ↓
    C FFI Layer (foundation_models.h)
         ↓
  Swift Implementation (foundation_models.swift)
         ↓
  FoundationModels Framework (Apple Intelligence)
```

## Performance

- Cython-compiled for near-C performance
- Direct Swift → FoundationModels calls (no intermediate libraries)
- Async streaming with delta-based chunk delivery
- No GIL during Swift library calls (when possible)

## Troubleshooting

### Apple Intelligence not available

If you get `NotAvailableError`:

1. Ensure you're running macOS 26.0 (Sequoia) or later
2. Check System Settings → Apple Intelligence → Enable
3. Wait for models to download (check with `Session.get_availability_reason()`)

### Import errors

If you get import errors after installation:

```bash
# Rebuild everything (Swift dylib + Cython extension)
pip install --force-reinstall --no-cache-dir -e .
```

### Compilation errors

Ensure you have Xcode command line tools:

```bash
xcode-select --install
```

If the Swift build fails during installation:

1. Verify macOS version: `sw_vers -productVersion` (should be 26.0+)
2. Check Swift compiler: `swiftc --version`
3. Clean and reinstall: `pip install --force-reinstall --no-cache-dir -e .`

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Links

- [FoundationModels Framework](https://developer.apple.com/documentation/FoundationModels)
- [Apple Intelligence Documentation](https://developer.apple.com/apple-intelligence/)
- [Issue Tracker](https://github.com/btucker/apple-foundation-models-py/issues)

## Acknowledgments

This project was inspired by and learned from several excellent works:

- **[libai](https://github.com/6over3/libai)** by 6over3 Institute - The original C library wrapper for FoundationModels that demonstrated the possibility of non-Objective-C access to Apple Intelligence. While we ultimately chose a direct Swift integration approach, the libai project's API design and documentation heavily influenced our Python API structure.

- **[apple-on-device-ai](https://github.com/Meridius-Labs/apple-on-device-ai)** by Meridius Labs - The Node.js bindings that showed the path to direct FoundationModels integration via Swift. Their architecture of using Swift → C FFI → JavaScript inspired our Swift → C FFI → Cython → Python approach, and their code examples were invaluable for understanding the FoundationModels API.
