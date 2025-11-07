# apple-foundation-models

Python bindings for Apple's FoundationModels framework - Direct access to on-device Apple Intelligence.

## Features

- **High-level Pythonic API**: Context managers, async/await, type hints
- **Structured Outputs**: JSON Schema and Pydantic model support
- **Async Streaming**: Native `async for` support for streaming responses
- **Type Safety**: Full type annotations with mypy support
- **Memory Safe**: Automatic resource cleanup, no manual memory management
- **Thread Safe**: All operations are thread-safe

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

### Basic Usage

```python
from applefoundationmodels import Client

# Create a client (library auto-initializes)
with Client() as client:
    # Check if Apple Intelligence is available
    if not client.is_ready():
        print("Apple Intelligence is not available")
        print(client.get_availability_reason())
        return

    # Create a session
    session = client.create_session(
        instructions="You are a helpful assistant.",
        enable_guardrails=True
    )

    # Generate a response
    response = session.generate("What is the capital of France?")
    print(response)

    # Get conversation history
    history = session.get_history()
    for msg in history:
        print(f"{msg['role']}: {msg['content']}")
```

### Async Streaming

```python
import asyncio
from applefoundationmodels import Client

async def main():
    with Client() as client:
        session = client.create_session()

        # Stream response chunks as they arrive
        async for chunk in session.generate_stream("Tell me a story about a robot"):
            print(chunk, end='', flush=True)
        print()  # Newline after stream

asyncio.run(main())
```

### Structured Output

```python
from applefoundationmodels import Client

with Client() as client:
    session = client.create_session()

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
    result = session.generate_structured(
        "Extract person info: Alice is 28 and lives in Paris",
        schema=schema
    )

    print(result)  # {'name': 'Alice', 'age': 28, 'city': 'Paris'}
```

#### Using Pydantic Models

You can also use Pydantic models for structured outputs (requires `pip install pydantic>=2.0`):

```python
from applefoundationmodels import Client
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    city: str

with Client() as client:
    session = client.create_session()

    # Pass Pydantic model directly - no need for JSON schema!
    result = session.generate_structured(
        "Extract person info: Alice is 28 and lives in Paris",
        schema=Person
    )

    print(result)  # {'name': 'Alice', 'age': 28, 'city': 'Paris'}

    # Parse directly into a Pydantic model for validation
    person = Person(**result)
    print(person.name, person.age, person.city)  # Alice 28 Paris
```

### Tool Calling

Tool calling allows the model to call your Python functions to access real-time data, perform actions, or integrate with external systems. Tools work with a simple decorator-based API:

```python
from applefoundationmodels import Client

with Client() as client:
    session = client.create_session()

    # Register a tool with the @session.tool decorator
    @session.tool(description="Get current weather for a location")
    def get_weather(location: str, units: str = "celsius") -> str:
        """Fetch weather information from your weather API."""
        # Your implementation here
        return f"Weather in {location}: 22°{units[0].upper()}, sunny"

    @session.tool()
    def calculate(expression: str) -> float:
        """Evaluate a mathematical expression safely."""
        # Your implementation here
        return eval(expression)  # Use safe_eval in production!

    # The model will automatically call tools when needed
    response = session.generate(
        "What's the weather in Paris and what's 15 times 23?"
    )
    print(response)
    # "The weather in Paris is 22°C and sunny. 15 times 23 equals 345."

    # View the full conversation including tool calls
    for entry in session.transcript:
        print(f"{entry['type']}: {entry.get('content', '')}")
```

**Features:**
- **Automatic schema generation** from Python type hints
- **Parallel tool execution** when the model calls multiple tools
- **Full transcript access** showing all tool calls and outputs
- **Error handling** with detailed error information
- **Type-safe** with complete type annotations

**Schema Extraction:**

The library automatically extracts JSON schemas from your Python functions:

```python
@session.tool(description="Search documentation")
def search_docs(query: str, limit: int = 10, category: str = "all") -> list:
    """Search the documentation database."""
    # Implementation...
    return results

# Automatically generates:
# {
#   "name": "search_docs",
#   "description": "Search documentation",
#   "parameters": {
#     "type": "object",
#     "properties": {
#       "query": {"type": "string"},
#       "limit": {"type": "integer"},
#       "category": {"type": "string"}
#     },
#     "required": ["query"]
#   }
# }
```

**Transcript Access:**

View the complete conversation history including tool interactions:

```python
# After generating with tools
for entry in session.transcript:
    match entry['type']:
        case 'prompt':
            print(f"User: {entry['content']}")
        case 'tool_calls':
            for call in entry['tool_calls']:
                print(f"Calling tool: {call['id']}")
        case 'tool_output':
            print(f"Tool result: {entry['content']}")
        case 'response':
            print(f"Assistant: {entry['content']}")
```

**Supported Parameter Types:**

Tool calling works with various parameter signatures:
- No parameters
- Single parameters (string, int, float, bool)
- Multiple parameters with mixed types
- Optional parameters with default values
- Lists and nested objects

See `examples/tool_calling_comprehensive.py` for complete examples of all supported patterns.

### Generation Parameters

```python
# Control generation with parameters
response = session.generate(
    "Write a creative story",
    temperature=1.5,      # Higher = more creative (0.0-2.0)
    max_tokens=500,       # Limit response length
    seed=42               # Reproducible outputs
)
```

### Session Management

```python
with Client() as client:
    # Create multiple sessions
    chat_session = client.create_session(
        instructions="You are a friendly chatbot"
    )
    code_session = client.create_session(
        instructions="You are a code review assistant"
    )

    # Each session maintains separate conversation history
    chat_response = chat_session.generate("Hello!")
    code_response = code_session.generate("Review this code: ...")

    # Clear history while keeping session
    chat_session.clear_history()

    # Manually add messages
    chat_session.add_message("system", "Be concise")
```

### Statistics

```python
with Client() as client:
    session = client.create_session()

    # Generate some responses
    for i in range(5):
        session.generate(f"Question {i}")

    # Get statistics
    stats = client.get_stats()
    print(f"Total requests: {stats['total_requests']}")
    print(f"Success rate: {stats['successful_requests'] / stats['total_requests'] * 100:.1f}%")
    print(f"Avg response time: {stats['average_response_time']:.2f}s")

    # Reset statistics
    client.reset_stats()
```

## API Reference

### Client

The main entry point for using libai.

```python
class Client:
    def __init__() -> None: ...
    def __enter__() -> Client: ...
    def __exit__(...) -> None: ...

    @staticmethod
    def check_availability() -> Availability: ...
    @staticmethod
    def get_availability_reason() -> str: ...
    @staticmethod
    def is_ready() -> bool: ...
    @staticmethod
    def get_version() -> str: ...
    @staticmethod
    def get_supported_languages() -> List[str]: ...

    def create_session(...) -> Session: ...
    def get_stats() -> Stats: ...
    def reset_stats() -> None: ...
    def close() -> None: ...
```

### Session

Manages conversation state and text generation.

```python
class Session:
    def __enter__() -> Session: ...
    def __exit__(...) -> None: ...

    def generate(prompt: str, **params) -> str: ...
    def generate_structured(prompt: str, schema: dict, **params) -> dict: ...
    async def generate_stream(prompt: str, **params) -> AsyncIterator[str]: ...

    def tool(description: str = None, name: str = None) -> Callable: ...
    @property
    def transcript() -> List[dict]: ...

    def get_history() -> List[dict]: ...
    def clear_history() -> None: ...
    def add_message(role: str, content: str) -> None: ...
    def close() -> None: ...
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
    tools_json: Optional[str]
    enable_guardrails: bool
    prewarm: bool

class GenerationParams(TypedDict):
    temperature: float
    max_tokens: int
    seed: int

class Stats(TypedDict):
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_tokens_generated: int
    average_response_time: float
    total_processing_time: float
```

### Exceptions

All exceptions inherit from `FoundationModelsError`:

- `InitializationError` - Library initialization failed
- `NotAvailableError` - Apple Intelligence not available
- `InvalidParametersError` - Invalid parameters
- `MemoryError` - Memory allocation failed
- `JSONParseError` - JSON parsing error
- `GenerationError` - Text generation failed
- `TimeoutError` - Operation timeout
- `SessionNotFoundError` - Session not found
- `StreamNotFoundError` - Stream not found
- `GuardrailViolationError` - Content blocked by safety filters
- `ToolNotFoundError` - Tool not registered
- `ToolExecutionError` - Tool execution failed
- `ToolCallError` - Tool call error (validation, schema, etc.)
- `UnknownError` - Unknown error

## Examples

See the `examples/` directory for complete working examples:

- `basic_chat.py` - Simple conversation
- `streaming_chat.py` - Async streaming
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
│   ├── client.py       # High-level Client
│   ├── session.py      # Session management
│   ├── types.py        # Type definitions
│   ├── exceptions.py   # Exception classes
│   └── swift/          # Swift FoundationModels bindings
│       ├── foundation_models.swift  # Swift implementation
│       └── foundation_models.h      # C FFI header
├── lib/                # Swift dylib and modules (auto-generated)
│   └── libfoundation_models.dylib    # Compiled Swift library
├── examples/           # Example scripts
└── tests/              # Unit tests
```

## Architecture

apple-foundation-models-py uses a layered architecture for optimal performance:

```
Python API (client.py, session.py)
         ↓
    Cython FFI (_foundationmodels.pyx)
         ↓
    C FFI Layer (foundation_models.h)
         ↓
  Swift Implementation (foundation_models.swift)
         ↓
  FoundationModels Framework (Apple Intelligence)
```

**Key Design Decisions:**

- **Direct FoundationModels Integration**: No intermediate C library - Swift calls FoundationModels directly
- **Minimal Overhead**: C FFI layer provides thin wrapper for Python/Swift communication
- **Async Coordination**: Uses semaphores to bridge Swift's async/await with synchronous C calls
- **Streaming**: Real-time delta calculation from FoundationModels snapshot-based streaming

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
3. Wait for models to download (check with `client.get_availability_reason()`)

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
