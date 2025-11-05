# libai-py

Python bindings for [libai](https://github.com/6over3/libai) - Apple Intelligence C library for accessing on-device Foundation models.

## Features

- **High-level Pythonic API**: Context managers, async/await, type hints
- **Async Streaming**: Native `async for` support for streaming responses
- **Type Safety**: Full type annotations with mypy support
- **Memory Safe**: Automatic resource cleanup, no manual memory management
- **Thread Safe**: All operations are thread-safe
- **Zero Dependencies**: Only requires Python 3.8+ and typing-extensions

## Requirements

- macOS 26.0+ (macOS Sequoia or later)
- Python 3.8 or higher
- Apple Intelligence enabled on your device
- C compiler (for installation from source)

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/6over3/libai-py.git
cd libai-py

# Install in development mode
pip install -e .
```

### From PyPI (when available)

```bash
pip install libai-py
```

## Quick Start

### Basic Usage

```python
from libai import Client

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
from libai import Client

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
from libai import Client

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

    print(result['object'])  # {'name': 'Alice', 'age': 28, 'city': 'Paris'}
```

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

All exceptions inherit from `LibAIError`:

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
- `UnknownError` - Unknown error

## Examples

See the `examples/` directory for complete working examples:

- `basic_chat.py` - Simple conversation
- `streaming_chat.py` - Async streaming
- `tool_calling.py` - Tool registration (coming soon)
- `structured_output.py` - JSON schema validation

## Development

### Building from Source

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy libai

# Format code
black libai examples
```

### Project Structure

```
libai-py/
├── libai/              # Python package
│   ├── __init__.py     # Public API
│   ├── _libai.pyx      # Cython bindings
│   ├── _libai.pxd      # C declarations
│   ├── client.py       # High-level Client
│   ├── session.py      # Session management
│   ├── types.py        # Type definitions
│   └── exceptions.py   # Exception classes
├── lib/                # Bundled libai.a
├── include/            # C headers
├── examples/           # Example scripts
└── tests/              # Unit tests
```

## Performance

- Cython-compiled for near-C performance
- Statically linked with libai for zero-overhead calls
- Async streaming with minimal latency
- No GIL during C library calls (when possible)

## Troubleshooting

### Apple Intelligence not available

If you get `NotAvailableError`:

1. Ensure you're running macOS 26.0 (Sequoia) or later
2. Check System Settings → Apple Intelligence → Enable
3. Wait for models to download (check with `client.get_availability_reason()`)

### Import errors

If you get import errors after installation:

```bash
# Rebuild the extension
pip install --force-reinstall --no-cache-dir -e .
```

### Compilation errors

Ensure you have Xcode command line tools:

```bash
xcode-select --install
```

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

- [libai C library](https://github.com/6over3/libai)
- [Apple Intelligence Documentation](https://developer.apple.com/apple-intelligence/)
- [Issue Tracker](https://github.com/6over3/libai-py/issues)

## Acknowledgments

Built on top of [libai](https://github.com/6over3/libai) by 6over3 Institute.
