# Test Suite

## Overview

The test suite for `apple-foundation-models` is organized into unit tests and integration tests.

## Test Files

### `conftest.py`
Pytest configuration and shared fixtures:
- `check_availability`: Skips tests if Apple Intelligence unavailable
- `client`: Provides a Client instance
- `session`: Provides a Session instance

### `test_client.py` (Unit Tests)
Tests for `foundationmodels.Client`:
- **Availability checking**: `check_availability()`, `get_availability_reason()`, `is_ready()`
- **Client information**: `get_version()`, `get_supported_languages()`
- **Lifecycle management**: Context managers, `close()`, multiple clients
- **Session creation**: Basic, with instructions, multiple sessions
- **Statistics**: `get_stats()`, `reset_stats()`

### `test_session.py` (Unit Tests)
Tests for `foundationmodels.Session`:
- **Text generation**: Basic generation, temperature control, token limits
- **Streaming**: Async streaming, temperature variations
- **History management**: `get_history()`, `clear_history()`, `add_message()`
- **Lifecycle**: Context managers, `close()`
- **Structured output**: NotImplementedError verification

### `test_integration.py` (Integration Tests)
Comprehensive end-to-end tests:
- Availability checking
- Version information retrieval
- Basic text generation (math, knowledge, creative)
- Multi-turn conversation with context
- Async streaming generation
- Temperature variations
- Multiple session management
- Error handling (empty prompts, long prompts)
- Context manager functionality

## Running Tests

### Run all tests:
```bash
pytest
```

### Run only unit tests:
```bash
pytest tests/test_client.py tests/test_session.py
```

### Run only integration tests:
```bash
pytest tests/test_integration.py
```

### Run with coverage:
```bash
pip install pytest-cov
pytest --cov=foundationmodels --cov-report=html
```

### Skip tests if Apple Intelligence unavailable:
Tests automatically skip if Apple Intelligence is not available using the `check_availability` fixture.

## Test Coverage Summary

### ✅ Well-Covered Areas:
- Client initialization and lifecycle
- Session creation and management  
- Text generation (sync)
- Streaming generation (async)
- Temperature control
- Context managers
- Availability checking
- Statistics tracking

### ⚠️ Limited Coverage:
- Structured output (not yet implemented)
- Tool calling (not yet implemented)
- Error edge cases
- Memory management under load
- Thread safety
- Large-scale concurrent operations

### ❌ Not Covered:
- Performance benchmarks
- Memory leak detection
- Stress testing
- Cross-platform compatibility (only macOS)
- Different Swift dylib versions

## Requirements

Tests require:
- macOS 26.0+ with Apple Intelligence enabled
- `pytest>=7.0`
- `pytest-asyncio>=0.20`

Install dev dependencies:
```bash
pip install -e ".[dev]"
```

## CI/CD Considerations

Since tests require Apple Intelligence on macOS 26.0+:
- Cannot run in standard CI environments
- Need macOS runners with Apple Intelligence
- Consider mock/stub layer for basic CI
- Integration tests should be manual or local-only
