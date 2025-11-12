"""
Pytest configuration for apple-foundation-models-py tests
"""

import pytest
import applefoundationmodels


@pytest.fixture(scope="session")
def check_availability():
    """Check if Apple Intelligence is available before running tests that need it."""
    status = applefoundationmodels.Session.check_availability()
    if status != applefoundationmodels.Availability.AVAILABLE:
        reason = applefoundationmodels.Session.get_availability_reason()
        pytest.skip(f"Apple Intelligence not available: {reason}")
    return True


@pytest.fixture
def session(check_availability):
    """Provide a applefoundationmodels Session instance."""
    # Create session without instructions to avoid transcript pollution
    session = applefoundationmodels.Session(instructions=None)
    yield session
    session.close()


# Test helper functions


def assert_valid_response(response, min_length=0, message="Response validation failed"):
    """
    Assert response is a valid GenerationResponse or raw string (legacy).

    Args:
        response: The GenerationResponse object or string to validate
        min_length: Minimum expected length (default: 0)
        message: Custom error message prefix

    Returns:
        The validated response text for chaining
    """
    from applefoundationmodels.types import GenerationResponse

    # Handle both new GenerationResponse and legacy string responses
    if isinstance(response, GenerationResponse):
        text = response.text
    elif isinstance(response, str):
        text = response
    else:
        raise TypeError(
            f"{message}: should be GenerationResponse or string, got {type(response)}"
        )

    assert (
        len(text) > min_length
    ), f"{message}: should have content (got {len(text)} chars)"
    return text


def assert_valid_chunks(chunks):
    """
    Assert chunks are valid for streaming responses.

    Args:
        chunks: List of StreamChunk objects or strings (legacy) received from streaming

    Returns:
        The combined full response string
    """
    from applefoundationmodels.types import StreamChunk

    assert len(chunks) > 0, "Should receive at least one chunk"

    # Handle both new StreamChunk objects and legacy string chunks
    if all(isinstance(chunk, StreamChunk) for chunk in chunks):
        full_response = "".join(chunk.content for chunk in chunks)
    elif all(isinstance(chunk, str) for chunk in chunks):
        full_response = "".join(chunks)
    else:
        raise TypeError(
            "All chunks should be StreamChunk objects or all should be strings"
        )

    return full_response
