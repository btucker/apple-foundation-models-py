"""
Pytest configuration for apple-foundation-models-py tests
"""

import pytest
import applefoundationmodels


@pytest.fixture(scope="session")
def check_availability():
    """Check if Apple Intelligence is available before running tests."""
    status = applefoundationmodels.Client.check_availability()
    if status != applefoundationmodels.Availability.AVAILABLE:
        reason = applefoundationmodels.Client.get_availability_reason()
        pytest.skip(f"Apple Intelligence not available: {reason}")
    return True


@pytest.fixture
def client():
    """Provide a applefoundationmodels Client instance."""
    client = applefoundationmodels.Client()
    yield client
    client.close()


@pytest.fixture
def session(client):
    """Provide a applefoundationmodels Session instance."""
    session = client.create_session()
    yield session
    session.close()


# Test helper functions

def assert_valid_response(response, min_length=0, message="Response validation failed"):
    """
    Assert response is a valid non-empty string.

    Args:
        response: The response to validate
        min_length: Minimum expected length (default: 0)
        message: Custom error message prefix

    Returns:
        The validated response for chaining
    """
    assert isinstance(response, str), f"{message}: should be a string, got {type(response)}"
    assert len(response) > min_length, f"{message}: should have content (got {len(response)} chars)"
    return response


def assert_valid_chunks(chunks):
    """
    Assert chunks are valid for streaming responses.

    Args:
        chunks: List of chunks received from streaming

    Returns:
        The combined full response string
    """
    assert len(chunks) > 0, "Should receive at least one chunk"
    assert all(isinstance(chunk, str) for chunk in chunks), "All chunks should be strings"
    full_response = ''.join(chunks)
    assert len(full_response) > 0, "Combined response should not be empty"
    return full_response
