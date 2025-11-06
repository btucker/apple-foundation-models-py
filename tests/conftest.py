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
