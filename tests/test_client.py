"""
Unit tests for appleapplefoundationmodels.Client
"""

import pytest
import applefoundationmodels
from applefoundationmodels import Availability


class TestAvailability:
    """Tests for availability checking."""

    def test_check_availability(self):
        """Test availability check returns valid status."""
        status = applefoundationmodels.Client.check_availability()
        assert isinstance(status, Availability)
        assert status in [
            Availability.AVAILABLE,
            Availability.DEVICE_NOT_ELIGIBLE,
            Availability.NOT_ENABLED,
            Availability.MODEL_NOT_READY,
        ]

    def test_get_availability_reason(self):
        """Test availability reason returns a string."""
        reason = applefoundationmodels.Client.get_availability_reason()
        assert isinstance(reason, str)
        assert len(reason) > 0

    def test_is_ready(self):
        """Test is_ready returns a boolean."""
        ready = applefoundationmodels.Client.is_ready()
        assert isinstance(ready, bool)


class TestClientInfo:
    """Tests for client information methods."""

    def test_get_version(self, client):
        """Test version string is returned."""
        version = client.get_version()
        assert isinstance(version, str)
        assert len(version) > 0


class TestClientLifecycle:
    """Tests for client lifecycle management."""

    def test_client_context_manager(self):
        """Test client works as context manager."""
        with applefoundationmodels.Client() as client:
            assert client is not None
            version = client.get_version()
            assert isinstance(version, str)

    def test_client_close(self):
        """Test explicit client close."""
        client = applefoundationmodels.Client()
        version = client.get_version()
        assert isinstance(version, str)
        assert len(version) > 0
        client.close()
        # Close should complete without error

    def test_multiple_clients(self):
        """Test multiple clients can be created."""
        client1 = applefoundationmodels.Client()
        client2 = applefoundationmodels.Client()

        v1 = client1.get_version()
        v2 = client2.get_version()

        assert isinstance(v1, str)
        assert isinstance(v2, str)
        assert len(v1) > 0
        assert len(v2) > 0
        assert v1 == v2, "Both clients should report same version"

        client1.close()
        client2.close()


class TestSessionCreation:
    """Tests for session creation."""

    def test_create_session_basic(self, client, check_availability):
        """Test basic session creation."""
        session = client.create_session()
        assert session is not None
        session.close()

    def test_create_session_with_instructions(self, client, check_availability):
        """Test session creation with instructions."""
        instructions = "You are a helpful assistant."
        session = client.create_session(instructions=instructions)
        assert session is not None
        session.close()

    def test_create_multiple_sessions(self, client, check_availability):
        """Test creating multiple sessions."""
        session1 = client.create_session(instructions="You are a math tutor.")
        session2 = client.create_session(instructions="You are a poet.")

        assert session1 is not None
        assert session2 is not None

        session1.close()
        session2.close()
