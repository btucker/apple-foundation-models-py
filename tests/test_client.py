"""
Unit tests for foundationmodels.Client
"""

import pytest
import foundationmodels
from foundationmodels import Availability


class TestAvailability:
    """Tests for availability checking."""

    def test_check_availability(self):
        """Test availability check returns valid status."""
        status = foundationmodels.Client.check_availability()
        assert isinstance(status, Availability)
        assert status in [
            Availability.AVAILABLE,
            Availability.DEVICE_NOT_ELIGIBLE,
            Availability.NOT_ENABLED,
            Availability.MODEL_NOT_READY,
        ]

    def test_get_availability_reason(self):
        """Test availability reason returns a string."""
        reason = foundationmodels.Client.get_availability_reason()
        assert isinstance(reason, str)
        assert len(reason) > 0

    def test_is_ready(self):
        """Test is_ready returns a boolean."""
        ready = foundationmodels.Client.is_ready()
        assert isinstance(ready, bool)


class TestClientInfo:
    """Tests for client information methods."""

    def test_get_version(self, client):
        """Test version string is returned."""
        version = client.get_version()
        assert isinstance(version, str)
        assert len(version) > 0

    def test_get_supported_languages(self, client):
        """Test supported languages returns a list."""
        languages = client.get_supported_languages()
        assert isinstance(languages, list)
        # May be empty if not available
        for lang in languages:
            assert isinstance(lang, str)


class TestClientLifecycle:
    """Tests for client lifecycle management."""

    def test_client_context_manager(self):
        """Test client works as context manager."""
        with foundationmodels.Client() as client:
            assert client is not None
            version = client.get_version()
            assert isinstance(version, str)

    def test_client_close(self):
        """Test explicit client close."""
        client = foundationmodels.Client()
        version = client.get_version()
        assert isinstance(version, str)
        client.close()
        # Should not raise

    def test_multiple_clients(self):
        """Test multiple clients can be created."""
        client1 = foundationmodels.Client()
        client2 = foundationmodels.Client()

        v1 = client1.get_version()
        v2 = client2.get_version()

        assert v1 == v2

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


class TestStats:
    """Tests for statistics tracking."""

    def test_get_stats(self, client):
        """Test getting statistics."""
        stats = client.get_stats()
        assert isinstance(stats, dict)
        assert "total_requests" in stats
        assert "successful_requests" in stats
        assert "failed_requests" in stats
        assert "total_tokens_generated" in stats
        assert "average_response_time" in stats
        assert "total_processing_time" in stats

    def test_reset_stats(self, client):
        """Test resetting statistics."""
        client.reset_stats()
        # Should not raise
