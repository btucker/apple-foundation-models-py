"""
Unit tests for applefoundationmodels.Session static methods and lifecycle
"""

import pytest
import applefoundationmodels
from applefoundationmodels import Availability


class TestAvailability:
    """Tests for availability checking."""

    def test_check_availability(self):
        """Test availability check returns valid status."""
        status = applefoundationmodels.Session.check_availability()
        assert isinstance(status, Availability)
        assert status in [
            Availability.AVAILABLE,
            Availability.DEVICE_NOT_ELIGIBLE,
            Availability.NOT_ENABLED,
            Availability.MODEL_NOT_READY,
            Availability.AVAILABILITY_UNKNOWN,
        ]

    def test_get_availability_reason(self):
        """Test availability reason returns None or a non-empty string."""
        reason = applefoundationmodels.Session.get_availability_reason()
        assert reason is None or (isinstance(reason, str) and len(reason) > 0)

    def test_is_ready(self):
        """Test is_ready returns a boolean."""
        ready = applefoundationmodels.Session.is_ready()
        assert isinstance(ready, bool)


class TestSessionInfo:
    """Tests for session information methods."""

    def test_get_version(self):
        """Test version string is returned."""
        version = applefoundationmodels.Session.get_version()
        assert isinstance(version, str)
        assert len(version) > 0


class TestSessionLifecycle:
    """Tests for session lifecycle management."""

    def test_session_context_manager(self, check_availability):
        """Test session works as context manager."""
        with applefoundationmodels.Session() as session:
            assert session is not None
            version = session.get_version()
            assert isinstance(version, str)

    def test_session_close(self, check_availability):
        """Test explicit session close."""
        session = applefoundationmodels.Session()
        version = session.get_version()
        assert isinstance(version, str)
        assert len(version) > 0
        session.close()
        # Close should complete without error

    def test_multiple_sessions(self, check_availability):
        """Test multiple sessions can be created."""
        session1 = applefoundationmodels.Session()
        session2 = applefoundationmodels.Session()

        v1 = session1.get_version()
        v2 = session2.get_version()

        assert isinstance(v1, str)
        assert isinstance(v2, str)
        assert len(v1) > 0
        assert len(v2) > 0
        assert v1 == v2, "Both sessions should report same version"

        session1.close()
        session2.close()


class TestSessionCreation:
    """Tests for session creation with parameters."""

    def test_create_session_basic(self, check_availability):
        """Test basic session creation."""
        session = applefoundationmodels.Session()
        assert session is not None
        session.close()

    def test_create_session_with_instructions(self, check_availability):
        """Test session creation with instructions."""
        instructions = "You are a helpful assistant."
        session = applefoundationmodels.Session(instructions=instructions)
        assert session is not None
        session.close()

    def test_create_multiple_sessions_with_instructions(self, check_availability):
        """Test creating multiple sessions with different instructions."""
        session1 = applefoundationmodels.Session(instructions="You are a math tutor.")
        session2 = applefoundationmodels.Session(instructions="You are a poet.")

        assert session1 is not None
        assert session2 is not None

        session1.close()
        session2.close()
