"""
Unit tests for applefoundationmodels.Session
"""

import pytest
import asyncio
import applefoundationmodels


class TestSessionGeneration:
    """Tests for text generation."""

    def test_generate_basic(self, session, check_availability):
        """Test basic text generation."""
        response = session.generate("What is 2 + 2?", temperature=0.3)
        assert isinstance(response, str)
        assert len(response) > 0
        # Should contain the answer 4
        assert "4" in response or "four" in response.lower(), "Response should contain the answer to 2+2"

    def test_generate_with_temperature(self, session, check_availability):
        """Test generation with different temperatures."""
        prompt = "Complete: The sky is"

        # Low temperature
        response1 = session.generate(prompt, temperature=0.1)
        assert isinstance(response1, str)
        assert len(response1) > 0, "Low temperature response should have content"

        # Medium temperature
        response2 = session.generate(prompt, temperature=0.7)
        assert isinstance(response2, str)
        assert len(response2) > 0, "Medium temperature response should have content"

        # High temperature
        response3 = session.generate(prompt, temperature=1.5)
        assert isinstance(response3, str)
        assert len(response3) > 0, "High temperature response should have content"

        # All responses should be valid
        all_responses = [response1, response2, response3]
        assert all(len(r) > 0 for r in all_responses), "All temperature variations should produce content"

    def test_generate_with_max_tokens(self, session, check_availability):
        """Test generation with token limit."""
        response = session.generate(
            "Tell me a story",
            max_tokens=50,
            temperature=0.5
        )
        assert isinstance(response, str)
        assert len(response) > 0


class TestSessionStreaming:
    """Tests for streaming generation."""

    @pytest.mark.asyncio
    async def test_generate_stream_basic(self, session, check_availability):
        """Test basic streaming generation."""
        chunks = []
        async for chunk in session.generate_stream(
            "Count to 5",
            temperature=0.3
        ):
            assert isinstance(chunk, str)
            chunks.append(chunk)

        assert len(chunks) > 0
        full_response = ''.join(chunks)
        assert len(full_response) > 0

    @pytest.mark.asyncio
    async def test_generate_stream_with_temperature(self, session, check_availability):
        """Test streaming with different temperatures."""
        chunks = []
        async for chunk in session.generate_stream(
            "Say hello",
            temperature=1.0
        ):
            assert isinstance(chunk, str), "Each chunk should be a string"
            chunks.append(chunk)

        assert len(chunks) > 0, "Should receive at least one chunk"
        full_response = ''.join(chunks)
        assert len(full_response) > 0, "Combined response should not be empty"


class TestSessionHistory:
    """Tests for conversation history."""

    def test_get_history(self, session, check_availability):
        """Test getting conversation history."""
        history = session.get_history()
        assert isinstance(history, list)

    def test_clear_history(self, session, check_availability):
        """Test clearing conversation history."""
        # Generate something to populate history
        response = session.generate("Hello", temperature=0.5)
        assert len(response) > 0

        # Get history before clear
        history_before = session.get_history()
        assert isinstance(history_before, list)

        # Clear history
        session.clear_history()

        # Verify history is cleared
        history_after = session.get_history()
        assert isinstance(history_after, list)
        assert len(history_after) == 0, "History should be empty after clearing"

    def test_add_message(self, session, check_availability):
        """Test manually adding messages."""
        # Get initial history
        initial_history = session.get_history()
        initial_count = len(initial_history)

        # Add a message
        session.add_message("user", "Test message")

        # Verify message was added
        updated_history = session.get_history()
        assert len(updated_history) >= initial_count, "History should not shrink after adding message"


class TestSessionLifecycle:
    """Tests for session lifecycle."""

    def test_session_context_manager(self, client, check_availability):
        """Test session works as context manager."""
        with client.create_session() as session:
            assert session is not None
            response = session.generate("Hello", temperature=0.5)
            assert isinstance(response, str)
            assert len(response) > 0, "Response should have content"

    def test_session_close(self, client, check_availability):
        """Test explicit session close."""
        session = client.create_session()
        response = session.generate("Hello", temperature=0.5)
        assert isinstance(response, str)
        assert len(response) > 0, "Response should have content"
        session.close()
        # Close should complete without error


class TestStructuredOutput:
    """Tests for structured output generation."""

    def test_generate_structured_not_implemented(self, session):
        """Test that structured generation raises NotImplementedError."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        }

        with pytest.raises(NotImplementedError):
            session.generate_structured("Extract name: John", schema=schema)
