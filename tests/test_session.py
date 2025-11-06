"""
Unit tests for foundationmodels.Session
"""

import pytest
import asyncio
import foundationmodels


class TestSessionGeneration:
    """Tests for text generation."""

    def test_generate_basic(self, session, check_availability):
        """Test basic text generation."""
        response = session.generate("What is 2 + 2?", temperature=0.3)
        assert isinstance(response, str)
        assert len(response) > 0

    def test_generate_with_temperature(self, session, check_availability):
        """Test generation with different temperatures."""
        prompt = "Complete: The sky is"

        # Low temperature
        response1 = session.generate(prompt, temperature=0.1)
        assert isinstance(response1, str)

        # Medium temperature
        response2 = session.generate(prompt, temperature=0.7)
        assert isinstance(response2, str)

        # High temperature
        response3 = session.generate(prompt, temperature=1.5)
        assert isinstance(response3, str)

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
            chunks.append(chunk)

        assert len(chunks) > 0


class TestSessionHistory:
    """Tests for conversation history."""

    def test_get_history(self, session, check_availability):
        """Test getting conversation history."""
        history = session.get_history()
        assert isinstance(history, list)

    def test_clear_history(self, session, check_availability):
        """Test clearing conversation history."""
        # Generate something
        session.generate("Hello", temperature=0.5)

        # Clear history
        session.clear_history()

        # Should not raise
        history = session.get_history()
        assert isinstance(history, list)

    def test_add_message(self, session, check_availability):
        """Test manually adding messages."""
        # This is a stub in current implementation
        session.add_message("user", "Test message")
        # Should not raise


class TestSessionLifecycle:
    """Tests for session lifecycle."""

    def test_session_context_manager(self, client, check_availability):
        """Test session works as context manager."""
        with client.create_session() as session:
            response = session.generate("Hello", temperature=0.5)
            assert isinstance(response, str)

    def test_session_close(self, client, check_availability):
        """Test explicit session close."""
        session = client.create_session()
        response = session.generate("Hello", temperature=0.5)
        assert isinstance(response, str)
        session.close()
        # Should not raise


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
