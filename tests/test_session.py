"""
Unit tests for applefoundationmodels.Session
"""

import pytest
import asyncio
import applefoundationmodels
from conftest import assert_valid_response, assert_valid_chunks


class TestSessionGeneration:
    """Tests for text generation."""

    def test_generate_basic(self, session, check_availability):
        """Test basic text generation."""
        response = session.generate("What is 2 + 2?", temperature=0.3)
        assert_valid_response(response)
        assert (
            "4" in response or "four" in response.lower()
        ), "Response should contain the answer to 2+2"

    def test_generate_with_temperature(self, session, check_availability):
        """Test generation with different temperatures."""
        prompt = "Complete: The sky is"

        # Low temperature
        response1 = session.generate(prompt, temperature=0.1)
        assert_valid_response(response1)

        # Medium temperature
        response2 = session.generate(prompt, temperature=0.7)
        assert_valid_response(response2)

        # High temperature
        response3 = session.generate(prompt, temperature=1.5)
        assert_valid_response(response3)

    def test_generate_with_max_tokens(self, session, check_availability):
        """Test generation with token limit."""
        # Generate with very low token limit
        response_short = session.generate(
            "Write a long story about space exploration", max_tokens=20, temperature=0.5
        )
        assert_valid_response(response_short)

        # Generate with higher token limit on same prompt
        response_long = session.generate(
            "Write a long story about space exploration",
            max_tokens=200,
            temperature=0.5,
        )
        assert_valid_response(response_long)

        # The longer response should be significantly longer
        # (accounting for some variance, but should be noticeably different)
        assert len(response_long) > len(response_short), (
            f"Higher max_tokens should produce longer response: "
            f"short={len(response_short)} chars, long={len(response_long)} chars"
        )


class TestSessionStreaming:
    """Tests for streaming generation."""

    @pytest.mark.asyncio
    async def test_generate_stream_basic(self, session, check_availability):
        """Test basic streaming generation."""
        chunks = []
        async for chunk in session.generate_stream("Count to 5", temperature=0.3):
            chunks.append(chunk)

        assert_valid_chunks(chunks)

    @pytest.mark.asyncio
    async def test_generate_stream_with_temperature(self, session, check_availability):
        """Test streaming with different temperatures."""
        chunks = []
        async for chunk in session.generate_stream("Say hello", temperature=1.0):
            chunks.append(chunk)

        assert_valid_chunks(chunks)


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
        assert_valid_response(response)

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
        assert (
            len(updated_history) >= initial_count
        ), "History should not shrink after adding message"


class TestSessionLifecycle:
    """Tests for session lifecycle."""

    def test_session_context_manager(self, client, check_availability):
        """Test session works as context manager."""
        with client.create_session() as session:
            assert session is not None
            response = session.generate("Hello", temperature=0.5)
            assert_valid_response(response)

    def test_session_close(self, client, check_availability):
        """Test explicit session close."""
        session = client.create_session()
        response = session.generate("Hello", temperature=0.5)
        assert_valid_response(response)
        session.close()
        # Close should complete without error


class TestStructuredOutput:
    """Tests for structured output generation."""

    def test_generate_structured_basic(self, session):
        """Test basic structured output generation."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }

        result = session.generate_structured(
            "Extract information: John is 30 years old", schema=schema
        )

        # Verify response structure
        assert isinstance(result, dict), "Result should be a dictionary"
        assert "name" in result, "Result should have 'name' field"
        assert "age" in result, "Result should have 'age' field"
        assert isinstance(result["name"], str), "Name should be a string"
        assert isinstance(result["age"], int), "Age should be an integer"

    def test_generate_structured_pydantic(self, session):
        """Test structured output with Pydantic model."""
        try:
            from pydantic import BaseModel
        except ImportError:
            pytest.skip("Pydantic not installed")

        class Person(BaseModel):
            name: str
            age: int

        result = session.generate_structured(
            "Extract information: John is 30 years old", schema=Person
        )

        # Verify response structure
        assert isinstance(result, dict), "Result should be a dictionary"
        assert "name" in result, "Result should have 'name' field"
        assert "age" in result, "Result should have 'age' field"
        assert isinstance(result["name"], str), "Name should be a string"
        assert isinstance(result["age"], int), "Age should be an integer"

        # Should be able to parse directly into Pydantic model
        person = Person(**result)
        assert person.name == result["name"]
        assert person.age == result["age"]
