"""
Unit tests for applefoundationmodels.Session
"""

import pytest
import asyncio
import applefoundationmodels
from applefoundationmodels import Session
from conftest import assert_valid_response, assert_valid_chunks


class TestSessionGeneration:
    """Tests for text generation."""

    def test_generate_basic(self, session, check_availability):
        """Test basic text generation."""
        response = session.generate("What is 2 + 2?", temperature=0.3)
        assert isinstance(response.text, str), "Response should have text property"
        assert (
            "4" in response.text or "four" in response.text.lower()
        ), "Response should contain the answer to 2+2"

    def test_generate_with_temperature(self, session, check_availability):
        """Test generation with different temperatures."""
        prompt = "Complete: The sky is"

        # Low temperature
        response1 = session.generate(prompt, temperature=0.1)
        assert isinstance(response1.text, str), "Response should have text property"

        # Medium temperature
        response2 = session.generate(prompt, temperature=0.7)
        assert isinstance(response2.text, str), "Response should have text property"

        # High temperature
        response3 = session.generate(prompt, temperature=1.5)
        assert isinstance(response3.text, str), "Response should have text property"

    def test_generate_with_max_tokens(self, session, check_availability):
        """Test generation with token limit."""
        # Generate with very low token limit
        response_short = session.generate(
            "Write a long story about space exploration", max_tokens=20, temperature=0.5
        )
        assert isinstance(
            response_short.text, str
        ), "Response should have text property"

        # Generate with higher token limit on same prompt
        response_long = session.generate(
            "Write a long story about space exploration",
            max_tokens=200,
            temperature=0.5,
        )
        assert isinstance(response_long.text, str), "Response should have text property"

        # The longer response should be significantly longer
        # (accounting for some variance, but should be noticeably different)
        assert len(response_long.text) > len(response_short.text), (
            f"Higher max_tokens should produce longer response: "
            f"short={len(response_short.text)} chars, long={len(response_long.text)} chars"
        )


class TestSessionStreaming:
    """Tests for streaming generation."""

    def test_generate_stream_basic(self, session, check_availability):
        """Test basic streaming generation."""
        chunks = []
        for chunk in session.generate("Count to 5", stream=True, temperature=0.3):
            chunks.append(chunk)

        assert len(chunks) > 0, "Should receive at least one chunk"
        for chunk in chunks:
            assert hasattr(chunk, "content"), "Chunk should have content attribute"
            assert isinstance(chunk.content, str), "Chunk content should be string"

    def test_generate_stream_with_temperature(self, session, check_availability):
        """Test streaming with different temperatures."""
        chunks = []
        for chunk in session.generate("Say hello", stream=True, temperature=1.0):
            chunks.append(chunk)

        assert len(chunks) > 0, "Should receive at least one chunk"
        for chunk in chunks:
            assert hasattr(chunk, "content"), "Chunk should have content attribute"


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
        assert isinstance(response.text, str), "Response should have text property"

        # Get history before clear
        history_before = session.get_history()
        assert isinstance(history_before, list)

        # Clear history
        session.clear_history()

        # Verify history is cleared
        history_after = session.get_history()
        assert isinstance(history_after, list)
        assert len(history_after) == 0, "History should be empty after clearing"


class TestSessionLifecycle:
    """Tests for session lifecycle."""

    def test_session_context_manager(self, check_availability):
        """Test session works as context manager."""
        with Session() as session:
            assert session is not None
            response = session.generate("Hello", temperature=0.5)
            assert isinstance(response.text, str), "Response should have text property"

    def test_session_close(self, check_availability):
        """Test explicit session close."""
        session = Session()
        response = session.generate("Hello", temperature=0.5)
        assert isinstance(response.text, str), "Response should have text property"
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

        response = session.generate(
            "Extract information: John is 30 years old", schema=schema
        )

        # Verify response structure
        result = response.parsed
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

        response = session.generate(
            "Extract information: John is 30 years old", schema=Person
        )

        # Verify response structure
        result = response.parsed
        assert isinstance(result, dict), "Result should be a dictionary"
        assert "name" in result, "Result should have 'name' field"
        assert "age" in result, "Result should have 'age' field"
        assert isinstance(result["name"], str), "Name should be a string"
        assert isinstance(result["age"], int), "Age should be an integer"

        # Should be able to parse directly into Pydantic model
        person = Person(**result)
        assert person.name == result["name"]
        assert person.age == result["age"]

        # Test parse_as helper
        person2 = response.parse_as(Person)
        assert person2.name == result["name"]
        assert person2.age == result["age"]


class TestTranscriptTracking:
    """Tests for transcript and last_generation_transcript properties."""

    def test_last_generation_transcript_empty(self, session, check_availability):
        """Test last_generation_transcript is empty before any generation."""
        last_transcript = session.last_generation_transcript
        assert isinstance(last_transcript, list), "Should return a list"
        assert len(last_transcript) == 0, "Should be empty before any generation"

    def test_last_generation_transcript_single_generation(
        self, session, check_availability
    ):
        """Test last_generation_transcript after a single generation."""
        # Perform one generation
        response = session.generate("What is 2 + 2?", temperature=0.3)
        assert isinstance(response.text, str), "Response should have text property"

        # Get last generation transcript
        last_transcript = session.last_generation_transcript
        assert isinstance(last_transcript, list), "Should return a list"
        assert len(last_transcript) > 0, "Should have entries after generation"

        # Verify transcript contains expected entry types
        entry_types = {entry["type"] for entry in last_transcript}
        assert "prompt" in entry_types, "Should contain prompt entry"
        assert "response" in entry_types, "Should contain response entry"

        # Last entries of full transcript should match last_generation_transcript
        full_transcript = session.transcript
        assert len(full_transcript) >= len(
            last_transcript
        ), "Full transcript should be at least as long as last_generation_transcript"
        assert (
            full_transcript[-len(last_transcript) :] == last_transcript
        ), "Last entries of full transcript should match last_generation_transcript"

    def test_last_generation_transcript_multiple_generations(
        self, session, check_availability
    ):
        """Test last_generation_transcript only returns entries from the last call."""
        # First generation
        response1 = session.generate("What is 2 + 2?", temperature=0.3)
        assert isinstance(response1.text, str), "Response should have text property"
        last_transcript1 = session.last_generation_transcript
        last_transcript1_len = len(last_transcript1)
        assert last_transcript1_len > 0, "Should have entries after first generation"

        # Second generation
        response2 = session.generate("What is 5 + 7?", temperature=0.3)
        assert isinstance(response2.text, str), "Response should have text property"
        last_transcript2 = session.last_generation_transcript
        last_transcript2_len = len(last_transcript2)
        assert last_transcript2_len > 0, "Should have entries after second generation"

        # Get full transcript
        full_transcript = session.transcript
        full_transcript_len = len(full_transcript)

        # Verify last_generation_transcript only contains entries from second call
        assert (
            last_transcript2_len < full_transcript_len
        ), "last_generation_transcript should be shorter than full transcript"

        # Verify full transcript contains both generations
        assert full_transcript_len >= (
            last_transcript1_len + last_transcript2_len
        ), "Full transcript should contain entries from both generations"

        # Verify last entries in full transcript match last_generation_transcript
        assert (
            full_transcript[-last_transcript2_len:] == last_transcript2
        ), "Last entries of full transcript should match last_generation_transcript"

    def test_last_generation_transcript_with_structured_output(self, session):
        """Test last_generation_transcript with structured output."""
        schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "integer"},
            },
            "required": ["answer"],
        }

        # First generate regular text
        response = session.generate("Hello", temperature=0.3)
        assert isinstance(response.text, str), "Response should have text property"

        # Then generate structured output
        result = session.generate(
            "What is 2 + 2? Respond with just the number.", schema=schema
        )
        assert isinstance(result.parsed, dict), "Result should have parsed property"

        # Get last generation transcript
        last_transcript = session.last_generation_transcript
        assert (
            len(last_transcript) > 0
        ), "Should have entries after structured generation"

        # Full transcript should have more entries than last_generation_transcript
        full_transcript = session.transcript
        assert len(full_transcript) > len(
            last_transcript
        ), "Full transcript should include previous generation"

    def test_last_generation_transcript_with_streaming(
        self, session, check_availability
    ):
        """Test last_generation_transcript with streaming."""
        # First generation (regular)
        response1 = session.generate("Count to 3", temperature=0.3)
        assert isinstance(response1.text, str), "Response should have text property"

        # Second generation (streaming)
        chunks = []
        for chunk in session.generate("Say hello", stream=True, temperature=0.3):
            chunks.append(chunk)
        assert len(chunks) > 0, "Should receive chunks"

        # Get last generation transcript
        last_transcript = session.last_generation_transcript
        assert (
            len(last_transcript) > 0
        ), "Should have entries after streaming generation"

        # Verify full transcript includes both generations
        full_transcript = session.transcript
        assert len(full_transcript) > len(
            last_transcript
        ), "Full transcript should include both regular and streaming generations"

    def test_last_generation_transcript_after_clear_history(
        self, session, check_availability
    ):
        """Test last_generation_transcript behavior after clearing history."""
        # Generate something
        response = session.generate("Hello", temperature=0.3)
        assert isinstance(response.text, str), "Response should have text property"

        # Clear history
        session.clear_history()

        # last_generation_transcript should be empty after clear
        last_transcript = session.last_generation_transcript
        assert len(last_transcript) == 0, "Should be empty after clearing history"

        # Generate again
        response2 = session.generate("What is 3 plus 3?", temperature=0.3)
        assert isinstance(response2.text, str), "Response should have text property"

        # Now should have entries from the new generation
        last_transcript2 = session.last_generation_transcript
        assert len(last_transcript2) > 0, "Should have entries from new generation"

        # Last entries of full transcript should match last_generation_transcript
        full_transcript = session.transcript
        assert len(full_transcript) >= len(
            last_transcript2
        ), "Full transcript should be at least as long as last_generation_transcript"
        assert (
            full_transcript[-len(last_transcript2) :] == last_transcript2
        ), "Last entries of full transcript should match last_generation_transcript"

    def test_last_generation_transcript_entry_format(self, session, check_availability):
        """Test that last_generation_transcript entries have expected format."""
        response = session.generate("What is the capital of France?", temperature=0.3)
        assert isinstance(response.text, str), "Response should have text property"

        last_transcript = session.last_generation_transcript
        assert len(last_transcript) > 0, "Should have entries"

        # Verify each entry has required fields
        for entry in last_transcript:
            assert isinstance(entry, dict), "Each entry should be a dictionary"
            assert "type" in entry, "Each entry should have a 'type' field"
            assert entry["type"] in [
                "instructions",
                "prompt",
                "response",
                "tool_call",
                "tool_output",
            ], f"Entry type should be valid, got: {entry['type']}"

            # Content field should exist for text entries
            if entry["type"] in ["instructions", "prompt", "response"]:
                assert "content" in entry, "Text entry should have 'content' field"
                assert isinstance(entry["content"], str), "Content should be a string"

    def test_last_generation_transcript_with_tool_calling(self, check_availability):
        """Test last_generation_transcript includes tool_call and tool_output entries."""
        tool_call_count = {"first": 0, "second": 0}

        def get_temperature(city: str) -> str:
            """Get the current temperature in a city."""
            # Track which generation called the tool
            if tool_call_count["first"] == 0:
                tool_call_count["first"] += 1
            else:
                tool_call_count["second"] += 1
            return f"The temperature in {city} is 72°F"

        # Create session with tools
        session = Session(
            instructions="You are a helpful assistant. Use tools when appropriate.",
            tools=[get_temperature],
        )

        # First generation - call tool
        response1 = session.generate(
            "What's the temperature in Boston?", temperature=0.3
        )
        assert isinstance(response1.text, str), "Response should have text property"

        # Get last_generation_transcript from first call
        last_transcript1 = session.last_generation_transcript
        assert len(last_transcript1) > 0, "Should have entries after first generation"

        # Verify tool_call and tool_output are in last_generation_transcript
        entry_types1 = {entry["type"] for entry in last_transcript1}
        assert "prompt" in entry_types1, "Should contain prompt"
        assert "tool_call" in entry_types1, "Should contain tool_call entry"
        assert "tool_output" in entry_types1, "Should contain tool_output entry"
        assert "response" in entry_types1, "Should contain response"

        # Verify tool_call entry structure
        tool_calls1 = [e for e in last_transcript1 if e["type"] == "tool_call"]
        assert len(tool_calls1) > 0, "Should have tool_call entries"
        tool_call = tool_calls1[0]
        assert "tool_id" in tool_call, "tool_call should have tool_id"
        assert "tool_name" in tool_call, "tool_call should have tool_name"
        assert "arguments" in tool_call, "tool_call should have arguments"
        assert tool_call["tool_name"] == "get_temperature"

        # Verify tool_output entry structure
        tool_outputs1 = [e for e in last_transcript1 if e["type"] == "tool_output"]
        assert len(tool_outputs1) > 0, "Should have tool_output entries"
        tool_output = tool_outputs1[0]
        assert "tool_id" in tool_output, "tool_output should have tool_id"
        assert "content" in tool_output, "tool_output should have content"
        assert "72°F" in tool_output["content"], "tool_output should contain the result"

        # Second generation - also call tool
        response2 = session.generate("What about Seattle?", temperature=0.3)
        assert isinstance(response2.text, str), "Response should have text property"

        # Get last_generation_transcript from second call
        last_transcript2 = session.last_generation_transcript
        assert len(last_transcript2) > 0, "Should have entries after second generation"

        # Verify last_generation_transcript only contains second generation's tool calls
        entry_types2 = {entry["type"] for entry in last_transcript2}
        assert "prompt" in entry_types2, "Should contain prompt from second call"
        assert "tool_call" in entry_types2, "Should contain tool_call from second call"
        assert (
            "tool_output" in entry_types2
        ), "Should contain tool_output from second call"

        # Verify full transcript contains both generations
        full_transcript = session.transcript
        full_tool_calls = [e for e in full_transcript if e["type"] == "tool_call"]
        assert (
            len(full_tool_calls) >= 2
        ), "Full transcript should have tool calls from both generations"

        last_tool_calls = [e for e in last_transcript2 if e["type"] == "tool_call"]
        assert len(last_tool_calls) < len(
            full_tool_calls
        ), "last_generation_transcript should have fewer tool calls than full transcript"

        # Verify last_generation_transcript doesn't include first generation's entries
        # by checking that its length is less than full transcript
        assert len(last_transcript2) < len(
            full_transcript
        ), "last_generation_transcript should be shorter than full transcript after multiple generations"

        # Verify the last entries in full transcript match last_generation_transcript
        assert (
            full_transcript[-len(last_transcript2) :] == last_transcript2
        ), "Last entries of full transcript should match last_generation_transcript"

        session.close()
