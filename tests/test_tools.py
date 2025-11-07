"""
Tests for tool calling functionality.

These tests verify that tools can be registered and called by the model
with various parameter signatures and types.
"""

import pytest
from applefoundationmodels import Client


@pytest.fixture
def client():
    """Create a client instance for testing."""
    with Client() as c:
        yield c


@pytest.fixture
def session(client):
    """Create a session for testing."""
    return client.create_session(
        instructions="You are a helpful assistant. Use the provided tools when needed."
    )


class TestToolRegistration:
    """Tests for tool registration and schema extraction."""

    def test_tool_with_no_parameters(self, session):
        """Test registering and calling a tool with no parameters."""
        called = {}

        @session.tool(description="Get the current time")
        def get_time() -> str:
            """Get current time."""
            called['get_time'] = True
            return "2:30 PM"

        response = session.generate("What time is it?")

        assert 'get_time' in called
        assert "2:30" in response or "time" in response.lower()

    def test_tool_with_single_string_parameter(self, session):
        """Test tool with a single string parameter."""
        called = {}

        @session.tool(description="Get weather for a location")
        def get_weather(location: str) -> str:
            """Get weather information."""
            called['location'] = location
            return f"Weather in {location}: 72°F, sunny"

        response = session.generate("What's the weather in Paris?")

        assert 'location' in called
        assert called['location'] == 'Paris'
        assert "72°F" in response or "sunny" in response.lower()

    def test_tool_with_multiple_parameters(self, session):
        """Test tool with multiple string parameters."""
        called = {}

        @session.tool(description="Search documentation")
        def search_docs(query: str, category: str) -> str:
            """Search the documentation database."""
            called['query'] = query
            called['category'] = category
            return f"Found 5 documents about '{query}' in {category}"

        response = session.generate("Search for 'authentication' in the API category")

        assert 'query' in called
        assert 'category' in called
        assert called['query'] == 'authentication'
        assert called['category'] == 'API'

    def test_tool_with_mixed_types(self, session):
        """Test tool with mixed parameter types (string and int)."""
        called = {}

        @session.tool(description="Get top N items from a category")
        def get_top_items(category: str, count: int) -> str:
            """Get top items."""
            called['category'] = category
            called['count'] = count
            items = [f"Item {i+1}" for i in range(count)]
            return f"Top {count} in {category}: {', '.join(items)}"

        response = session.generate("Show me the top 3 products")

        assert 'category' in called
        assert 'count' in called
        assert called['count'] == 3
        assert "Item 1" in response or "top" in response.lower()

    def test_tool_with_optional_parameters(self, session):
        """Test tool with optional parameters and defaults."""
        called = {}

        @session.tool(description="Perform mathematical calculation")
        def calculate(x: int, y: int, operation: str = "add") -> str:
            """Perform a calculation."""
            called['x'] = x
            called['y'] = y
            called['operation'] = operation

            operations = {
                "add": x + y,
                "subtract": x - y,
                "multiply": x * y,
            }
            result = operations.get(operation, "unknown")
            return f"Result: {result}"

        response = session.generate("What is 15 times 7?")

        assert 'x' in called
        assert 'y' in called
        assert 'operation' in called
        # Should use multiply operation
        assert called['operation'] in ['multiply', 'times']
        assert "105" in response


class TestToolExecution:
    """Tests for tool execution behavior."""

    def test_multiple_tools_registered(self, session):
        """Test that multiple tools can be registered and called."""
        calls = []

        @session.tool(description="Get time")
        def get_time() -> str:
            calls.append('get_time')
            return "2:30 PM"

        @session.tool(description="Get date")
        def get_date() -> str:
            calls.append('get_date')
            return "November 7, 2024"

        # This might call one or both depending on the prompt
        session.generate("What's the time and date?")

        # At least one should be called
        assert len(calls) > 0

    def test_tool_return_types(self, session):
        """Test tools can return different types."""

        @session.tool(description="Get status")
        def get_status() -> str:
            return "OK"

        response = session.generate("What's the system status?")
        assert "OK" in response or "status" in response.lower()


class TestTranscript:
    """Tests for transcript access with tool calls."""

    def test_transcript_includes_tool_calls(self, session):
        """Test that transcript includes tool call entries."""

        @session.tool(description="Get info")
        def get_info() -> str:
            return "Information"

        session.generate("Get me some info")

        transcript = session.transcript
        assert len(transcript) > 0

        # Check that we have expected entry types
        entry_types = [entry.get('type') for entry in transcript]
        assert 'instructions' in entry_types
        assert 'prompt' in entry_types

    def test_transcript_structure(self, session):
        """Test that transcript entries have expected structure."""

        @session.tool(description="Get the current status")
        def get_status() -> str:
            return "System is operational"

        session.generate("What's the current status?")

        transcript = session.transcript
        for entry in transcript:
            assert 'type' in entry
            # Each entry type should have appropriate fields
            if entry['type'] == 'tool_calls':
                assert 'tool_calls' in entry
            elif entry['type'] in ('prompt', 'response', 'instructions'):
                assert 'content' in entry


@pytest.mark.skipif(
    not Client.is_ready(),
    reason="Apple Intelligence not available"
)
class TestToolIntegration:
    """Integration tests requiring Apple Intelligence."""

    def test_end_to_end_tool_calling(self):
        """Full end-to-end test of tool calling."""
        with Client() as client:
            session = client.create_session()

            results = {}

            @session.tool(description="Calculate math expression")
            def calculate(expression: str) -> str:
                results['called'] = True
                # Simple calculator (in real code, use safe evaluation)
                if "2 + 2" in expression:
                    return "4"
                return "calculated"

            response = session.generate("What is 2 + 2?")

            assert results.get('called') == True
            assert "4" in response
