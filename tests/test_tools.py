"""
Tests for tool calling functionality.

These tests verify that tools can be registered and called by the model
with various parameter signatures and types.
"""

import pytest
from typing import Callable, Dict, Any, Optional
from functools import wraps
from applefoundationmodels import Session


class ToolTestHarness:
    """Helper for testing tool calling with less boilerplate."""

    def __init__(self):
        self.calls = []
        self.tools = []

    def wrap_tool(self, func: Callable) -> Callable:
        """
        Wrap a tool function to capture calls.

        Usage:
            harness = ToolTestHarness()

            def my_tool(param: str) -> str:
                '''Tool description.'''
                return "result"

            wrapped = harness.wrap_tool(my_tool)
            response = session.generate("prompt", tools=[wrapped])
        """
        original_func = func

        @wraps(func)
        def wrapper(*args, **kwargs):
            call_info = {
                "args": args,
                "kwargs": kwargs,
            }
            result = original_func(*args, **kwargs)
            call_info["result"] = result
            self.calls.append(call_info)
            return result

        # Copy function metadata for schema extraction
        wrapper.__name__ = original_func.__name__
        wrapper.__annotations__ = original_func.__annotations__
        wrapper.__doc__ = original_func.__doc__

        self.tools.append(wrapper)
        return wrapper

    def assert_called_once(self) -> Dict[str, Any]:
        """Assert tool was called exactly once and return the call info."""
        assert len(self.calls) == 1, f"Expected 1 call, got {len(self.calls)}"
        return self.calls[0]

    def assert_called_with(self, **expected_kwargs) -> Dict[str, Any]:
        """Assert tool was called once with specific kwargs."""
        call = self.assert_called_once()
        for key, expected_value in expected_kwargs.items():
            assert key in call["kwargs"], f"Expected kwarg '{key}' not found"
            assert (
                call["kwargs"][key] == expected_value
            ), f"Expected {key}={expected_value}, got {call['kwargs'][key]}"
        return call

    def get_call_kwargs(self) -> Dict[str, Any]:
        """Get the kwargs from the single call."""
        call = self.assert_called_once()
        return call["kwargs"]


@pytest.fixture
def session(check_availability):
    """Create a session for testing."""
    with Session(
        instructions="You are a helpful assistant. Use the provided tools when needed."
    ) as s:
        yield s


class TestToolRegistration:
    """Tests for tool registration and schema extraction."""

    def test_tool_with_no_parameters(self, check_availability):
        """Test registering and calling a tool with no parameters."""
        called = {}

        def get_time() -> str:
            """Get current time."""
            called["get_time"] = True
            return "2:30 PM"

        session = Session(tools=[get_time])
        response = session.generate("What time is it?")

        assert "get_time" in called
        assert "2:30" in response.text or "time" in response.text.lower()

    def test_tool_with_single_string_parameter(self, check_availability):
        """Test tool with a single string parameter."""
        harness = ToolTestHarness()

        def get_weather(location: str) -> str:
            """Get weather information."""
            return f"Weather in {location}: 72°F, sunny"

        wrapped = harness.wrap_tool(get_weather)
        session = Session(tools=[wrapped])
        response = session.generate("What's the weather in Paris?")

        harness.assert_called_with(location="Paris")
        assert "72°F" in response.text or "sunny" in response.text.lower()

    def test_tool_with_multiple_parameters(self, check_availability):
        """Test tool with multiple string parameters."""
        called = {}

        def search_docs(query: str, category: str) -> str:
            """Search the documentation database."""
            called["query"] = query
            called["category"] = category
            return f"Found 5 documents about '{query}' in {category}"

        session = Session(tools=[search_docs])
        response = session.generate("Search for 'authentication' in the API category")

        assert "query" in called
        assert "category" in called
        assert called["query"] == "authentication"
        assert called["category"] == "API"
        # Validate response is a non-empty string with expected content
        assert isinstance(response.text, str), "Response should be a string"
        assert response, "Response should not be empty"
        assert (
            "5 documents" in response.text or "authentication" in response.text.lower()
        )

    def test_tool_with_mixed_types(self, check_availability):
        """Test tool with mixed parameter types (string and int)."""
        harness = ToolTestHarness()

        def get_top_items(category: str, count: int) -> str:
            """Get top items in a category."""
            items = [f"Item {i+1}" for i in range(count)]
            return f"Top {count} in {category}: {', '.join(items)}"

        wrapped = harness.wrap_tool(get_top_items)
        session = Session(
            instructions="You are a helpful assistant. Always use the tools provided and include their results in your response.",
            tools=[wrapped],
        )
        response = session.generate(
            "Show me the top 3 electronics products. Use the get_top_items tool."
        )

        # Verify tool was called with correct parameters
        kwargs = harness.get_call_kwargs()
        assert kwargs["count"] == 3
        assert "category" in kwargs

        # Verify the tool was called (check tool_calls property)
        assert response.tool_calls is not None, "Tool should have been called"
        assert len(response.tool_calls) > 0, "At least one tool call expected"
        assert response.tool_calls[0].function.name == "get_top_items"

    def test_tool_with_optional_parameters(self, check_availability):
        """Test tool with optional parameters and defaults."""
        called = {}

        def calculate(x: int, y: int, operation: str = "add") -> str:
            """Perform a calculation."""
            called["x"] = x
            called["y"] = y
            called["operation"] = operation

            operations = {
                "add": x + y,
                "subtract": x - y,
                "multiply": x * y,
                "times": x * y,
                "multiplication": x * y,
            }
            result = operations.get(operation, "unknown")
            return f"Result: {result}"

        session = Session(tools=[calculate])
        response = session.generate("What is 15 times 7?")

        assert "x" in called
        assert "y" in called
        assert "operation" in called
        # Should use multiply operation
        assert called["operation"] in ["multiply", "times", "multiplication"]
        assert "105" in response.text


class TestToolExecution:
    """Tests for tool execution behavior."""

    def test_multiple_tools_registered(self, check_availability):
        """Test that multiple tools can be registered and called."""
        calls = []

        def get_time() -> str:
            """Get time."""
            calls.append("get_time")
            return "2:30 PM"

        def get_date() -> str:
            """Get date."""
            calls.append("get_date")
            return "November 7, 2024"

        # This might call one or both depending on the prompt
        session = Session(tools=[get_time, get_date])
        session.generate("What's the time and date?")

        # At least one should be called
        assert len(calls) > 0

    def test_tool_return_types(self, check_availability):
        """Test tools can return different types."""
        called = {}

        def get_status() -> str:
            """Get the current system status."""
            called["invoked"] = True
            return "System operational"

        session = Session(
            instructions="You must use the get_status tool to answer status questions.",
            tools=[get_status],
        )
        response = session.generate(
            "What's the system status? Use the get_status tool."
        )
        # Verify tool was called and response contains relevant content
        assert called.get("invoked"), "Tool should have been called"
        assert (
            "operational" in response.text.lower() or "status" in response.text.lower()
        )

    def test_tool_with_optional_type_annotation(self, check_availability):
        """Test that Optional[...] type annotations are properly handled."""
        called = {}

        def get_weather(location: Optional[str] = None, units: str = "celsius") -> str:
            """Get weather information."""
            called["location"] = location
            called["units"] = units

            if location is None:
                return "Weather for current location: 20°C, cloudy"
            return f"Weather in {location}: 22°{units[0].upper()}, sunny"

        session = Session(tools=[get_weather])
        response = session.generate("What's the weather in Paris?")

        # Verify the tool was called with location set
        assert "location" in called
        assert called["location"] == "Paris"
        assert "22°" in response.text or "sunny" in response.text.lower()


class TestToolCallsProperty:
    """Tests for tool_calls and finish_reason properties on GenerationResponse."""

    def test_tool_calls_property_without_tools(self, session):
        """Test that tool_calls is None when no tools are registered."""
        response = session.generate("What is 2 plus 2?")

        # Verify response has tool_calls and finish_reason properties
        assert hasattr(response, "tool_calls")
        assert hasattr(response, "finish_reason")

        # No tools registered, so tool_calls should be None
        assert response.tool_calls is None
        # finish_reason should be "stop" since no tools were called
        assert response.finish_reason == "stop"

    def test_tool_calls_property_with_tools_not_called(self, check_availability):
        """Test that tool_calls is None when tools exist but aren't called."""

        def get_weather(location: str) -> str:
            """Get weather for a location."""
            return f"Weather in {location}: 20°C"

        # Generate something that doesn't trigger the tool
        session = Session(tools=[get_weather])
        response = session.generate("What is 2 plus 2?")

        # Tool wasn't called, so tool_calls should be None
        assert response.tool_calls is None
        assert response.finish_reason == "stop"

    def test_tool_calls_property_with_tools_called(self, check_availability):
        """Test that tool_calls is populated when tools are called."""

        def get_weather(location: str) -> str:
            """Get weather for a location."""
            return f"Weather in {location}: 22°C"

        session = Session(tools=[get_weather])
        response = session.generate("What's the weather in Paris?")

        # Verify tool_calls is populated
        assert response.tool_calls is not None
        assert len(response.tool_calls) > 0

        # Verify finish_reason is "tool_calls"
        assert response.finish_reason == "tool_calls"

        # Verify structure of tool call
        tool_call = response.tool_calls[0]
        assert hasattr(tool_call, "id")
        assert hasattr(tool_call, "type")
        assert hasattr(tool_call, "function")
        assert tool_call.type == "function"

        # Verify function structure
        assert hasattr(tool_call.function, "name")
        assert hasattr(tool_call.function, "arguments")
        assert tool_call.function.name == "get_weather"

        # Verify arguments is a JSON string
        import json

        args = json.loads(tool_call.function.arguments)
        assert "location" in args
        assert args["location"] == "Paris"

    def test_tool_calls_property_with_multiple_tools(self, check_availability):
        """Test tool_calls with multiple tool invocations."""

        def get_weather(location: str) -> str:
            """Get weather for a location."""
            return f"Weather in {location}: 22°C"

        def get_time(timezone: str = "UTC") -> str:
            """Get current time for a timezone."""
            return f"Time in {timezone}: 12:00 PM"

        session = Session(tools=[get_weather, get_time])
        response = session.generate("What's the weather and time in Paris?")

        # May call one or both tools depending on model behavior
        if response.tool_calls:
            assert response.finish_reason == "tool_calls"
            assert len(response.tool_calls) >= 1

            # Verify all tool calls have proper structure
            for tool_call in response.tool_calls:
                assert tool_call.type == "function"
                assert tool_call.function.name in ["get_weather", "get_time"]
                assert isinstance(tool_call.function.arguments, str)


class TestTranscript:
    """Tests for transcript access with tool calls."""

    def test_transcript_includes_tool_calls(self, check_availability):
        """Test that transcript includes tool call entries."""

        def get_info() -> str:
            """Get info."""
            return "Information"

        session = Session(tools=[get_info])
        session.generate("Get me some info")

        transcript = session.transcript
        assert len(transcript) > 0

        # Check that we have expected entry types
        entry_types = [entry.get("type") for entry in transcript]
        assert "instructions" in entry_types or "prompt" in entry_types
        assert "prompt" in entry_types
        # Verify tool_call and tool_output entries are present
        assert "tool_call" in entry_types, "transcript should contain tool_call entries"
        assert (
            "tool_output" in entry_types
        ), "transcript should contain tool_output entries"

    def test_transcript_structure(self, check_availability):
        """Test that transcript entries have expected structure."""

        def get_status() -> str:
            """Get the current status."""
            return "System is operational"

        session = Session(tools=[get_status])
        session.generate("What's the current status?")

        transcript = session.transcript
        for entry in transcript:
            assert "type" in entry
            # Each entry type should have appropriate fields
            if entry["type"] == "tool_call":
                # Individual tool call entries
                assert "tool_id" in entry
                assert "tool_name" in entry
                assert "arguments" in entry
            elif entry["type"] == "tool_output":
                # Tool output entries
                assert "tool_id" in entry
                assert "content" in entry
            elif entry["type"] in ("prompt", "response", "instructions"):
                assert "content" in entry

    def test_transcript_tool_call_shape(self, check_availability):
        """Test exact shape and content of tool_call and tool_output entries."""

        def calculate_sum(a: int, b: int) -> str:
            """Calculate sum."""
            return f"Result: {a + b}"

        session = Session(tools=[calculate_sum])
        session.generate("What is 5 plus 3?")

        transcript = session.transcript

        # Find tool_call entry
        tool_calls = [e for e in transcript if e["type"] == "tool_call"]
        assert len(tool_calls) > 0, "Should have at least one tool_call entry"

        tool_call = tool_calls[0]
        # Validate required fields
        assert "tool_id" in tool_call, "tool_call must have tool_id"
        assert "tool_name" in tool_call, "tool_call must have tool_name"
        assert "arguments" in tool_call, "tool_call must have arguments"

        # Validate tool_name is correct
        assert tool_call["tool_name"] == "calculate_sum"

        # Validate arguments is a JSON string
        import json

        args = json.loads(tool_call["arguments"])
        assert isinstance(args, dict), "arguments should be a JSON object"

        # Find tool_output entry
        tool_outputs = [e for e in transcript if e["type"] == "tool_output"]
        assert len(tool_outputs) > 0, "Should have at least one tool_output entry"

        tool_output = tool_outputs[0]
        # Validate required fields
        assert "tool_id" in tool_output, "tool_output must have tool_id"
        assert "content" in tool_output, "tool_output must have content"

        # Validate content is not empty
        assert tool_output["content"], "tool_output content should not be empty"
        assert (
            "Result:" in tool_output["content"]
        ), "tool_output should contain actual result"


@pytest.mark.skipif(not Session.is_ready(), reason="Apple Intelligence not available")
class TestToolIntegration:
    """Integration tests requiring Apple Intelligence."""

    def test_end_to_end_tool_calling(self, check_availability):
        """Full end-to-end test of tool calling."""
        results = {}

        def calculate(expression: str) -> str:
            """Calculate math expression."""
            results["called"] = True
            # Simple calculator (in real code, use safe evaluation)
            if "2 + 2" in expression:
                return "4"
            return "calculated"

        with Session(tools=[calculate]) as session:
            response = session.generate("What is 2 + 2?")

            assert results.get("called")
            assert "4" in response.text

    def test_large_tool_output(self, check_availability):
        """Test that tools can return outputs larger than the initial 16KB buffer."""
        called = {}

        # Create a large output (20KB) to test buffer resizing
        # Use a pattern that we can verify wasn't truncated
        large_data = "START-" + ("x" * 20470) + "-END"  # 20KB total

        def get_system_logs() -> str:
            """Get system diagnostic data that includes large logs."""
            called["invoked"] = True
            return large_data

        with Session(tools=[get_system_logs]) as session:
            # More explicit prompt to trigger tool call
            response = session.generate(
                "Use the get_system_logs tool to retrieve the system diagnostic data"
            )

            # Verify the tool was called
            assert called.get("invoked"), "Tool should have been called"

            # Get the transcript to verify the actual tool output
            transcript = session.transcript
            tool_outputs = [e for e in transcript if e.get("type") == "tool_output"]
            assert (
                len(tool_outputs) > 0
            ), "Should have at least one tool output in transcript"

            # Get the actual tool output content
            tool_output_content = tool_outputs[0]["content"]

            # Verify output wasn't truncated - check for both start and end markers
            assert (
                "START-" in tool_output_content
            ), "Tool output should contain START marker"
            assert (
                "-END" in tool_output_content
            ), "Tool output should contain END marker"

            # Verify the output length is close to expected (20KB)
            assert (
                len(tool_output_content) >= 20480
            ), f"Tool output should be at least 20KB, got {len(tool_output_content)} bytes"

            # Verify the exact data was preserved
            assert (
                tool_output_content == large_data
            ), "Tool output should exactly match the returned data"
