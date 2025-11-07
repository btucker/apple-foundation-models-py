#!/usr/bin/env python3
"""
Comprehensive demonstration of tool calling with different parameter signatures.

This example showcases the full capabilities of tool calling in apple-foundation-models,
demonstrating that tools can have:
- No parameters
- Single parameters
- Multiple parameters
- Mixed types (strings, integers, etc.)
- Optional parameters with defaults

All tools are automatically registered and called by the model when appropriate.
"""

from applefoundationmodels import Client


def main():
    print("=" * 70)
    print("TOOL CALLING DEMONSTRATION")
    print("=" * 70)
    print()

    with Client() as client:
        if not client.is_ready():
            print(f"❌ Apple Intelligence not available: {client.get_availability_reason()}")
            return 1

        print("✓ Apple Intelligence ready\n")

        session = client.create_session(
            instructions="You are a helpful assistant. Use the provided tools when you need information."
        )

        # Example 1: Tool with no parameters
        @session.tool(description="Get the current time")
        def get_time() -> str:
            """Returns the current time."""
            return "2:30 PM"

        # Example 2: Tool with single string parameter
        @session.tool(description="Get weather for a location")
        def get_weather(location: str) -> str:
            """
            Get weather information for a specific location.

            Args:
                location: City or location name
            """
            return f"Weather in {location}: 72°F, sunny"

        # Example 3: Tool with multiple string parameters
        @session.tool(description="Search documentation")
        def search_docs(query: str, category: str) -> str:
            """
            Search the documentation database.

            Args:
                query: Search query string
                category: Documentation category to search in
            """
            return f"Found 5 documents about '{query}' in {category} category"

        # Example 4: Tool with mixed parameter types
        @session.tool(description="Get top N items from a category")
        def get_top_items(category: str, count: int) -> str:
            """
            Get the top items from a category.

            Args:
                category: The category to get items from
                count: Number of items to return
            """
            items = [f"Item {i+1}" for i in range(count)]
            return f"Top {count} in {category}: {', '.join(items)}"

        # Example 5: Tool with optional parameters
        @session.tool(description="Perform mathematical calculation")
        def calculate(x: int, y: int, operation: str = "add") -> str:
            """
            Perform a mathematical operation.

            Args:
                x: First number
                y: Second number
                operation: Operation to perform (add, subtract, multiply)
            """
            operations = {
                "add": x + y,
                "subtract": x - y,
                "multiply": x * y,
            }
            result = operations.get(operation, "unknown")
            return f"Result: {result}"

        print("✓ Registered 5 tools with various signatures\n")
        print("=" * 70)

        # Now let's test the tools with different queries
        test_queries = [
            ("What time is it?", "get_time"),
            ("What's the weather in Paris?", "get_weather"),
            ("Search for 'API authentication' in the developer docs", "search_docs"),
            ("Show me the top 3 products", "get_top_items"),
            ("What is 42 times 13?", "calculate"),
        ]

        for i, (query, expected_tool) in enumerate(test_queries, 1):
            print(f"\nTest {i}: {query}")
            print("-" * 70)

            try:
                response = session.generate(query)
                print(f"Response: {response}")
                print("✓ Success")
            except Exception as e:
                print(f"❌ Error: {e}")
                return 1

        print("\n" + "=" * 70)
        print("✅ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY")
        print("=" * 70)

        # Show transcript summary
        transcript = session.transcript
        tool_calls_count = sum(1 for entry in transcript if entry.get('type') == 'tool_calls')
        print(f"\nTotal tool calls made: {tool_calls_count}")
        print("\nTranscript summary:")
        for entry in transcript[-10:]:  # Show last 10 entries
            entry_type = entry.get('type', 'unknown')
            if entry_type == 'tool_calls':
                calls = entry.get('tool_calls', [])
                print(f"  - Tool calls: {len(calls)} call(s)")
            elif entry_type in ('instructions', 'prompt', 'response'):
                content = entry.get('content', '')[:50]
                print(f"  - {entry_type}: {content}...")
            else:
                print(f"  - {entry_type}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
