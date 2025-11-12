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

from applefoundationmodels import Session


def main():
    print("=" * 70)
    print("TOOL CALLING DEMONSTRATION")
    print("=" * 70)
    print()

    # Define all tools before creating the session

    # Example 1: Tool with no parameters
    def get_time() -> str:
        """Get the current time."""
        return "2:30 PM"

    # Example 2: Tool with single string parameter
    def get_weather(location: str) -> str:
        """
        Get weather information for a specific location.

        Args:
            location: City or location name
        """
        return f"Weather in {location}: 72°F, sunny"

    # Example 3: Tool with multiple string parameters
    def search_docs(query: str, category: str) -> str:
        """
        Search the documentation database.

        Args:
            query: Search query string
            category: Documentation category to search in
        """
        return f"Found 5 documents about '{query}' in {category} category"

    # Example 4: Tool with mixed parameter types
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

    # Create session with all tools registered
    with Session(
        instructions="You are a helpful assistant. Use the provided tools when you need information.",
        tools=[get_time, get_weather, search_docs, get_top_items, calculate],
    ) as session:
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
                print(f"Response: {response.text}")
                print("✓ Success")
            except Exception as e:
                print(f"❌ Error: {e}")
                return 1

        print("\n" + "=" * 70)
        print("✅ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY")
        print("=" * 70)

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
