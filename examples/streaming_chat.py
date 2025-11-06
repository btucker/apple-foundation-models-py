#!/usr/bin/env python3
"""
Async streaming chat example using libai-py.

Demonstrates:
- Async streaming with async/await
- Real-time response display
- Stream handling
"""

import asyncio
from applefoundationmodels import Client
from utils import check_availability_or_exit, print_stats


async def stream_question(session, question: str):
    """Ask a question and stream the response."""
    print(f"\nUser: {question}")
    print("Assistant: ", end="", flush=True)

    # Stream response chunks
    async for chunk in session.generate_stream(question, temperature=0.8):
        print(chunk, end="", flush=True)

    print()  # Newline after stream completes


async def main():
    # Check availability
    if not check_availability_or_exit(verbose=False):
        return

    # Create client and session
    with Client() as client:
        session = client.create_session(
            instructions="You are a creative storyteller. Write engaging narratives."
        )

        # Stream multiple questions
        questions = [
            "Tell me a short story about a brave robot.",
            "What happened next?",
            "How does the story end?",
        ]

        for question in questions:
            await stream_question(session, question)
            # Small delay between questions
            await asyncio.sleep(0.5)

        # Show final statistics
        print_stats(client, verbose=False)


if __name__ == "__main__":
    asyncio.run(main())
