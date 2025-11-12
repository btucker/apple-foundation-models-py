#!/usr/bin/env python3
"""
Async streaming chat example.

Demonstrates:
- Async streaming with async/await
- Real-time response display
- Stream handling
"""

import asyncio
from applefoundationmodels import AsyncSession
from utils import check_availability_or_exit


async def stream_question(session, question: str):
    """Ask a question and stream the response."""
    print(f"\nUser: {question}")
    print("Assistant: ", end="", flush=True)

    # Stream response chunks
    async for chunk in session.generate(question, temperature=0.8, stream=True):
        print(chunk.content, end="", flush=True)

    print()  # Newline after stream completes


async def main():
    # Check availability
    if not check_availability_or_exit(verbose=False):
        return

    # Create async session
    async with AsyncSession(
        instructions="You are a creative storyteller. Write engaging narratives."
    ) as session:
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


if __name__ == "__main__":
    asyncio.run(main())
