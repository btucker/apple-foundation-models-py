#!/usr/bin/env python3
"""
Basic async chat example.

Demonstrates:
- Creating an async session
- Async text generation without streaming
- Conversation history in async context
- Error handling with async/await
"""

import asyncio
from applefoundationmodels import AsyncSession
from utils import check_availability_or_exit


async def main():
    # Check availability
    if not check_availability_or_exit():
        return

    # Create an async session with instructions
    async with AsyncSession(
        instructions="You are a helpful assistant. Be concise."
    ) as session:
        # Have a conversation
        questions = [
            "What is the capital of France?",
            "What is it famous for?",
            "How many people live there?",
        ]

        for question in questions:
            print(f"User: {question}")

            # Generate response using await (not streaming)
            response = await session.generate(question, temperature=0.7)

            print(f"Assistant: {response.text}")
            print()

        # Show conversation history
        print("=" * 60)
        print("Conversation History:")
        print("=" * 60)
        history = await session.get_history()
        for msg in history:
            role = msg["role"].capitalize()
            content = msg["content"]
            print(f"{role}: {content[:100]}...")
            print()


if __name__ == "__main__":
    asyncio.run(main())
