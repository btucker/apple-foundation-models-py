#!/usr/bin/env python3
"""
Basic chat example.

Demonstrates:
- Creating a session
- Synchronous text generation
- Conversation history
- Error handling
"""

from applefoundationmodels import Session
from utils import check_availability_or_exit


def main():
    # Check availability
    if not check_availability_or_exit():
        return

    print(f"Version: {Session.get_version()}")
    print()

    # Create a session with instructions
    with Session(instructions="You are a helpful assistant. Be concise.") as session:
        # Have a conversation
        questions = [
            "What is the capital of France?",
            "What is it famous for?",
            "How many people live there?",
        ]

        for question in questions:
            print(f"User: {question}")

            # Generate response
            response = session.generate(question, temperature=0.7)

            print(f"Assistant: {response.text}")
            print()

        # Show conversation history
        print("=" * 60)
        print("Conversation History:")
        print("=" * 60)
        history = session.get_history()
        for msg in history:
            role = msg["role"].capitalize()
            content = msg["content"]
            print(f"{role}: {content[:100]}...")
            print()


if __name__ == "__main__":
    main()
