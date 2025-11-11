#!/usr/bin/env python3
"""
Basic chat example using libai-py.

Demonstrates:
- Creating a client and session
- Synchronous text generation
- Conversation history
- Error handling
"""

from applefoundationmodels import Client, NotAvailableError
from utils import check_availability_or_exit, print_stats, handle_example_error


def main():
    try:
        # Check availability before creating client
        if not check_availability_or_exit():
            return

        print(f"libai version: {Client.get_version()}")
        print()

        # Create client
        with Client() as client:
            # Create a session with instructions
            session = client.create_session(
                instructions="You are a helpful assistant. Be concise.",
                enable_guardrails=True,
            )

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

                print(f"Assistant: {response}")
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

            # Show statistics
            print_stats(client, verbose=True)

    except NotAvailableError as e:
        handle_example_error(e)
    except Exception as e:
        handle_example_error(e)


if __name__ == "__main__":
    main()
