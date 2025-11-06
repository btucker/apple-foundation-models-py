#!/usr/bin/env python3
"""
Basic chat example using libai-py.

Demonstrates:
- Creating a client and session
- Synchronous text generation
- Conversation history
- Error handling
"""

from applefoundationmodels import Client, NotAvailableError, Availability


def main():
    try:
        # Check availability before creating client
        status = Client.check_availability()
        if status != Availability.AVAILABLE:
            print("Apple Intelligence is not available:")
            print(Client.get_availability_reason())
            return

        print(f"libai version: {Client.get_version()}")
        print(f"Supported languages: {', '.join(Client.get_supported_languages()[:5])}")
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
            stats = client.get_stats()
            print("=" * 60)
            print("Statistics:")
            print("=" * 60)
            print(f"Total requests: {stats['total_requests']}")
            print(f"Successful: {stats['successful_requests']}")
            print(f"Failed: {stats['failed_requests']}")
            print(f"Avg response time: {stats['average_response_time']:.2f}s")

    except NotAvailableError as e:
        print(f"Error: {e.message}")
        print("Make sure Apple Intelligence is enabled in System Settings")
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
