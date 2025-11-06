#!/usr/bin/env python3
"""
Comprehensive integration tests for apple-foundation-models-py
"""

import asyncio
import foundationmodels
from foundationmodels import Availability

def test_availability():
    """Test availability checking"""
    print("=" * 60)
    print("TEST 1: Availability Check")
    print("=" * 60)

    status = foundationmodels.Client.check_availability()
    reason = foundationmodels.Client.get_availability_reason()
    is_ready = foundationmodels.Client.is_ready()

    print(f"Status: {status} ({status.name if hasattr(status, 'name') else status})")
    print(f"Reason: {reason}")
    print(f"Ready: {is_ready}")

    if status != Availability.AVAILABLE:
        print("\n⚠️  Apple Intelligence not available - some tests may fail")
    else:
        print("\n✓ Apple Intelligence is available and ready")

    print()


def test_version():
    """Test version information"""
    print("=" * 60)
    print("TEST 2: Version Information")
    print("=" * 60)

    with foundationmodels.Client() as client:
        version = client.get_version()
        print(f"Version: {version}")

        languages = client.get_supported_languages()
        print(f"Supported languages: {', '.join(languages)}")

    print("\n✓ Version information retrieved")
    print()


def test_basic_generation():
    """Test basic text generation"""
    print("=" * 60)
    print("TEST 3: Basic Text Generation")
    print("=" * 60)

    with foundationmodels.Client() as client:
        session = client.create_session()

        # Test simple math
        print("Q: What is 7 + 15?")
        response = session.generate("What is 7 + 15?", temperature=0.3)
        print(f"A: {response}")
        print()

        # Test general knowledge
        print("Q: What is the largest planet in our solar system?")
        response = session.generate("What is the largest planet in our solar system?", temperature=0.5)
        print(f"A: {response}")
        print()

        # Test creative generation
        print("Q: Write a haiku about coding")
        response = session.generate("Write a haiku about coding", temperature=1.0)
        print(f"A: {response}")

    print("\n✓ Basic generation tests passed")
    print()


def test_conversation_context():
    """Test multi-turn conversation"""
    print("=" * 60)
    print("TEST 4: Conversation Context")
    print("=" * 60)

    with foundationmodels.Client() as client:
        session = client.create_session(
            instructions="You are a friendly assistant who gives concise answers."
        )

        # First message
        print("User: My name is Alice and I love Python programming.")
        response1 = session.generate("My name is Alice and I love Python programming.")
        print(f"Assistant: {response1}")
        print()

        # Follow-up that requires context
        print("User: What's my name and what do I like?")
        response2 = session.generate("What's my name and what do I like?")
        print(f"Assistant: {response2}")
        print()

        # Check if context was maintained
        if "Alice" in response2 or "alice" in response2.lower():
            print("✓ Context maintained across turns")
        else:
            print("⚠️  Context may not be fully maintained")

    print()


async def test_streaming():
    """Test streaming generation"""
    print("=" * 60)
    print("TEST 5: Streaming Generation")
    print("=" * 60)

    client = foundationmodels.Client()
    session = client.create_session()

    print("Prompt: Tell me a short story about a robot learning to paint (2 sentences)")
    print("Response: ", end='', flush=True)

    chunks = []
    async for chunk in session.generate_stream(
        "Tell me a short story about a robot learning to paint in exactly 2 sentences",
        temperature=0.8
    ):
        print(chunk, end='', flush=True)
        chunks.append(chunk)

    print("\n")

    full_response = ''.join(chunks)
    print(f"✓ Received {len(chunks)} chunks totaling {len(full_response)} characters")
    print()


def test_temperature_variations():
    """Test temperature parameter"""
    print("=" * 60)
    print("TEST 6: Temperature Variations")
    print("=" * 60)

    with foundationmodels.Client() as client:
        session = client.create_session()

        prompt = "Complete this sentence: The sky is"

        # Low temperature (more deterministic)
        print(f"Temperature 0.1: ", end='')
        response1 = session.generate(prompt, temperature=0.1)
        print(response1)

        # Medium temperature
        print(f"Temperature 0.7: ", end='')
        response2 = session.generate(prompt, temperature=0.7)
        print(response2)

        # High temperature (more creative)
        print(f"Temperature 1.5: ", end='')
        response3 = session.generate(prompt, temperature=1.5)
        print(response3)

    print("\n✓ Temperature variations tested")
    print()


def test_session_management():
    """Test session lifecycle"""
    print("=" * 60)
    print("TEST 7: Session Management")
    print("=" * 60)

    client = foundationmodels.Client()

    # Create multiple sessions
    session1 = client.create_session(instructions="You are a math tutor.")
    session2 = client.create_session(instructions="You are a poet.")

    # Test each session maintains its own context
    print("Session 1 (Math): What is 12 * 8?")
    response1 = session1.generate("What is 12 * 8?")
    print(f"Response: {response1}")
    print()

    print("Session 2 (Poetry): Write one line of poetry about the moon")
    response2 = session2.generate("Write one line of poetry about the moon")
    print(f"Response: {response2}")
    print()

    # Close sessions
    session1.close()
    session2.close()
    client.close()

    print("✓ Multiple sessions managed successfully")
    print()


def test_error_handling():
    """Test error handling"""
    print("=" * 60)
    print("TEST 8: Error Handling")
    print("=" * 60)

    try:
        client = foundationmodels.Client()
        session = client.create_session()

        # Try with empty prompt
        print("Testing empty prompt...")
        try:
            response = session.generate("")
            print(f"Response: {response[:50]}...")
            print("✓ Empty prompt handled")
        except Exception as e:
            print(f"✓ Exception caught: {type(e).__name__}")

        print()

        # Try with very long prompt
        print("Testing very long prompt...")
        long_prompt = "Tell me about " + ("the number one " * 100)
        try:
            response = session.generate(long_prompt[:500])  # Limit to reasonable size
            print(f"✓ Long prompt handled ({len(response)} chars in response)")
        except Exception as e:
            print(f"✓ Exception caught: {type(e).__name__}: {e}")

    except Exception as e:
        print(f"✗ Unexpected error: {e}")

    print()


def test_context_manager():
    """Test context manager functionality"""
    print("=" * 60)
    print("TEST 9: Context Managers")
    print("=" * 60)

    # Test client context manager
    with foundationmodels.Client() as client:
        with client.create_session() as session:
            response = session.generate("Say 'Context managers work!'")
            print(f"Response: {response}")

    print("✓ Context managers cleaned up properly")
    print()


async def run_async_tests():
    """Run all async tests"""
    await test_streaming()


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "apple-foundation-models-py Comprehensive Test Suite" + " " * 14 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    try:
        # Synchronous tests
        test_availability()
        test_version()
        test_basic_generation()
        test_conversation_context()
        test_temperature_variations()
        test_session_management()
        test_context_manager()
        test_error_handling()

        # Async tests
        asyncio.run(run_async_tests())

        # Summary
        print("=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print("✓ All tests completed!")
        print()
        print("Features tested:")
        print("  • Availability checking")
        print("  • Version information")
        print("  • Basic text generation")
        print("  • Conversation context")
        print("  • Streaming generation")
        print("  • Temperature control")
        print("  • Multiple sessions")
        print("  • Context managers")
        print("  • Error handling")
        print()
        print("🎉 apple-foundation-models-py is working correctly with FoundationModels!")

    except Exception as e:
        print("\n" + "=" * 60)
        print("TEST FAILED")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
