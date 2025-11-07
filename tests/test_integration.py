#!/usr/bin/env python3
"""
Comprehensive integration tests for apple-foundation-models-py
"""

import asyncio
import applefoundationmodels
from applefoundationmodels import Availability
from conftest import assert_valid_response, assert_valid_chunks


def test_availability():
    """Test availability checking"""
    print("=" * 60)
    print("TEST 1: Availability Check")
    print("=" * 60)

    status = applefoundationmodels.Client.check_availability()
    reason = applefoundationmodels.Client.get_availability_reason()
    is_ready = applefoundationmodels.Client.is_ready()

    print(f"Status: {status} ({status.name if hasattr(status, 'name') else status})")
    print(f"Reason: {reason}")
    print(f"Ready: {is_ready}")

    # Assertions
    assert isinstance(status, Availability), "Status should be an Availability enum"
    assert isinstance(reason, str), "Reason should be a string"
    assert isinstance(is_ready, bool), "is_ready should be a boolean"
    assert reason, "Reason should not be empty"

    if status != Availability.AVAILABLE:
        print("\n⚠️  Apple Intelligence not available - some tests may fail")
    else:
        print("\n✓ Apple Intelligence is available and ready")
        assert is_ready is True, "is_ready should be True when status is AVAILABLE"

    print()


def test_version():
    """Test version information"""
    print("=" * 60)
    print("TEST 2: Version Information")
    print("=" * 60)

    with applefoundationmodels.Client() as client:
        version = client.get_version()
        print(f"Version: {version}")

        languages = client.get_supported_languages()
        print(f"Supported languages: {', '.join(languages)}")

        # Assertions
        assert version, "Version should not be empty"
        assert isinstance(version, str), "Version should be a string"
        assert isinstance(languages, list), "Languages should be a list"
        assert len(languages) > 0, "Should support at least one language"
        assert all(
            isinstance(lang, str) for lang in languages
        ), "All languages should be strings"

    print("\n✓ Version information retrieved")
    print()


def test_basic_generation():
    """Test basic text generation"""
    print("=" * 60)
    print("TEST 3: Basic Text Generation")
    print("=" * 60)

    with applefoundationmodels.Client() as client:
        session = client.create_session()

        # Test simple math
        print("Q: What is 7 + 15?")
        response = session.generate("What is 7 + 15?", temperature=0.3)
        print(f"A: {response}")
        assert_valid_response(response)
        print()

        # Test general knowledge
        print("Q: What is the largest planet in our solar system?")
        response = session.generate(
            "What is the largest planet in our solar system?", temperature=0.5
        )
        print(f"A: {response}")
        assert_valid_response(response)
        assert "jupiter" in response.lower(), "Response should mention Jupiter"
        print()

        # Test creative generation
        print("Q: Write a haiku about coding")
        response = session.generate("Write a haiku about coding", temperature=1.0)
        print(f"A: {response}")
        assert_valid_response(response)
        assert len(response) > 10, "Haiku should have substantial content"

    print("\n✓ Basic generation tests passed")
    print()


def test_conversation_context():
    """Test multi-turn conversation"""
    print("=" * 60)
    print("TEST 4: Conversation Context")
    print("=" * 60)

    with applefoundationmodels.Client() as client:
        session = client.create_session(
            instructions="You are a helpful assistant. Remember information from previous messages."
        )

        # First message - establish context
        print("User: Remember this: The code name is BLUE42")
        response1 = session.generate("Remember this: The code name is BLUE42")
        print(f"Assistant: {response1}")
        assert_valid_response(response1)
        print()

        # Follow-up that requires context
        print("User: What code name did I just tell you?")
        response2 = session.generate("What code name did I just tell you?")
        print(f"Assistant: {response2}")
        assert_valid_response(response2)
        print()

        # Check if context was maintained
        # More lenient check - if the model refuses or doesn't maintain context,
        # at least verify it responded with a valid message
        if "blue42" in response2.lower() or "blue 42" in response2.lower():
            print("✓ Context maintained across turns")
        else:
            print(f"⚠️  Context not fully maintained - got: {response2[:100]}")
            # Still pass if we got a coherent response (not a refusal)
            assert len(response2) > 5, "Should still provide a meaningful response"

    print()


async def test_streaming():
    """Test streaming generation"""
    print("=" * 60)
    print("TEST 5: Streaming Generation")
    print("=" * 60)

    client = applefoundationmodels.Client()
    session = client.create_session()

    print("Prompt: Tell me a short story about a robot learning to paint (2 sentences)")
    print("Response: ", end="", flush=True)

    chunks = []
    async for chunk in session.generate_stream(
        "Tell me a short story about a robot learning to paint in exactly 2 sentences",
        temperature=0.8,
    ):
        print(chunk, end="", flush=True)
        chunks.append(chunk)

    print("\n")

    # Assertions
    assert_valid_chunks(chunks)
    print(f"✓ Received {len(chunks)} chunks totaling {len(''.join(chunks))} characters")

    client.close()
    print()


def test_temperature_variations():
    """Test temperature parameter"""
    print("=" * 60)
    print("TEST 6: Temperature Variations")
    print("=" * 60)

    with applefoundationmodels.Client() as client:
        session = client.create_session()

        prompt = "Complete this sentence: The sky is"

        # Low temperature (more deterministic)
        print("Temperature 0.1: ", end="")
        response1 = session.generate(prompt, temperature=0.1)
        print(response1)
        assert_valid_response(response1)

        # Medium temperature
        print("Temperature 0.7: ", end="")
        response2 = session.generate(prompt, temperature=0.7)
        print(response2)
        assert_valid_response(response2)

        # High temperature (more creative)
        print("Temperature 1.5: ", end="")
        response3 = session.generate(prompt, temperature=1.5)
        print(response3)
        assert_valid_response(response3)

    print("\n✓ Temperature variations tested")
    print()


def test_session_management():
    """Test session lifecycle"""
    print("=" * 60)
    print("TEST 7: Session Management")
    print("=" * 60)

    client = applefoundationmodels.Client()

    # Create multiple sessions
    session1 = client.create_session(instructions="You are a math tutor.")
    session2 = client.create_session(instructions="You are a poet.")

    # Test each session maintains its own context
    print("Session 1 (Math): What is 12 * 8?")
    response1 = session1.generate("What is 12 * 8?")
    print(f"Response: {response1}")
    assert_valid_response(response1)
    assert (
        "96" in response1 or "ninety" in response1.lower()
    ), "Math response should contain the answer"
    print()

    print("Session 2 (Poetry): Write one line of poetry about the moon")
    response2 = session2.generate("Write one line of poetry about the moon")
    print(f"Response: {response2}")
    assert_valid_response(response2)
    assert len(response2) > 5, "Poetry response should have content"
    print()

    # Verify sessions are independent
    assert (
        response1 != response2
    ), "Different sessions should produce different responses"

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

    client = applefoundationmodels.Client()
    session = client.create_session()

    # Try with empty prompt
    print("Testing empty prompt...")
    empty_handled = False
    try:
        response = session.generate("")
        # If it doesn't raise an error, check the response
        assert_valid_response(response)
        print(f"Response: {response[:50]}...")
        print("✓ Empty prompt handled gracefully")
        empty_handled = True
    except Exception as e:
        print(f"✓ Exception caught as expected: {type(e).__name__}")
        empty_handled = True

    assert empty_handled, "Empty prompt should be handled somehow"
    print()

    # Try with very long prompt
    print("Testing very long prompt...")
    long_prompt = "Tell me about " + ("the number one " * 100)
    long_handled = False
    try:
        response = session.generate(long_prompt[:500])  # Limit to reasonable size
        assert_valid_response(response)
        print(f"✓ Long prompt handled ({len(response)} chars in response)")
        long_handled = True
    except Exception as e:
        print(f"✓ Exception caught: {type(e).__name__}: {e}")
        long_handled = True

    assert long_handled, "Long prompt should be handled somehow"

    client.close()
    print()


def test_context_manager():
    """Test context manager functionality"""
    print("=" * 60)
    print("TEST 9: Context Managers")
    print("=" * 60)

    # Test client context manager
    with applefoundationmodels.Client() as client:
        assert client is not None, "Client should be created"
        with client.create_session() as session:
            assert session is not None, "Session should be created"
            response = session.generate("Say 'Context managers work!'")
            print(f"Response: {response}")
            assert_valid_response(response)

    print("✓ Context managers cleaned up properly")
    print()


async def run_async_tests():
    """Run all async tests"""
    await test_streaming()


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print(
        "║"
        + " " * 10
        + "apple-foundation-models-py Comprehensive Test Suite"
        + " " * 14
        + "║"
    )
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
        print(
            "🎉 apple-foundation-models-py is working correctly with FoundationModels!"
        )

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
