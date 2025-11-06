#!/usr/bin/env python3
"""Quick test for structured output debugging."""

from applefoundationmodels import Client

with Client() as client:
    if not client.is_ready():
        print("Apple Intelligence not available")
        print(client.get_availability_reason())
        exit(1)

    session = client.create_session()

    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name", "age"]
    }

    try:
        result = session.generate_structured(
            "Extract: Alice is 28",
            schema=schema
        )
        print(f"Success! Result: {result}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
