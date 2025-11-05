#!/usr/bin/env python3
"""
Structured output example using libai-py.

Demonstrates:
- JSON schema validation
- Structured data extraction
- Type-safe responses
"""

import json
from libai import Client, Availability


def main():
    # Check availability
    if Client.check_availability() != Availability.AVAILABLE:
        print("Apple Intelligence not available")
        return

    with Client() as client:
        session = client.create_session()

        # Example 1: Extract person information
        print("Example 1: Person Information Extraction")
        print("=" * 60)

        person_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "city": {"type": "string"},
                "occupation": {"type": "string"}
            },
            "required": ["name", "age", "city"]
        }

        result = session.generate_structured(
            "Extract person info: Alice is 28 years old, lives in Paris, and works as a software engineer.",
            schema=person_schema
        )

        print("Extracted data:")
        print(json.dumps(result['object'], indent=2))
        print()

        # Example 2: Extract list of items
        print("Example 2: Shopping List Extraction")
        print("=" * 60)

        shopping_schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "quantity": {"type": "integer"},
                            "category": {"type": "string"}
                        }
                    }
                }
            }
        }

        result = session.generate_structured(
            "I need to buy: 2 apples, 1 loaf of bread, 3 bottles of milk, and 1 box of eggs.",
            schema=shopping_schema
        )

        print("Shopping list:")
        for item in result['object']['items']:
            print(f"  - {item['quantity']} x {item['name']} ({item.get('category', 'N/A')})")
        print()

        # Example 3: Sentiment analysis
        print("Example 3: Sentiment Analysis")
        print("=" * 60)

        sentiment_schema = {
            "type": "object",
            "properties": {
                "sentiment": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral"]
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["sentiment", "confidence"]
        }

        texts = [
            "This product is amazing! I love it!",
            "Terrible experience, would not recommend.",
            "It's okay, nothing special."
        ]

        for text in texts:
            result = session.generate_structured(
                f"Analyze sentiment of: {text}",
                schema=sentiment_schema
            )
            data = result['object']
            print(f"Text: {text}")
            print(f"  Sentiment: {data['sentiment']} ({data['confidence']:.2f} confidence)")
            print(f"  Keywords: {', '.join(data.get('keywords', []))}")
            print()


if __name__ == "__main__":
    main()
