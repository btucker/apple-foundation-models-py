"""
Pydantic compatibility utilities for applefoundationmodels.

Provides optional Pydantic integration for structured output generation.
"""

from typing import Any, Dict, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import BaseModel

# Try to import Pydantic, but don't fail if it's not installed
try:
    from pydantic import BaseModel

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = None  # type: ignore


def require_pydantic() -> None:
    """
    Raise ImportError if Pydantic is not available.

    Raises:
        ImportError: If Pydantic is not installed
    """
    if not PYDANTIC_AVAILABLE:
        raise ImportError(
            "Pydantic is not installed. Install it with: pip install pydantic>=2.0"
        )


def model_to_schema(model: "BaseModel") -> Dict[str, Any]:
    """
    Convert a Pydantic model to JSON Schema.

    Args:
        model: Pydantic BaseModel class or instance

    Returns:
        JSON Schema dictionary

    Raises:
        ImportError: If Pydantic is not installed
        ValueError: If model is not a valid Pydantic model

    Example:
        >>> from pydantic import BaseModel
        >>> class Person(BaseModel):
        ...     name: str
        ...     age: int
        >>> schema = model_to_schema(Person)
        >>> print(schema)
        {'type': 'object', 'properties': {...}, 'required': [...]}
    """
    require_pydantic()

    if not hasattr(model, "model_json_schema"):
        raise ValueError(
            f"Expected Pydantic BaseModel, got {type(model).__name__}. "
            "Make sure your model inherits from pydantic.BaseModel"
        )

    # Get JSON Schema from Pydantic model
    schema = model.model_json_schema()

    # Clean up schema (remove title, $defs if empty, etc.)
    schema = _clean_schema(schema)

    return schema


def _clean_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean up Pydantic-generated schema for better compatibility.

    Removes unnecessary fields like 'title' and simplifies nested definitions.
    """
    cleaned = schema.copy()

    # Remove title if present
    cleaned.pop("title", None)

    # Remove $defs if empty or inline them if simple
    if "$defs" in cleaned:
        defs = cleaned["$defs"]
        if not defs:
            del cleaned["$defs"]

    return cleaned


def is_pydantic_model(obj: Any) -> bool:
    """
    Check if an object is a Pydantic model class or instance.

    Args:
        obj: Object to check

    Returns:
        True if obj is a Pydantic BaseModel class or instance
    """
    if not PYDANTIC_AVAILABLE:
        return False

    # Check if it has the Pydantic model_json_schema method
    # This works for both classes and instances
    if hasattr(obj, "model_json_schema"):
        return True

    return False


def normalize_schema(schema: Union[Dict[str, Any], "BaseModel"]) -> Dict[str, Any]:
    """
    Normalize schema input to JSON Schema dict.

    Accepts either a JSON Schema dictionary or a Pydantic model,
    and returns a JSON Schema dictionary.

    Args:
        schema: JSON Schema dict or Pydantic BaseModel

    Returns:
        JSON Schema dictionary

    Raises:
        TypeError: If schema is neither dict nor Pydantic model
        ImportError: If Pydantic is needed but not installed
    """
    # If it's already a dict, return it
    if isinstance(schema, dict):
        return schema

    # If it's a Pydantic model, convert it
    if is_pydantic_model(schema):
        return model_to_schema(schema)

    # Otherwise, raise an error
    raise TypeError(
        f"Expected JSON Schema dict or Pydantic BaseModel, got {type(schema).__name__}"
    )
