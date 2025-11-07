"""
Tool calling utilities for applefoundationmodels.

Provides utilities for extracting JSON schemas from Python functions
and managing tool registrations.
"""

import inspect
from typing import Callable, Dict, Any, Optional, get_type_hints, get_origin, get_args
from .exceptions import ToolCallError


def python_type_to_json_schema(python_type: Any) -> Dict[str, Any]:
    """
    Convert a Python type hint to a JSON Schema type definition.

    Args:
        python_type: Python type hint

    Returns:
        JSON Schema type definition

    Raises:
        ToolCallError: If type cannot be converted
    """
    # Handle None type
    if python_type is type(None):
        return {"type": "null"}

    # Get the origin type for generic types (e.g., List, Dict, Optional)
    origin = get_origin(python_type)

    # Handle Optional[X] as union of X and null
    if origin is type(None) or (
        hasattr(python_type, "__origin__") and python_type.__origin__ is type(None)
    ):
        return {"type": "null"}

    # Handle basic types
    if python_type is str or python_type == "str":
        return {"type": "string"}
    elif python_type is int or python_type == "int":
        return {"type": "integer"}
    elif python_type is float or python_type == "float":
        return {"type": "number"}
    elif python_type is bool or python_type == "bool":
        return {"type": "boolean"}
    elif python_type is list or origin is list:
        # Get the item type if specified
        args = get_args(python_type)
        if args:
            items_schema = python_type_to_json_schema(args[0])
            return {"type": "array", "items": items_schema}
        return {"type": "array"}
    elif python_type is dict or origin is dict:
        # Get key/value types if specified
        args = get_args(python_type)
        if len(args) == 2:
            # For Dict[str, X], we can specify value type
            if args[0] is str:
                value_schema = python_type_to_json_schema(args[1])
                return {"type": "object", "additionalProperties": value_schema}
        return {"type": "object"}
    elif origin is not None:
        # Handle other generic types - try to extract the origin
        if origin is list:
            return {"type": "array"}
        elif origin is dict:
            return {"type": "object"}

    # If we can't determine the type, default to string
    return {"type": "string"}


def extract_function_schema(func: Callable) -> Dict[str, Any]:
    """
    Extract JSON Schema from a Python function's signature and docstring.

    Args:
        func: Python function to extract schema from

    Returns:
        Dictionary containing:
        - name: Function name
        - description: Function description from docstring
        - parameters: JSON Schema for function parameters

    Raises:
        ToolCallError: If schema cannot be extracted
    """
    try:
        # Get function signature
        sig = inspect.signature(func)

        # Get type hints
        try:
            type_hints = get_type_hints(func)
        except Exception:
            # If type hints fail (e.g., forward references), inspect parameters directly
            type_hints = {}

        # Extract parameter schemas
        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            # Skip self and cls parameters
            if param_name in ("self", "cls"):
                continue

            # Get type annotation
            param_type = type_hints.get(param_name, param.annotation)

            # Handle parameters without type hints
            if param_type is inspect.Parameter.empty:
                # Default to string type if no annotation
                param_schema = {"type": "string"}
            else:
                param_schema = python_type_to_json_schema(param_type)

            properties[param_name] = param_schema

            # Mark as required if no default value
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        # Build parameters schema
        parameters_schema = {
            "type": "object",
            "properties": properties,
        }

        if required:
            parameters_schema["required"] = required

        # Extract description from docstring
        description = ""
        if func.__doc__:
            # Get the first line or paragraph of the docstring
            lines = func.__doc__.strip().split("\n")
            description = lines[0].strip()

        # Get function name
        name = func.__name__

        return {
            "name": name,
            "description": description,
            "parameters": parameters_schema,
        }

    except Exception as e:
        raise ToolCallError(
            f"Failed to extract schema from function '{func.__name__}': {e}"
        )


def tool(
    description: Optional[str] = None,
    name: Optional[str] = None,
) -> Callable:
    """
    Decorator to mark a function as a tool and attach metadata.

    Args:
        description: Optional tool description (uses docstring if not provided)
        name: Optional tool name (uses function name if not provided)

    Returns:
        Decorated function with tool metadata attached

    Example:
        @tool(description="Get the current weather")
        def get_weather(location: str) -> str:
            return f"Weather in {location}: sunny"
    """

    def decorator(func: Callable) -> Callable:
        # Extract schema
        schema = extract_function_schema(func)

        # Override with provided values
        if description is not None:
            schema["description"] = description
        if name is not None:
            schema["name"] = name

        # Attach metadata to function
        func._tool_name = schema["name"]
        func._tool_description = schema["description"]
        func._tool_parameters = schema["parameters"]

        return func

    return decorator
