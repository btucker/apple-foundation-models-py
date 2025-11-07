"""
Tool calling utilities for applefoundationmodels.

Provides utilities for extracting JSON schemas from Python functions
and managing tool registrations.
"""

import inspect
from typing import (
    Callable,
    Dict,
    Any,
    Optional,
    get_type_hints,
    get_origin,
    get_args,
    Union,
)
from .exceptions import ToolCallError


# Type mapping tables for efficient schema generation
_BASIC_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}

_STRING_TYPE_MAP = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
}


def is_optional_type(python_type: Any) -> bool:
    """
    Check if a type is Optional (Union[X, None]).

    Args:
        python_type: Python type hint

    Returns:
        True if the type is Optional, False otherwise
    """
    origin = get_origin(python_type)
    if origin is Union:
        args = get_args(python_type)
        return type(None) in args
    return False


def unwrap_optional(python_type: Any) -> Any:
    """
    Extract the non-None type from Optional[X] or Union[X, None].

    Args:
        python_type: Python type hint that may be Optional

    Returns:
        The unwrapped type (X from Optional[X])
    """
    origin = get_origin(python_type)
    if origin is Union:
        args = get_args(python_type)
        # Filter out None and return the first non-None type
        non_none_types = [arg for arg in args if arg is not type(None)]
        if non_none_types:
            return non_none_types[0]
    return python_type


def _handle_list_type(python_type: Any) -> Dict[str, Any]:
    """
    Extract list/array type with optional item schema.

    Args:
        python_type: List or generic list type hint

    Returns:
        JSON Schema for array type
    """
    args = get_args(python_type)
    if args:
        items_schema = python_type_to_json_schema(args[0])
        return {"type": "array", "items": items_schema}
    return {"type": "array"}


def _handle_dict_type(python_type: Any) -> Dict[str, Any]:
    """
    Extract dict/object type with optional value schema.

    Args:
        python_type: Dict or generic dict type hint

    Returns:
        JSON Schema for object type
    """
    args = get_args(python_type)
    if len(args) == 2 and args[0] is str:
        value_schema = python_type_to_json_schema(args[1])
        return {"type": "object", "additionalProperties": value_schema}
    return {"type": "object"}


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

    # Handle Optional[X] / Union[X, None] - unwrap and process the inner type
    if is_optional_type(python_type):
        inner_type = unwrap_optional(python_type)
        return python_type_to_json_schema(inner_type)

    # Check basic types via lookup table
    if python_type in _BASIC_TYPE_MAP:
        return {"type": _BASIC_TYPE_MAP[python_type]}

    # Check string type annotations
    if isinstance(python_type, str) and python_type in _STRING_TYPE_MAP:
        return {"type": _STRING_TYPE_MAP[python_type]}

    # Get origin for generic types
    origin = get_origin(python_type)

    # Handle container types
    if python_type is list or origin is list:
        return _handle_list_type(python_type)

    if python_type is dict or origin is dict:
        return _handle_dict_type(python_type)

    # Default fallback for unknown types
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
        ) from e


def attach_tool_metadata(
    func: Callable,
    schema: Dict[str, Any],
    description: Optional[str] = None,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Attach tool metadata to a function and return final schema.

    This is a shared helper used by both the standalone @tool decorator
    and the Session.tool() method to avoid code duplication.

    Args:
        func: Function to attach metadata to
        schema: Base schema from extract_function_schema
        description: Optional override for description
        name: Optional override for name

    Returns:
        Final schema with overrides applied
    """
    # Override with provided values
    if description is not None:
        schema["description"] = description
    if name is not None:
        schema["name"] = name

    # Attach metadata to function
    func._tool_name = schema["name"]  # type: ignore[attr-defined]
    func._tool_description = schema["description"]  # type: ignore[attr-defined]
    func._tool_parameters = schema["parameters"]  # type: ignore[attr-defined]

    return schema


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

    Note:
        Tool output size limits:
        - Initial buffer: 16KB
        - Maximum size: 1MB (automatically retried with larger buffers)
        - Tools returning outputs larger than 1MB will raise an error
        - For large outputs, consider returning references or summaries

    Example:
        @tool(description="Get the current weather")
        def get_weather(location: str) -> str:
            return f"Weather in {location}: sunny"
    """

    def decorator(func: Callable) -> Callable:
        # Extract schema and attach metadata
        schema = extract_function_schema(func)
        attach_tool_metadata(func, schema, description, name)
        return func

    return decorator
