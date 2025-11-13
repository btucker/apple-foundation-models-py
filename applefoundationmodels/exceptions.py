"""
Exception hierarchy for applefoundationmodels Python bindings.

Exceptions are generated from a single error_codes.json definition so both
Python and Swift share the same source of truth for error metadata.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import Any, Dict, List, Optional, Tuple, Type

__all__ = [
    "FoundationModelsError",
    "GenerationError",
    "ERROR_CODE_TO_EXCEPTION",
    "raise_for_error_code",
    "get_error_definitions",
]


class FoundationModelsError(Exception):
    """Base exception for all FoundationModels errors."""

    def __init__(self, message: str, error_code: Optional[int] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code


class GenerationError(FoundationModelsError):
    """Text generation error."""

    pass


@dataclass(frozen=True)
class ErrorDefinition:
    """Structured representation of an error entry from the JSON definition."""

    code: int
    name: str
    parent: str
    description: str
    swift_case: Optional[str]


def _load_error_definitions() -> Tuple[ErrorDefinition, ...]:
    data_path = resources.files(__package__).joinpath("error_codes.json")
    with data_path.open("r", encoding="utf-8") as f:
        raw: List[Dict[str, Any]] = json.load(f)

    definitions: List[ErrorDefinition] = []
    for entry in raw:
        definitions.append(
            ErrorDefinition(
                code=int(entry["code"]),
                name=str(entry["name"]),
                parent=str(entry.get("parent", "FoundationModelsError")),
                description=str(entry["description"]),
                swift_case=entry.get("swift_case") or None,
            )
        )
    return tuple(definitions)


_ERROR_DEFINITIONS = _load_error_definitions()

_PARENT_CLASS_MAP: Dict[str, Type[FoundationModelsError]] = {
    "FoundationModelsError": FoundationModelsError,
    "GenerationError": GenerationError,
}


def get_error_definitions() -> Tuple[ErrorDefinition, ...]:
    """Return the error definitions used to build the exception hierarchy."""

    return _ERROR_DEFINITIONS


ERROR_CODE_TO_EXCEPTION: Dict[int, Type[FoundationModelsError]] = {}


def _register_exception_classes() -> None:
    for definition in _ERROR_DEFINITIONS:
        parent_class = _PARENT_CLASS_MAP.get(definition.parent, FoundationModelsError)

        if definition.name == "GenerationError":
            ERROR_CODE_TO_EXCEPTION[definition.code] = GenerationError
            continue

        exception_class = type(
            definition.name,
            (parent_class,),
            {"__doc__": definition.description},
        )
        globals()[definition.name] = exception_class
        __all__.append(definition.name)
        ERROR_CODE_TO_EXCEPTION[definition.code] = exception_class


_register_exception_classes()


def raise_for_error_code(error_code: int, message: str) -> None:
    """
    Raise the appropriate exception for a given error code.

    Args:
        error_code: The error code from the Swift API
        message: Error message to include in the exception

    Raises:
        FoundationModelsError: The appropriate exception subclass for the error code
    """

    exception_class = ERROR_CODE_TO_EXCEPTION.get(
        error_code, ERROR_CODE_TO_EXCEPTION.get(-99, FoundationModelsError)
    )
    raise exception_class(message, error_code)
