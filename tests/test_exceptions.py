"""
Tests for exception handling and propagation.

Verifies that all error codes from the Swift layer properly propagate
through the FFI layer and raise the correct Python exception types.

Test Coverage:
1. TestErrorCodeMapping: Verifies error code → exception class mappings
2. TestRaiseForErrorCode: Tests raise_for_error_code() for all error codes
3. TestExceptionInheritance: Tests exception hierarchy and catch behavior
4. TestErrorJSONParsing: End-to-end tests for Swift → Cython → Python flow
   - Simulates the exact Cython error handling logic
   - Verifies error JSON format contract between Swift and Python
   - Tests edge cases (missing error_code, malformed JSON, etc.)

All exception types are auto-generated from error_codes.json, ensuring
Python and Swift share the same source of truth for error definitions.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import pytest

from applefoundationmodels import exceptions as exc_mod
from applefoundationmodels.exceptions import (
    ERROR_CODE_TO_EXCEPTION,
    FoundationModelsError,
    GenerationError,
    raise_for_error_code,
    get_error_definitions,
)


@dataclass(frozen=True)
class ErrorCase:
    code: int
    exception: type[FoundationModelsError]
    message: str
    swift_key: Optional[str] = None


ERROR_CASES: Tuple[ErrorCase, ...] = tuple(
    ErrorCase(
        definition.code,
        getattr(exc_mod, definition.name),
        definition.description,
        definition.swift_case,
    )
    for definition in get_error_definitions()
)

ERROR_CASES_BY_CODE = {case.code: case for case in ERROR_CASES}
GENERATION_ERROR_CODES = tuple(
    case.code for case in ERROR_CASES if issubclass(case.exception, GenerationError)
)


class TestErrorCodeMapping:
    """Test that error codes map to the correct exception classes."""

    def test_all_error_codes_mapped(self):
        """Verify all error codes have corresponding exception classes."""
        for case in ERROR_CASES:
            assert (
                ERROR_CODE_TO_EXCEPTION[case.code] == case.exception
            ), f"Error code {case.code} should map to {case.exception.__name__}"

    def test_generation_errors_inherit_from_generation_error(self):
        """Verify all generation-related errors inherit from GenerationError."""
        for code in GENERATION_ERROR_CODES:
            exception_class = ERROR_CODE_TO_EXCEPTION[code]
            assert issubclass(
                exception_class, GenerationError
            ), f"{exception_class.__name__} should inherit from GenerationError"

    def test_all_errors_inherit_from_base(self):
        """Verify all errors inherit from FoundationModelsError."""
        for exception_class in ERROR_CODE_TO_EXCEPTION.values():
            assert issubclass(
                exception_class, FoundationModelsError
            ), f"{exception_class.__name__} should inherit from FoundationModelsError"


class TestRaiseForErrorCode:
    """Test the raise_for_error_code function."""

    @pytest.mark.parametrize(
        "case",
        ERROR_CASES,
        ids=lambda case: f"{case.code}:{case.exception.__name__}",
    )
    def test_error_code_raises_correct_exception(self, case: ErrorCase):
        """Test that each error code raises the correct exception type."""
        with pytest.raises(case.exception) as exc_info:
            raise_for_error_code(case.code, case.message)
        assert case.message in str(exc_info.value)
        assert exc_info.value.error_code == case.code

    def test_unknown_error_code_defaults_to_unknown_error(self):
        """Test that unmapped error codes raise UnknownError."""
        unknown_case = ERROR_CASES_BY_CODE[-99]
        with pytest.raises(unknown_case.exception) as exc_info:
            raise_for_error_code(-999, "Unexpected error code")
        assert str(exc_info.value) == "Unexpected error code"


class TestExceptionInheritance:
    """Test exception inheritance for proper exception handling."""

    def test_catch_generation_error_catches_specific_types(self):
        """Test that catching GenerationError catches all generation-specific errors."""
        for code in (-14, -16):
            case = ERROR_CASES_BY_CODE[code]
            try:
                raise_for_error_code(code, case.message)
            except GenerationError as exc:
                assert isinstance(exc, case.exception)

    def test_catch_foundation_models_error_catches_all(self):
        """Test that catching FoundationModelsError catches all exception types."""
        test_codes = [-1, -2, -6, -14, -16, -21, -99]

        for code in test_codes:
            try:
                raise_for_error_code(code, f"Error {code}")
            except FoundationModelsError:
                pass  # Successfully caught
            except Exception as e:
                pytest.fail(
                    f"Error code {code} not caught by FoundationModelsError: {e}"
                )


class TestErrorJSONParsing:
    """Test error JSON parsing logic that mimics Cython layer behavior."""

    def _simulate_cython_error_handling(self, result_str: str) -> None:
        """
        Simulate the error handling logic from _foundationmodels.pyx.

        This mirrors the code in lines 364-371 of _foundationmodels.pyx:
        ```
        if result_str.startswith('{"error"'):
            try:
                error_data = json.loads(result_str)
                error_msg = error_data.get('error', 'Unknown error')
                error_code = error_data.get('error_code', -6)
                raise_for_error_code(error_code, error_msg)
            except json.JSONDecodeError:
                pass
        ```
        """
        import json

        if result_str.startswith('{"error"'):
            try:
                error_data = json.loads(result_str)
                error_msg = error_data.get("error", "Unknown error")
                error_code = error_data.get("error_code", -6)
                raise_for_error_code(error_code, error_msg)
            except json.JSONDecodeError:
                pass  # Not JSON, treat as normal response

    @pytest.mark.parametrize(
        "case",
        ERROR_CASES,
        ids=lambda case: f"{case.code}:{case.exception.__name__}",
    )
    def test_error_json_format_and_parsing(self, case: ErrorCase):
        """
        Test that error JSON format is correctly parsed.

        This verifies the contract between Swift and Python layers:
        - Swift produces: {"error": "message", "error_code": -N}
        - Cython parses it and calls raise_for_error_code()
        - Python raises the correct exception type
        """
        import json

        # Create error JSON in the format Swift layer produces
        error_json = json.dumps({"error": case.message, "error_code": case.code})

        # Verify the JSON format is valid
        parsed = json.loads(error_json)
        assert "error" in parsed
        assert "error_code" in parsed
        assert parsed["error"] == case.message
        assert parsed["error_code"] == case.code

        # Test that our error handling logic raises the correct exception
        with pytest.raises(case.exception) as exc_info:
            self._simulate_cython_error_handling(error_json)

        assert case.message in str(exc_info.value)
        assert exc_info.value.error_code == case.code

    def test_error_json_without_error_code_defaults_to_generation_error(self):
        """Test that missing error_code defaults to GenerationError (-6)."""
        import json

        # Old format without error_code (for backwards compatibility)
        error_json = json.dumps({"error": "Something went wrong"})

        with pytest.raises(GenerationError) as exc_info:
            self._simulate_cython_error_handling(error_json)

        assert exc_info.value.error_code == -6

    def test_non_error_json_does_not_raise(self):
        """Test that non-error responses don't trigger error handling."""
        import json

        # Normal success response
        success_json = json.dumps({"content": "Hello, world!"})

        # Should not raise - just return normally
        self._simulate_cython_error_handling(success_json)  # No exception

    def test_malformed_error_json_is_handled_gracefully(self):
        """Test that malformed JSON is handled without crashing."""
        # Invalid JSON that starts with {"error"
        malformed = '{"error": "incomplete'

        # Should not raise - JSONDecodeError is caught
        self._simulate_cython_error_handling(malformed)  # No exception
