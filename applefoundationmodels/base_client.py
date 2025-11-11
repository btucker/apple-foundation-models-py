"""
Base Client implementation for applefoundationmodels Python bindings.

Provides shared logic for both sync and async clients.
"""

import platform
from typing import Optional, List, Dict, Any
from abc import ABC

from . import _foundationmodels
from .base import ContextManagedResource
from .types import Availability
from .exceptions import NotAvailableError


class BaseClient(ContextManagedResource, ABC):
    """
    Base class for Client and AsyncClient with shared logic.

    This class contains all the common functionality between the sync
    and async client implementations to avoid duplication.
    """

    _initialized: bool = False

    @staticmethod
    def _validate_platform() -> None:
        """
        Validate platform requirements for Apple Intelligence.

        Raises:
            NotAvailableError: If platform is not supported or version is insufficient
        """
        # Check platform requirements
        if platform.system() != "Darwin":
            raise NotAvailableError(
                "Apple Intelligence is only available on macOS. "
                f"Current platform: {platform.system()}"
            )

        # Check macOS version
        mac_ver = platform.mac_ver()[0]
        if mac_ver:
            try:
                major_version = int(mac_ver.split(".")[0])
                if major_version < 26:
                    raise NotAvailableError(
                        f"Apple Intelligence requires macOS 26.0 or later. "
                        f"Current version: {mac_ver}"
                    )
            except (ValueError, IndexError):
                # If we can't parse the version, let it try anyway
                pass

    @staticmethod
    def _initialize_library() -> None:
        """
        Initialize the FoundationModels library if not already initialized.

        This is called automatically on first client creation.
        """
        if not BaseClient._initialized:
            _foundationmodels.init()
            BaseClient._initialized = True

    @staticmethod
    def check_availability() -> Availability:
        """
        Check Apple Intelligence availability on this device.

        This is a static method that can be called without creating a client.

        Returns:
            Availability status enum value

        Example:
            >>> from applefoundationmodels import Client, Availability
            >>> status = Client.check_availability()
            >>> if status == Availability.AVAILABLE:
            ...     print("Apple Intelligence is available!")
        """
        return Availability(_foundationmodels.check_availability())

    @staticmethod
    def get_availability_reason() -> Optional[str]:
        """
        Get detailed availability status message.

        Returns:
            Detailed status description with actionable guidance,
            or None if library not initialized
        """
        return _foundationmodels.get_availability_reason()

    @staticmethod
    def is_ready() -> bool:
        """
        Check if Apple Intelligence is ready for immediate use.

        Returns:
            True if ready for use, False otherwise
        """
        return _foundationmodels.is_ready()

    @staticmethod
    def get_version() -> str:
        """
        Get library version string.

        Returns:
            Version string in format "major.minor.patch"
        """
        return _foundationmodels.get_version()

    @staticmethod
    def get_supported_languages() -> List[str]:
        """
        Get list of languages supported by Apple Intelligence.

        Returns:
            List of localized language display names
        """
        count = _foundationmodels.get_supported_languages_count()
        return [
            lang
            for i in range(count)
            if (lang := _foundationmodels.get_supported_language(i)) is not None
        ]

    @staticmethod
    def _build_session_config(
        instructions: Optional[str],
        tools_json: Optional[str],
        enable_guardrails: bool,
        prewarm: bool,
    ) -> Optional[Dict[str, Any]]:
        """
        Build session configuration dictionary.

        Args:
            instructions: Optional system instructions
            tools_json: Optional tool definitions (not used in simplified API)
            enable_guardrails: Enable guardrails (not used in simplified API)
            prewarm: Prewarm session (not used in simplified API)

        Returns:
            Configuration dictionary or None if empty
        """
        config = {}
        if instructions is not None:
            config["instructions"] = instructions
        # Note: tools_json, enable_guardrails, prewarm not supported in simplified API
        return config if config else None
