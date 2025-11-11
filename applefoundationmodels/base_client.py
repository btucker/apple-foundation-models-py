"""
Base Client implementation for applefoundationmodels Python bindings.

Provides shared logic for both sync and async clients.
"""

import platform
from typing import Optional, List, Dict, Any, Callable, TYPE_CHECKING, Type, Union
from abc import ABC, abstractmethod

from . import _foundationmodels
from .base import ContextManagedResource
from .types import Availability
from .exceptions import NotAvailableError

if TYPE_CHECKING:
    from .session import Session
    from .async_session import AsyncSession


class BaseClient(ContextManagedResource, ABC):
    """
    Base class for Client and AsyncClient with shared logic.

    This class contains all the common functionality between the sync
    and async client implementations to avoid duplication.
    """

    _initialized: bool = False

    def __init__(self):
        """
        Initialize the client with platform validation and library initialization.

        This is called by both Client and AsyncClient constructors.

        Raises:
            InitializationError: If library initialization fails
            NotAvailableError: If Apple Intelligence is not available
            RuntimeError: If platform is not supported
        """
        self._validate_platform()
        self._initialize_library()
        self._sessions: List[Any] = []

    @property
    @abstractmethod
    def _session_class(self) -> Type[Union["Session", "AsyncSession"]]:
        """
        Return the session class to use for this client.

        This is implemented by subclasses to return either Session or AsyncSession.
        """
        pass

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

    def _create_session_impl(
        self,
        instructions: Optional[str],
        tools: Optional[List[Callable]],
    ) -> Union["Session", "AsyncSession"]:
        """
        Shared implementation for creating sessions.

        This method contains the common logic for both sync and async session creation.
        Subclasses don't need to override this.

        Args:
            instructions: Optional system instructions
            tools: Optional list of tool functions to register

        Returns:
            New session instance (Session or AsyncSession based on _session_class)
        """
        config = self._build_session_config(instructions, tools)
        session_id = _foundationmodels.create_session(config)
        session = self._session_class(session_id, config)
        self._sessions.append(session)
        return session

    @staticmethod
    def _build_session_config(
        instructions: Optional[str],
        tools: Optional[List[Callable]],
    ) -> Optional[Dict[str, Any]]:
        """
        Build session configuration dictionary and register tools.

        Args:
            instructions: Optional system instructions
            tools: Optional list of tool functions to register

        Returns:
            Configuration dictionary or None if empty
        """
        # Register tools if provided
        if tools:
            from .tools import register_tool_for_function

            # Build tool dictionary with function objects
            tool_dict = {}
            for func in tools:
                schema = register_tool_for_function(func)
                tool_name = schema["name"]
                tool_dict[tool_name] = func

            # Register with FFI
            _foundationmodels.register_tools(tool_dict)

        config = {}
        if instructions is not None:
            config["instructions"] = instructions
        return config if config else None
