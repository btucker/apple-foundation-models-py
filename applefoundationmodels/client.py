"""
High-level Client API for applefoundationmodels Python bindings.

Provides a Pythonic interface to Apple's FoundationModels framework with
context managers, automatic resource cleanup, and integration with the Session class.
"""

from typing import Optional, List, Callable, Type
from contextlib import contextmanager

from . import _foundationmodels
from .base_client import BaseClient
from .session import Session


class Client(BaseClient):
    """
    High-level client for Apple Intelligence operations.

    This class provides a Pythonic interface to Apple's FoundationModels framework,
    managing sessions with automatic resource cleanup.

    Usage:
        with Client() as client:
            session = client.create_session()
            response = session.generate("Hello!")
            print(response)
    """

    def __init__(self):
        """
        Create a new FoundationModels client.

        The library is automatically initialized on first client creation.

        Raises:
            InitializationError: If library initialization fails
            NotAvailableError: If Apple Intelligence is not available
            RuntimeError: If platform is not supported
        """
        super().__init__()

    @property
    def _session_class(self) -> Type[Session]:
        """Return Session class for sync client."""
        return Session

    def close(self) -> None:
        """
        Close the client and cleanup all resources.

        Closes all sessions created by this client.
        """
        for session in self._sessions:
            session.close()
        self._sessions.clear()

    def create_session(
        self,
        instructions: Optional[str] = None,
        tools: Optional[List[Callable]] = None,
    ) -> Session:
        """
        Create a new AI session.

        Sessions maintain conversation state and can be configured with
        tools and instructions.

        Args:
            instructions: Optional system instructions to guide AI behavior
            tools: Optional list of tool functions to make available to the model

        Returns:
            New Session instance

        Raises:
            Various FoundationModelsError subclasses on failure

        Example:
            >>> def get_weather(location: str) -> str:
            ...     '''Get current weather for a location.'''
            ...     return f"Weather in {location}: 22°C"
            >>> session = client.create_session(
            ...     instructions="You are a helpful assistant.",
            ...     tools=[get_weather]
            ... )
        """
        return self._create_session_impl(instructions, tools)


@contextmanager
def client():
    """
    Context manager factory for creating a Client.

    Usage:
        with client() as c:
            session = c.create_session()
            response = session.generate("Hello!")

    Yields:
        Client instance with automatic cleanup
    """
    c = Client()
    try:
        yield c
    finally:
        c.close()
