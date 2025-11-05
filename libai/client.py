"""
High-level Client API for libai Python bindings.

Provides a Pythonic interface to the libai library with context managers,
automatic resource cleanup, and integration with the Session class.
"""

from typing import Optional, List
from contextlib import contextmanager

from . import _libai
from .types import Availability, Stats
from .session import Session


class Client:
    """
    High-level client for Apple Intelligence operations.

    This class provides a Pythonic interface to the libai library, managing
    contexts and sessions with automatic resource cleanup.

    Usage:
        with Client() as client:
            session = client.create_session()
            response = session.generate("Hello!")
            print(response)
    """

    _initialized: bool = False

    def __init__(self):
        """
        Create a new libai client.

        The library is automatically initialized on first client creation.

        Raises:
            InitializationError: If library initialization fails
            NotAvailableError: If Apple Intelligence is not available
        """
        # Initialize library on first client creation
        if not Client._initialized:
            _libai.init()
            Client._initialized = True

        # Create context
        self._context = _libai.Context()
        self._sessions: List[Session] = []

    def __enter__(self) -> "Client":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit with automatic cleanup."""
        self.close()

    def close(self) -> None:
        """
        Close the client and cleanup all resources.

        Destroys all sessions and frees the context.
        """
        # Close all sessions
        for session in self._sessions:
            session.close()
        self._sessions.clear()

        # Context will be freed automatically by Cython __dealloc__
        self._context = None

    @staticmethod
    def check_availability() -> Availability:
        """
        Check Apple Intelligence availability on this device.

        This is a static method that can be called without creating a client.

        Returns:
            Availability status enum value

        Example:
            >>> from libai import Client, Availability
            >>> status = Client.check_availability()
            >>> if status == Availability.AVAILABLE:
            ...     print("Apple Intelligence is available!")
        """
        return Availability(_libai.check_availability())

    @staticmethod
    def get_availability_reason() -> Optional[str]:
        """
        Get detailed availability status message.

        Returns:
            Detailed status description with actionable guidance,
            or None if library not initialized
        """
        return _libai.get_availability_reason()

    @staticmethod
    def is_ready() -> bool:
        """
        Check if Apple Intelligence is ready for immediate use.

        Returns:
            True if ready for use, False otherwise
        """
        return _libai.is_ready()

    @staticmethod
    def get_version() -> str:
        """
        Get library version string.

        Returns:
            Version string in format "major.minor.patch"
        """
        return _libai.get_version()

    @staticmethod
    def get_supported_languages() -> List[str]:
        """
        Get list of languages supported by Apple Intelligence.

        Returns:
            List of localized language display names
        """
        count = _libai.get_supported_languages_count()
        return [_libai.get_supported_language(i) for i in range(count)]

    def create_session(
        self,
        instructions: Optional[str] = None,
        tools_json: Optional[str] = None,
        enable_guardrails: bool = True,
        prewarm: bool = False
    ) -> Session:
        """
        Create a new AI session.

        Sessions maintain conversation state and can be configured with
        tools and instructions.

        Args:
            instructions: Optional system instructions to guide AI behavior
            tools_json: Optional JSON array of tool definitions in Claude format
            enable_guardrails: Whether to enable content safety filtering
            prewarm: Whether to preload session resources for faster first response

        Returns:
            New Session instance

        Raises:
            Various LibAIError subclasses on failure

        Example:
            >>> session = client.create_session(
            ...     instructions="You are a helpful assistant.",
            ...     enable_guardrails=True
            ... )
        """
        config = {}
        if instructions is not None:
            config['instructions'] = instructions
        if tools_json is not None:
            config['tools_json'] = tools_json
        config['enable_guardrails'] = enable_guardrails
        config['prewarm'] = prewarm

        session_id = self._context.create_session(config)
        session = Session(self._context, session_id)
        self._sessions.append(session)
        return session

    def get_stats(self) -> Stats:
        """
        Get generation statistics for this client.

        Returns:
            Dictionary with statistics fields including request counts,
            success rates, and performance metrics

        Example:
            >>> stats = client.get_stats()
            >>> print(f"Total requests: {stats['total_requests']}")
            >>> print(f"Average response time: {stats['average_response_time']:.2f}s")
        """
        return self._context.get_stats()

    def reset_stats(self) -> None:
        """
        Reset generation statistics for this client.

        Clears all accumulated statistics, resetting counters to zero.
        """
        self._context.reset_stats()

    def get_last_error(self) -> str:
        """
        Get the last error message for this client.

        Returns:
            Human-readable error description
        """
        return self._context.get_last_error()


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
