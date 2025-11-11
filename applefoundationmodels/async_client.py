"""
Async Client API for applefoundationmodels Python bindings.

Provides async/await interface following OpenAI's AsyncClient pattern.
"""

from typing import Optional, List, Callable

from . import _foundationmodels
from .base_client import BaseClient
from .async_session import AsyncSession


class AsyncClient(BaseClient):
    """
    Async client for Apple Intelligence operations.

    This class provides an async interface to Apple's FoundationModels framework,
    following OpenAI's AsyncClient pattern. All session methods return AsyncSession
    instances that work with async/await.

    Usage:
        async with AsyncClient() as client:
            session = await client.create_session()
            response = await session.generate("Hello!")
            print(response.text)

        # Or with explicit lifecycle:
        client = AsyncClient()
        session = await client.create_session()
        response = await session.generate("Hello!")
        await client.close()
    """

    def __init__(self):
        """
        Create a new async FoundationModels client.

        The library is automatically initialized on first client creation.

        Raises:
            InitializationError: If library initialization fails
            NotAvailableError: If Apple Intelligence is not available
            RuntimeError: If platform is not supported
        """
        self._validate_platform()
        self._initialize_library()
        self._sessions: List[AsyncSession] = []

    async def close(self) -> None:
        """
        Close the client and cleanup all resources.

        Closes all async sessions asynchronously.
        """
        for session in self._sessions:
            await session.close()
        self._sessions.clear()

    async def create_session(
        self,
        instructions: Optional[str] = None,
        tools: Optional[List[Callable]] = None,
    ) -> AsyncSession:
        """
        Create a new async AI session.

        AsyncSession provides async/await support for all operations including
        streaming. Sessions maintain conversation state and can be configured with
        tools and instructions.

        Args:
            instructions: Optional system instructions to guide AI behavior
            tools: Optional list of tool functions to make available to the model

        Returns:
            New AsyncSession instance

        Raises:
            Various FoundationModelsError subclasses on failure

        Example:
            >>> def get_weather(location: str) -> str:
            ...     '''Get current weather for a location.'''
            ...     return f"Weather in {location}: 22°C"
            >>> session = await client.create_session(
            ...     instructions="You are a helpful assistant.",
            ...     tools=[get_weather]
            ... )
            >>> response = await session.generate("Hello!")
            >>> print(response.text)
        """
        config = self._build_session_config(instructions, tools)
        session_id = _foundationmodels.create_session(config)
        session = AsyncSession(session_id, config)
        self._sessions.append(session)
        return session


# Note: asynccontextmanager is available in Python 3.7+
# Using typing_extensions for older Python versions if needed
