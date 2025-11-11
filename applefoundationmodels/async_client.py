"""
Async Client API for applefoundationmodels Python bindings.

Provides async/await interface following OpenAI's AsyncClient pattern.
"""

from typing import Optional, List, Callable, Type

from . import _foundationmodels
from .base_client import BaseClient
from .base import AsyncContextManagedResource
from .async_session import AsyncSession


class AsyncClient(BaseClient, AsyncContextManagedResource):
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
        super().__init__()

    @property
    def _session_class(self) -> Type[AsyncSession]:
        """Return AsyncSession class for async client."""
        return AsyncSession

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
        return self._create_session_impl(instructions, tools)
