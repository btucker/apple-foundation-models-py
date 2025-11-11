"""
Async Client API for applefoundationmodels Python bindings.

Provides async/await interface following OpenAI's AsyncClient pattern.
"""

from typing import Optional, List, cast

from . import _foundationmodels
from .base_client import BaseClient
from .types import Stats
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
        tools_json: Optional[str] = None,
        enable_guardrails: bool = True,
        prewarm: bool = False,
    ) -> AsyncSession:
        """
        Create a new async AI session.

        AsyncSession provides async/await support for all operations including
        streaming. Sessions maintain conversation state and can be configured with
        tools and instructions.

        Args:
            instructions: Optional system instructions to guide AI behavior
            tools_json: Optional JSON array of tool definitions in Claude format
            enable_guardrails: Whether to enable content safety filtering
            prewarm: Whether to preload session resources for faster first response

        Returns:
            New AsyncSession instance

        Raises:
            Various FoundationModelsError subclasses on failure

        Example:
            >>> session = await client.create_session(
            ...     instructions="You are a helpful assistant.",
            ...     enable_guardrails=True
            ... )
            >>> response = await session.generate("Hello!")
            >>> print(response.text)
        """
        config = self._build_session_config(
            instructions, tools_json, enable_guardrails, prewarm
        )
        session_id = _foundationmodels.create_session(config)
        session = AsyncSession(session_id, config)
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
        return cast(Stats, _foundationmodels.get_stats())

    def reset_stats(self) -> None:
        """
        Reset generation statistics for this client.

        Clears all accumulated statistics, resetting counters to zero.
        """
        _foundationmodels.reset_stats()


# Note: asynccontextmanager is available in Python 3.7+
# Using typing_extensions for older Python versions if needed
