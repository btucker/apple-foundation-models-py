"""
Base classes for applefoundationmodels.

Provides base functionality for context-managed resources.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import TypeVar

T = TypeVar("T", bound="ContextManagedResource")
AT = TypeVar("AT", bound="AsyncContextManagedResource")


class ContextManagedResource(ABC):
    """
    Base class for resources that support context manager protocol.

    Provides standard __enter__ and __exit__ methods that call the
    close() method on exit. Subclasses must implement close().
    """

    def __enter__(self: T) -> T:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit with automatic cleanup."""
        self.close()

    @abstractmethod
    def close(self) -> None:
        """
        Close and cleanup resources.

        Must be implemented by subclasses.
        """
        pass


class AsyncContextManagedResource(ABC):
    """
    Base class for resources that support async context manager protocol.

    Provides standard __aenter__ and __aexit__ methods that call the
    aclose() method on exit.
    """

    async def __aenter__(self: AT) -> AT:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit with automatic cleanup."""
        await self.aclose()

    @abstractmethod
    async def aclose(self) -> None:
        """
        Close and cleanup resources asynchronously.

        Must be implemented by subclasses. This is called by the async
        context manager (__aexit__).
        """
        pass
