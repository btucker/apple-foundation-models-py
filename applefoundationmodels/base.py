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
    aclose() method on exit. Also provides a close() bridge method that
    detects async context and delegates to aclose() appropriately.
    """

    async def __aenter__(self: AT) -> AT:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit with automatic cleanup."""
        await self.aclose()

    def close(self) -> None:
        """
        Close and cleanup resources synchronously.

        This method bridges sync and async cleanup by detecting the event loop
        context. If called from outside an async context, it runs aclose() via
        asyncio.run(). If called from within an async context, it raises an error
        directing the user to use aclose() instead.

        Raises:
            RuntimeError: If called from within an async context (user should use aclose())

        Note:
            Prefer using aclose() or the async context manager in async code.
            This method is primarily for compatibility and cleanup from sync contexts.
        """
        try:
            # Check if we're in an async context with a running event loop
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop, safe to use asyncio.run()
            asyncio.run(self.aclose())
        else:
            # Running loop - user should call aclose() instead
            class_name = self.__class__.__name__
            raise RuntimeError(
                f"close() called from async context. "
                f"Use 'await {class_name.lower()}.aclose()' instead."
            )

    @abstractmethod
    async def aclose(self) -> None:
        """
        Close and cleanup resources asynchronously.

        Must be implemented by subclasses. This is called by the async
        context manager (__aexit__) and by close() when in non-async contexts.
        """
        pass
