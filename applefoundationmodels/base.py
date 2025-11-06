"""
Base classes for applefoundationmodels.

Provides base functionality for context-managed resources.
"""

from abc import ABC, abstractmethod
from typing import TypeVar

T = TypeVar("T", bound="ContextManagedResource")


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
