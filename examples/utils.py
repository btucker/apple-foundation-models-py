"""
Utility functions for examples.

Provides common functionality used across multiple example scripts.
"""

import sys
from applefoundationmodels import Session, Availability, NotAvailableError


def check_availability_or_exit(verbose=True):
    """
    Check Apple Intelligence availability and exit if not available.

    Args:
        verbose: If True, print detailed availability information

    Returns:
        True if available, False otherwise (will exit if not available)
    """
    status = Session.check_availability()
    if status != Availability.AVAILABLE:
        if verbose:
            print("Apple Intelligence is not available:")
            print(Session.get_availability_reason())
        else:
            print("Apple Intelligence not available")
        return False
    return True


def print_example_header(title, width=60):
    """
    Print a formatted example header.

    Args:
        title: The title to display
        width: Width of the header line (default: 60)
    """
    print(f"\n{'=' * width}")
    print(f"{title}")
    print(f"{'=' * width}\n")


def handle_example_error(e):
    """
    Standard error handling for examples.

    Args:
        e: The exception to handle
    """
    if isinstance(e, NotAvailableError):
        print(f"Error: {e.message}")
        print("Make sure Apple Intelligence is enabled in System Settings")
        sys.exit(1)
    else:
        print(f"Error: {e}")
        raise
