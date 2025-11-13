"""
Utility functions for examples.

Provides common functionality used across multiple example scripts.
"""

from applefoundationmodels import Session, apple_intelligence_available


def check_availability_or_exit(verbose=True):
    """
    Check Apple Intelligence availability and exit if not available.

    Args:
        verbose: If True, print detailed availability information

    Returns:
        True if available, False otherwise (will exit if not available)
    """
    if not apple_intelligence_available():
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
