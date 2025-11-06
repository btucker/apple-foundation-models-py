"""
Utility functions for examples.

Provides common functionality used across multiple example scripts.
"""

import sys
from applefoundationmodels import Client, Availability, NotAvailableError


def check_availability_or_exit(verbose=True):
    """
    Check Apple Intelligence availability and exit if not available.

    Args:
        verbose: If True, print detailed availability information

    Returns:
        True if available, False otherwise (will exit if not available)
    """
    status = Client.check_availability()
    if status != Availability.AVAILABLE:
        if verbose:
            print("Apple Intelligence is not available:")
            print(Client.get_availability_reason())
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


def print_stats(client, verbose=True, width=60):
    """
    Print client statistics in a formatted way.

    Args:
        client: The Client instance to get stats from
        verbose: If True, show detailed stats; if False, show summary
        width: Width of the separator line (default: 60)
    """
    stats = client.get_stats()
    print(f"\n{'=' * width}")
    if verbose:
        print("Statistics:")
        print("=" * width)
        print(f"Total requests: {stats['total_requests']}")
        print(f"Successful: {stats['successful_requests']}")
        print(f"Failed: {stats['failed_requests']}")
        print(f"Avg response time: {stats['average_response_time']:.2f}s")
        print(f"Total processing time: {stats['total_processing_time']:.2f}s")
    else:
        print(f"Completed {stats['successful_requests']} requests")
        print(f"Total time: {stats['total_processing_time']:.2f}s")


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
