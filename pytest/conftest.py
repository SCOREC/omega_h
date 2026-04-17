"""
Pytest configuration for PyOmega_h tests.

This conftest.py handles MPI/Kokkos cleanup issues by forcing a clean exit
before Python's normal cleanup process, which can conflict with MPI finalization.
"""

import sys
import os


def pytest_sessionfinish(session, exitstatus):
    """
    Hook called after all tests complete but before pytest exits.
    
    For MPI programs, we need to exit immediately to avoid Python's cleanup
    attempting to use MPI after it has been finalized. This prevents the
    "Attempting to use an MPI routine after finalizing MPICH" error.
    """
    # Force immediate exit without Python cleanup to avoid MPI finalization conflicts
    os._exit(exitstatus)
