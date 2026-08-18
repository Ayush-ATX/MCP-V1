"""conftest.py — pytest configuration for the mahasamvaad-mcp test suite."""
import sys
import os

# Ensure the project root is on sys.path so all imports resolve correctly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

collect_ignore = [
    # Standalone integration scripts — not pytest test modules.
    # They are run directly: python tests/test_stability_e2e.py
    # Excluded here because they import `test_reformulation_stability` from
    # server_reformulation.main, which pytest mistakenly collects as a fixture.
    "tests/test_stability_e2e.py",
    "tests/test_nvidia_nim.py",
]
