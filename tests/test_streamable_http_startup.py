"""Regression tests for the synchronous Streamable HTTP startup path."""
from __future__ import annotations

from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest
from mahasamvaad_eval import server as unified_server


def test_streamable_http_main_uses_synchronous_sdk_runner():
    """The HTTP branch must run the SDK entry point, not create an unawaited coroutine."""
    async_runner = MagicMock()

    with patch.object(
        unified_server,
        "_parse_args",
        return_value=Namespace(transport="streamable-http", host="127.0.0.1", port=8001),
    ), patch.object(unified_server.mcp, "run") as run, patch.object(
        unified_server.mcp, "run_streamable_http_async", async_runner
    ):
        unified_server.main()

    run.assert_called_once_with(transport="streamable-http", host="127.0.0.1", port=8001)
    async_runner.assert_not_called()
