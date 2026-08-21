"""Regression tests for the synchronous Streamable HTTP startup path."""
from __future__ import annotations

from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

from server_grounding import main as grounding_main
from server_observability import main as observability_main
from server_reformulation import main as reformulation_main


@pytest.mark.parametrize(
    ("server_module", "host", "port"),
    [
        (observability_main, "127.0.0.1", 8101),
        (grounding_main, "127.0.0.1", 8102),
        (reformulation_main, "127.0.0.1", 8103),
    ],
)
def test_streamable_http_main_uses_synchronous_sdk_runner(server_module, host, port):
    """The HTTP branch must run the SDK entry point, not create an unawaited coroutine."""
    async_runner = MagicMock()

    with patch.object(
        server_module,
        "_parse_args",
        return_value=Namespace(transport="streamable-http", host=host, port=port),
    ), patch.object(server_module.mcp, "run") as run, patch.object(
        server_module.mcp, "run_streamable_http_async", async_runner
    ):
        server_module.main()

    run.assert_called_once_with(transport="streamable-http", host=host, port=port)
    async_runner.assert_not_called()
