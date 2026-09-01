"""server.py — Single unified MCP Server entrypoint for MahaSamvaad Eval MCP Suite."""
from __future__ import annotations

import argparse
import logging
import sys

from mcp.server.mcpserver import MCPServer

import config
from tools import register_all_tools

logging.basicConfig(
    stream=sys.stderr,
    level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s [mahasamvaad-eval-mcp] %(message)s",
)
logger = logging.getLogger(__name__)

mcp = MCPServer(
    name="mahasamvaad-eval-mcp",
    version="2.0.0",
    description="Unified Observability, Grounding v2, Reformulation, and AI-Evaluation Suite for MahaSamvaad GR Chat API",
)

# Register all 8 tool modules
register_all_tools(mcp)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MahaSamvaad Unified Eval MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default=config.MCP_TRANSPORT,
        help="Transport mode (stdio or streamable-http)",
    )
    parser.add_argument(
        "--host",
        default=config.MCP_HTTP_HOST,
        help="Host to bind for streamable-http (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=config.MCP_HTTP_PORT,
        help="Port for streamable-http (default: 8001)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logger.info("Initializing MahaSamvaad Eval MCP Server v2.0.0 (Grounding Method: %s)", config.GROUNDING_METHOD)

    if args.transport == "streamable-http":
        logger.info("Starting unified MCP server on http://%s:%d (streamable-http)", args.host, args.port)
        mcp.run(transport="streamable-http", host=args.host, port=args.port)
    else:
        logger.info("Starting unified MCP server on stdio")
        mcp.run_stdio_async()


if __name__ == "__main__":
    main()
