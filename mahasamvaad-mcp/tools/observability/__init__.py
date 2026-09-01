"""tools/observability package."""
from tools.observability.tool import (
    handle_get_dashboard_stats,
    handle_list_recent_queries,
    handle_query_and_log,
    register,
)

__all__ = [
    "register",
    "handle_query_and_log",
    "handle_get_dashboard_stats",
    "handle_list_recent_queries",
]
