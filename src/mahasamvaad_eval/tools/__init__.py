"""tools package registering all 8 evaluation and observability tool modules."""
from mcp.server.mcpserver import MCPServer

from mahasamvaad_eval.tools.bilingual_parity import register as register_bilingual_parity
from mahasamvaad_eval.tools.corpus_web_precedence import register as register_corpus_web_precedence
from mahasamvaad_eval.tools.grounding import register as register_grounding
from mahasamvaad_eval.tools.hallucinated_entity import register as register_hallucinated_entity
from mahasamvaad_eval.tools.lineage_correctness import register as register_lineage_correctness
from mahasamvaad_eval.tools.observability import register as register_observability
from mahasamvaad_eval.tools.reformulation import register as register_reformulation
from mahasamvaad_eval.tools.refusal_redteam import register as register_refusal_redteam


def register_all_tools(mcp: MCPServer) -> None:
    """Register all 8 tool domains on the MCP server."""
    register_observability(mcp)
    register_grounding(mcp)
    register_reformulation(mcp)
    register_hallucinated_entity(mcp)
    register_lineage_correctness(mcp)
    register_bilingual_parity(mcp)
    register_refusal_redteam(mcp)
    register_corpus_web_precedence(mcp)


__all__ = [
    "register_all_tools",
    "register_observability",
    "register_grounding",
    "register_reformulation",
    "register_hallucinated_entity",
    "register_lineage_correctness",
    "register_bilingual_parity",
    "register_refusal_redteam",
    "register_corpus_web_precedence",
]
