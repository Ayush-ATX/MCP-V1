"""clients package — External HTTP & Database client connectors."""
from mahasamvaad_eval.clients.chat_api_client import ChatRequest, ChatResult, call_chat
from mahasamvaad_eval.clients.embedding_client import EmbeddingClient, compute_similarity, cosine_similarity
from mahasamvaad_eval.clients.falkordb_client import FalkorDBClient, SecurityViolationError, validate_read_only_cypher
from mahasamvaad_eval.clients.llm_judge_client import JudgeEvaluationResult, LLMJudgeClient, default_judge_client
from mahasamvaad_eval.clients.serper_client import SerperSearchResult, search_web
from mahasamvaad_eval.clients.storage_client import StorageDocumentResult, fetch_document

__all__ = [
    "ChatRequest",
    "ChatResult",
    "call_chat",
    "fetch_document",
    "StorageDocumentResult",
    "EmbeddingClient",
    "compute_similarity",
    "cosine_similarity",
    "FalkorDBClient",
    "SecurityViolationError",
    "validate_read_only_cypher",
    "LLMJudgeClient",
    "JudgeEvaluationResult",
    "default_judge_client",
    "search_web",
    "SerperSearchResult",
]
