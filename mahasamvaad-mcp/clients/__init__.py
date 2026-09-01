"""clients package for MahaSamvaad Eval MCP."""
from clients.chat_api_client import ChatRequest, ChatResult, call_chat
from clients.embedding_client import EmbeddingClient, compute_similarity, cosine_similarity
from clients.llm_judge_client import JudgeEvaluationResult, LLMJudgeClient, default_judge_client
from clients.serper_client import SerperSearchResult, search_web
from clients.storage_client import StorageDocumentResult, fetch_document

__all__ = [
    "ChatRequest",
    "ChatResult",
    "call_chat",
    "fetch_document",
    "StorageDocumentResult",
    "EmbeddingClient",
    "compute_similarity",
    "cosine_similarity",
    "LLMJudgeClient",
    "JudgeEvaluationResult",
    "default_judge_client",
    "search_web",
    "SerperSearchResult",
]
