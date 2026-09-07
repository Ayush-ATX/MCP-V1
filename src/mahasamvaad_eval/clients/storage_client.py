"""clients/storage_client.py — Storage endpoint client for corpus PDF / GR documents."""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Any

import httpx

import config

logger = logging.getLogger(__name__)

try:
    import pypdf
except ImportError:
    pypdf = None


@dataclass
class StorageDocumentResult:
    ok: bool
    filepath: str
    content_bytes: bytes | None = None
    extracted_text: str | None = None
    page_count: int | None = None
    http_status: int | None = None
    error: str | None = None


async def fetch_document(
    filepath: str,
    *,
    base_url: str | None = None,
    timeout_s: float | None = None,
    extract_text: bool = True,
) -> StorageDocumentResult:
    """Fetch a document from the storage API and optionally extract its text.

    Never raises — returns StorageDocumentResult(ok=False, ...) on any failure.
    """
    base = (base_url or config.STORAGE_API_BASE_URL).rstrip("/")
    clean_path = filepath.strip()
    if clean_path.startswith("/api/v1/storage"):
        clean_path = clean_path[len("/api/v1/storage"):]
    if not clean_path.startswith("/"):
        clean_path = f"/{clean_path}"

    url = f"{base}{clean_path}"
    timeout = httpx.Timeout(float(timeout_s or config.MCP_HTTP_TIMEOUT_S))

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                return StorageDocumentResult(
                    ok=False,
                    filepath=filepath,
                    http_status=resp.status_code,
                    error=f"Storage GET returned HTTP {resp.status_code}: {resp.text[:200]}",
                )

            content = resp.content
            text: str | None = None
            page_count: int | None = None

            if extract_text and pypdf and (clean_path.endswith(".pdf") or content.startswith(b"%PDF")):
                try:
                    reader = pypdf.PdfReader(io.BytesIO(content))
                    page_count = len(reader.pages)
                    extracted_pages = []
                    for p in reader.pages:
                        extracted_pages.append(p.extract_text() or "")
                    text = "\n\n".join(extracted_pages)
                except Exception as p_err:
                    logger.warning("PDF extraction failed for %s: %s", filepath, p_err)
                    text = None
            elif extract_text:
                try:
                    text = content.decode("utf-8", errors="replace")
                except Exception:
                    text = None

            return StorageDocumentResult(
                ok=True,
                filepath=filepath,
                content_bytes=content,
                extracted_text=text,
                page_count=page_count,
                http_status=200,
            )

    except Exception as exc:
        logger.warning("fetch_document error for %s: %s", filepath, exc)
        return StorageDocumentResult(
            ok=False,
            filepath=filepath,
            error=f"Storage request error: {exc}",
            http_status=None,
        )
