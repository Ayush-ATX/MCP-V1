"""clients/falkordb_client.py — Production FalkorDB Knowledge Graph Client with Strict 4-Layer Read-Only Guardrails."""
from __future__ import annotations

import logging
import re
from typing import Any

from falkordb import FalkorDB

import config

logger = logging.getLogger(__name__)


class SecurityViolationError(ValueError):
    """Raised when a non-read-only or mutating Cypher query is detected."""
    pass


# Forbidden mutation keywords in OpenCypher
_MUTATING_KEYWORDS_REGEX = re.compile(
    r"\b(?:CREATE|MERGE|SET|DELETE|DETACH|REMOVE|DROP|ALTER|LOAD\s+CSV|FOREACH)\b|"
    r"\bCALL\s+(?:db\b|dbms\b|graph\.delete|apoc\.trigger)",
    re.IGNORECASE,
)

# Whitelist allowed query prefix keywords
_ALLOWED_READ_PREFIX_REGEX = re.compile(
    r"^\s*(?:MATCH|OPTIONAL\s+MATCH|WITH|UNWIND|RETURN|EXPLAIN|PROFILE)\b",
    re.IGNORECASE,
)


def validate_read_only_cypher(cypher: str) -> None:
    """Validate that the Cypher query contains NO mutation or write clauses.

    Raises:
        SecurityViolationError: If any mutating keyword, procedure, or invalid clause is detected.
    """
    if not cypher or not cypher.strip():
        raise SecurityViolationError("Empty Cypher query.")

    clean_query = cypher.strip()

    # 1. Reject mutating keywords
    match = _MUTATING_KEYWORDS_REGEX.search(clean_query)
    if match:
        raise SecurityViolationError(
            f"Mutating Cypher keyword/procedure '{match.group(0)}' is strictly prohibited on production FalkorDB."
        )

    # 2. Must start with an allowed read clause
    if not _ALLOWED_READ_PREFIX_REGEX.match(clean_query):
        raise SecurityViolationError(
            "Query must start with a read-only clause (MATCH, OPTIONAL MATCH, WITH, UNWIND, RETURN)."
        )


class FalkorDBClient:
    """Read-only client wrapper for production FalkorDB Knowledge Graph."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        username: str | None = None,
        password: str | None = None,
        graph_name: str | None = None,
        ssl: bool | None = None,
    ) -> None:
        self.host = host or config.FALKORDB_HOST
        self.port = port if port is not None else config.FALKORDB_PORT
        self.username = username or config.FALKORDB_USER
        self.password = password or config.FALKORDB_PASSWORD
        self.graph_name = graph_name or config.FALKORDB_GRAPH_NAME
        self.ssl = ssl if ssl is not None else config.FALKORDB_SSL

        self._db: FalkorDB | None = None
        self._graph: Any = None

    def _get_graph(self) -> Any:
        if self._graph is None:
            self._db = FalkorDB(
                host=self.host,
                port=self.port,
                username=self.username if self.username else None,
                password=self.password if self.password else None,
                ssl=self.ssl,
            )
            self._graph = self._db.select_graph(self.graph_name)
        return self._graph

    def ro_query(self, cypher: str, params: dict[str, Any] | None = None) -> list[list[Any]]:
        """Execute a validated read-only Cypher query against FalkorDB.

        Enforces:
        - Layer 1: Lexical security firewall (`validate_read_only_cypher`)
        - Layer 2: Protocol-level read-only call (`graph.ro_query`)
        """
        validate_read_only_cypher(cypher)
        graph = self._get_graph()
        result = graph.ro_query(cypher, params or {})
        return result.result_set if result else []

    def is_available(self) -> bool:
        """Check if FalkorDB connection and graph are reachable."""
        try:
            res = self.ro_query("RETURN 1 AS ping")
            return len(res) > 0 and res[0][0] == 1
        except Exception as exc:
            logger.debug("FalkorDB is_available check failed: %s", exc)
            return False

    def get_gr_lineage(self, gr_identifier: str) -> list[dict[str, Any]]:
        """Fetch all incoming and outgoing lineage relations for a given GR.

        Relations tracked: SUPERSEDES, AMENDS, REVOKES, EXTENDS, SUSPENDS, REINSTATES.
        """
        if not gr_identifier or not gr_identifier.strip():
            return []

        gr_clean = gr_identifier.strip()
        lineage_entries: list[dict[str, Any]] = []

        # Outgoing: (gr)-[r]->(target) e.g., gr SUPERSEDES target (meaning target is superseded by gr)
        cypher_outgoing = """
        MATCH (a:GR)-[r:SUPERSEDES|AMENDS|REVOKES|EXTENDS|SUSPENDS|REINSTATES]->(b:GR)
        WHERE a.gr_number = $gr_num OR a.gr_number_norm = $gr_num OR a.gr_code = $gr_num OR a.name = $gr_num
        RETURN a.gr_number AS src_gr, a.department AS src_dept, a.is_active AS src_active,
               type(r) AS rel_type,
               b.gr_number AS tgt_gr, b.department AS tgt_dept, b.is_active AS tgt_active, b.title_en AS tgt_title
        LIMIT 25
        """
        try:
            res_out = self.ro_query(cypher_outgoing, {"gr_num": gr_clean})
            for row in res_out:
                lineage_entries.append({
                    "source_gr": str(row[0] or ""),
                    "source_dept": str(row[1] or ""),
                    "source_is_active": row[2] if row[2] is not None else True,
                    "relation_type": str(row[3]).lower(),
                    "target_gr": str(row[4] or ""),
                    "target_dept": str(row[5] or ""),
                    "target_is_active": row[6] if row[6] is not None else False,
                    "direction": "outgoing",  # source -> relation -> target
                    "description": f"{row[0]} {str(row[3]).lower()} {row[4]}",
                })
        except Exception as exc:
            logger.warning("FalkorDB get_gr_lineage outgoing query failed for '%s': %s", gr_clean, exc)

        # Incoming: (source)-[r]->(gr) e.g., source SUPERSEDES gr (meaning gr is superseded by source)
        cypher_incoming = """
        MATCH (a:GR)-[r:SUPERSEDES|AMENDS|REVOKES|EXTENDS|SUSPENDS|REINSTATES]->(b:GR)
        WHERE b.gr_number = $gr_num OR b.gr_number_norm = $gr_num OR b.gr_code = $gr_num OR b.name = $gr_num
        RETURN a.gr_number AS src_gr, a.department AS src_dept, a.is_active AS src_active, a.title_en AS src_title,
               type(r) AS rel_type,
               b.gr_number AS tgt_gr, b.department AS tgt_dept, b.is_active AS tgt_active
        LIMIT 25
        """
        try:
            res_in = self.ro_query(cypher_incoming, {"gr_num": gr_clean})
            for row in res_in:
                lineage_entries.append({
                    "source_gr": str(row[0] or ""),
                    "source_dept": str(row[1] or ""),
                    "source_is_active": row[2] if row[2] is not None else True,
                    "relation_type": str(row[4]).lower(),
                    "target_gr": str(row[5] or ""),
                    "target_dept": str(row[6] or ""),
                    "target_is_active": row[7] if row[7] is not None else False,
                    "direction": "incoming",  # source -> relation -> target(gr)
                    "description": f"{row[5]} is {str(row[4]).lower()}_by {row[0]}",
                })
        except Exception as exc:
            logger.warning("FalkorDB get_gr_lineage incoming query failed for '%s': %s", gr_clean, exc)

        return lineage_entries

    def get_gr_metadata(self, gr_identifier: str) -> dict[str, Any] | None:
        """Fetch node metadata for a given GR."""
        if not gr_identifier or not gr_identifier.strip():
            return None

        gr_clean = gr_identifier.strip()
        cypher = """
        MATCH (a:GR)
        WHERE a.gr_number = $gr_num OR a.gr_number_norm = $gr_num OR a.gr_code = $gr_num OR a.name = $gr_num
        RETURN a.gr_number, a.gr_code, a.department, a.issue_date, a.title_en, a.marathi_title, a.is_active, a.summary_en
        LIMIT 1
        """
        try:
            res = self.ro_query(cypher, {"gr_num": gr_clean})
            if res and len(res) > 0:
                row = res[0]
                return {
                    "gr_number": row[0],
                    "gr_code": row[1],
                    "department": row[2],
                    "issue_date": row[3],
                    "title_en": row[4],
                    "marathi_title": row[5],
                    "is_active": row[6] if row[6] is not None else True,
                    "summary_en": row[7],
                }
        except Exception as exc:
            logger.warning("FalkorDB get_gr_metadata failed for '%s': %s", gr_clean, exc)

        return None
