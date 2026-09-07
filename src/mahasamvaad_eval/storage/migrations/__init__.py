"""storage.migrations package."""
from mahasamvaad_eval.storage.migrations.schema import apply_schema, migrate_legacy_db

__all__ = ["apply_schema", "migrate_legacy_db"]
