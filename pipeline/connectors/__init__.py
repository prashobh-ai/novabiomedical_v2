"""Source connectors — the fabric's ingestion boundary."""
from .base import Connector, KnowledgeRecord, registry_from, sanitize

__all__ = ["Connector", "KnowledgeRecord", "registry_from", "sanitize"]
