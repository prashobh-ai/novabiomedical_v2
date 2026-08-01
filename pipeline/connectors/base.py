"""Connector contract — every source, however different, lands as KnowledgeRecords.

The fabric thesis: a PDF page, an FDA clearance row, a Jira ticket, and a Confluence
page are all the same shape once ingested. Only the connector knows the difference.

A connector yields KnowledgeRecords. The pipeline never learns what a CSV is.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Protocol


@dataclass
class KnowledgeRecord:
    """One atomic, retrievable, citable unit of enterprise knowledge."""

    source_type: str            # 'document' | 'fda_regulatory' | future: 'jira' | 'confluence'
    source_system: str          # human label, e.g. 'Nova IFU Library', 'FDA openFDA'
    source_id: str              # stable id within the source system
    title: str
    text: str                   # the retrievable body
    section_path: list[str] = field(default_factory=list)
    page: int = 1
    url: str = ""               # provenance — where a human verifies this
    metadata: dict = field(default_factory=dict)   # typed, filterable facets
    entities: list[tuple[str, str]] = field(default_factory=list)  # (kind, name) seeds


class Connector(Protocol):
    """Anything that can produce knowledge. Implement one method."""

    name: str
    source_type: str

    def fetch(self) -> Iterator[KnowledgeRecord]:
        ...


def sanitize(value) -> str:
    """CSV/JSON fields arrive dirty. Normalise to clean single-line text."""
    if value is None:
        return ""
    s = str(value).strip()
    if s.lower() in {"nan", "none", "null", "n/a"}:
        return ""
    return " ".join(s.split())


def registry_from(paths: dict[str, Path]) -> list:
    """Build the active connector set. Sources absent on disk are skipped, not fatal —
    a fabric must degrade gracefully when one system is unavailable."""
    from .documents import DocumentConnector
    from .fda import FDARegulatoryConnector

    connectors = []
    docs = paths.get("documents")
    if docs and docs.exists():
        connectors.append(DocumentConnector(docs))
    fda = paths.get("fda_structured")
    if fda and fda.exists():
        connectors.append(FDARegulatoryConnector(fda))
    return connectors
