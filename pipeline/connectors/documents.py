"""Document connector — wraps the existing Phase 1 parsers behind the Connector contract.

Deliberately thin: parsers.py already handles PDF/DOCX/MD/TXT with heading-aware
structure. This adapts its output to KnowledgeRecords and classifies each file into
a knowledge domain so the fabric can reason about provenance.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from ..parsers import iter_sources, parse_any
from .base import KnowledgeRecord

# Folder name -> (source_system label, domain facet)
DOMAIN_MAP = {
    "fda_regulatory": ("FDA Clearance Documents", "Regulatory"),
}
DEFAULT_DOMAIN = ("Nova IFU Library", "Product Documentation")


def _classify(path: Path, root: Path) -> tuple[str, str]:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return DEFAULT_DOMAIN
    top = rel.parts[0] if len(rel.parts) > 1 else ""
    return DOMAIN_MAP.get(top, DEFAULT_DOMAIN)


def _infer_product(name: str) -> str:
    n = name.lower()
    for key, label in [
        ("statstrip_2.0", "StatStrip Glucose (Gen 2)"),
        ("statstrip_xpress", "StatStrip Xpress2 Glucose"),
        ("statstrip_glucose", "StatStrip Glucose"),
        ("statstrip_lactate", "StatStrip Lactate"),
        ("statsensor", "StatSensor Creatinine"),
        ("lactate_plus", "Lactate Plus"),
        ("prime_plus", "Stat Profile Prime Plus"),
        ("prime_es", "Stat Profile Prime ES Comp Plus"),
        ("allegro", "Nova Allegro"),
        ("nova_primary", "Nova Primary Glucose Analyzer"),
        ("nova_max", "Nova Max"),
    ]:
        if key in n:
            return label
    return "Unclassified"


def _infer_doc_type(name: str) -> str:
    n = name.lower()
    if "_review_" in n:
        return "FDA Review Memorandum"
    if n.startswith("k") and n[1:3].isdigit():
        return "FDA 510(k) Summary"
    if "quick_reference" in n or "reference_manual" in n:
        return "Quick Reference Guide"
    if "ifu" in n or "instruction" in n:
        return "Instructions For Use"
    return "Document"


class DocumentConnector:
    name = "Document Library"
    source_type = "document"

    def __init__(self, root: Path):
        self.root = Path(root)

    def fetch(self) -> Iterator[KnowledgeRecord]:
        for path in iter_sources(self.root):
            try:
                parsed = parse_any(path)
            except Exception as exc:  # one bad file must not kill the fabric
                print(f"  ! skip {path.name}: {exc}")
                continue

            system, domain = _classify(path, self.root)
            product = _infer_product(path.name)
            doc_type = _infer_doc_type(path.name)
            k_number = path.name.split("_")[0] if path.name.startswith("K") else ""

            for para in parsed.paragraphs:
                if len(para.text) < 40:      # drop page furniture
                    continue
                yield KnowledgeRecord(
                    source_type=self.source_type,
                    source_system=system,
                    source_id=f"{path.name}#p{para.paragraph_index}",
                    title=parsed.name,
                    text=para.text,
                    section_path=para.section_path,
                    page=para.page,
                    url="",
                    metadata={
                        "record_type": "document_paragraph",
                        "document_name": parsed.name,
                        "domain": domain,
                        "product": product,
                        "doc_type": doc_type,
                        "k_number": k_number,
                        "page_count": parsed.page_count,
                        "paragraph_index": para.paragraph_index,
                    },
                    entities=([("Product", product)] if product != "Unclassified" else [])
                             + ([("Clearance", k_number)] if k_number else []),
                )
