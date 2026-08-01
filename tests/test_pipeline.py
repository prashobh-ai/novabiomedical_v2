"""Pipeline tests — verify parser, chunker, entity extraction, and BM25 build."""
from __future__ import annotations

import json
import pathlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.bm25_index import build_bm25_index
from pipeline.chunker import chunk_paragraphs, tokenize
from pipeline.entities import build_relationships, extract_entities_from_chunks
from pipeline.parsers import parse_markdown, parse_text


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture
def md_doc(tmp_path):
    p = tmp_path / "sample.md"
    p.write_text("""# Top Heading

## Subsection One

The Knowledge Fabric ingests documents and builds a graph. BM25 retrieves passages with citations.

## Subsection Two

The Knowledge Fabric grounds every answer in a document, page, section, and paragraph. BM25 ranking is the retrieval engine.

### Deeper Section

Citations from the Knowledge Fabric include the document name and page number. BM25 scores drive the ranking.
""")
    return p


@pytest.fixture
def docs_source_dir():
    return Path(__file__).resolve().parent.parent / "docs_source"


# =============================================================================
# Parser tests
# =============================================================================
def test_markdown_parser_extracts_headings(md_doc):
    parsed = parse_markdown(md_doc)
    assert len(parsed.paragraphs) >= 3
    sections = [p.section_path for p in parsed.paragraphs]
    assert any("Subsection One" in path for path in sections)
    assert any("Subsection Two" in path for path in sections)
    assert any("Deeper Section" in path for path in sections)


def test_text_parser_handles_caps_headings(tmp_path):
    p = tmp_path / "sample.txt"
    p.write_text("INTRODUCTION\n\nThis is the body of the introduction.\n\nDETAILS\n\nThis is the body of the details section.")
    parsed = parse_text(p)
    assert len(parsed.paragraphs) == 2
    assert parsed.paragraphs[0].section_path == ["INTRODUCTION"]
    assert parsed.paragraphs[1].section_path == ["DETAILS"]


# =============================================================================
# Chunker tests
# =============================================================================
def test_chunker_preserves_breadcrumbs(md_doc):
    parsed = parse_markdown(md_doc)
    chunks = chunk_paragraphs(parsed.paragraphs, document_id=0, document_name=md_doc.name)
    assert chunks
    for c in chunks:
        assert c.document_name == md_doc.name
        assert c.text
        assert c.tokens
        assert isinstance(c.section_path, list)


def test_tokenize_lowercases_and_filters():
    tokens = tokenize("Knowledge Fabric uses BM25.")
    assert "knowledge" in tokens
    assert "fabric" in tokens
    assert "bm25" in tokens
    assert "." not in tokens


# =============================================================================
# Entity extraction tests
# =============================================================================
def test_entity_extraction_picks_up_proper_nouns(md_doc):
    parsed = parse_markdown(md_doc)
    chunks = chunk_paragraphs(parsed.paragraphs, document_id=0, document_name=md_doc.name)
    entities, mentions = extract_entities_from_chunks(chunks)
    names = {e["canonical"] for e in entities}
    # Either "Knowledge Fabric" as a phrase, or BM25 as an acronym
    assert any(n in names for n in {"knowledge fabric", "bm25"})


# =============================================================================
# Relationships
# =============================================================================
def test_relationships_built_from_real_corpus(docs_source_dir):
    from pipeline.parsers import iter_sources, parse_any

    sources = list(iter_sources(docs_source_dir))
    assert sources, "no source documents found"

    all_chunks = []
    next_id = 0
    for did, src in enumerate(sources):
        parsed = parse_any(src)
        cs = chunk_paragraphs(parsed.paragraphs, document_id=did, document_name=parsed.name, start_chunk_id=next_id)
        next_id += len(cs)
        all_chunks.extend(cs)

    entities, mentions = extract_entities_from_chunks(all_chunks)
    rels = build_relationships(entities, mentions)
    assert len(entities) >= 5
    assert len(rels) >= 1
    for r in rels:
        assert r["weight"] >= 2
        assert r["evidence_chunks"]


# =============================================================================
# BM25 index
# =============================================================================
def test_bm25_index_shape(md_doc):
    parsed = parse_markdown(md_doc)
    chunks = chunk_paragraphs(parsed.paragraphs, document_id=0, document_name=md_doc.name)
    idx = build_bm25_index(chunks)
    assert idx["vocab"]
    assert len(idx["idf"]) == len(idx["vocab"])
    assert len(idx["doc_len"]) == len(chunks)
    assert idx["avgdl"] > 0
    assert idx["k1"] == 1.5
    assert idx["b"] == 0.75


# =============================================================================
# End-to-end: build full index from docs_source
# =============================================================================
def test_full_pipeline_emits_valid_index(tmp_path, docs_source_dir):
    from pipeline.build_index import build

    out = tmp_path / "index.json"
    sem = tmp_path / "semantic.json"
    fda = pathlib.Path("knowledge_sources/fda_structured")
    index = build(docs_source_dir, fda, out, sem)

    assert out.exists()
    loaded = json.loads(out.read_text())
    assert loaded["version"] == "2.0"
    assert loaded["stats"]["document_count"] >= 1
    assert loaded["stats"]["chunk_count"] >= 1
    assert loaded["stats"]["entity_count"] >= 1
    assert loaded["bm25"]["vocab"]
    # Every chunk must remain fully citable
    chunk = loaded["chunks"][0]
    for field in ("id", "document_name", "page", "section_path",
                  "paragraph_indices", "text", "paragraph_excerpt", "entities"):
        assert field in chunk, f"chunk missing field: {field}"
    # Phase 2 additions
    for field in ("source_type", "source_system", "meta"):
        assert field in chunk, f"chunk missing phase-2 field: {field}"


# =============================================================================
# DOCX edge cases — regression for the None-style crash and table content
# =============================================================================
def test_docx_parser_handles_none_style_and_tables(tmp_path):
    """Real-world DOCX files often have paragraphs where p.style is None
    (deleted style refs, headers/footers, third-party generators). The parser
    must not crash on those, and must also index content inside tables."""
    docx_mod = pytest.importorskip("docx")
    from pipeline.parsers import parse_docx

    doc = docx_mod.Document()
    doc.add_heading("Section One", level=1)
    doc.add_paragraph("First paragraph with a normal style.")

    p = doc.add_paragraph("Paragraph with a nulled style reference.")
    p.style = None  # The exact failure mode from production

    doc.add_heading("Section Two", level=2)
    doc.add_paragraph("Tail paragraph.")

    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Knowledge Fabric content inside a table cell."
    table.rows[0].cells[1].text = "Should also be indexed."

    out = tmp_path / "edge.docx"
    doc.save(str(out))

    parsed = parse_docx(out)
    texts = [p.text for p in parsed.paragraphs]
    assert any("nulled style" in t for t in texts), "None-style paragraph dropped"
    assert any("table cell" in t for t in texts), "Table content not indexed"
    # Headings must still build the section path for content underneath them
    assert any("Section One" in p.section_path for p in parsed.paragraphs)


# =============================================================================
# Phase 2 — connectors, semantic layer, typed graph
# =============================================================================
def test_fda_connector_verbalises_structured_records():
    """Structured rows must become natural-language text, or they are invisible
    to both lexical and semantic retrieval."""
    from pipeline.connectors.fda import FDARegulatoryConnector

    root = pathlib.Path("knowledge_sources/fda_structured")
    if not root.exists():
        pytest.skip("FDA source data not present")

    records = list(FDARegulatoryConnector(root).fetch())
    assert len(records) > 100

    kinds = {r.metadata["record_type"] for r in records}
    assert {"clearance", "recall", "classification"} <= kinds

    clearance = next(r for r in records if r.metadata["record_type"] == "clearance")
    assert clearance.metadata["k_number"].startswith("K")
    assert len(clearance.text) > 120                 # verbalised, not a raw dump
    assert clearance.url.startswith("https://")      # provenance is mandatory
    assert " " in clearance.text.strip()


def test_semantic_index_round_trips_a_query():
    """The pipeline's query embedding must be reproducible — the browser runs the
    identical operation, so any drift here is a silent retrieval bug in production."""
    from pipeline.semantic import build_semantic_index, embed_query, semantic_scores

    # LSA learns association from co-occurrence, so the corpus must actually
    # contain the renal/kidney/creatinine cluster it is expected to recover.
    texts = [
        "Creatinine testing supports renal function screening and kidney assessment.",
        "Renal impairment is detected by creatinine and kidney function testing.",
        "Kidney disease screening uses creatinine measurement and renal markers.",
        "The meter requires quality control testing with control solution vials.",
        "Quality control solution must be run at defined control intervals daily.",
        "Control solution vials are stored and the control test repeated weekly.",
        "Glucose is measured by amperometric biosensor with hematocrit correction.",
        "Hematocrit correction improves glucose biosensor amperometric accuracy.",
    ]
    sem = build_semantic_index(texts)
    assert sem["enabled"]
    assert sem["dims"] >= 4

    qv = embed_query(sem, "renal kidney function")
    assert qv is not None
    scores = semantic_scores(sem, qv)
    assert len(scores) == len(texts)
    # The renal cluster (0-2) must outrank the quality-control cluster (3-5).
    assert max(scores[0], scores[1], scores[2]) > max(scores[3], scores[4], scores[5])


def test_semantic_handles_out_of_vocabulary_query():
    from pipeline.semantic import build_semantic_index, embed_query

    sem = build_semantic_index(["alpha beta gamma delta", "epsilon zeta eta theta",
                                "iota kappa lambda mu", "nu xi omicron pi"])
    if sem.get("enabled"):
        assert embed_query(sem, "zzzz qqqq") is None   # degrade, never crash


def test_graph_builds_typed_cross_source_edges():
    """The whole point of the fabric: an edge that exists in no single source."""
    from pipeline.connectors.base import KnowledgeRecord
    from pipeline.graph import build_knowledge_graph, canonical_product

    assert canonical_product("StatStrip Glucose Hospital Meter System") == "StatStrip Glucose"
    assert canonical_product("Totally Unrelated Widget") is None

    records = [
        KnowledgeRecord(
            source_type="fda_regulatory", source_system="FDA", source_id="K232075",
            title="K232075", text="FDA 510(k) K232075 covers StatStrip Glucose Hospital Meter System.",
            metadata={"record_type": "clearance", "k_number": "K232075",
                      "product_code": "PZI", "device_name": "StatStrip Glucose Hospital Meter System"},
            entities=[("Clearance", "K232075"),
                      ("Product", "StatStrip Glucose Hospital Meter System"),
                      ("ProductCode", "PZI")],
        ),
        KnowledgeRecord(
            source_type="document", source_system="Nova IFU Library",
            source_id="ifu#1", title="StatStrip_Glucose_QRG.pdf",
            text="The StatStrip Glucose meter measures glucose and corrects for hematocrit.",
            metadata={"record_type": "document_paragraph",
                      "document_name": "StatStrip_Glucose_QRG.pdf",
                      "product": "StatStrip Glucose"},
            entities=[("Product", "StatStrip Glucose")],
        ),
    ]
    graph = build_knowledge_graph(records, [0, 1])

    types = {e["type"] for e in graph["edges"]}
    assert "CLEARED_UNDER" in types
    assert "GOVERNED_BY" in types
    assert "MEASURES" in types

    product = next(n for n in graph["nodes"] if n["kind"] == "Product")
    assert set(product["sources"]) == {"fda_regulatory", "document"}
    assert product["cross_source"] is True
    assert graph["stats"]["cross_source_products"] >= 1


def test_connector_registry_skips_missing_sources(tmp_path):
    """A fabric must degrade gracefully when one system is offline."""
    from pipeline.connectors.base import registry_from

    assert registry_from({"documents": tmp_path / "nope",
                          "fda_structured": tmp_path / "also_nope"}) == []
