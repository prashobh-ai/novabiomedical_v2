"""Knowledge Fabric build orchestrator.

    python -m pipeline.build_index

Pipeline:
    connectors  ->  records  ->  chunks  ->  BM25 (lexical)
                                        ->  LSA   (semantic)
                                        ->  typed knowledge graph
                                        ->  facets
                             ->  site/data/index.json + semantic.json

The output is a superset of the Phase 1 schema, so the existing site keeps working
while new capabilities light up progressively.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from pathlib import Path as _Path

from .bm25_index import build_bm25_index
from .chunker import Chunk, tokenize
from .connectors.base import registry_from
from .docmeta import resolve as resolve_date
from .graph import build_knowledge_graph
from .semantic import build_semantic_index
from .textnorm import analyse, build_vocabulary, chunk_quality, normalize

TARGET_CHARS = 700


def records_to_chunks(records: list) -> tuple[list[Chunk], list]:
    """Documents are packed to a target size; structured records stay atomic.

    An FDA clearance row is already exactly one complete thought - splitting or
    merging it would damage both retrieval and citation precision. Document
    paragraphs, by contrast, are fragments and need packing.
    """
    chunks: list[Chunk] = []
    aligned: list = []
    cid = 0
    doc_ids: dict[str, int] = {}
    buffer: list = []
    buf_key = None

    def doc_id_for(name: str) -> int:
        if name not in doc_ids:
            doc_ids[name] = len(doc_ids)
        return doc_ids[name]

    def flush():
        nonlocal buffer, cid
        if not buffer:
            return
        first = buffer[0]
        text = "\n\n".join(r.text for r in buffer)
        name = first.metadata.get("document_name") or first.title
        chunks.append(Chunk(
            chunk_id=cid, document_id=doc_id_for(name), document_name=name,
            page=first.page, section_path=list(first.section_path),
            paragraph_indices=[r.metadata.get("paragraph_index", 0) for r in buffer],
            text=text, paragraph_excerpt=first.text[:200], tokens=tokenize(text),
        ))
        aligned.append(first)
        cid += 1
        buffer = []

    for rec in records:
        if rec.metadata.get("record_type") != "document_paragraph":
            flush()
            name = rec.metadata.get("document_name") or rec.title
            chunks.append(Chunk(
                chunk_id=cid, document_id=doc_id_for(name), document_name=name,
                page=rec.page, section_path=list(rec.section_path),
                paragraph_indices=[0], text=rec.text,
                paragraph_excerpt=rec.text[:200], tokens=tokenize(rec.text),
            ))
            aligned.append(rec)
            cid += 1
            continue
        key = (rec.metadata.get("document_name"), tuple(rec.section_path), rec.page)
        if buf_key is not None and key != buf_key:
            flush()
        buf_key = key
        buffer.append(rec)
        if sum(len(r.text) for r in buffer) >= TARGET_CHARS:
            flush()
    flush()
    return chunks, aligned


def build_facets(records: list) -> dict:
    facets: dict = defaultdict(lambda: defaultdict(int))
    for r in records:
        facets["source_type"][r.source_type] += 1
        facets["source_system"][r.source_system] += 1
        for key in ("record_type", "product", "doc_type", "domain", "year",
                    "product_code", "advisory_committee", "root_cause"):
            v = r.metadata.get(key)
            if v:
                facets[key][str(v)] += 1
    return {k: dict(sorted(v.items(), key=lambda kv: -kv[1])[:60]) for k, v in facets.items()}


def build(source_dir: Path, fda_dir: Path, out_path: Path, semantic_path: Path) -> dict:
    print("=" * 66)
    print("  KNOWLEDGE FABRIC - BUILD")
    print("=" * 66)

    connectors = registry_from({"documents": source_dir, "fda_structured": fda_dir})
    if not connectors:
        print(f"[!] No sources found ({source_dir}, {fda_dir})", file=sys.stderr)
        sys.exit(1)

    print(f"\n[1/6] Ingesting from {len(connectors)} connector(s)")
    records: list = []
    for c in connectors:
        got = list(c.fetch())
        records.extend(got)
        print(f"      {c.name:28s} {len(got):>6} records  [{c.source_type}]")
    print(f"      {'TOTAL':28s} {len(records):>6} records")

    # ---- Text normalisation -------------------------------------------------
    # Runs before chunking so that BM25, LSA and the graph all index repaired
    # text. Repairing at query time would be too late: the indexes would already
    # have been built over "point -of-care" and "Mea surement".
    print("\n[2/7] Normalising text")
    vocab = build_vocabulary([r.text for r in records])
    repaired = 0
    for r in records:
        cleaned = normalize(r.text, vocab)
        if cleaned != r.text:
            repaired += 1
        r.text = cleaned
    print(f"      corpus lexicon {len(vocab)} words · {repaired}/{len(records)} records repaired")

    print("\n[3/7] Chunking")
    chunks, aligned = records_to_chunks(records)
    docs_n = sum(1 for r in aligned if r.source_type == "document")
    print(f"      {len(chunks)} chunks  ({docs_n} from documents, {len(chunks)-docs_n} structured)")

    print("\n[4/7] Lexical index (BM25)")
    bm25 = build_bm25_index(chunks)
    print(f"      vocab={len(bm25['vocab'])}  avgdl={bm25['avgdl']:.1f}")

    print("\n[5/7] Semantic index (LSA)")
    semantic = build_semantic_index([c.text for c in chunks])
    if semantic.get("enabled"):
        print(f"      dims={semantic['dims']}  vocab={semantic['stats']['vocabulary']}  "
              f"variance={semantic['explained_variance']:.1%}")
    else:
        print(f"      disabled: {semantic.get('reason')}")

    print("\n[6/7] Knowledge graph")
    graph = build_knowledge_graph(aligned, [c.chunk_id for c in chunks])
    gs = graph["stats"]
    print(f"      {gs['node_count']} nodes  {gs['edge_count']} edges  "
          f"{gs['cross_source_products']} cross-source products")
    for et, n in sorted(gs["edges_by_type"].items(), key=lambda kv: -kv[1]):
        print(f"        {et:20s} {n}")

    print("\n[7/7] Assembling index")

    # The browser reads entities using the Phase 1 field names (mention_count,
    # chunk_ids, document_ids). Those names are a published contract: graph.js,
    # insights.js and main.js all index them directly, and a renamed field does
    # not fail loudly — it silently yields `undefined` deep inside a click
    # handler. Emit both the legacy names and the Phase 2 additions.
    chunk_to_doc = {c.chunk_id: c.document_id for c in chunks}
    entities = []
    for n in graph["nodes"]:
        chunk_ids = n["chunks"]
        document_ids = sorted({chunk_to_doc[c] for c in chunk_ids if c in chunk_to_doc})
        entities.append({
            "id": n["id"],
            "name": n["name"],
            "canonical": n["name"].lower(),
            "kind": n["kind"],
            # --- Phase 1 contract (consumed by site/js) ---
            "mention_count": n["mentions"],
            "chunk_ids": chunk_ids,
            "document_ids": document_ids,
            # --- Phase 2 additions ---
            "count": n["mentions"],
            "mentions": n["mentions"],
            "chunks": chunk_ids,
            "sources": n["sources"],
            "cross_source": n.get("cross_source", False),
            "degree": n.get("degree", 0),
        })

    relationships = [{
        "source": e["source"], "target": e["target"], "weight": e["weight"],
        "kind": e["type"], "type": e["type"], "evidence_chunks": e["evidence"],
    } for e in graph["edges"]]

    chunk_entities: list[list[int]] = [[] for _ in chunks]
    idx_of = {c.chunk_id: i for i, c in enumerate(chunks)}
    for n in graph["nodes"]:
        for ch in n["chunks"]:
            i = idx_of.get(ch)
            if i is not None and n["id"] not in chunk_entities[i]:
                chunk_entities[i].append(n["id"])

    documents_meta: dict = {}
    doc_text_head: dict = {}
    for c, rec in zip(chunks, aligned):
        doc_text_head.setdefault(c.document_name, "")
        if len(doc_text_head[c.document_name]) < 4000:
            doc_text_head[c.document_name] += " " + c.text

    for c, rec in zip(chunks, aligned):
        d = documents_meta.setdefault(c.document_name, {
            "id": c.document_id, "name": c.document_name,
            "source_type": rec.source_type, "source_system": rec.source_system,
            "domain": rec.metadata.get("domain", "Regulatory Data"),
            "product": rec.metadata.get("product", ""),
            "doc_type": rec.metadata.get("doc_type", rec.metadata.get("record_type", "")),
            "url": rec.url, "page_count": rec.metadata.get("page_count", 1),
            "chunk_count": 0,
            **{f"date_{k}": v for k, v in resolve_date(
                source_path=_Path(rec.metadata["source_path"])
                if rec.metadata.get("source_path") else None,
                text=doc_text_head.get(c.document_name, ""),
                regulatory_date=rec.metadata.get("decision_date")
                or rec.metadata.get("date_initiated")
                or rec.metadata.get("publish_date")
                or rec.metadata.get("created_date"),
            ).items()},
        })
        d["chunk_count"] += 1

    # Sentence layer. Shipped as offsets rather than duplicated strings — the
    # client slices chunk.text, so this costs ~4 numbers per sentence instead of
    # a second copy of the corpus.
    serialized = []
    kind_counts: dict = defaultdict(int)
    for i, (c, rec) in enumerate(zip(chunks, aligned)):
        infos = analyse(c.text)
        for si in infos:
            kind_counts[si.kind] += 1
        serialized.append({
            "id": c.chunk_id, "document_id": c.document_id, "document_name": c.document_name,
            "page": c.page, "section_path": c.section_path,
            "paragraph_indices": c.paragraph_indices, "text": c.text,
            "paragraph_excerpt": c.paragraph_excerpt, "entities": chunk_entities[i],
            "source_type": rec.source_type, "source_system": rec.source_system,
            "url": rec.url, "meta": rec.metadata,
            "quality": chunk_quality(infos),
            "sents": [{"o": si.offset, "l": si.length, "k": si.kind, "q": si.quality}
                      for si in infos],
        })
    total_sents = sum(kind_counts.values())
    usable = kind_counts.get("prose", 0)
    print(f"      sentences {total_sents} · usable prose {usable} "
          f"({usable / max(total_sents, 1):.0%})")
    for k, n in sorted(kind_counts.items(), key=lambda kv: -kv[1]):
        print(f"        {k:10} {n}")

    index = {
        "version": "2.0",
        "generator": "knowledge-fabric/phase-2",
        "documents": list(documents_meta.values()),
        "chunks": serialized,
        "entities": entities,
        "relationships": relationships,
        "graph": {"node_types": graph["node_types"], "edge_types": graph["edge_types"], "stats": gs},
        "facets": build_facets(aligned),
        "bm25": bm25,
        "retrieval": {
            "modes": ["lexical", "semantic", "hybrid"],
            "default": "hybrid" if semantic.get("enabled") else "lexical",
            "fusion": "reciprocal_rank_fusion", "rrf_k": 60,
        },
        "stats": {
            "document_count": len(documents_meta), "chunk_count": len(chunks),
            "entity_count": len(entities), "relationship_count": len(relationships),
            "vocab_size": len(bm25["vocab"]), "source_count": len(connectors),
            "record_count": len(records),
            "cross_source_products": gs["cross_source_products"],
            "semantic_enabled": bool(semantic.get("enabled")),
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, separators=(",", ":"))
    with open(semantic_path, "w", encoding="utf-8") as f:
        json.dump(semantic, f, separators=(",", ":"))

    i_mb = out_path.stat().st_size / 1024 / 1024
    s_mb = semantic_path.stat().st_size / 1024 / 1024
    print(f"\n[OK] {out_path}      {i_mb:.2f} MB")
    print(f"[OK] {semantic_path}  {s_mb:.2f} MB")
    print(f"[OK] total payload    {i_mb + s_mb:.2f} MB")
    print("=" * 66)
    return index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=Path("docs_source"))
    ap.add_argument("--fda", type=Path, default=Path("knowledge_sources/fda_structured"))
    ap.add_argument("--out", type=Path, default=Path("site/data/index.json"))
    ap.add_argument("--semantic-out", type=Path, default=Path("site/data/semantic.json"))
    args = ap.parse_args()
    build(args.source, args.fda, args.out, args.semantic_out)


if __name__ == "__main__":
    main()
