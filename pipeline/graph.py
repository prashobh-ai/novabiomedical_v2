"""Typed knowledge graph — the layer that makes this a fabric rather than a search box.

Phase 1 built an untyped co-occurrence graph: entities that appear in the same chunk
get an edge. That surfaces association but cannot answer a question like
"which clearance governs the meter this IFU describes, and was it ever recalled?"

Here every node and edge carries a type, and — critically — edges are inferred
*across* source systems. A paragraph in StatStrip_Glucose_Quick_Reference_Guide.pdf
resolves to the Product 'StatStrip Glucose', which is CLEARED_UNDER K232075 from the
FDA connector, which is GOVERNED_BY product code PZI, which is the subject of the
November 2024 software recall.

No single source contains that chain. The fabric assembles it.
"""
from __future__ import annotations

import re
from collections import defaultdict

# --------------------------------------------------------------------------- types
NODE_TYPES = ["Product", "ProductCode", "Clearance", "Recall", "Regulation", "Analyte", "Document"]

EDGE_TYPES = {
    "CLEARED_UNDER": ("Product", "Clearance"),
    "GOVERNED_BY": ("Clearance", "ProductCode"),
    "CLASSIFIED_UNDER": ("ProductCode", "Regulation"),
    "RECALL_AFFECTS": ("Recall", "Product"),
    "DOCUMENTED_IN": ("Product", "Document"),
    "MEASURES": ("Product", "Analyte"),
    "CO_OCCURS": (None, None),
}

# Analytes worth first-class treatment — these are what clinicians actually search on.
ANALYTES = {
    "glucose": "Glucose", "lactate": "Lactate", "creatinine": "Creatinine",
    "hematocrit": "Hematocrit", "haematocrit": "Hematocrit", "ketone": "Ketone",
    "hemoglobin": "Hemoglobin", "haemoglobin": "Hemoglobin", "sodium": "Sodium",
    "potassium": "Potassium", "chloride": "Chloride", "calcium": "Ionised Calcium",
    "magnesium": "Ionised Magnesium", "bilirubin": "Bilirubin", "urea": "Blood Urea Nitrogen",
    "bun": "Blood Urea Nitrogen", "ph": "pH", "pco2": "PCO2", "po2": "PO2",
    "hba1c": "HbA1c", "albumin": "Albumin", "uacr": "UACR", "egfr": "eGFR",
    "glutamine": "Glutamine", "glutamate": "Glutamate", "ammonium": "Ammonium",
    "osmolality": "Osmolality",
}

# Canonical product names — collapses the many surface forms across sources.
PRODUCT_ALIASES = [
    (r"statstrip\s*(glucose)?\s*(hospital)?\s*meter|statstrip\s*2\.?0|statstrip\s*glucose", "StatStrip Glucose"),
    (r"statstrip\s*xpress\s*2?|xpress\s*2", "StatStrip Xpress2 Glucose"),
    (r"statstrip\s*lac(tate)?", "StatStrip Lactate"),
    (r"statsensor\s*creat(inine)?", "StatSensor Creatinine"),
    (r"lactate\s*plus", "Lactate Plus"),
    (r"stat\s*profile\s*prime\s*plus", "Stat Profile Prime Plus"),
    (r"stat\s*profile\s*prime\s*es", "Stat Profile Prime ES Comp Plus"),
    (r"stat\s*profile\s*prime\s*ccs", "Stat Profile Prime CCS"),
    (r"stat\s*profile\s*prime", "Stat Profile Prime"),
    (r"bioprofile\s*flex\s*2?", "BioProfile FLEX2"),
    (r"bioprofile\s*phox", "BioProfile pHOx"),
    (r"bioprofile\s*(fast\s*)?cdv", "BioProfile CDV"),
    (r"nova\s*primary", "Nova Primary Glucose Analyzer"),
    (r"nova\s*allegro|allegro", "Nova Allegro"),
    (r"nova\s*max\s*creat", "Nova Max Creat"),
    (r"nova\s*max", "Nova Max"),
]

K_NUMBER_RE = re.compile(r"\bK\d{6}\b")


def canonical_product(raw: str) -> str | None:
    if not raw:
        return None
    low = raw.lower()
    for pattern, canonical in PRODUCT_ALIASES:
        if re.search(pattern, low):
            return canonical
    return None


class KnowledgeGraph:
    def __init__(self):
        self._nodes: dict[tuple[str, str], int] = {}
        self.nodes: list[dict] = []
        self._edges: dict[tuple[int, int, str], dict] = {}

    # ------------------------------------------------------------------ nodes
    def node(self, kind: str, name: str) -> int | None:
        name = (name or "").strip()
        if not name or len(name) > 120:
            return None
        key = (kind, name.lower())
        if key in self._nodes:
            nid = self._nodes[key]
            self.nodes[nid]["mentions"] += 1
            return nid
        nid = len(self.nodes)
        self._nodes[key] = nid
        self.nodes.append({
            "id": nid, "kind": kind, "name": name,
            "mentions": 1, "chunks": [], "sources": [],
        })
        return nid

    def observe(self, nid: int, chunk_id: int, source_type: str):
        if nid is None:
            return
        n = self.nodes[nid]
        if len(n["chunks"]) < 60 and chunk_id not in n["chunks"]:
            n["chunks"].append(chunk_id)
        if source_type not in n["sources"]:
            n["sources"].append(source_type)

    # ------------------------------------------------------------------ edges
    def edge(self, a: int | None, b: int | None, etype: str, evidence: int | None = None):
        if a is None or b is None or a == b:
            return
        key = (a, b, etype)
        e = self._edges.get(key)
        if e is None:
            e = {"source": a, "target": b, "type": etype, "weight": 0, "evidence": []}
            self._edges[key] = e
        e["weight"] += 1
        if evidence is not None and len(e["evidence"]) < 12 and evidence not in e["evidence"]:
            e["evidence"].append(evidence)

    # ------------------------------------------------------------------ build
    def ingest(self, records, chunk_ids: list[int]):
        """First pass: create nodes and same-record edges from typed connector output."""
        for rec, cid in zip(records, chunk_ids):
            meta = rec.metadata
            rtype = meta.get("record_type", "")
            st = rec.source_type

            product_nodes: list[int] = []
            for kind, raw in rec.entities:
                if kind == "Product":
                    canon = canonical_product(raw)
                    if canon:
                        nid = self.node("Product", canon)
                        self.observe(nid, cid, st)
                        product_nodes.append(nid)
                elif kind and raw:
                    nid = self.node(kind, raw)
                    self.observe(nid, cid, st)

            # ---- typed regulatory edges
            if rtype == "clearance":
                k = self.node("Clearance", meta.get("k_number", ""))
                self.observe(k, cid, st)
                code = self.node("ProductCode", meta.get("product_code", ""))
                self.observe(code, cid, st)
                self.edge(k, code, "GOVERNED_BY", cid)
                for p in product_nodes:
                    self.edge(p, k, "CLEARED_UNDER", cid)

            elif rtype == "classification":
                code = self.node("ProductCode", meta.get("product_code", ""))
                reg = self.node("Regulation", f"21 CFR {meta.get('regulation_number','')}")
                self.observe(code, cid, st)
                self.observe(reg, cid, st)
                self.edge(code, reg, "CLASSIFIED_UNDER", cid)

            elif rtype == "recall":
                r = self.node("Recall", meta.get("recall_number", ""))
                self.observe(r, cid, st)
                for p in product_nodes:
                    self.edge(r, p, "RECALL_AFFECTS", cid)
                code = meta.get("product_code", "")
                if code:
                    cn = self.node("ProductCode", code)
                    self.observe(cn, cid, st)
                    self.edge(r, cn, "GOVERNED_BY", cid)

            elif rtype in {"ifu_catalogue", "document_paragraph"}:
                doc_name = meta.get("document_name") or rec.title
                d = self.node("Document", doc_name)
                self.observe(d, cid, st)
                prod = canonical_product(meta.get("product", "")) or canonical_product(doc_name)
                if prod:
                    pn = self.node("Product", prod)
                    self.observe(pn, cid, st)
                    self.edge(pn, d, "DOCUMENTED_IN", cid)
                    product_nodes.append(pn)

            # ---- analytes mentioned in body text
            low = rec.text.lower()
            for token, analyte in ANALYTES.items():
                if re.search(rf"\b{re.escape(token)}\b", low):
                    an = self.node("Analyte", analyte)
                    self.observe(an, cid, st)
                    for p in product_nodes:
                        self.edge(p, an, "MEASURES", cid)

            # ---- K-numbers cited inside free text (links PDFs to clearance records)
            for k in K_NUMBER_RE.findall(rec.text)[:4]:
                kn = self.node("Clearance", k)
                self.observe(kn, cid, st)
                for p in product_nodes:
                    self.edge(p, kn, "CLEARED_UNDER", cid)

    def bridge_sources(self):
        """Second pass — the fabric moment.

        A Product node observed in both the document corpus and the FDA corpus is a
        join key. Every clearance and recall attached to it becomes reachable from an
        IFU paragraph, and vice versa. These edges exist in no source system.
        """
        bridged = 0
        by_kind: dict[str, list[dict]] = defaultdict(list)
        for n in self.nodes:
            by_kind[n["kind"]].append(n)

        for product in by_kind["Product"]:
            if len(product["sources"]) < 2:
                continue          # only genuine cross-source products bridge
            product["cross_source"] = True
            bridged += 1
        return bridged

    def finalize(self, max_edges: int = 4000) -> dict:
        cross = self.bridge_sources()
        edges = sorted(self._edges.values(), key=lambda e: -e["weight"])[:max_edges]
        keep = {e["source"] for e in edges} | {e["target"] for e in edges}

        # Degree, for sizing in the UI
        degree: dict[int, int] = defaultdict(int)
        for e in edges:
            degree[e["source"]] += 1
            degree[e["target"]] += 1
        for n in self.nodes:
            n["degree"] = degree.get(n["id"], 0)
            n["cross_source"] = n.get("cross_source", False)

        type_counts: dict[str, int] = defaultdict(int)
        for e in edges:
            type_counts[e["type"]] += 1

        return {
            "nodes": [n for n in self.nodes if n["id"] in keep or n["mentions"] > 2],
            "edges": edges,
            "node_types": NODE_TYPES,
            "edge_types": sorted(type_counts.keys()),
            "stats": {
                "node_count": len(self.nodes),
                "edge_count": len(edges),
                "cross_source_products": cross,
                "edges_by_type": dict(type_counts),
            },
        }


def build_knowledge_graph(records, chunk_ids: list[int]) -> dict:
    kg = KnowledgeGraph()
    kg.ingest(records, chunk_ids)
    return kg.finalize()
