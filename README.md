<h2 align="center">Nova Biomedical</h2>

<h1 align="center">Knowledge Fabric</h1>

<p align="center">
Transforming Enterprise Knowledge into an Explainable AI-Powered Intelligence Network
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Phase-2%20Hybrid%20Retrieval-blue" />
  <img src="https://img.shields.io/badge/Open%20Source-Yes-green" />
  <img src="https://img.shields.io/badge/Knowledge%20Graph-Typed-purple" />
  <img src="https://img.shields.io/badge/Explainable%20AI-Citation%20Backed-orange" />
  <img src="https://img.shields.io/badge/Retrieval-Evaluated-red" />
  <img src="https://img.shields.io/badge/Infra%20Cost-%240-lightgrey" />
</p>

---

> **Innovation Initiative / Proof of Concept**
>
> A Knowledge Fabric that turns heterogeneous enterprise knowledge — documents,
> structured regulatory records, and the relationships between them — into a
> searchable, explainable, evaluated knowledge network.
>
> Phase 2 adds semantic retrieval, hybrid fusion, a typed cross-source knowledge
> graph, and a retrieval evaluation harness that gates deployment.

---

# Why Knowledge Fabric

Enterprise knowledge is scattered across documents, PDFs, spreadsheets, Confluence,
Jira, and repositories. Traditional search helps you find *documents*.

**Knowledge Fabric helps you find answers, relationships, evidence, and context —
and shows its working.**

The distinguishing claim is testable: *can it answer a question that no single
source system can answer alone?* Phase 2 can, and the evaluation harness proves it.

---

# What is running today

| Capability | Status | Implementation |
|---|---|---|
| Multi-source ingestion | Live | `pipeline/connectors/` — pluggable connector contract |
| Document intelligence | Live | PDF / DOCX / MD / TXT, heading-aware, page-accurate |
| Structured-record ingestion | Live | FDA regulatory records, verbalised for retrieval |
| Lexical retrieval | Live | BM25, client-side |
| **Semantic retrieval** | **Live** | **LSA over TF-IDF, embedded in-browser** |
| **Hybrid retrieval** | **Live** | **Reciprocal Rank Fusion (k=60)** |
| **Typed knowledge graph** | **Live** | **6 edge types, cross-source inference** |
| **Retrieval evaluation** | **Live** | **16-case gold set, Recall/MRR/nDCG, CI gate** |
| Citation-backed answers | Live | Answer → source → page → excerpt |
| Faceted filtering | Live | product, doc type, year, product code, source |
| Zero-cost deployment | Live | GitHub Pages, static, no backend |

---

# Architecture

```
                      +--------------------------------------+
  docs_source/ ------>|  DocumentConnector                   |
  (PDF/DOCX/MD)       |  heading-aware, page-accurate        |
                      +--------------------------------------+
  knowledge_sources/ >|  FDARegulatoryConnector              |--+
  (structured CSV)    |  verbalises rows into retrievable    |  |
                      |  prose + typed facets                |  |
                      +--------------------------------------+  |
                                                                v
                                                    +-----------------------+
                                                    |   KnowledgeRecord     |
                                                    |  one shape for all    |
                                                    +-----------+-----------+
                                                                v
                            +---------------+-------------------+-------------------+
                            v               v                   v                   v
                      +----------+   +------------+   +------------------+   +----------+
                      |  BM25    |   |  LSA/SVD   |   |  Typed Knowledge |   |  Facets  |
                      | lexical  |   |  semantic  |   |      Graph       |   |          |
                      +----+-----+   +-----+------+   +--------+---------+   +----+-----+
                           +-------+-------+                   |                  |
                                   v                           |                  |
                        +--------------------+                 |                  |
                        |  Reciprocal Rank   |                 |                  |
                        |   Fusion (k=60)    |                 |                  |
                        +---------+----------+                 |                  |
                                  +--------------+-------------+------------------+
                                                 v
                                   site/  -  static, client-side, $0
```

Everything after ingestion runs **in the browser**. There is no server, no vector
database, no API key, and no inference cost.

---

# The two engineering decisions worth explaining

### 1. Semantics without a transformer

A sentence-transformer would mean a ~90 MB model at build time and either a serving
process or a 20 MB+ ONNX payload in the browser. That breaks the zero-cost static
hosting constraint this project is built on.

Instead: **Truncated SVD over TF-IDF** (Latent Semantic Analysis). The pipeline ships
a term-to-vector table, IDF weights, and per-chunk vectors — int8 quantised, 1.4 MB
total. The browser embeds a query as an IDF-weighted mean of its term vectors,
L2-normalised. `pipeline/semantic.py::embed_query` and `site/js/semantic.js::embed`
perform the identical operation, so build and client agree exactly on what "similar"
means.

It is weaker than a transformer on paraphrase. It is dramatically better than BM25
alone on **vocabulary mismatch**, which is the real failure mode in this corpus:
a clinician asks about "kidney function", the source says "creatinine clearance".
BM25 scores that zero.

### 2. Rank fusion, not score blending

BM25 scores are unbounded and corpus-dependent; cosine is bounded [-1, 1]. Normalising
them onto a shared scale needs tuning constants that go stale the moment the corpus
changes. **RRF ignores magnitudes and fuses on rank position** — no tuning, stable as
the corpus grows.

```
score(d) = SUM over retrievers of  1 / (k + rank(d)),   k = 60
```

Agreement between retrievers becomes a genuine confidence signal: a chunk both
methods surface independently is far more likely to be on target, and the UI says so.

---

# The fabric moment

The typed graph infers edges **across** source systems. No single source contains
this chain:

```
StatStrip_Glucose_Quick_Reference_Guide.pdf   (document corpus)
        |  DOCUMENTED_IN
        v
   StatStrip Glucose            <- Product node, observed in BOTH corpora
        |  CLEARED_UNDER
        v
      K232075                   (FDA corpus)
        |  GOVERNED_BY
        v
        PZI                     <- the critically-ill glucose product code
        |  CLASSIFIED_UNDER
        v
  21 CFR 862.1345
```

A Product node observed in two source systems is a join key. Every clearance and
recall attached to it becomes reachable from an IFU paragraph, and vice versa.

Current build bridges **10 products** across the document and regulatory corpora.

Edge types: `CLEARED_UNDER` · `GOVERNED_BY` · `CLASSIFIED_UNDER` · `RECALL_AFFECTS`
· `DOCUMENTED_IN` · `MEASURES`

---

# Measured retrieval quality

Not asserted — measured, on every build, with a gold set that declares what correct
retrieval looks like independent of answer phrasing.

```
corpus: 1,360 chunks · 391 documents · 2 sources
gold set: 16 cases

mode          recall@5  recall@10      MRR   nDCG@10    latency
-----------------------------------------------------------------
lexical          0.844      0.844    0.771     0.807     28.4ms
semantic         0.844      0.906    0.709     0.789     26.8ms
hybrid           0.875      0.875    0.828     0.836     27.0ms
```

**Hybrid wins on ranking quality** (MRR +0.057, nDCG +0.029 over lexical) — it puts
the right evidence *higher*, which is what determines whether it survives into a
context window.

**An honest finding:** semantic-only achieves higher recall@10 (0.906) than hybrid
(0.875). Equal-weight RRF lets BM25 pull down queries where lexical matching is
actively misleading. Retriever weights are configurable, but tuning them against a
16-case set would be overfitting. **The correct next step is expanding the gold set,
not tuning the constant** — which is exactly why the harness exists.

Recall@10 by category:

| Category | Recall@10 | |
|---|---|---|
| cross_source | 1.00 | the fabric test — passes |
| lexical | 1.00 | |
| regulatory | 1.00 | |
| robustness | 1.00 | degenerate queries correctly rejected |
| semantic | 0.50 | see known gaps |

### Known gaps

- **`sem-004` (cell culture / bioprocessing) fails** — a genuine *corpus coverage*
  gap, not a retrieval failure. No BioProfile documentation is in `docs_source/`.
  The harness correctly surfaces missing coverage rather than hiding it.
- **`sem-003` (recalls via "safety problems") fails** — real semantic distance the
  96-dim LSA projection does not span. A transformer would likely close this; that
  is the Phase 3 trade-off to evaluate.

---

# Quick start

```bash
pip install -r pipeline/requirements.txt

# Build the fabric
python -m pipeline.build_index

# Measure retrieval quality
python -m eval.evaluate

# Fail on regression (what CI runs)
python -m eval.evaluate --gate

# Serve locally
cd site && python -m http.server 8000
```

---

# Repository structure

```
docs_source/                  Source documents
  fda_regulatory/               FDA clearance + review PDFs
knowledge_sources/
  fda_structured/               Structured regulatory records (CSV)
pipeline/
  connectors/                   Source connector layer
    base.py                       KnowledgeRecord contract
    documents.py                  PDF/DOCX/MD/TXT
    fda.py                        FDA regulatory records
  parsers.py                    Heading-aware document parsing
  chunker.py                    Citation-preserving chunking
  bm25_index.py                 Lexical index
  semantic.py                   LSA semantic index
  graph.py                      Typed cross-source knowledge graph
  build_index.py                Build orchestrator
eval/
  gold_set.yaml                 Declarative retrieval gold set
  evaluate.py                   Recall / MRR / nDCG + CI gate
site/
  js/semantic.js                In-browser query embedding
  js/hybrid.js                  RRF fusion + facet filtering
tests/                        14 tests
.github/workflows/            CI + Pages deploy with quality gate
```

---

# Data provenance

**FDA regulatory data** — U.S. FDA public records retrieved via the openFDA API and
`accessdata.fda.gov`. Government public records, published for programmatic access,
no redistribution restriction.

**Nova product documentation** — Nova Biomedical copyright, obtained through Nova's
published channels. Included here to demonstrate the ingestion pipeline.

---

# Roadmap

### Phase 1 — Foundation (done)
Document ingestion · citation engine · knowledge graph · lineage · static deployment

### Phase 2 — Enterprise Search (done)
Semantic retrieval · hybrid RRF · typed cross-source graph · faceted filtering ·
retrieval evaluation with CI gating

### Phase 3 — Enterprise Knowledge Fabric
Confluence / Jira / SharePoint / GitHub connectors (the connector contract is already
in place) · automated sync · transformer embeddings evaluated against the LSA baseline
using the existing harness · graph engine (Neo4j / FalkorDB) for multi-hop traversal

### Phase 4 — Agentic AI
Research agents · impact-analysis agents · enterprise copilots · autonomous knowledge
workflows — grounded on an evaluated retrieval layer rather than an unmeasured one

---

# Why this matters

Most AI initiatives struggle because they lack trusted enterprise context, and most
RAG demos never measure whether retrieval actually works.

Knowledge Fabric provides trusted context, explainable answers, cross-source
relationships — and a number that tells you whether it is getting better or worse.

The chatbot is one interface. **The fabric is the foundation.**
