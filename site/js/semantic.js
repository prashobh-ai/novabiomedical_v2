// ============================================================================
// Semantic retrieval — dense vectors, in the browser, with no backend
// ============================================================================
//
// The pipeline ships an LSA projection: a term->vector table, IDF weights, and
// one vector per chunk, all int8 quantised. Embedding a query here is the exact
// same operation the pipeline performs (pipeline/semantic.py::embed_query), so
// client and build agree bit-for-bit on what "similar" means.
//
// Cost: ~1.4 MB fetched once, lazily, after first paint. Scoring 1,360 chunks
// against a 96-dim query is a single pass of ~130k multiply-adds — sub-millisecond.
//
// This is what closes the vocabulary gap. A clinician asks about "kidney
// function"; the IFU says "creatinine clearance". BM25 scores that zero.

const TOKEN_RE = /[A-Za-z0-9]{2,}/g;

export class SemanticIndex {
  constructor(payload) {
    this.enabled = Boolean(payload && payload.enabled);
    if (!this.enabled) {
      this.reason = (payload && payload.reason) || 'not built';
      return;
    }

    this.dims = payload.dims;
    this.method = payload.method;
    this.explainedVariance = payload.explained_variance;

    // term -> row index
    this.termIndex = new Map();
    payload.terms.forEach((t, i) => this.termIndex.set(t, i));
    this.idf = Float32Array.from(payload.idf);

    // Dequantise once at load. Flat typed arrays keep scoring cache-friendly.
    this.termVectors = this._flatten(payload.term_vectors, payload.term_scale);
    this.docVectors = this._flatten(payload.doc_vectors, payload.doc_scale);
    this.docCount = payload.doc_vectors.length;
  }

  _flatten(rows, scale) {
    const d = rows[0].length;
    const out = new Float32Array(rows.length * d);
    const f = scale / 127.0;
    for (let i = 0; i < rows.length; i++) {
      const row = rows[i];
      const base = i * d;
      for (let j = 0; j < d; j++) out[base + j] = row[j] * f;
    }
    return out;
  }

  /** IDF-weighted mean of term vectors, L2-normalised. Null when nothing is in vocab. */
  embed(query) {
    if (!this.enabled) return null;
    const tokens = String(query).toLowerCase().match(TOKEN_RE);
    if (!tokens) return null;

    const d = this.dims;
    const vec = new Float32Array(d);
    let hits = 0;

    for (const tok of tokens) {
      const ti = this.termIndex.get(tok);
      if (ti === undefined) continue;
      const w = this.idf[ti];
      const base = ti * d;
      for (let j = 0; j < d; j++) vec[j] += this.termVectors[base + j] * w;
      hits++;
    }
    if (!hits) return null;

    let norm = 0;
    for (let j = 0; j < d; j++) norm += vec[j] * vec[j];
    norm = Math.sqrt(norm);
    if (!norm) return null;
    for (let j = 0; j < d; j++) vec[j] /= norm;
    return vec;
  }

  /** Ranked [{ chunkIdx, score }] by cosine similarity. Vectors are pre-normalised,
   *  so cosine is a plain dot product.
   *
   *  The `chunkIdx` key is not arbitrary: it is the retrieval contract the whole
   *  app is built on (search.js BM25, answer.js citations, explain.js traces all
   *  index `chunks[r.chunkIdx]`). A retriever that invents its own key name
   *  silently produces `chunks[undefined]` downstream. */
  search(query, topK = 20) {
    const qv = this.embed(query);
    if (!qv) return [];

    const d = this.dims;
    const scored = new Array(this.docCount);
    for (let i = 0; i < this.docCount; i++) {
      const base = i * d;
      let s = 0;
      for (let j = 0; j < d; j++) s += this.docVectors[base + j] * qv[j];
      scored[i] = { chunkIdx: i, score: s };
    }
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, topK).filter(r => r.score > 0.01);
  }

  /** Vocabulary coverage for a query — drives the honest "why" in the UI.
   *  Low coverage means semantic results should be trusted less. */
  coverage(query) {
    if (!this.enabled) return 0;
    const tokens = String(query).toLowerCase().match(TOKEN_RE);
    if (!tokens || !tokens.length) return 0;
    let hit = 0;
    for (const t of tokens) if (this.termIndex.has(t)) hit++;
    return hit / tokens.length;
  }
}

/** Lazy fetch. A failure here must never take the app down — the fabric falls
 *  back to lexical-only retrieval and says so. */
export async function loadSemanticIndex(url = 'data/semantic.json') {
  try {
    const res = await fetch(url, { cache: 'no-cache' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return new SemanticIndex(await res.json());
  } catch (err) {
    console.warn('[fabric] semantic layer unavailable, using lexical only:', err.message);
    return new SemanticIndex({ enabled: false, reason: err.message });
  }
}
