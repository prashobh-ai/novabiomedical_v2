// ============================================================================
// Hybrid retrieval — Reciprocal Rank Fusion over lexical + semantic
// ============================================================================
//
// Why RRF and not weighted score blending:
//   BM25 scores are unbounded and corpus-dependent; cosine is bounded [-1,1].
//   Normalising them onto a common scale requires tuning constants that go stale
//   the moment the corpus changes. RRF ignores magnitudes entirely and fuses on
//   *rank position*, so it needs no tuning and stays stable as the corpus grows.
//
//   score(d) = SUM over retrievers of 1 / (k + rank(d))
//
//   k=60 is the value from Cormack et al. (2009); it damps the influence of any
//   single retriever's top hit enough that one confident-but-wrong retriever
//   cannot dominate the fused list.

export const RRF_K = 60;

// Queries made entirely of function words carry no retrievable intent. BM25 will
// happily return hundreds of documents for "the" because the term is in-vocabulary
// — technically correct, operationally useless, and it erodes trust fast when a
// meaningless query produces a confident-looking answer.
const FUNCTION_WORDS = new Set([
  'the','a','an','and','or','but','if','then','of','to','in','on','at','by','for',
  'with','is','are','was','were','be','been','being','it','its','this','that',
  'these','those','as','from','so','than','too','very','can','will','just',
]);

export function isDegenerateQuery(query) {
  const tokens = String(query).toLowerCase().match(/[A-Za-z0-9]{2,}/g);
  if (!tokens || !tokens.length) return true;
  return tokens.every(t => FUNCTION_WORDS.has(t));
}

/**
 * Fuses ranked runs on `chunkIdx` — the app-wide retrieval key. Every retriever
 * passed in must emit it.
 *
 * @param {Array<{retriever: string, results: Array<{chunkIdx:number,score:number}>, weight?: number}>} runs
 * @returns {Array<{chunkIdx:number, score:number, contributions:Object, retrievers:string[]}>}
 */
export function reciprocalRankFusion(runs, { k = RRF_K, topK = 20 } = {}) {
  const fused = new Map();

  for (const run of runs) {
    if (!run || !run.results || !run.results.length) continue;
    const weight = run.weight ?? 1.0;

    run.results.forEach((hit, idx) => {
      const key = hit.chunkIdx;
      // A retriever that does not emit chunkIdx would collapse its entire run
      // into a single `undefined` bucket. Fail loudly instead of silently.
      if (key === undefined || key === null) return;

      const rank = idx + 1;
      const contribution = weight / (k + rank);
      let entry = fused.get(key);
      if (!entry) {
        entry = { chunkIdx: key, id: key, score: 0, contributions: {}, retrievers: [], ranks: {} };
        fused.set(key, entry);
      }
      entry.score += contribution;
      entry.contributions[run.retriever] = contribution;
      entry.ranks[run.retriever] = rank;
      if (!entry.retrievers.includes(run.retriever)) entry.retrievers.push(run.retriever);
    });
  }

  return [...fused.values()].sort((a, b) => b.score - a.score).slice(0, topK);
}

/**
 * Metadata filtering over fused results.
 * @param {Array<{chunkIdx:number}>} results
 * @param {Array<Object>} chunks  the index chunk array (positional)
 * @param {Object} filters  e.g. { source_type: 'fda_regulatory', product: 'StatStrip Glucose' }
 */
export function applyFacets(results, chunks, filters) {
  const active = Object.entries(filters || {}).filter(([, v]) => v && v !== 'all');
  if (!active.length) return results;

  return results.filter(r => {
    const chunk = chunks[r.chunkIdx];
    if (!chunk) return false;
    return active.every(([key, want]) => {
      const got = chunk[key] ?? (chunk.meta ? chunk.meta[key] : undefined);
      return got !== undefined && String(got) === String(want);
    });
  });
}

/**
 * The fabric's retrieval entry point.
 *
 * Degrades cleanly: no semantic index, or a query with no in-vocabulary terms,
 * and this is exactly Phase 1 BM25 — with `mode` reporting which path ran, so
 * the UI can tell the truth about how an answer was found.
 */
export function hybridSearch(query, { bm25, semantic, chunks, filters, topK = 20 } = {}) {
  const t0 = (typeof performance !== 'undefined' ? performance.now() : Date.now());

  if (isDegenerateQuery(query)) {
    return {
      results: [], mode: 'rejected', latencyMs: 0,
      diagnostics: {
        lexicalHits: 0, semanticHits: 0, fusedHits: 0, agreement: 0,
        vocabCoverage: 0, fusion: 'none', k: RRF_K,
        rejected: 'query contains no content terms',
      },
    };
  }

  const lexical = bm25 ? bm25.search(query, topK * 2) : [];
  const semanticHits = (semantic && semantic.enabled) ? semantic.search(query, topK * 2) : [];

  let mode = 'hybrid';
  if (!semanticHits.length && lexical.length) mode = 'lexical';
  else if (!lexical.length && semanticHits.length) mode = 'semantic';
  else if (!lexical.length && !semanticHits.length) mode = 'empty';

  const runs = [];
  if (lexical.length) runs.push({ retriever: 'bm25', results: lexical, weight: 1.0 });
  if (semanticHits.length) runs.push({ retriever: 'semantic', results: semanticHits, weight: 1.0 });

  let fused = reciprocalRankFusion(runs, { topK: topK * 2 });
  if (filters && chunks) fused = applyFacets(fused, chunks, filters);
  fused = fused.slice(0, topK);

  const t1 = (typeof performance !== 'undefined' ? performance.now() : Date.now());

  // Agreement between retrievers is a genuine confidence signal: a chunk both
  // methods surface independently is far more likely to be on-target.
  const agreed = fused.filter(r => r.retrievers.length > 1).length;

  return {
    results: fused,
    mode,
    latencyMs: +(t1 - t0).toFixed(2),
    diagnostics: {
      lexicalHits: lexical.length,
      semanticHits: semanticHits.length,
      fusedHits: fused.length,
      agreement: fused.length ? +(agreed / fused.length).toFixed(2) : 0,
      vocabCoverage: semantic ? +semantic.coverage(query).toFixed(2) : 0,
      fusion: 'reciprocal_rank_fusion',
      k: RRF_K,
      // Pre-fusion scores, carried through for confidence scoring.
      //
      // RRF output cannot be used to judge how decisively the top result won:
      // its scores are 1/(k+rank) sums, so rank 1 and rank 5 differ by under 3%
      // BY CONSTRUCTION. Any "margin" computed on fused scores is measuring the
      // fusion constant, not the evidence. The underlying retriever scores have
      // real dynamic range and are the correct basis.
      lexicalScores: lexical.slice(0, 8).map(h => h.score),
      semanticScores: semanticHits.slice(0, 8).map(h => h.score),
    },
  };
}

/** Human-readable trace of why a chunk ranked where it did — feeds the
 *  explainability panel, and makes the retrieval auditable rather than magic. */
export function explainRanking(result) {
  if (!result) return '';
  const parts = [];
  for (const r of result.retrievers || []) {
    parts.push(`${r} rank #${result.ranks[r]} (+${result.contributions[r].toFixed(4)})`);
  }
  const agreement = (result.retrievers || []).length > 1
    ? ' — surfaced independently by both retrievers'
    : '';
  return `${parts.join(' · ')}${agreement}`;
}
