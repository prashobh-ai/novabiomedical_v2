// =============================================================================
// Answer composer — query-focused extractive summarisation
// =============================================================================
//
// Replaces the previous "top-N sentences by query-term overlap, concatenated"
// composer. That approach produced the failures seen in demo 2:
//
//   Q "clinical significance of measuring lactate"
//     -> a UDI registration record, because it repeats "lactate" densely
//   Q "intended use of the StatStrip Glucose meter"
//     -> the same intended-use sentence four times, lightly reworded
//   Q "how does hematocrit affect creatinine measurement"
//     -> opened with a warranty disclaimer
//
// The pipeline now is:
//
//   NLU  ->  evidence routing  ->  sentence scoring  ->  MMR  ->  assembly
//
// with a real confidence figure computed from the retrieval run rather than a
// constant. Each stage is separately inspectable, which is what makes the
// answer defensible in front of a technical audience.

import { analyseQuestion, rerankByIntent } from './nlu.js?v=7';
import { summarise } from './summarize.js?v=7';
import { computeConfidence } from './confidence.js?v=7';

/**
 * @param {string} query
 * @param {Array} ranked      [{chunkIdx, score}]
 * @param {Array} chunks      index chunk array
 * @param {Object} cohesion   from cohereByDocument (retained for compatibility)
 * @param {Object} opts       { semantic, bm25, diagnostics }
 */
export function buildAnswer(query, ranked, chunks, cohesion = {}, opts = {}) {
  const { semantic = null, bm25 = null, diagnostics = null } = opts;

  if (!ranked || !ranked.length) {
    return {
      answerHtml: 'No passage in the corpus matched that question.',
      citations: [], lowConfidence: true, confidence: null, analysis: null,
    };
  }

  const analysis = analyseQuestion(query);

  const idf = (term) => {
    if (!bm25 || !bm25.termId || !bm25.idf) return 1;
    const id = bm25.termId[term];
    return (id === undefined) ? 2.2 : (bm25.idf[id] ?? 1);
  };

  // Evidence routing runs here, on the retrieved pool, before summarisation.
  // Applying it later would be too late — the pool would already be full of
  // records that match the words but cannot answer the question.
  const routed = rerankByIntent(ranked, chunks, analysis.intent);
  const pool = routed.length >= 3 ? routed : ranked;

  const summary = summarise(query, pool, chunks, { analysis, semantic, bm25 });

  // ---- nothing survived routing + quality filters --------------------------
  if (!summary.sentences.length) {
    const top = chunks[ranked[0].chunkIdx];
    const conf = computeConfidence({ analysis, ranked: pool, sentences: [], diagnostics, idf });
    const routedOut = summary.rejected > 0;
    return {
      answerHtml:
        `<p class="answer-none">No passage in the corpus answers this directly.</p>` +
        `<p class="answer-none-detail">` +
        (routedOut
          ? `${summary.rejected} candidate passage${summary.rejected === 1 ? ' was' : 's were'} ` +
            `found but filtered out — they matched the words in your question without ` +
            `containing ${analysis.intent.wants || 'a usable answer'}. `
          : '') +
        `The nearest related document is <strong>${top.document_name}</strong>.</p>`,
      citations: [{ num: 1, chunkIdx: ranked[0].chunkIdx, chunk: top, score: ranked[0].score, confidence: 0.2 }],
      lowConfidence: true,
      confidence: conf,
      analysis,
      summary,
    };
  }

  // ---- citation numbering, in order of first appearance --------------------
  const citationsByChunk = new Map();
  const pieces = [];

  for (const s of summary.sentences) {
    if (!citationsByChunk.has(s.chunkIdx)) {
      citationsByChunk.set(s.chunkIdx, {
        num: citationsByChunk.size + 1,
        chunkIdx: s.chunkIdx,
        chunk: s.chunk,
        score: s.final,
        signals: s.signals,
      });
    }
    const cite = citationsByChunk.get(s.chunkIdx);
    pieces.push(
      `<span class="answer-sent">${escapeHtml(s.display)}` +
      `<sup class="cite-ref" data-cite="${cite.num}">[${cite.num}]</sup></span>`
    );
  }

  const citations = [...citationsByChunk.values()];
  const maxScore = Math.max(...citations.map(c => c.score), 1e-6);
  for (const c of citations) c.confidence = c.score / maxScore;

  const confidence = computeConfidence({
    analysis, ranked: pool, sentences: summary.sentences, diagnostics, idf,
  });

  // A lead-in line naming what kind of answer this is makes the routing visible
  // and sets the reader's expectation before the evidence.
  const lead = analysis.intent.name && analysis.intent.name !== 'GENERAL'
    ? `<p class="answer-lead">${labelFor(analysis)}</p>`
    : '';

  return {
    answerHtml: lead + `<p class="answer-body">${pieces.join(' ')}</p>`,
    citations,
    lowConfidence: confidence.percent < 45,
    confidence,
    analysis,
    summary,
    primarySource: cohesion.dominantDoc || null,
  };
}

function labelFor(analysis) {
  const focus = analysis.focus.length
    ? analysis.focus.slice(0, 2).map(titleCase).join(' · ')
    : null;
  const intent = analysis.intent.name.replace(/_/g, ' ').toLowerCase();
  return `Interpreted as a <strong>${escapeHtml(intent)}</strong> question` +
         (focus ? ` about <strong>${escapeHtml(focus)}</strong>` : '') +
         ` — looking for ${escapeHtml(analysis.intent.wants || 'a direct answer')}.`;
}

function titleCase(s) {
  return String(s).replace(/\b\w/g, c => c.toUpperCase());
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
