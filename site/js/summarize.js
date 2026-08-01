// ============================================================================
// Query-focused extractive summarisation
// ============================================================================
//
// What was shipping before: take the top sentences by raw query-term overlap and
// concatenate them. That produces the observed failures —
//
//   * the same intended-use statement repeated four times with minor wording
//     differences, because exact-string dedup does not catch near-duplicates;
//   * a warranty disclaimer opening an answer about creatinine, because it is
//     long, contains "creatinine", and nothing scored it as legal boilerplate;
//   * "The sponsor determined the following substances did not cause
//     interference at the concentrations listed below:" with no list, because
//     the sentence that scored well was a lead-in to a table.
//
// This module scores sentences on six independent signals, then selects with
// Maximal Marginal Relevance so each added sentence must contribute something
// the answer does not already contain. That is what removes the repetition and
// forces coverage of different facets of the question.
//
//   MMR:  argmax [ λ · relevance(s) − (1−λ) · max similarity(s, already chosen) ]
//
// λ = 0.72 — relevance-leaning, but enough diversity pressure that a paraphrase
// of an already-selected sentence cannot win.

import { evidencePrior, contentTokens, META_REGULATORY_RE, familyMembers } from './nlu.js?v=7';

const MMR_LAMBDA = 0.72;
const MAX_SENTENCES = 5;
const MIN_SENTENCE_CHARS = 30;
const MAX_ANSWER_CHARS = 900;

// Sentence kinds that are never answer material, whatever they score.
const REJECTED_KINDS = new Set(['header', 'fragment', 'equation', 'legal']);

// For these intents a table IS the answer — an interference limits table or a
// specification block is exactly what was asked for. Blanket-rejecting tables
// was removing the only passage that could satisfy the question.
const TABLE_ADMITTING_INTENTS = new Set(['INTERFERENCE', 'SPECIFICATION', 'COMPARISON']);

// ---------------------------------------------------------------------------
// Similarity helpers
// ---------------------------------------------------------------------------
function tokenSet(text) { return new Set(contentTokens(text)); }

function jaccard(a, b) {
  if (!a.size || !b.size) return 0;
  let inter = 0;
  for (const t of a) if (b.has(t)) inter++;
  return inter / (a.size + b.size - inter);
}

/** Containment — catches the "same statement, extra clause" case that Jaccard
 *  under-weights when one sentence is much longer than the other. */
function containment(a, b) {
  if (!a.size || !b.size) return 0;
  let inter = 0;
  for (const t of a) if (b.has(t)) inter++;
  return inter / Math.min(a.size, b.size);
}

function redundancy(a, b) {
  return Math.max(jaccard(a, b), containment(a, b) * 0.95);
}

// ---------------------------------------------------------------------------
// Sentence candidate extraction
// ---------------------------------------------------------------------------
function sentencesOf(chunk) {
  // Build-time sentence layer: offsets + kind + quality. Fall back to a naive
  // split for any index built before the sentence layer existed.
  if (Array.isArray(chunk.sents) && chunk.sents.length) {
    return chunk.sents.map((s, i) => ({
      text: chunk.text.substr(s.o, s.l).trim(),
      kind: s.k, quality: s.q, position: i, total: chunk.sents.length,
    }));
  }
  const parts = String(chunk.text || '').split(/(?<=[.!?])\s+/);
  return parts.map((t, i) => ({
    text: t.trim(), kind: 'prose', quality: 0.6, position: i, total: parts.length,
  }));
}

// ---------------------------------------------------------------------------
// Relevance scoring — six signals, each independently defensible
// ---------------------------------------------------------------------------
function scoreSentence(sent, ctx) {
  const { queryTokens, idf, intent, focus, semantic } = ctx;
  const sTokens = tokenSet(sent.text);
  if (!sTokens.size) return null;

  // 1. IDF-weighted query coverage. Raw overlap counts treat "the meter" and
  //    "hematocrit" as equally informative; IDF does not.
  let covered = 0, totalWeight = 0;
  for (const t of queryTokens) {
    const w = idf(t);
    totalWeight += w;
    if (sTokens.has(t)) covered += w;
  }
  const coverage = totalWeight > 0 ? covered / totalWeight : 0;

  // 2. Semantic similarity to the question — bridges vocabulary mismatch, so a
  //    sentence saying "renal" can answer a question saying "kidney".
  let semSim = 0;
  if (semantic && semantic.enabled && ctx.queryVec) {
    const v = semantic.embed(sent.text);
    if (v) {
      let dot = 0;
      for (let i = 0; i < v.length; i++) dot += v[i] * ctx.queryVec[i];
      semSim = Math.max(0, dot);
    }
  }

  // 3. Intent cue match — does this sentence look like the *kind* of statement
  //    the question asked for. Cues are weighted: a sentence that STATES the
  //    fact ("is intended for ...") must outrank one that merely MENTIONS the
  //    concept ("...equivalent in terms of intended use...").
  let cueScore = 0;
  for (const entry of (intent.answerCues || [])) {
    const [re, w] = Array.isArray(entry) ? entry : [entry, 0.6];
    if (re.test(sent.text)) cueScore = Math.max(cueScore, w);
  }
  let cue = Math.min(1, cueScore);

  // Regulatory meta-commentary is the right answer to a regulatory question and
  // a distraction from any other kind.
  if (intent.name !== 'REGULATORY' && META_REGULATORY_RE.test(sent.text)) {
    cue *= 0.25;
  }

  // 4. Focus — is this sentence about the thing that was asked about?
  //
  // This is a gate, not a nudge. "What substances interfere with glucose
  // results?" was being answered with the StatSensor *Creatinine* interference
  // table: excellent cue match, excellent quality, wrong analyte. A sentence
  // that names a competing member of the same entity family while never naming
  // the one asked about is answering a different question.
  let focusHit = 0;
  let focusConflict = 1.0;
  if (focus.length) {
    const low = sent.text.toLowerCase();
    focusHit = focus.some(f => low.includes(f)) ? 1 : 0;

    for (const family of ['analyte', 'product']) {
      const asked = focus.filter(f => familyMembers(f, family).length);
      if (!asked.length) continue;
      const present = familyMembers(sent.text, family);
      if (!present.length) continue;
      const mentionsAsked = present.some(p => asked.some(a => a.includes(p) || p.includes(a)));
      if (!mentionsAsked) focusConflict = Math.min(focusConflict, 0.15);
    }
  }

  // 5. Intrinsic sentence quality from the build-time classifier.
  const quality = sent.quality ?? 0.5;

  // 6. Position prior — the opening sentences of a passage carry its topic
  //    statement more often than the tail does.
  const position = 1 - Math.min(1, sent.position / Math.max(sent.total, 1)) * 0.35;

  if (coverage === 0 && semSim < 0.2 && cueScore === 0) return null;

  let metaPenalty = 1.0;
  if (intent.name !== 'REGULATORY' && META_REGULATORY_RE.test(sent.text)) metaPenalty = 0.55;

  const relevance = metaPenalty * focusConflict * (
      0.26 * coverage +
      0.22 * semSim +
      0.18 * cue +
      0.16 * focusHit +
      0.12 * quality +
      0.06 * position);

  return {
    ...sent,
    tokens: sTokens,
    relevance,
    signals: {
      coverage: +coverage.toFixed(3),
      semantic: +semSim.toFixed(3),
      cue: +cue.toFixed(3),
      focus: focusHit,
      focusConflict: +focusConflict.toFixed(2),
      quality: +quality.toFixed(3),
      position: +position.toFixed(3),
    },
  };
}

// ---------------------------------------------------------------------------
// Presentation cleanup
// ---------------------------------------------------------------------------
function polish(text) {
  let t = text.trim()
    .replace(/\s{2,}/g, ' ')
    .replace(/\s+([,.;:)])/g, '$1')
    .replace(/\(\s+/g, '(')
    .replace(/^[•\-–—*]\s*/, '')
    .replace(/^\d{1,3}\s+(?=[A-Z])/, '');       // stray leading page number
  if (t && !/[.!?:]$/.test(t)) t += '.';
  return t.charAt(0).toUpperCase() + t.slice(1);
}

/** A sentence that promises a list it does not contain ("...as follows:",
 *  "...listed below:") is a lead-in to a table that did not survive extraction.
 *  Presenting it alone reads as a broken answer. */
function isDanglingLeadIn(text) {
  return /(as follows|listed below|shown below|following (substances|table|list)|below)\s*:?\s*$/i.test(text.trim());
}

/** Sentences that describe the document rather than its subject: "This section
 *  introduces the meter and covers requirements, tests performed...". They score
 *  well on coverage because they enumerate topic words, and they answer nothing. */
function isMetaNavigational(text) {
  const t = text.trim();
  if (/^(this|the following)\s+(section|chapter|manual|guide|document|appendix|part)\b/i.test(t)) return true;
  if (/\b(covers|describes|introduces|explains|outlines|provides an overview)\b/i.test(t)
      && /\b(section|chapter|manual|guide|document|appendix)\b/i.test(t)) return true;
  return false;
}

// ---------------------------------------------------------------------------
// Main entry
// ---------------------------------------------------------------------------
/**
 * @returns {{sentences:Array, coverage:number, facets:number, rejected:number}}
 */
export function summarise(query, ranked, chunks, opts = {}) {
  const { analysis, semantic, bm25, maxSentences = MAX_SENTENCES } = opts;
  const intent = analysis.intent;
  const queryTokens = analysis.terms;
  const focus = analysis.focus;

  // IDF from the shipped BM25 table — same weighting the retriever used.
  const idf = (term) => {
    if (!bm25 || !bm25.termId || !bm25.idf) return 1;
    const id = bm25.termId[term];
    return (id === undefined) ? 2.2 : (bm25.idf[id] ?? 1);
  };

  const queryVec = (semantic && semantic.enabled) ? semantic.embed(query) : null;
  const ctx = { queryTokens, idf, intent, focus, semantic, queryVec };

  // ---- candidate pool, weighted by evidence prior --------------------------
  const candidates = [];
  let rejected = 0;

  for (const r of ranked.slice(0, 20)) {
    const chunk = chunks[r.chunkIdx];
    if (!chunk) continue;
    const prior = evidencePrior(chunk, intent);
    if (prior < 0.12) { rejected++; continue; }   // routed out by intent

    for (const sent of sentencesOf(chunk)) {
      if (sent.text.length < MIN_SENTENCE_CHARS) continue;
      const tablesOk = TABLE_ADMITTING_INTENTS.has(intent.name);
      if (REJECTED_KINDS.has(sent.kind)) { rejected++; continue; }
      if (sent.kind === 'table' && !tablesOk) { rejected++; continue; }
      if (isDanglingLeadIn(sent.text)) { rejected++; continue; }
      if (isMetaNavigational(sent.text)) { rejected++; continue; }

      const scored = scoreSentence(sent, ctx);
      if (!scored) continue;

      candidates.push({
        ...scored,
        chunkIdx: r.chunkIdx,
        chunk,
        prior,
        // Retrieval score is a weak prior here: it ranks passages, not sentences.
        final: scored.relevance * prior * (0.85 + 0.15 * Math.min(1, r.score * 8)),
      });
    }
  }

  if (!candidates.length) return { sentences: [], coverage: 0, facets: 0, rejected };

  candidates.sort((a, b) => b.final - a.final);

  // ---- MMR selection -------------------------------------------------------
  const selected = [];
  const pool = candidates.slice(0, 60);
  let chars = 0;

  while (selected.length < maxSentences && pool.length) {
    let bestIdx = -1, bestScore = -Infinity;

    for (let i = 0; i < pool.length; i++) {
      const cand = pool[i];
      let maxSim = 0;
      for (const chosen of selected) {
        const sim = redundancy(cand.tokens, chosen.tokens);
        if (sim > maxSim) maxSim = sim;
      }
      const mmr = MMR_LAMBDA * cand.final - (1 - MMR_LAMBDA) * maxSim;
      if (mmr > bestScore) { bestScore = mmr; bestIdx = i; }
    }
    if (bestIdx < 0) break;

    const pick = pool.splice(bestIdx, 1)[0];
    // Hard redundancy floor — a near-paraphrase adds nothing, whatever MMR says.
    const dup = selected.some(s => redundancy(pick.tokens, s.tokens) > 0.62);
    if (dup) { rejected++; continue; }

    const text = polish(pick.text);
    if (chars + text.length > MAX_ANSWER_CHARS && selected.length >= 2) break;

    selected.push({ ...pick, display: text });
    chars += text.length;
  }

  // ---- ordering: group by document, preserve original reading order --------
  selected.sort((a, b) => {
    if (a.chunk.document_id !== b.chunk.document_id) {
      return b.final - a.final;                 // strongest document first
    }
    return a.position - b.position;             // then natural order within it
  });

  // ---- how much of the question did we actually cover ---------------------
  const answered = new Set();
  for (const s of selected) for (const t of queryTokens) if (s.tokens.has(t)) answered.add(t);
  const coverage = queryTokens.length ? answered.size / queryTokens.length : 0;
  const facets = new Set(selected.map(s => s.chunk.document_id)).size;

  return { sentences: selected, coverage, facets, rejected, candidateCount: candidates.length };
}
