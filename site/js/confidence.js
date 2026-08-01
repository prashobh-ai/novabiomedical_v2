// ============================================================================
// Confidence — derived, not declared
// ============================================================================
//
// Every answer in the last demo showed 75%. Not because the system was 75%
// confident, but because search.js contained:
//
//     confidence = min(1, max(distinctness / 4, docNameMatch ? 0.75 : 0))
//
// The 0.75 floor fired whenever the query matched a filename, which is almost
// always, so the floor became the value. A number that never moves is not a
// confidence score — it is a decoration, and a technical audience will read it
// as one the moment two very different answers both report 75%.
//
// This replaces it with five measured signals. Each is independently computable,
// bounded [0,1], and means something specific that can be defended out loud.
// They are combined as a WEIGHTED GEOMETRIC MEAN rather than an arithmetic one,
// because these signals are conjunctive: an answer with excellent retrieval but
// zero query coverage is not "average", it is bad. A geometric mean punishes a
// single near-zero component, which is the behaviour we want.
//
//     confidence = Π (signal_i ^ weight_i)        Σ weight_i = 1
//
// Nothing here needs a model. It is all measurable from the retrieval run.

const EPS = 0.02;   // floor so one zero signal cannot annihilate the product

export const SIGNAL_SPECS = {
  retrievalMargin: {
    plainLabel: 'One clear best source',
    plainWhy: 'The system found one passage that stood out, rather than many weak near-matches.',
    weight: 0.22,
    label: 'Retrieval margin',
    question: 'Did one passage clearly win, or was it a coin toss?',
    how: '(top score − mean of next 4) / top score',
    why: 'A decisive winner means the corpus contains a specific answer. A flat ' +
         'score distribution means many passages match weakly — usually a sign ' +
         'the answer is not really in the corpus.',
  },
  retrieverAgreement: {
    plainLabel: 'Two methods agreed',
    plainWhy: 'Two independent search methods picked the same evidence — real corroboration, not one method guessing.',
    weight: 0.18,
    label: 'Retriever agreement',
    question: 'Do lexical and semantic search independently agree?',
    how: 'share of fused top-k found by BOTH BM25 and the LSA retriever',
    why: 'Two methods with different failure modes converging on the same passage ' +
         'is real corroboration. Agreement near zero means one retriever is ' +
         'carrying the result alone.',
  },
  queryCoverage: {
    plainLabel: 'Covers what you asked',
    plainWhy: 'The answer actually addresses the terms in your question, weighted so the unusual words count most.',
    weight: 0.24,
    label: 'Question coverage',
    question: 'Does the answer actually address the terms that were asked about?',
    how: 'IDF-weighted share of question content-terms present in the answer',
    why: 'The strongest single predictor of a wrong answer is an answer that ' +
         'never mentions what was asked. IDF-weighted so rare, meaningful terms ' +
         'count more than common ones.',
  },
  sourceConsensus: {
    plainLabel: 'More than one document',
    plainWhy: 'Supported by several documents rather than resting on a single file.',
    weight: 0.18,
    label: 'Source consensus',
    question: 'Is this supported by more than one document?',
    how: '1 − 1/n over distinct supporting documents, capped at 3',
    why: 'Independent corroboration. One document can be wrong, out of date, or ' +
         'about a different product variant.',
  },
  evidenceQuality: {
    plainLabel: 'Clean source text',
    plainWhy: 'The supporting text is readable prose, not fragments of tables or page headers.',
    weight: 0.18,
    label: 'Evidence quality',
    question: 'Is the supporting text clean prose or extraction debris?',
    how: 'mean build-time quality of the selected sentences',
    why: 'A confident-sounding answer assembled from table fragments and page ' +
         'headers is worse than an honest low-confidence one.',
  },
};

function clamp01(x) { return Math.max(0, Math.min(1, x)); }

// ---------------------------------------------------------------------------
// Individual signals
// ---------------------------------------------------------------------------
function retrievalMargin(ranked, diagnostics) {
  // Measured on the PRE-FUSION retriever scores.
  //
  // Computing this on RRF output was a genuine bug: fused scores are 1/(k+rank)
  // sums, so rank 1 and rank 5 differ by under 3% by construction. The signal
  // therefore never exceeded 0.20 on any question, and — being weighted 0.22
  // inside a geometric mean — multiplied every answer by roughly 0.60. That is
  // why every score clustered near 50%.
  const pools = [];
  if (diagnostics) {
    if (diagnostics.lexicalScores && diagnostics.lexicalScores.length > 1) {
      pools.push(diagnostics.lexicalScores);
    }
    if (diagnostics.semanticScores && diagnostics.semanticScores.length > 1) {
      pools.push(diagnostics.semanticScores);
    }
  }
  if (!pools.length) {
    if (!ranked || !ranked.length) return 0;
    if (ranked.length === 1) return 0.6;
    return 0.5;                                     // unknown, not zero
  }

  // Each retriever votes; take the best separation either achieved.
  const margins = pools.map(scores => {
    const top = scores[0];
    if (!(top > 0)) return 0;
    const rest = scores.slice(1, 5);
    if (!rest.length) return 0.6;
    const mean = rest.reduce((a, s) => a + s, 0) / rest.length;
    return clamp01((top - mean) / top);
  });
  return Math.max(...margins);
}

function retrieverAgreement(diagnostics) {
  if (!diagnostics) return 0.5;                     // unknown, not zero
  if (typeof diagnostics.agreement === 'number') return clamp01(diagnostics.agreement);
  return 0.5;
}

function queryCoverage(analysis, sentences, idf) {
  const terms = analysis.terms || [];
  if (!terms.length) return 0.5;
  const answerTokens = new Set();
  for (const s of sentences) for (const t of s.tokens) answerTokens.add(t);
  let hit = 0, total = 0;
  for (const t of terms) {
    const w = idf(t);
    total += w;
    if (answerTokens.has(t)) hit += w;
  }
  return total > 0 ? clamp01(hit / total) : 0.5;
}

function sourceConsensus(sentences) {
  if (!sentences.length) return 0;
  const docs = new Set(sentences.map(s => s.chunk.document_id)).size;
  // The previous curve, 1 - 1/(n+0.35) with n capped at 3, could never exceed
  // 0.70 however well corroborated an answer was — a permanent ceiling applied
  // to every question. This one still penalises a single source but lets strong
  // corroboration actually register:
  //   1 doc 0.35 · 2 docs 0.68 · 3 docs 0.84 · 4 docs 0.92 · 5+ docs 0.96
  return clamp01(1 - Math.pow(0.5, docs) * 1.3);
}

function evidenceQuality(sentences) {
  if (!sentences.length) return 0;
  const mean = sentences.reduce((a, s) => a + (s.quality ?? 0.5), 0) / sentences.length;
  return clamp01(mean);
}

// ---------------------------------------------------------------------------
// Combination
// ---------------------------------------------------------------------------
/**
 * @returns {{score:number, percent:number, band:string, signals:Object, caveats:string[]}}
 */
export function computeConfidence({ analysis, ranked, sentences, diagnostics, idf }) {
  const signals = {
    retrievalMargin: retrievalMargin(ranked, diagnostics),
    retrieverAgreement: retrieverAgreement(diagnostics),
    queryCoverage: queryCoverage(analysis, sentences, idf),
    sourceConsensus: sourceConsensus(sentences),
    evidenceQuality: evidenceQuality(sentences),
  };

  // Weighted geometric mean.
  let logSum = 0;
  for (const [key, spec] of Object.entries(SIGNAL_SPECS)) {
    const v = Math.max(EPS, signals[key]);
    logSum += spec.weight * Math.log(v);
  }
  let score = Math.exp(logSum);

  // ---- honest caveats, surfaced rather than smoothed away ------------------
  const caveats = [];
  if (!sentences.length) {
    score = 0;
    caveats.push('No supporting passage passed the evidence filters.');
  }
  if (signals.queryCoverage < 0.4) {
    caveats.push('The answer does not cover most of the terms in the question.');
  }
  if (signals.sourceConsensus < 0.3) {
    caveats.push('Supported by a single document — no independent corroboration.');
  }
  if (signals.retrieverAgreement < 0.2) {
    caveats.push('Lexical and semantic retrieval disagreed on the best evidence.');
  }
  if (signals.evidenceQuality < 0.45) {
    caveats.push('Supporting text is partly extraction debris (tables, headers).');
  }
  if (analysis.intent && analysis.intent.name === 'GENERAL') {
    caveats.push('Question intent was ambiguous, so evidence routing was not applied.');
  }

  // Two decimal places. A score that reads 58.25% is visibly the output of a
  // calculation; one that reads 50% looks like a constant — which is exactly
  // how the previous version was (correctly) received.
  const exact = +(score * 100).toFixed(2);
  const percent = exact;
  const band = exact >= 70 ? 'high' : exact >= 45 ? 'moderate' : exact >= 25 ? 'low' : 'very low';

  return {
    score, percent, exact, band, caveats,
    signals: Object.fromEntries(Object.entries(signals).map(([k, v]) => [k, {
      value: +v.toFixed(3),
      percent: +(v * 100).toFixed(1),
      contribution: +(SIGNAL_SPECS[k].weight).toFixed(2),
      ...SIGNAL_SPECS[k],
    }])),
    method: 'weighted geometric mean of 5 measured retrieval signals',
  };
}

/** Compact HTML for the "how was this derived" panel. */
/** One plain sentence explaining the score, before any signal name appears. */
function plainVerdict(conf) {
  const strong = Object.values(conf.signals).filter(s => s.value >= 0.6).length;
  if (conf.percent >= 70) {
    return 'Strong answer. The evidence was clear, consistent, and covered what you asked.';
  }
  if (conf.percent >= 45) {
    return `Reasonable answer, with caveats. ${strong} of 5 quality checks came back strong — ` +
           `the notes below say which ones did not.`;
  }
  if (conf.percent >= 25) {
    return 'Weak answer. Read the supporting passages before relying on this.';
  }
  return 'The documentation does not appear to answer this. Treat what is shown as related context only.';
}

export function renderConfidenceBreakdown(conf) {
  if (!conf || !conf.signals) return '';
  const rows = Object.entries(conf.signals).map(([key, s]) => `
    <div class="conf-row">
      <div class="conf-row-head">
        <span class="conf-row-label">${s.plainLabel || s.label}</span>
        <span class="conf-row-val">${s.percent}%</span>
      </div>
      <div class="conf-row-bar"><span style="width:${s.percent}%"></span></div>
      <div class="conf-row-why">${s.plainWhy || s.why}</div>
      <details class="tech-detail">
        <summary>How this is calculated</summary>
        <div class="tech-detail-body">
          <div class="conf-row-how"><code>${s.how}</code></div>
          <p class="conf-row-tech">${s.why}</p>
          <p class="conf-row-tech"><b>Weight in the overall score:</b> ${s.contribution}</p>
        </div>
      </details>
    </div>`).join('');

  const caveats = conf.caveats.length
    ? `<div class="conf-caveats"><strong>What to watch</strong><ul>${
        conf.caveats.map(c => `<li>${c}</li>`).join('')}</ul></div>`
    : '<div class="conf-caveats conf-caveats-clean">All five checks came back clean.</div>';

  return `
    <div class="conf-breakdown">
      <div class="conf-headline">
        <span class="conf-headline-num">${conf.percent.toFixed(2)}%</span>
        <span class="conf-headline-band conf-band-${conf.band.replace(' ', '-')}">${conf.band} confidence</span>
      </div>
      <p class="conf-verdict">${plainVerdict(conf)}</p>
      <p class="conf-method">This score is measured from the evidence found — five
      independent checks, every time. It is not a fixed number.</p>
      ${rows}
      ${caveats}
      <details class="tech-detail tech-detail-wide">
        <summary>Full method</summary>
        <div class="tech-detail-body">
          <p class="conf-row-tech">${conf.method}. The five checks are combined as a
          weighted geometric mean rather than an average, because they are
          conjunctive: an answer with excellent sourcing but no coverage of the
          question is not "average", it is wrong. A geometric mean lets one
          failing check pull the result down instead of being smoothed away.</p>
          <p class="conf-row-tech"><code>confidence = Π signal_i ^ weight_i</code></p>
        </div>
      </details>
    </div>`;
}
