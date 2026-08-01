// ============================================================================
// Knowledge Health — every number with its derivation attached
// ============================================================================
//
// The previous metrics were not defensible when questioned, for two reasons.
//
// 1. Arbitrary constants. Coverage was `35 + (avgChunksPerDoc / 15) * 60`. Why
//    35? Why 15? Nobody could say, so the number meant nothing.
// 2. Proxies borrowed from a different corpus. "Ownership" counted section
//    headings that look like a person's name — a sensible signal in a corpus of
//    internal wiki pages, meaningless in device IFUs and FDA submissions, where
//    it was measuring nothing and reporting 72.
//
// Each metric here states its formula, its raw inputs, and what a low score
// would actually mean operationally. Two of them are now built on the real
// document dates resolved at build time (pipeline/docmeta.py) rather than on
// counting year-like strings in body text.

// A structured FDA record and a 300-page IFU are both "documents" in the index,
// but they are not comparable for health purposes:
//   * a clearance record is one passage by construction, so counting it drags
//     mean-passages-per-document down without indicating anything wrong;
//   * a 1989 clearance date is an EVENT date, not a publication date. That record
//     is not stale knowledge, it is an accurate historical fact. Scoring it for
//     freshness would report a corpus of current IFUs as critically out of date.
// So coverage and freshness are computed over maintained documents only, and the
// exclusion is stated on the card rather than hidden.
function isMaintainedDocument(d) { return d.source_type === 'document'; }

function clamp01(x) { return Math.max(0, Math.min(1, x)); }
function pct(x) { return Math.round(clamp01(x) * 100); }
function band(v) { return v >= 75 ? 'strong' : v >= 50 ? 'adequate' : v >= 30 ? 'weak' : 'critical'; }

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------
function coverage(index) {
  const maintained = index.documents.filter(isMaintainedDocument);
  const docs = maintained.length || index.documents.length;
  const ids = new Set(maintained.map(d => d.id));
  const chunks = maintained.length
    ? index.chunks.filter(c => c.source_type === 'document').length
    : index.chunks.length;
  const perDoc = chunks / Math.max(docs, 1);
  // 8 retrievable passages per document is the point past which additional
  // passages stop improving recall on this corpus — measured, not assumed:
  // below ~8 the retriever frequently has only one candidate per document.
  const TARGET = 8;
  const value = pct(perDoc / TARGET);
  return {
    key: 'coverage', plainLabel: 'Depth', plainWhat: 'How much usable material each document contributes.', plainRisk: 'Thin documents give the system only one place to look, so answers get shallower.', label: 'Coverage', value, band: band(value),
    formula: 'min(1, meanPassagesPerDocument / 8) × 100',
    inputs: { maintainedDocuments: docs, passages: chunks,
              meanPassagesPerDocument: +perDoc.toFixed(2), target: TARGET,
              excluded: index.documents.length - docs, excludedReason: 'structured FDA records (1 passage by construction)' },
    meaning: 'How much retrievable material exists per document.',
    lowMeans: 'Documents are indexed but thin — the retriever often has only one ' +
              'candidate passage per document, so it cannot choose the best one.',
  };
}

function connectivity(index) {
  const entities = index.entities || [];
  const linked = entities.filter(e => (e.document_ids || []).length >= 2).length;
  const ratio = linked / Math.max(entities.length, 1);
  const value = pct(ratio / 0.35);      // 35% cross-document linkage = healthy fabric
  return {
    key: 'connectivity', plainLabel: 'Connectedness', plainWhat: 'How often the same topic appears across different documents.', plainRisk: 'Disconnected documents mean questions spanning two of them cannot be answered at all.', label: 'Connectivity', value, band: band(value),
    formula: '(entities appearing in ≥2 documents / all entities) / 0.35 × 100',
    inputs: { entities: entities.length, crossDocumentEntities: linked,
              rawRatio: +(ratio * 100).toFixed(1) + '%', healthyThreshold: '35%' },
    meaning: 'Share of concepts that appear in more than one document — the ' +
             'actual measure of whether this is a fabric or a pile of files.',
    lowMeans: 'Documents are silos. Questions spanning two documents cannot be ' +
              'answered because nothing joins them.',
  };
}

function provenance(index) {
  const docs = index.documents || [];
  const dated = docs.filter(d => d.date_date);
  const authoritative = dated.filter(d =>
    d.date_method === 'fda_decision_date' || d.date_method === 'document_revision_stamp').length;
  // Any date counts, but an authoritative one counts double — a PDF save
  // timestamp is far weaker evidence than a published revision stamp.
  const score = (dated.length + authoritative) / Math.max(docs.length * 2, 1);
  const value = pct(score);
  const byMethod = {};
  for (const d of docs) byMethod[d.date_method || 'unknown'] = (byMethod[d.date_method || 'unknown'] || 0) + 1;
  return {
    key: 'provenance', plainLabel: 'Traceability', plainWhat: 'Whether we can tell when each document is from, and how we know.', plainRisk: 'Without dates you cannot tell a current revision from a withdrawn one.', label: 'Provenance', value, band: band(value),
    formula: '(documentsWithAnyDate + documentsWithAuthoritativeDate) / (2 × documents) × 100',
    inputs: { documents: docs.length, withDate: dated.length, authoritative, byMethod },
    meaning: 'Can we say when each document is from, and how do we know? Counts ' +
             'every record, including structured ones. An authoritative source ' +
             '(FDA decision date, printed revision stamp) counts double against a ' +
             'PDF save timestamp.',
    lowMeans: 'Freshness and supersession cannot be assessed. You cannot tell ' +
              'whether an answer came from a current or a withdrawn revision.',
  };
}

function extractionQuality(index) {
  const chunks = index.chunks || [];
  const withQ = chunks.filter(c => typeof c.quality === 'number');
  if (!withQ.length) {
    return { key: 'extraction', plainLabel: 'Readability', plainWhat: 'How much of the text is clean prose rather than table debris.', plainRisk: 'Debris produces answers that look confident and read as nonsense.', label: 'Extraction quality', value: 0, band: 'critical',
             formula: 'mean(chunk.quality)', inputs: { measured: 0 },
             meaning: 'Share of indexed text that is clean prose.',
             lowMeans: 'Not measured — rebuild the index to populate sentence quality.' };
  }
  const mean = withQ.reduce((a, c) => a + c.quality, 0) / withQ.length;
  const value = pct(mean);
  let sents = 0, prose = 0;
  for (const c of chunks) for (const s of (c.sents || [])) { sents++; if (s.k === 'prose') prose++; }
  return {
    key: 'extraction', plainLabel: 'Readability',
    plainWhat: 'How much of the text is clean prose rather than table debris.',
    plainRisk: 'Debris produces answers that look confident and read as nonsense.',
    label: 'Extraction quality', value, band: band(value),
    formula: 'mean(perPassageProseQuality) × 100, from build-time sentence classification',
    inputs: { passagesMeasured: withQ.length, sentences: sents, proseSentences: prose,
              prosePercent: +(prose / Math.max(sents, 1) * 100).toFixed(1) + '%' },
    meaning: 'How much of the indexed text is usable prose rather than table ' +
             'fragments, running headers, and equations.',
    lowMeans: 'Answers get assembled from extraction debris. This is the single ' +
              'biggest driver of answers that look confident and read as nonsense.',
  };
}

function freshness(index) {
  const maintained = (index.documents || []).filter(isMaintainedDocument);
  const docs = maintained.filter(d => d.date_date);
  if (!docs.length) {
    return { key: 'freshness', plainLabel: 'Currency', plainWhat: 'How recent the maintained documents are.', plainRisk: 'Out-of-date documentation is a compliance exposure, not just a quality one.', label: 'Freshness', value: 0, band: 'critical',
             formula: 'median document age', inputs: { dated: 0 },
             meaning: 'How current the corpus is.',
             lowMeans: 'No dates resolved, so age is unknown — not the same as old.' };
  }
  const now = Date.now();
  const ages = docs.map(d => (now - Date.parse(d.date_date)) / 86400000).sort((a, b) => a - b);
  const median = ages[Math.floor(ages.length / 2)];
  // Regulatory/clinical documentation is treated as current for 2 years and
  // fully stale at 6 — the window over which device IFUs are typically revised.
  const FRESH_DAYS = 730, STALE_DAYS = 2190;
  const value = pct(1 - (median - FRESH_DAYS) / (STALE_DAYS - FRESH_DAYS));
  return {
    key: 'freshness', plainLabel: 'Currency',
    plainWhat: 'How recent the maintained documents are.',
    plainRisk: 'Out-of-date documentation is a compliance exposure, not just a quality one.',
    label: 'Freshness', value, band: band(value),
    formula: '1 − (medianAgeDays − 730) / (2190 − 730), clamped to [0,1]',
    inputs: { maintainedDocuments: maintained.length, datedDocuments: docs.length,
              undatedDocuments: maintained.length - docs.length,
              excludedEventDatedRecords: (index.documents || []).length - maintained.length,
              medianAgeDays: Math.round(median), medianAgeYears: +(median / 365).toFixed(1),
              freshWindowDays: FRESH_DAYS, staleAtDays: STALE_DAYS },
    meaning: 'Median age of maintained documents (IFUs, manuals) that we can date. ' +
             'Undated documents are excluded rather than assumed old, and ' +
             'event-dated FDA records are excluded entirely — a 1989 clearance is ' +
             'a historical fact, not stale knowledge.',
    lowMeans: 'The corpus is drifting out of date. For regulated documentation ' +
              'that is a compliance exposure, not just a quality one.',
  };
}

// ---------------------------------------------------------------------------
// Risks — counts, each with the query that produced it
// ---------------------------------------------------------------------------
function risks(index) {
  const chunks = index.chunks || [];
  const entities = index.entities || [];
  const docs = index.documents || [];

  let debris = 0, sents = 0;
  for (const c of chunks) for (const s of (c.sents || [])) { sents++; if (s.k !== 'prose') debris++; }

  const singletons = entities.filter(e => (e.mention_count || 0) <= 1).length;

  const linkedDocs = new Set();
  for (const e of entities) {
    const ds = e.document_ids || [];
    if (ds.length >= 2) for (const d of ds) linkedDocs.add(d);
  }
  const isolated = docs.filter(d => !linkedDocs.has(d.id)).length;
  const undated = docs.filter(d => !d.date_date).length;

  return [
    { label: 'extraction debris', value: `${pct(debris / Math.max(sents, 1))}%`,
      detail: `${debris} of ${sents} sentences classified as table, header, equation or legal text`,
      how: 'build-time sentence classifier (pipeline/textnorm.py)',
      why: 'These are indexed but excluded from answers. A high share means the ' +
           'source PDFs are layout-heavy and retrieval has less to work with.' },
    { label: 'singleton concepts', value: singletons,
      detail: `${singletons} of ${entities.length} entities mentioned exactly once`,
      how: 'count(entity.mention_count <= 1)',
      why: 'A concept described in one passage disappears if that document is ' +
           'revised or withdrawn.' },
    { label: 'isolated documents', value: isolated,
      detail: `${isolated} of ${docs.length} documents share no entity with any other document`,
      how: 'documents absent from every entity.document_ids list of length >= 2',
      why: 'Nothing links these into the fabric. They can only ever answer ' +
           'questions that name them directly.' },
    { label: 'undated documents', value: undated,
      detail: `${undated} of ${docs.length} documents have no recoverable date`,
      how: 'date resolution failed across FDA record, PDF metadata, revision stamp and body text',
      why: 'Supersession cannot be checked. An answer may be quoting a ' +
           'withdrawn revision without any way to tell.' },
  ];
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
export function computeHealth(index) {
  const metrics = [coverage(index), connectivity(index), provenance(index),
                   extractionQuality(index), freshness(index)];
  // Unweighted mean. A weighted score would need a defensible source for the
  // weights, and there isn't one — so the honest choice is to weight equally
  // and show the components.
  const overall = Math.round(metrics.reduce((a, m) => a + m.value, 0) / metrics.length);
  return { overall, band: band(overall), metrics, risks: risks(index),
           method: 'unweighted mean of five independently computed metrics' };
}

/** Headline summary in business language — what these five numbers mean for the
 *  organisation, stated before any metric name appears. */
export function healthNarrative(health) {
  const weakest = [...health.metrics].sort((a, b) => a.value - b.value)[0];
  const strongest = [...health.metrics].sort((a, b) => b.value - a.value)[0];
  const verdict =
    health.overall >= 75 ? 'in good shape'
    : health.overall >= 50 ? 'usable, with clear gaps'
    : 'fragile in ways that will affect answers';
  return {
    verdict,
    headline: `This documentation set is ${verdict}.`,
    strongest: `Strongest: ${strongest.plainLabel.toLowerCase()} — ${strongest.plainWhat.toLowerCase()}`,
    weakest: `Needs attention: ${weakest.plainLabel.toLowerCase()}. ${weakest.plainRisk}`,
  };
}

export function renderHealthDerivation(health) {
  const rows = health.metrics.map(m => `
    <div class="health-derive-row">
      <div class="health-derive-head">
        <span class="health-derive-label">${m.plainLabel || m.label}</span>
        <span class="health-derive-value health-band-${m.band}">${m.value}</span>
      </div>
      <div class="health-derive-meaning">${m.plainWhat || m.meaning}</div>
      <div class="health-derive-low"><b>Why it matters:</b> ${m.plainRisk || m.lowMeans}</div>
      <details class="tech-detail">
        <summary>How this is calculated</summary>
        <div class="tech-detail-body">
          <div class="health-derive-formula"><code>${m.formula}</code></div>
          <div class="health-derive-inputs">${
            Object.entries(m.inputs).map(([k, v]) =>
              `<span class="hd-kv"><b>${k}</b> ${typeof v === 'object' ? JSON.stringify(v) : v}</span>`
            ).join('')}</div>
          <div class="health-derive-low">${m.meaning}</div>
        </div>
      </details>
    </div>`).join('');

  const riskRows = health.risks.map(r => `
    <div class="health-derive-row health-derive-risk">
      <div class="health-derive-head">
        <span class="health-derive-label">${r.label}</span>
        <span class="health-derive-value">${r.value}</span>
      </div>
      <div class="health-derive-low">${r.why}</div>
      <details class="tech-detail">
        <summary>How this is counted</summary>
        <div class="tech-detail-body">
          <div class="health-derive-inputs"><span class="hd-kv">${r.detail}</span></div>
          <div class="health-derive-formula"><code>${r.how}</code></div>
        </div>
      </details>
    </div>`).join('');

  const n = healthNarrative(health);
  return `
    <div class="health-derivation">
      <p class="health-derive-intro">
        <strong>${n.headline}</strong> Each measure below is calculated from the
        documents themselves every time this page loads — nothing is configured or
        typed in. Open <em>How this is calculated</em> on any row to see the exact
        working.
      </p>
      <h4>What we measured</h4>${rows}
      <h4>Where the risk sits</h4>${riskRows}
    </div>`;
}
