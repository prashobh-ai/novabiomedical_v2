// ============================================================================
// Graph value — answering "what is the graph actually doing for us?"
// ============================================================================
//
// A fair challenge from a reviewer: the graph is rendered beautifully, but the
// visual never explains what it CONTRIBUTES. It looks like decoration.
//
// The contribution is specific and demonstrable. Keyword search can only find
// documents that share vocabulary with the question. The graph finds documents
// that share a SUBJECT — even when they share almost no words.
//
// A product manual says "insert the test strip and apply the blood drop".
// Its FDA clearance says "substantially equivalent to the predicate device".
// No word in either sentence appears in the other. A reader looking at one of
// them would never be led to the other. They are both about StatStrip Glucose,
// and the graph is the only thing in the system that knows that.
//
// So after every answer we compute, from the answer itself:
//   * which documents were used
//   * which shared concepts link them
//   * how much vocabulary those documents actually have in common
//   * therefore: which links keyword search alone could not have made
//
// and state it in a sentence a non-technical reviewer can evaluate on the spot.

const WORD_RE = /[a-z][a-z0-9]{3,}/g;

const COMMON = new Set([
  'the', 'and', 'for', 'with', 'that', 'this', 'from', 'nova', 'test', 'used',
  'use', 'using', 'shall', 'must', 'been', 'have', 'has', 'are', 'was', 'were',
  'will', 'can', 'may', 'each', 'when', 'them', 'they', 'than', 'then', 'also',
  'into', 'over', 'under', 'such', 'these', 'those', 'their', 'there', 'which',
  'page', 'section', 'table', 'system', 'device', 'results', 'result',
]);

function vocabOf(text) {
  const set = new Set();
  for (const w of String(text).toLowerCase().match(WORD_RE) || []) {
    if (!COMMON.has(w)) set.add(w);
  }
  return set;
}

function overlapRatio(a, b) {
  if (!a.size || !b.size) return 1;
  let shared = 0;
  for (const w of a) if (b.has(w)) shared++;
  return shared / Math.min(a.size, b.size);
}

const KIND_PHRASE = {
  Product: 'the product',
  ProductCode: 'the FDA product code',
  Clearance: 'the FDA clearance',
  Recall: 'the recall record',
  Analyte: 'the measured analyte',
  Regulation: 'the regulation',
  Document: 'the document',
};

/**
 * What the graph contributed to THIS answer.
 *
 * @param {Array} citations   answer citations (carry .chunk)
 * @param {Object} index      the built index
 * @returns {Object|null}
 */
export function explainGraphContribution(citations, index, analysis = null) {
  if (!citations || citations.length < 2) return null;

  const docIds = [...new Set(citations.map(c => c.chunk.document_id))];
  if (docIds.length < 2) return null;

  const docsById = new Map((index.documents || []).map(d => [d.id, d]));

  // Vocabulary per contributing document, sampled from the passages actually used.
  const vocabByDoc = new Map();
  for (const c of citations) {
    const id = c.chunk.document_id;
    if (!vocabByDoc.has(id)) vocabByDoc.set(id, new Set());
    for (const w of vocabOf(c.chunk.text)) vocabByDoc.get(id).add(w);
  }

  // Concepts present in more than one of the contributing documents. These are
  // the graph nodes doing the joining.
  // A bridge is only interesting if it is SPECIFIC and RELEVANT.
  //
  // Ranking by document count alone surfaced "Sodium" as the concept linking an
  // answer about glucose — technically true (sodium is measured by both
  // analysers) and useless to a reader, because a concept appearing in 32 of 391
  // documents joins almost everything and therefore explains nothing.
  //
  // So: concepts named in the question rank first, then rarer concepts, then
  // reach. Specificity is 1 - (documents containing it / all documents).
  const totalDocs = Math.max((index.documents || []).length, 1);
  const focusTerms = ((analysis && analysis.focus) || []).map(f => String(f).toLowerCase());

  const bridges = [];
  for (const e of (index.entities || [])) {
    const shared = (e.document_ids || []).filter(id => docIds.includes(id));
    if (shared.length < 2) continue;
    const reach = (e.document_ids || []).length;
    const nameLow = e.name.toLowerCase();
    const onTopic = focusTerms.some(f => nameLow.includes(f) || f.includes(nameLow));
    const specificity = 1 - Math.min(1, reach / totalDocs);
    bridges.push({
      name: e.name,
      kind: e.kind,
      docs: shared,
      crossSource: Boolean(e.cross_source),
      reach,
      onTopic,
      specificity,
      rank: (onTopic ? 3 : 0) + (e.cross_source ? 1.2 : 0) +
            specificity * 1.5 + Math.min(shared.length / docIds.length, 1),
    });
  }
  if (!bridges.length) return null;

  bridges.sort((a, b) => b.rank - a.rank);


  // FDA clearance PDFs are ingested as documents, so source_type alone does not
  // separate them from product manuals. The document's domain does.
  const classOf = (id) => {
    const d = docsById.get(id);
    if (!d) return 'unknown';
    if (d.source_type === 'fda_regulatory') return 'regulatory-record';
    if (d.domain === 'Regulatory' || /^K\d{6}/.test(d.name || '')) return 'regulatory-filing';
    return 'product-documentation';
  };
  const classes = new Set(docIds.map(classOf));
  const spansSources = classes.size > 1;

  // Document pairs with little vocabulary in common — the links keyword search
  // could not have made on its own.
  const weakPairs = [];
  for (let i = 0; i < docIds.length; i++) {
    for (let j = i + 1; j < docIds.length; j++) {
      const a = docIds[i], b = docIds[j];
      const ratio = overlapRatio(vocabByDoc.get(a) || new Set(), vocabByDoc.get(b) || new Set());
      // Attribute the pair to the highest-ranked concept that actually joins it,
      // not merely the first one found.
      const link = bridges.find(br => br.docs.includes(a) && br.docs.includes(b));
      if (!link) continue;
      weakPairs.push({
        a, b, ratio,
        aName: docsById.get(a)?.name || `document ${a}`,
        bName: docsById.get(b)?.name || `document ${b}`,
        aType: classOf(a),
        bType: classOf(b),
        via: link,
      });
    }
  }
  // Lowest overlap makes the strongest point, but only if the joining concept is
  // one the reader asked about. A pair genuinely linked by "Sodium" is true and
  // reads as a non-sequitur under a question about creatinine.
  weakPairs.sort((x, y) =>
    (y.via.onTopic - x.via.onTopic) ||
    (y.via.specificity - x.via.specificity) ||
    (x.ratio - y.ratio));
  const headline = weakPairs[0] || null;

  // The concept the answer will NAME. Chosen once here, so the prose and the
  // diagram can never disagree — a card whose sentence says "Creatinine" beside
  // a picture labelled "Hemoglobin" destroys trust faster than no picture at all.
  const headlineWeak = weakPairs[0] && weakPairs[0].ratio < 0.22 ? weakPairs[0] : null;
  const primary = headlineWeak ? headlineWeak.via : bridges[0];

  return {
    documentCount: docIds.length,
    classes: [...classes],
    primary,
    bridges: bridges.slice(0, 4),
    weakPairs: weakPairs.slice(0, 3),
    spansSources,
    headline,
    // The claim we can defend: pairs joined despite low shared vocabulary.
    graphOnlyLinks: weakPairs.filter(p => p.ratio < 0.22).length,
  };
}

// ---------------------------------------------------------------------------
// Mini-graph — the specific subgraph behind THIS answer
// ---------------------------------------------------------------------------
//
// The hero galaxy shows 323 nodes at once, which is impressive and unreadable.
// What a reviewer needs beside the explanation is the four or five nodes that
// actually produced this answer: the documents used, and the concept that joined
// them. Rendered as inline SVG — no library, no layout engine, ~2 KB.
//
// Layout is a deliberate bipartite fan rather than a force simulation: the
// bridging concept sits at the centre, contributing documents fan out around it.
// That shape IS the argument — every document connects through the middle, and
// nothing connects document-to-document directly.
function shortLabel(name, max = 26) {
  const clean = stripExt(name).replace(/\s+/g, ' ').trim();
  return clean.length > max ? clean.slice(0, max - 1) + '\u2026' : clean;
}

export function renderMiniGraph(contribution, index) {
  if (!contribution || !contribution.bridges.length) return '';

  // Always the concept the prose names. Density is not worth an inconsistency.
  const hub = contribution.primary || contribution.bridges[0];
  const docsById = new Map((index.documents || []).map(d => [d.id, d]));
  const docIds = hub.docs.slice(0, 5);
  if (docIds.length < 2) return '';

  const W = 300, H = 200;
  const cx = W / 2, cy = H / 2;
  const R = 74;

  // Fan the documents across an arc so labels never collide.
  const n = docIds.length;
  const spread = n === 2 ? 140 : 300;
  const start = -90 - spread / 2;

  const nodes = docIds.map((id, i) => {
    const angle = (start + (spread / Math.max(n - 1, 1)) * i) * Math.PI / 180;
    const d = docsById.get(id);
    const isRegulatory = d && (d.source_type === 'fda_regulatory' ||
      d.domain === 'Regulatory' || /^K\d{6}/.test(d.name || ''));
    return {
      id,
      x: cx + R * Math.cos(angle),
      y: cy + R * Math.sin(angle) * 0.82,
      label: shortLabel(d ? d.name : `doc ${id}`),
      full: d ? stripExt(d.name) : `document ${id}`,
      regulatory: isRegulatory,
    };
  });

  const edges = nodes.map((nd, i) => `
    <line class="mg-edge" x1="${cx}" y1="${cy}" x2="${nd.x.toFixed(1)}" y2="${nd.y.toFixed(1)}"
          style="animation-delay:${i * 90}ms" />`).join('');

  const docNodes = nodes.map((nd, i) => `
    <g class="mg-doc" style="animation-delay:${140 + i * 90}ms">
      <title>${escapeHtml(nd.full)}</title>
      <circle cx="${nd.x.toFixed(1)}" cy="${nd.y.toFixed(1)}" r="7"
              class="${nd.regulatory ? 'mg-node-reg' : 'mg-node-doc'}" />
      <text x="${nd.x.toFixed(1)}" y="${(nd.y + (nd.y < cy ? -13 : 19)).toFixed(1)}"
            text-anchor="middle" class="mg-label">${escapeHtml(nd.label)}</text>
    </g>`).join('');

  return `
    <figure class="mini-graph">
      <svg viewBox="0 0 ${W} ${H}" role="img"
           aria-label="${escapeHtml(docIds.length)} documents connected through ${escapeHtml(hub.name)}">
        <g class="mg-edges">${edges}</g>
        <g class="mg-hub">
          <circle cx="${cx}" cy="${cy}" r="19" class="mg-node-hub-halo" />
          <circle cx="${cx}" cy="${cy}" r="11" class="mg-node-hub" />
          <text x="${cx}" y="${cy + 36}" text-anchor="middle" class="mg-hub-label">
            ${escapeHtml(hub.name)}
          </text>
        </g>
        ${docNodes}
      </svg>
      <figcaption class="mg-caption">
        <span class="mg-key"><i class="mg-swatch mg-swatch-hub"></i>shared concept</span>
        <span class="mg-key"><i class="mg-swatch mg-swatch-doc"></i>documentation</span>
        <span class="mg-key"><i class="mg-swatch mg-swatch-reg"></i>regulatory</span>
      </figcaption>
    </figure>`;
}

function stripExt(name) {
  return String(name).replace(/\.[^.]+$/, '').replace(/_/g, ' ');
}

function sourceLabel(type) {
  return type === 'fda_regulatory' ? 'FDA regulatory record' : 'product documentation';
}

/** Plain-language render. Leads with the finding; the mechanism is secondary. */
export function renderGraphContribution(contribution, index = null) {
  if (!contribution) return '';

  const { documentCount, bridges, headline, spansSources, graphOnlyLinks } = contribution;
  const top = contribution.primary || bridges[0];

  let lead;
  if (headline && headline.ratio < 0.22) {
    lead = `This answer drew on <strong>${documentCount} documents</strong>. ` +
           `Two of them — <em>${escapeHtml(stripExt(headline.aName))}</em> and ` +
           `<em>${escapeHtml(stripExt(headline.bName))}</em> — share almost no wording ` +
           `(${Math.round(headline.ratio * 100)}% vocabulary in common). ` +
           `Keyword search would never have returned them together. ` +
           `They were connected because both describe ` +
           `<strong>${escapeHtml(top.name)}</strong>.`;
  } else {
    lead = `This answer drew on <strong>${documentCount} documents</strong>, linked through ` +
           `<strong>${escapeHtml(top.name)}</strong> — a concept that appears in ` +
           `${top.reach} documents across the corpus.`;
  }

  const crossNote = spansSources
    ? `<p class="gv-cross">It combined product documentation with regulatory filings. ` +
      `Those are written by different teams, in different language, and normally sit in ` +
      `different systems — the graph is what puts them in one answer.</p>`
    : '';

  const chips = bridges.map(b => `
    <span class="gv-chip" data-kind="${escapeHtml(b.kind)}">
      <span class="gv-chip-kind">${escapeHtml(KIND_PHRASE[b.kind] || b.kind)}</span>
      <span class="gv-chip-name">${escapeHtml(b.name)}</span>
      <span class="gv-chip-reach">${b.docs.length} docs here</span>
    </span>`).join('');

  const mini = index ? renderMiniGraph(contribution, index) : '';

  return `
    <div class="graph-value${mini ? ' graph-value-split' : ''}">
      <div class="gv-main">
        <div class="gv-head">
          <span class="gv-eyebrow">What the graph contributed</span>
        </div>
        <p class="gv-lead">${lead}</p>
        ${crossNote}
        <div class="gv-chips">${chips}</div>
        ${graphOnlyLinks > 0 ? `
          <p class="gv-metric">
            <strong>${graphOnlyLinks}</strong> of the connections behind this answer
            could not have been made by keyword matching alone.
          </p>` : ''}
        <details class="tech-detail">
          <summary>How this was determined</summary>
          <div class="tech-detail-body">
            <p class="gv-tech">For every pair of documents used in the answer we measure
            how much distinctive vocabulary they share, then check whether the knowledge
            graph links them through a common subject. A pair with low shared vocabulary
            but a graph link is a connection keyword search could not have found.</p>
            <p class="gv-tech"><code>overlap = |sharedTerms| / min(|termsA|, |termsB|)</code>
            — common words and boilerplate excluded. Below 22% is treated as
            "no meaningful vocabulary in common".</p>
          </div>
        </details>
      </div>
      ${mini ? `<div class="gv-viz">${mini}</div>` : ''}
    </div>`;
}

/**
 * Standing explanation for the graph section — shown before any question is
 * asked, so a first-time viewer understands what they are looking at.
 */
export function graphPurposeCopy(index) {
  const entities = index.entities || [];
  const spanning = entities.filter(e => (e.document_ids || []).length >= 2).length;
  const crossSource = entities.filter(e => e.cross_source).length;
  return {
    headline: 'The map is not decoration. It is how separate documents become one answer.',
    body: `Search finds documents that share your words. This map finds documents that ` +
          `share a subject — even when they share no words at all. Across this corpus, ` +
          `${spanning} concepts appear in more than one document, and ${crossSource} of ` +
          `them link product documentation to its regulatory record.`,
    stat: spanning,
    statLabel: 'concepts linking two or more documents',
    stat2: crossSource,
    stat2Label: 'links between documentation and regulatory records',
  };
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
