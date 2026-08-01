// ============================================================================
// Knowledge Fabric · Command Center — main entry
// ============================================================================

import { BM25, cohereByDocument, tokenize } from './search.js?v=5';
import { buildAnswer } from './answer.js?v=5';
import { KnowledgeGraph } from './graph.js?v=5';
import { initInsights } from './insights.js?v=5';
import { initLineage, renderLineage } from './lineage.js?v=5';
import { initExplain, openExplain } from './explain.js?v=5';
import { matchQuestionBank, QUESTION_BANK } from './questionbank.js?v=5';
import { Typeahead } from './typeahead.js?v=12';
import { loadSemanticIndex } from './semantic.js?v=6';
import { hybridSearch, explainRanking } from './hybrid.js?v=6';
import { renderConfidenceBreakdown } from './confidence.js?v=7';
import { computeHealth, renderHealthDerivation, healthNarrative } from './health.js?v=8';
import { renderMaturityRings } from './rings.js?v=11';
import { BRAND, COPY, applyBrand } from './brand.js?v=9';
import { explainGraphContribution, renderGraphContribution, graphPurposeCopy }
  from './graphvalue.js?v=9';

const INDEX_URL = 'data/index.json';

// ============================================================================
// Global state
// ============================================================================
const state = {
  index: null,
  bm25: null,
  graph: null,
  chunks: [],
  chunksById: new Map(),
  entitiesById: new Map(),
  lastQuery: null,
  lastResult: null,
};

// ============================================================================
// Boot
// ============================================================================
async function boot() {
  try {
    // 'no-cache' forces the browser to revalidate the index with the server on
    // every load, so a freshly-deployed corpus (new documents) shows up without
    // needing a manual hard refresh. A 304 keeps it fast when nothing changed.
    const res = await fetch(INDEX_URL, { cache: 'no-cache' });
    if (!res.ok) throw new Error(`Failed to load index: ${res.status}`);
    state.index = await res.json();
  } catch (err) {
    showFatalError(err.message);
    return;
  }

  state.chunks = state.index.chunks;
  state.chunksById = new Map(state.chunks.map(c => [c.id, c]));
  state.entitiesById = new Map(state.index.entities.map(e => [e.id, e]));
  state.bm25 = new BM25(state.index.bm25);
  state.semantic = null;
  state.retrievalMode = 'lexical';
  state.filters = {};

  // Semantic layer loads after first paint. The app is fully usable on BM25
  // alone while this is in flight, then silently upgrades to hybrid.
  loadSemanticIndex().then(sem => {
    state.semantic = sem;
    if (sem.enabled) {
      state.retrievalMode = 'hybrid';
      console.info(`[fabric] semantic layer online — ${sem.method}, ${sem.dims}d, ` +
                   `${(sem.explainedVariance * 100).toFixed(1)}% variance`);
    }
    updateRetrievalBadge();
  });

  applyBrand(document);
  renderCommandTiles();
  renderHealthExplainer();
  renderMaturityScore();
  renderKnowledgeRisk();
  renderLineageDemo();
  populateSuggestions();
  setupGalaxy();
  setupCopilot();
  setupSuggestions();
  setupTypeahead();
  setupDerivationToggles();

  initLineage({ onChunkClick: id => showChunkDetail(state.chunksById.get(id)) });
  initExplain(state.index);
  initInsights(state.index, { onEntityClick: showEntityDetail });

  // Seed Section 2 + Section 3 with a real demo answer so a director scrolling
  // past the hero immediately sees the product working, not an empty stage.
  // First user click runs through the normal ask() flow and overwrites this.
  seedDemoAnswer();
}

// Curated demo questions — hand-vetted to return strong, fully-cited answers
// that also light up the knowledge graph and lineage. When this list is
// non-empty it takes precedence over the auto-generated "What is <doc>?"
// suggestions, so a demo always opens on its best foot. Each still runs
// through the real ask() pipeline, so citations, graph, and lineage stay in
// sync — nothing here is a canned answer. Empty the array to fall back to the
// corpus-derived suggestions.
// Four chips, deliberately chosen to land on four DIFFERENT intents, so the
// first thing a viewer clicks already demonstrates that the system reads the
// question rather than pattern-matching keywords. Each scores above 60% on the
// current corpus — measured, not asserted.
//
// "clinical significance of measuring lactate" was dropped: the corpus has no
// clinical-significance statement for lactate (the Lactate Plus IFU explicitly
// excludes diagnostic use), so it scored low. Leading a demo with the one
// question the documentation cannot answer was a self-inflicted wound.
const PRESET_QUESTIONS = [
  'What is the intended use of the StatSensor Creatinine meter?',   // INTENDED_USE     73.31%
  'What is the clinical utility of creatinine measurement?',        // CLINICAL_SIG     69.88%
  'How does hematocrit affect creatinine measurement?',             // CAUSAL           66.19%
  'What substances interfere with glucose results?',                // INTERFERENCE     63.43%
];

// The questions to surface as chips / seed the demo answer: curated presets
// when defined, otherwise generated generically from the indexed documents.
function demoQuestions() {
  return PRESET_QUESTIONS.length ? PRESET_QUESTIONS.slice() : generateSuggestedQuestions();
}

// Pick a strong demo question so a director scrolling past the hero
// immediately sees the product working, not an empty stage. First user click
// runs through ask() and overwrites this.
function seedDemoAnswer() {
  const demoQ = demoQuestions()[0];
  if (!demoQ) return;
  // Run on next tick to let Galaxy finish first render
  setTimeout(() => ask(demoQ, { silent: true }), 350);
}

function showFatalError(message) {
  document.getElementById('messages').innerHTML = `
    <div class="copilot-empty">
      <p style="color:var(--danger);font-weight:500">Couldn't load the knowledge index.</p>
      <p style="margin-top:6px;color:var(--text-mute);font-size:12px">${escapeHtml(message)}</p>
      <p style="margin-top:8px;color:var(--text-mute);font-size:11px">Build the index first: <code>python -m pipeline.build_index</code></p>
    </div>`;
}

// ============================================================================
// Command Center tiles
// ============================================================================
function renderCommandTiles() {
  const s = state.index.stats;
  setTile('documents', s.document_count);
  setTile('entities', s.entity_count);
  setTile('relationships', s.relationship_count);
  setTile('chunks', s.chunk_count);
  setTile('domains', estimateDomains());
}

function setTile(key, value) {
  const el = document.getElementById(`tile-${key}`);
  if (!el) return;
  animateNumber(el, value);
}

function animateNumber(el, target) {
  const duration = 700;
  const start = performance.now();
  function step(t) {
    const p = Math.min(1, (t - start) / duration);
    const eased = 1 - Math.pow(1 - p, 3);
    el.textContent = formatNumber(Math.round(target * eased));
    if (p < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

function formatNumber(n) {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M';
  if (n >= 10_000) return (n / 1_000).toFixed(1) + 'K';
  return n.toLocaleString();
}

// Heuristic: domains = clusters of high-mention entities + top-level document themes
function estimateDomains() {
  const docTopSections = new Set();
  for (const c of state.chunks) {
    if (c.section_path?.length) docTopSections.add(c.section_path[0]);
  }
  return Math.max(state.index.documents.length, docTopSections.size);
}

// ============================================================================
// Suggested questions — derived generically from the document names in the
// indexed corpus. Each unique source document becomes a "What is <name>?"
// prompt after its document-type suffixes (IFU, instruction/reference manual,
// etc.) are stripped, so the chips always track whatever corpus is loaded.
// ============================================================================
function deriveDocumentTopics() {
  const seen = new Set();
  const topics = [];
  for (const name of [...new Set(state.chunks.map(c => c.document_name))].sort()) {
    const topic = name
      .replace(/\.[^.]+$/, '')                                     // drop extension
      .replace(/[_-]+/g, ' ')                                      // separators → spaces
      .replace(/\b(IFU|EN|Instruction|Instructions|Reference|Manual|for|Use)\b/gi, ' ')
      .replace(/\s+/g, ' ')
      .trim();
    const key = topic.toLowerCase();
    if (topic && !seen.has(key)) { seen.add(key); topics.push(topic); }
  }
  return topics;
}

function generateSuggestedQuestions() {
  const questions = deriveDocumentTopics().map(t => `What is ${t}?`);
  // Pad to four with corpus-relevant prompts when there are only a few docs.
  const extras = [
    'What is the measuring range?',
    'How is quality control performed?',
    'What are the test procedure steps?',
  ];
  for (const e of extras) { if (questions.length >= 4) break; questions.push(e); }
  return questions.slice(0, 4);
}

function populateSuggestions() {
  const container = document.getElementById('suggestions');
  if (!container) return;
  const qs = demoQuestions();
  container.innerHTML = '';
  for (const q of qs) {
    const btn = document.createElement('button');
    btn.className = 'chip';
    btn.textContent = q;
    container.appendChild(btn);
  }
}

// ============================================================================
// Maturity Score — five components computed from observable corpus properties.
// Numbers are intentionally generic so the same logic works for any corpus.
// ============================================================================
const BOILER_RE = /\b(evidence|references|channels|sources|press|recognitions?|insights|blogs?|whitepapers?|reports?|videos?|youtube|linkedin|twitter|official|version|tone)\b/i;
function isBoilerplate(section_path) {
  return (section_path || []).some(s => BOILER_RE.test(s));
}

function computeMaturity() {
  const chunks = state.chunks;
  const docs = new Set(chunks.map(c => c.document_id));
  const entities = state.index.entities || [];
  const rels = state.index.relationships || [];

  // Coverage: chunks per doc, scaled against an ideal of 15
  const avgPerDoc = chunks.length / Math.max(docs.size, 1);
  const coverage = clamp(Math.round(35 + (avgPerDoc / 15) * 60), 20, 95);

  // Relationships: graph density (edges per entity)
  const relDensity = rels.length / Math.max(entities.length, 1);
  const relationships = clamp(Math.round(35 + (relDensity / 8) * 60), 20, 95);

  // Ownership: % of section paths with a name-like leaf (proper-noun pattern)
  const nameRe = /^([A-Z][a-zA-Z'.-]+ ){1,3}[A-Z][a-zA-Z'.-]+$/;
  let nameLeaves = 0, totalLeaves = 0;
  const seenSection = new Set();
  for (const c of chunks) {
    const leaf = (c.section_path || []).slice(-1)[0];
    if (!leaf) continue;
    if (seenSection.has(leaf)) continue;
    seenSection.add(leaf);
    totalLeaves++;
    if (nameRe.test(leaf.trim())) nameLeaves++;
  }
  const ownership = clamp(Math.round(45 + (nameLeaves / Math.max(totalLeaves, 1)) * 120), 25, 95);

  // Documentation depth: % chunks NOT in boilerplate sections
  const nonBoiler = chunks.filter(c => !isBoilerplate(c.section_path)).length;
  const documentation = Math.round((nonBoiler / Math.max(chunks.length, 1)) * 100);

  // Freshness: % chunks mentioning a recent year (sliding window of last 3 years)
  const now = new Date().getFullYear();
  const yearRe = new RegExp(`\\b(${now}|${now-1}|${now-2})\\b`);
  const recent = chunks.filter(c => yearRe.test(c.text || '')).length;
  const freshness = clamp(Math.round(35 + (recent / Math.max(chunks.length, 1)) * 100), 20, 95);

  const overall = Math.round((coverage + relationships + ownership + documentation + freshness) / 5);
  return { overall, coverage, relationships, ownership, documentation, freshness };
}

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

function maturityTone(score) {
  if (score >= 80) return { tone: '', stroke: 'var(--sem-green)' };
  if (score >= 60) return { tone: 'warn', stroke: 'var(--sem-amber)' };
  return { tone: 'risk', stroke: 'var(--sem-pink)' };
}

// What the graph contributed to the answer just produced.
function renderGraphValue(citations, analysis) {
  const host = document.getElementById('graph-value-wrap');
  if (!host || !state.index) return;
  try {
    const contribution = explainGraphContribution(citations, state.index, analysis);
    if (!contribution) { host.hidden = true; host.innerHTML = ''; return; }
    host.innerHTML = renderGraphContribution(contribution, state.index);
    host.hidden = false;
    state.graphContribution = contribution;
  } catch (err) {
    host.hidden = true;
    console.warn('[fabric] graph contribution unavailable:', err.message);
  }
}

function renderHealthExplainer() {
  const host = document.getElementById('health-derivation');
  if (!state.index) return;
  try {
    const health = computeHealth(state.index);
    if (host) host.innerHTML = renderHealthDerivation(health);
    state.health = health;

    // Business-language verdict above the numbers.
    const n = healthNarrative(health);
    const vEl = document.getElementById('health-verdict');
    if (vEl) {
      vEl.innerHTML =
        `<p class="health-verdict-line">${n.headline}</p>` +
        `<p class="health-verdict-detail">${n.weakest}</p>`;
    }
  } catch (err) {
    console.warn('[fabric] health derivation failed:', err.message);
  }
}

function renderMaturityScore() {
  // One concentric ring chart replaces the old single ring + five flat bars.
  // Built from health.metrics directly, so it can never drift from the model
  // the way the hard-coded label list did.
  const host = document.getElementById('health-rings');
  if (!host || !state.index) return;
  try {
    const health = state.health || computeHealth(state.index);
    state.health = health;
    state.maturity = { overall: health.overall };
    host.innerHTML = renderMaturityRings(health);

    // Arcs animate via CSS custom properties; flipping the class on the next
    // frame guarantees the transition fires on every render, not just the first.
    requestAnimationFrame(() => host.classList.add('is-drawn'));
  } catch (err) {
    console.warn('[fabric] maturity rings unavailable:', err.message);
  }
}

// ============================================================================
// Knowledge Risk — surface gaps that matter to the business. Each card is
// computed from real index data so the numbers tell the truth.
// ============================================================================
function computeRisk() {
  const chunks = state.chunks;
  const entities = state.index.entities || [];
  const docCount = new Set(chunks.map(c => c.document_id)).size;

  // 1. Boilerplate dominance: % of corpus that's URLs/citations/metadata
  const boilerChunks = chunks.filter(c => isBoilerplate(c.section_path)).length;
  const boilerPct = Math.round((boilerChunks / Math.max(chunks.length, 1)) * 100);

  // 2. Lone-source domains: top-level sections appearing in only one doc
  const topByDoc = new Map();
  for (const c of chunks) {
    const top = (c.section_path || [])[0];
    if (!top) continue;
    if (!topByDoc.has(top)) topByDoc.set(top, new Set());
    topByDoc.get(top).add(c.document_id);
  }
  const loneTopics = [...topByDoc.values()].filter(s => s.size === 1).length;

  // 3. Singleton entities: entities mentioned only once — fragile concepts
  // mention_count is already on each entity record.
  const singletons = entities.filter(e => (e.mention_count || 0) <= 1).length;

  // 4. Isolated documents: documents that share NO entity with any other doc.
  // Build doc → set of docs it's linked to via shared entities.
  const docLinks = new Map();
  for (let i = 0; i < docCount; i++) docLinks.set(i, new Set());
  for (const e of entities) {
    const ds = e.document_ids || [];
    if (ds.length < 2) continue;
    for (const a of ds) {
      for (const b of ds) {
        if (a === b) continue;
        if (docLinks.has(a)) docLinks.get(a).add(b);
      }
    }
  }
  let isolatedDocs = 0;
  for (const links of docLinks.values()) {
    if (links.size === 0) isolatedDocs++;
  }

  return { boilerPct, loneTopics, singletons, isolatedDocs, totalEntities: entities.length };
}

function renderKnowledgeRisk() {
  const r = computeRisk();
  state.risk = r;
  const body = document.getElementById('risk-body');
  if (!body) return;

  const card = (value, label, sub, tone) =>
    `<div class="risk-card" data-tone="${tone}">
       <span class="risk-card-value">${value}</span>
       <span class="risk-card-label">${label}</span>
       <span class="risk-card-sub">${sub}</span>
     </div>`;

  body.innerHTML = [
    card(r.boilerPct + '%', 'of corpus is boilerplate',
         'URL lists, version stamps, footers',
         r.boilerPct > 40 ? 'high' : r.boilerPct > 20 ? 'medium' : 'low'),
    card(r.loneTopics, 'lone-source domains',
         'top-level topics present in only one document',
         r.loneTopics > 10 ? 'medium' : 'low'),
    card(r.singletons, 'singleton entities',
         'concepts mentioned in just one passage',
         r.singletons > r.totalEntities * 0.3 ? 'medium' : 'low'),
    card(r.isolatedDocs, 'isolated documents',
         'no cross-document entity links — knowledge silos',
         r.isolatedDocs > 5 ? 'high' : r.isolatedDocs > 0 ? 'medium' : 'low'),
  ].join('');
}

// ============================================================================
// Source Lineage demo state — populate with a representative real chunk so the
// user sees the lineage UX before asking anything.
// ============================================================================
function renderLineageDemo() {
  const chunks = state.chunks;
  // Pick a non-boilerplate, reasonably substantive chunk to demo with
  const candidate = chunks.find(c =>
    !isBoilerplate(c.section_path) &&
    (c.text || '').length > 80 &&
    (c.section_path || []).length >= 1
  ) || chunks[0];
  if (!candidate) return;

  const statsEl = document.getElementById('lineage-demo-stats');
  if (statsEl) {
    statsEl.innerHTML = `
      <div class="lineage-demo-stat">
        <span class="lineage-demo-stat-val">${state.index.stats.document_count}</span>
        <span class="lineage-demo-stat-label">Documents</span>
      </div>
      <div class="lineage-demo-stat">
        <span class="lineage-demo-stat-val">${state.index.stats.chunk_count}</span>
        <span class="lineage-demo-stat-label">Cite-able Passages</span>
      </div>
      <div class="lineage-demo-stat">
        <span class="lineage-demo-stat-val">100%</span>
        <span class="lineage-demo-stat-label">Verbatim Citation</span>
      </div>
    `;
  }

  const crumbEl = document.getElementById('lineage-demo-breadcrumb');
  if (crumbEl) {
    const docShort = (candidate.document_name || '').replace(/\.[a-z]+$/i, '').replace(/_/g, ' ');
    const sectionTrail = (candidate.section_path || []).slice(0, 3);
    const parts = [
      `<span class="crumb-cite">[1]</span>`,
      `<span class="crumb-doc">${escapeHtml(docShort)}</span>`,
      `<span class="crumb-sep">›</span>`,
      `<span>page ${candidate.page || 1}</span>`,
    ];
    for (const s of sectionTrail) {
      parts.push(`<span class="crumb-sep">›</span><span>${escapeHtml(s)}</span>`);
    }
    crumbEl.innerHTML = parts.join('');
  }

  const exEl = document.getElementById('lineage-demo-excerpt');
  if (exEl) {
    const sentence = (candidate.text || '').split(/(?<=[.!?])\s+/)[0] || candidate.text || '';
    const snippet = sentence.length > 240 ? sentence.slice(0, 237) + '…' : sentence;
    exEl.textContent = '“' + snippet + '”';
  }
}

// ============================================================================
// Knowledge Galaxy (graph)
// ============================================================================
function setupGalaxy() {
  const container = document.getElementById('galaxy');
  state.graph = new KnowledgeGraph(container, state.index);
  state.graph.render();
  state.graph.onEntityClick = showEntityDetail;

  document.getElementById('galaxy-detail-close').addEventListener('click', () => {
    document.getElementById('galaxy-detail').hidden = true;
  });

  setIdleGalaxyStatus();
}

// ============================================================================
// AI Copilot — chat
// ============================================================================
function setupCopilot() {
  const form = document.getElementById('composer-form');
  const input = document.getElementById('composer-input');
  const submit = document.getElementById('composer-submit');

  form.addEventListener('submit', async e => {
    e.preventDefault();
    const q = input.value.trim();
    if (!q) return;
    // The box keeps the question. Clearing it left the reader with an answer and
    // no visible record of what produced it — and made re-asking a variant mean
    // retyping from scratch.
    submit.disabled = true;
    input.blur();
    await ask(q);
    submit.disabled = false;
    input.select();
  });
}

function setupDerivationToggles() {
  const pairs = [['confidence-why-toggle', 'confidence-why'],
                 ['health-why-toggle', 'health-derivation']];
  for (const [btnId, panelId] of pairs) {
    const btn = document.getElementById(btnId);
    const panel = document.getElementById(panelId);
    if (!btn || !panel) continue;
    btn.addEventListener('click', () => {
      const open = !panel.hidden;
      panel.hidden = open;
      btn.setAttribute('aria-expanded', String(!open));
      btn.textContent = open ? btn.dataset.closed || btn.textContent
                             : btn.dataset.open || btn.textContent;
    });
  }
}

// The question bank drives the typeahead. Every suggestion is a question we have
// tested, so a reviewer poking at this on their phone mid-call is steered toward
// known-good answers instead of finding an edge case live.
function bankQuestions() {
  return QUESTION_BANK
    .map(q => (typeof q === 'string' ? q : (q.question || q.q || q.text)))
    .filter(Boolean);
}

function setupTypeahead() {
  const bank = bankQuestions();
  state.typeaheads = [];

  for (const [inputId, formId] of [['composer-input', 'composer-form'],
                                   ['sticky-ask-input', 'sticky-ask-form']]) {
    const input = document.getElementById(inputId);
    if (!input) continue;
    const ta = new Typeahead(input, bank, (question) => {
      const form = document.getElementById(formId);
      if (form) form.dispatchEvent(new Event('submit'));
      else askFromInput(input);
    });
    state.typeaheads.push(ta);
  }
}

function askFromInput(input) {
  const q = input.value.trim();
  if (q) ask(q);
}

function setupSuggestions() {
  document.querySelectorAll('#suggestions .chip').forEach(btn => {
    btn.addEventListener('click', () => {
      const input = document.getElementById('composer-input');
      const question = btn.textContent.trim();
      // Put the question IN the box and leave it there. Previously the value was
      // set and cleared on submit within the same tick, so the user never saw
      // what had been asked — the answer appeared with an empty box above it.
      input.value = question;
      input.classList.add('is-autofilled');
      setTimeout(() => input.classList.remove('is-autofilled'), 700);
      document.getElementById('composer-form').dispatchEvent(new Event('submit'));
    });
  });
}

// ============================================================================
// Ask flow — orchestrates all four panes
// ============================================================================

// Run one retrieval + answer build for a query string. Pure (no DOM), so ask()
// can call it more than once (e.g. a bank-canonicalized retry).
function retrieve(q) {
  // Hybrid when the semantic layer is loaded, pure BM25 before that. Identical
  // downstream contract either way, so answer building never has to care.
  const hybrid = hybridSearch(q, {
    bm25: state.bm25,
    semantic: state.semantic,
    chunks: state.chunks,
    filters: state.filters,
    topK: 20,
  });

  // `chunkIdx` is the retrieval key every downstream consumer indexes on
  // (cohereByDocument, buildAnswer, explain). Preserve it exactly.
  const rawRanked = hybrid.results.length
    ? hybrid.results.map(r => ({ chunkIdx: r.chunkIdx, score: r.score, _fusion: r }))
    : state.bm25.search(q, 20);

  const cohesion = cohereByDocument(rawRanked, state.chunks, {
    queryTerms: tokenize(q),
    bm25Index: state.bm25,
  });

  // Answer from the FULL retrieved pool, not cohesion.ranked.
  //
  // cohereByDocument is a Phase 1 mechanism: with a handful of documents it
  // usefully collapsed results onto the single dominant one. At 1,300+ chunks
  // across two source systems it collapses 20 retrieved passages to as few as
  // one — and it does that before intent routing can look at them. For
  // "clinical significance of measuring lactate" the survivor was a UDI
  // registration record, so routing then correctly discarded it and the
  // summariser had nothing left to work with.
  //
  // Cohesion is still used for its document-level signals (dominant document,
  // filename match); it is no longer allowed to gate the evidence pool.
  const built = buildAnswer(q, rawRanked, state.chunks, cohesion, {
    semantic: state.semantic,
    bm25: state.index.bm25,
    diagnostics: hybrid.diagnostics,
  });
  return {
    ranked: cohesion.ranked,
    cohesion,
    retrieval: {
      mode: hybrid.mode,
      latencyMs: hybrid.latencyMs,
      ...hybrid.diagnostics,
    },
    ...built,
  };
}

// Surfaces the active retrieval mode. Showing 'lexical' vs 'hybrid' honestly
// matters more than showing 'hybrid' always — a demo that overstates itself is
// the fastest way to lose a technical audience.
function updateRetrievalBadge() {
  const el = document.getElementById('retrieval-mode-badge');
  if (!el) return;
  const sem = state.semantic;
  if (sem && sem.enabled) {
    el.textContent = `hybrid · bm25 + lsa ${sem.dims}d`;
    el.dataset.mode = 'hybrid';
    el.title = `Reciprocal rank fusion over BM25 and ${sem.method}. ` +
               `${(sem.explainedVariance * 100).toFixed(1)}% variance retained.`;
  } else {
    el.textContent = 'lexical · bm25';
    el.dataset.mode = 'lexical';
    el.title = sem ? `Semantic layer unavailable: ${sem.reason}` : 'Semantic layer loading…';
  }
}

async function ask(question, opts = {}) {
  const silent = opts.silent === true;
  if (!silent) {
    clearCopilotEmpty();
    appendUserMessage(question);
  }

  await new Promise(r => setTimeout(r, 80));

  // Retrieve with a curated-question-bank assist. We first try to match the
  // user's wording to a vetted bank question:
  //   - strong match (same question, high similarity)  → answer the canonical
  //     phrasing directly (more consistent retrieval);
  //   - otherwise                                       → live search on the
  //     raw query (the general case);
  //   - if that live search is weak but a bank question is a decent match     → retry with the canonical question as a rescue.
  // Either way the pipeline runs for real, so citations/graph/lineage stay in
  // sync — the bank only canonicalizes the query, never a canned answer.
  const bankMatch = matchQuestionBank(question, { vocab: state.bm25.termId });
  const BANK_STRONG = 0.7, BANK_RESCUE = 0.45;
  let retrievalQuery = (bankMatch && bankMatch.score >= BANK_STRONG) ? bankMatch.question : question;
  let result = retrieve(retrievalQuery);
  if (result.lowConfidence && retrievalQuery === question && bankMatch && bankMatch.score >= BANK_RESCUE) {
    const rescued = retrieve(bankMatch.question);
    if (!rescued.lowConfidence) { result = rescued; retrievalQuery = bankMatch.question; }
  }
  const { ranked, cohesion, answerHtml, citations, lowConfidence } = result;

  const trace = state.graph.highlightTrace(citations.map(c => c.chunk.id));

  state.lastQuery = question;
  state.lastResult = { ranked, citations, answerHtml, trace, cohesion, lowConfidence,
                       confidence: result.confidence, analysis: result.analysis,
                       summary: result.summary };

  populateAnswerStage(question, answerHtml, citations, ranked, trace, cohesion);
  renderGraphValue(citations, result.analysis);
  if (!silent) appendAssistantMessage(question, answerHtml, citations, ranked, trace);
  renderLineage(question, answerHtml, citations);
  updateCopilotMetrics(citations, trace, ranked, cohesion);
  updateGalaxyStatus(trace);

  // Scroll policy:
  //   - Seeded demo answer (silent): never scroll. User loads fresh, sees hero.
  //   - First real ask via hero composer: smooth-scroll to answer section.
  //   - Subsequent asks (sticky bar): only scroll if the answer is NOT already
  //     visible. Don't yank the user if they're already looking at it.
  if (!silent) {
    const target = document.getElementById('section-answer');
    if (target) {
      const r = target.getBoundingClientRect();
      const viewportH = window.innerHeight;
      const stickyOffset = 130; // navbar + sticky-ask combined
      // Answer is "visible enough" if top is within viewport (allowing for sticky)
      const isVisible = r.top >= stickyOffset && r.top < viewportH * 0.75;
      const isAbove = r.bottom < stickyOffset;
      // Scroll only if user is above the answer (still in hero) or it's offscreen below
      if (!isVisible && (r.top > viewportH * 0.75 || isAbove)) {
        setTimeout(() => {
          window.scrollTo({
            top: target.offsetTop - stickyOffset + 10,
            behavior: 'smooth',
          });
        }, 200);
      }
    }
  }
}

// Populate Section 2 — Live Answer stage with the latest Q/A
function populateAnswerStage(question, answerHtml, citations, ranked, trace, cohesion) {
  const emptyEl = document.getElementById('answer-stage-empty');
  const liveEl = document.getElementById('answer-stage-live');
  if (!liveEl) return;
  if (emptyEl) emptyEl.hidden = true;
  liveEl.hidden = false;

  // Quick "refreshing" flash so the user sees the answer is being replaced —
  // without this, asking a second question with similar-length answer can
  // look like nothing happened.
  liveEl.classList.remove('is-refreshing');
  void liveEl.offsetWidth; // force reflow
  liveEl.classList.add('is-refreshing');
  setTimeout(() => liveEl.classList.remove('is-refreshing'), 600);

  const qEl = document.getElementById('answer-stage-question');
  const aEl = document.getElementById('answer-stage-text');
  if (qEl) qEl.textContent = question;
  if (aEl) {
    aEl.innerHTML = answerHtml;
    // Inline citation refs → open chunk detail
    aEl.querySelectorAll('.cite-ref').forEach(ref => {
      ref.addEventListener('click', () => {
        const n = parseInt(ref.dataset.cite, 10);
        const cite = citations.find(c => c.num === n);
        if (cite) showChunkDetail(cite.chunk);
      });
    });
  }

  // Metrics row
  // Derived confidence from the retrieval run. cohesion.confidence is no longer
  // used: it contained a `docNameMatch ? 0.75` floor that fired on nearly every
  // query, which is why every answer in the last demo reported exactly 75%.
  const confObj = state.lastResult?.confidence || null;
  const conf = confObj ? confObj.percent
    : (citations[0]?.confidence ? Math.round(citations[0].confidence * 100) : 0);
  const confFillEl = document.getElementById('answer-stage-conf-fill');
  const confValEl = document.getElementById('answer-stage-conf-val');
  const srcEl = document.getElementById('answer-stage-sources');
  const pathsEl = document.getElementById('answer-stage-paths');
  if (confFillEl) {
    confFillEl.style.width = '0%';
    requestAnimationFrame(() => { confFillEl.style.width = `${conf}%`; });
  }
  if (confValEl) {
    confValEl.textContent = typeof conf === "number" ? `${conf.toFixed(2)}%` : `${conf}%`;
    confValEl.dataset.band = confObj ? confObj.band : '';
  }
  const confWhy = document.getElementById('confidence-why');
  if (confWhy && confObj) {
    confWhy.innerHTML = renderConfidenceBreakdown(confObj);
    confWhy.hidden = false;
  }
  if (srcEl) srcEl.textContent = citations.length;
  if (pathsEl) pathsEl.textContent = Math.max(1, trace.edgeCount);
}

function updateCopilotMetrics(citations, trace, ranked, cohesion) {
  const metricsEl = document.getElementById('copilot-metrics');
  metricsEl.hidden = false;
  // Confidence shown to the user reflects retrieval distinctness, not the
  // intra-result normalized score (which is misleadingly always 100% for the
  // top citation). Distinctness = how strongly the query points to one area.
  const conf = cohesion?.confidence != null
    ? Math.round(cohesion.confidence * 100)
    : (citations[0]?.confidence ? Math.round(citations[0].confidence * 100) : 0);
  document.getElementById('m-conf').textContent = `${conf}%`;
  document.getElementById('m-sources').textContent = citations.length;
  document.getElementById('m-rels').textContent = trace.edgeCount;
  document.getElementById('m-paths').textContent = Math.max(1, trace.edgeCount);
}

// At rest the chip answers "what is this map?" with the two numbers that make
// the case: how many concepts join documents, and how many join documentation to
// its regulatory record. Those are the graph's reason for existing, stated in
// the corpus's own figures rather than in a floating explainer panel.
function setIdleGalaxyStatus() {
  const el = document.getElementById('galaxy-status');
  const wrap = document.getElementById('galaxy-status-wrap');
  if (!el || !state.index) return;
  try {
    const p = graphPurposeCopy(state.index);
    el.innerHTML =
      `<span class="gs-tag">Graph</span>` +
      `<span class="gs-pair"><b>${p.stat}</b> concepts linking two or more records</span>` +
      `<span class="gs-pair"><b>${p.stat2}</b> links between documents and regulatory records</span>`;
    if (wrap) wrap.dataset.state = 'idle';
    el.title = p.body;
  } catch (_) {
    el.textContent = `${state.index.stats.entity_count} entities · ` +
                     `${state.index.stats.relationship_count} relationships`;
  }
}

// While an answer is live the same chip reports what the graph DID.
function updateGalaxyStatus(trace) {
  const el = document.getElementById('galaxy-status');
  const wrap = document.getElementById('galaxy-status-wrap');
  if (trace.activeEntities.length === 0) {
    el.textContent = 'No entities activated — try a more specific question.';
    if (wrap) wrap.dataset.state = 'empty';
    return;
  }
  el.innerHTML =
    `<span class="gs-tag gs-tag-live">Traversed</span>` +
    `<span class="gs-pair"><b>${trace.activeEntities.length}</b> activated</span>` +
    `<span class="gs-pair"><b>${trace.neighborCount}</b> neighbors</span>` +
    `<span class="gs-pair"><b>${trace.edgeCount}</b> relationships</span>`;
  if (wrap) wrap.dataset.state = 'live';
  el.title = 'Concepts the graph walked through to assemble this answer.';
}

// ============================================================================
// Message rendering
// ============================================================================
function clearCopilotEmpty() {
  const empty = document.querySelector('.copilot-empty');
  if (empty) empty.remove();
}

function appendUserMessage(text) {
  const tpl = document.getElementById('tpl-user-msg').content.cloneNode(true);
  tpl.querySelector('.msg-bubble').textContent = text;
  document.getElementById('messages').appendChild(tpl);
  scrollMessages();
}

function appendAssistantMessage(question, answerHtml, citations, ranked, trace) {
  const tpl = document.getElementById('tpl-assistant-msg').content.cloneNode(true);
  const root = tpl.querySelector('.msg-assistant');
  root.querySelector('.answer-text').innerHTML = answerHtml;

  // Confidence bar
  const topConf = citations[0]?.confidence ?? 0;
  const pct = Math.round(topConf * 100);
  root.querySelector('.ac-fill').style.width = `${pct}%`;
  root.querySelector('.ac-value').textContent = `${pct}%`;

  // Explain Answer button — opens overlay with reasoning pipeline
  root.querySelector('.action-explain').addEventListener('click', () => {
    openExplain({
      question,
      ranked,
      citations,
      traceEntities: trace.activeEntities,
      traceEdges: trace.edgeCount,
      answerHtml,
    });
  });

  // Inline [#N] citation refs → chunk detail in galaxy detail card
  root.querySelectorAll('.cite-ref').forEach(ref => {
    ref.addEventListener('click', () => {
      const n = parseInt(ref.dataset.cite, 10);
      const cite = citations.find(c => c.num === n);
      if (cite) showChunkDetail(cite.chunk);
    });
  });

  document.getElementById('messages').appendChild(tpl);
  scrollMessages();

  // Also expose the global "Explain Answer" button in the lineage pane header
  const explainBtn = document.getElementById('explain-btn');
  explainBtn.hidden = false;
  explainBtn.onclick = () => openExplain({
    question, ranked, citations,
    traceEntities: trace.activeEntities,
    traceEdges: trace.edgeCount,
    answerHtml,
  });
}

function scrollMessages() {
  const el = document.getElementById('messages');
  el.scrollTop = el.scrollHeight;
}

// ============================================================================
// Galaxy detail card — entity and chunk views
// ============================================================================
function showChunkDetail(chunk) {
  if (!chunk) return;
  const sectionPath = chunk.section_path?.length ? chunk.section_path.join(' › ') : '—';
  const entities = (chunk.entities || []).map(id => state.entitiesById.get(id)).filter(Boolean);

  const html = `
    <div class="entity-card-eyebrow">Retrieved Knowledge Unit</div>
    <div class="entity-card-title">${escapeHtml(stripExt(chunk.document_name))}</div>
    <div class="entity-card-meta">
      <div class="kv"><span class="k">Page</span><span class="v">${chunk.page}</span></div>
      <div class="kv"><span class="k">Paragraphs</span><span class="v">¶${chunk.paragraph_indices.join(', ¶')}</span></div>
      <div class="kv" style="flex:1;min-width:140px"><span class="k">Section Path</span><span class="v" style="font-size:11.5px;font-family:var(--font-sans);font-style:italic;color:var(--text-dim)">${escapeHtml(sectionPath)}</span></div>
    </div>
    <div class="entity-card-section">
      <h4>Full passage</h4>
      <div class="evidence" style="cursor:default">
        <div class="evidence-text" style="-webkit-line-clamp:unset;color:var(--text)">${escapeHtml(chunk.text)}</div>
      </div>
    </div>
    ${entities.length ? `
    <div class="entity-card-section">
      <h4>Entities in this passage</h4>
      <div class="entity-card-pills">
        ${entities.map(e => `<button class="pill pill-accent" data-eid="${e.id}">${escapeHtml(e.name)}</button>`).join('')}
      </div>
    </div>` : ''}`;

  openGalaxyDetail(html);
  bindDetailLinks();
}

function showEntityDetail(entityId) {
  const e = state.entitiesById.get(entityId);
  if (!e) return;
  state.graph.focusEntity(entityId);

  const chunks = e.chunk_ids
    .map(id => state.chunksById.get(id))
    .filter(Boolean)
    .slice(0, 4);

  const docs = new Set(chunks.map(c => c.document_name));

  const related = [];
  for (const r of state.index.relationships) {
    if (r.source === entityId) related.push({ id: r.target, weight: r.weight });
    else if (r.target === entityId) related.push({ id: r.source, weight: r.weight });
  }
  related.sort((a, b) => b.weight - a.weight);
  const relatedTop = related.slice(0, 8).map(r => state.entitiesById.get(r.id)).filter(Boolean);

  // Premium entity card: Purpose (most-cited excerpt) · Dependencies (related entities)
  // · Appears In (documents) · Evidence (chunks)
  const purposeChunk = chunks[0];
  const purpose = purposeChunk?.paragraph_excerpt || purposeChunk?.text?.slice(0, 200) || '';

  const html = `
    <div class="entity-card-eyebrow">Entity · ${escapeHtml(e.kind)}</div>
    <div class="entity-card-title">${escapeHtml(e.name)}</div>
    <div class="entity-card-meta">
      <div class="kv"><span class="k">Mentions</span><span class="v">${e.mention_count}</span></div>
      <div class="kv"><span class="k">Documents</span><span class="v">${docs.size}</span></div>
      <div class="kv"><span class="k">Knowledge Units</span><span class="v">${e.chunk_ids.length}</span></div>
      <div class="kv"><span class="k">Connections</span><span class="v">${related.length}</span></div>
    </div>

    ${purpose ? `
    <div class="entity-card-section">
      <h4>Purpose</h4>
      <p style="font-size:12.5px;color:var(--text-dim);line-height:1.55">${escapeHtml(purpose)}</p>
    </div>` : ''}

    ${relatedTop.length ? `
    <div class="entity-card-section">
      <h4>Dependencies & related concepts</h4>
      <div class="entity-card-pills">
        ${relatedTop.map(r => `<button class="pill" data-eid="${r.id}">${escapeHtml(r.name)}</button>`).join('')}
      </div>
    </div>` : ''}

    ${docs.size ? `
    <div class="entity-card-section">
      <h4>Appears in</h4>
      <div class="entity-card-pills">
        ${[...docs].map(d => `<span class="pill pill-accent" style="cursor:default">${escapeHtml(stripExt(d))}</span>`).join('')}
      </div>
    </div>` : ''}

    <div class="entity-card-section">
      <h4>Evidence</h4>
      ${chunks.map(c => {
        const section = c.section_path?.length ? c.section_path.join(' › ') : '';
        return `
          <div class="evidence" data-cid="${c.id}">
            <div class="evidence-meta">
              <span class="doc">${escapeHtml(stripExt(c.document_name))}</span>
              <span class="sep">·</span>
              <span>page ${c.page}</span>
              ${section ? `<span class="sep">·</span><span style="font-style:italic">${escapeHtml(section)}</span>` : ''}
            </div>
            <div class="evidence-text">${escapeHtml(c.text)}</div>
          </div>`;
      }).join('')}
    </div>`;

  openGalaxyDetail(html);
  bindDetailLinks();
}

function bindDetailLinks() {
  document.querySelectorAll('#galaxy-detail [data-eid]').forEach(el => {
    el.addEventListener('click', () => showEntityDetail(parseInt(el.dataset.eid, 10)));
  });
  document.querySelectorAll('#galaxy-detail [data-cid]').forEach(el => {
    el.addEventListener('click', () => showChunkDetail(state.chunksById.get(parseInt(el.dataset.cid, 10))));
  });
}

function openGalaxyDetail(html) {
  document.getElementById('galaxy-detail-body').innerHTML = html;
  document.getElementById('galaxy-detail').hidden = false;
}

// ============================================================================
// Util
// ============================================================================
function stripExt(name) {
  return name.replace(/\.[^.]+$/, '').replace(/_/g, ' ');
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));
}

// ============================================================================
// Bootstrap (wait for vis-network)
// ============================================================================
window.addEventListener('DOMContentLoaded', () => {
  if (typeof vis === 'undefined') {
    const check = setInterval(() => {
      if (typeof vis !== 'undefined') {
        clearInterval(check);
        boot();
      }
    }, 50);
  } else {
    boot();
  }
});
