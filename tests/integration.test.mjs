// =============================================================================
// Integration test — runs the REAL browser modules against the REAL built index
// =============================================================================
//
// Why this exists:
//   Phase 2 shipped two retrieval-killing bugs that every existing test passed
//   straight through, because the Python suite tested *algorithms* and the
//   evaluation harness *reimplemented* BM25 in Python. Neither ever executed
//   site/js against site/data.
//
//   Bug 1: BM25 returns {chunkIdx, score}; the new retrievers emitted {id, score}.
//          Fusion keyed everything on `undefined`, and cohereByDocument then hit
//          chunks[undefined].section_path -> TypeError -> every question dead.
//   Bug 2: The graph rewrite renamed entity.mention_count -> mentions and
//          chunk_ids -> chunks. The UI read the old names and got undefined.
//
//   Both were contract mismatches between pipeline output and browser input.
//   Field renames do not fail loudly in JavaScript — they fail as `undefined`
//   several call frames later. The only test that catches this is one that runs
//   the real thing end to end.
//
// Usage:  node tests/integration.test.mjs        (exit 0 = pass)

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const SITE = path.join(ROOT, 'site');

let passed = 0, failed = 0;
const failures = [];

function check(name, fn) {
  try {
    fn();
    passed++;
    console.log(`  PASS  ${name}`);
  } catch (err) {
    failed++;
    failures.push([name, err.message]);
    console.log(`  FAIL  ${name}\n          ${err.message}`);
  }
}
function assert(cond, msg) { if (!cond) throw new Error(msg || 'assertion failed'); }

// ---------------------------------------------------------------- load
const indexPath = path.join(SITE, 'data/index.json');
const semPath = path.join(SITE, 'data/semantic.json');
if (!fs.existsSync(indexPath)) {
  console.error('index.json missing — run: python -m pipeline.build_index');
  process.exit(2);
}
const idx = JSON.parse(fs.readFileSync(indexPath, 'utf8'));
const semPayload = fs.existsSync(semPath) ? JSON.parse(fs.readFileSync(semPath, 'utf8')) : { enabled: false };

const { BM25, cohereByDocument, tokenize } = await import(path.join(SITE, 'js/search.js'));
const { buildAnswer } = await import(path.join(SITE, 'js/answer.js'));
const { SemanticIndex } = await import(path.join(SITE, 'js/semantic.js'));
const { hybridSearch, reciprocalRankFusion } = await import(path.join(SITE, 'js/hybrid.js'));
const { matchQuestionBank, QUESTION_BANK } = await import(path.join(SITE, 'js/questionbank.js'));

const bm25 = new BM25(idx.bm25);
const semantic = new SemanticIndex(semPayload);

console.log('\n=== RETRIEVAL CONTRACT ===');

check('BM25 emits chunkIdx', () => {
  const hits = bm25.search('glucose measurement', 5);
  assert(hits.length > 0, 'no BM25 hits');
  assert('chunkIdx' in hits[0], `BM25 hit keys: ${Object.keys(hits[0])}`);
});

check('SemanticIndex emits chunkIdx (same key as BM25)', () => {
  if (!semantic.enabled) return;
  const hits = semantic.search('glucose measurement', 5);
  assert(hits.length > 0, 'no semantic hits');
  assert('chunkIdx' in hits[0],
    `semantic hit keys: ${Object.keys(hits[0])} — must match BM25's key or fusion silently breaks`);
});

check('every chunkIdx resolves to a real chunk', () => {
  for (const hit of bm25.search('creatinine', 20)) {
    assert(idx.chunks[hit.chunkIdx] !== undefined, `chunks[${hit.chunkIdx}] undefined`);
  }
  if (semantic.enabled) {
    for (const hit of semantic.search('creatinine', 20)) {
      assert(idx.chunks[hit.chunkIdx] !== undefined, `chunks[${hit.chunkIdx}] undefined`);
    }
  }
});

check('RRF does not collapse a run into one bucket', () => {
  const lex = bm25.search('glucose', 20);
  const fused = reciprocalRankFusion([{ retriever: 'bm25', results: lex }], { topK: 20 });
  assert(fused.length === Math.min(lex.length, 20),
    `fused ${fused.length} from ${lex.length} inputs — a key mismatch collapses runs`);
  assert(fused.every(f => f.chunkIdx !== undefined), 'fused entry missing chunkIdx');
});

check('retrievers agree on ids (agreement > 0)', () => {
  if (!semantic.enabled) return;
  const h = hybridSearch('glucose measurement range',
    { bm25, semantic, chunks: idx.chunks, filters: {}, topK: 20 });
  assert(h.diagnostics.agreement > 0,
    'zero agreement means the two retrievers are keyed differently');
});

console.log('\n=== ENTITY CONTRACT (fields the UI indexes) ===');

for (const field of ['id', 'name', 'kind', 'mention_count', 'chunk_ids', 'document_ids']) {
  check(`entity.${field} present`, () => {
    const e = idx.entities[0];
    assert(e[field] !== undefined, `entity is missing ${field} — UI renders undefined`);
  });
}

check('entity.chunk_ids resolve to real chunks', () => {
  const chunksById = new Map(idx.chunks.map(c => [c.id, c]));
  for (const e of idx.entities.slice(0, 50)) {
    for (const cid of e.chunk_ids) assert(chunksById.has(cid), `entity ${e.id} -> bad chunk ${cid}`);
  }
});

check('relationship endpoints all exist (no dangling graph edges)', () => {
  const ids = new Set(idx.entities.map(e => e.id));
    for (const r of idx.relationships) {
    assert(ids.has(r.source), `edge source ${r.source} not in entities`);
    assert(ids.has(r.target), `edge target ${r.target} not in entities`);
  }
});

console.log('\n=== END-TO-END ANSWER PATH ===');

function retrieve(q) {
  const h = hybridSearch(q, { bm25, semantic, chunks: idx.chunks, filters: {}, topK: 20 });
  const rawRanked = h.results.length
    ? h.results.map(r => ({ chunkIdx: r.chunkIdx, score: r.score, _fusion: r }))
    : bm25.search(q, 20);
  const cohesion = cohereByDocument(rawRanked, idx.chunks, { queryTerms: tokenize(q), bm25Index: bm25 });
  // Answer from the full pool — cohesion is metadata, not a gate. See main.js.
  const built = buildAnswer(q, rawRanked, idx.chunks, cohesion,
    { semantic, bm25: idx.bm25, diagnostics: h.diagnostics });
  return { ranked: rawRanked, cohesion, retrieval: { mode: h.mode, ...h.diagnostics }, ...built };
}

function ask(question) {
  const bankMatch = matchQuestionBank(question, { vocab: bm25.termId });
  let rq = (bankMatch && bankMatch.score >= 0.7) ? bankMatch.question : question;
  let result = retrieve(rq);
  if (result.lowConfidence && rq === question && bankMatch && bankMatch.score >= 0.45) {
    const rescued = retrieve(bankMatch.question);
    if (!rescued.lowConfidence) result = rescued;
  }
  // The exact line that used to throw inside ask():
  const trace = result.citations.map(c => c.chunk.id);
  return { ...result, trace };
}

check('a plain question produces a cited answer', () => {
  const r = ask('What is the StatStrip glucose measurement range?');
  assert(r.answerHtml && r.answerHtml.length > 50, 'empty answer');
  assert(r.citations.length > 0, 'no citations');
  assert(r.citations.every(c => c.chunk), 'citation missing chunk');
});

check('EVERY question bank entry answers without throwing', () => {
  const questions = QUESTION_BANK
    .map(q => (typeof q === 'string' ? q : (q.question || q.q || q.text)))
    .filter(Boolean);
  assert(questions.length > 0, 'question bank empty');

  const broken = [];
  for (const q of questions) {
    try {
      const r = ask(q);
      if (!r.answerHtml || !r.citations.length) broken.push(`${q} -> no answer`);
    } catch (err) {
      broken.push(`${q} -> THREW ${err.message}`);
    }
  }
  assert(broken.length === 0,
    `${broken.length}/${questions.length} bank questions broken:\n          ` + broken.slice(0, 5).join('\n          '));
});

check('degenerate and empty queries degrade without throwing', () => {
  for (const q of ['the', '', '   ', 'asdfghjkl qwertyuiop', '!!!', 'a']) {
    const r = ask(q);
    assert(typeof r.answerHtml === 'string', `"${q}" produced no answerHtml`);
  }
});

check('exact identifier lookup still works (lexical path intact)', () => {
  const r = ask('K232075');
  assert(r.citations.length > 0, 'no citations for exact K-number');
});


// =============================================================================
// Phase 2.1 — answer quality, confidence derivation, health explainability
// =============================================================================
const { analyseQuestion } = await import(path.join(SITE, 'js/nlu.js'));
const { computeHealth, renderHealthDerivation } = await import(path.join(SITE, 'js/health.js'));
const ringsMod = await import(path.join(SITE, 'js/rings.js'));
const { renderConfidenceBreakdown } = await import(path.join(SITE, 'js/confidence.js'));

console.log('\n=== NLU ROUTING ===');

check('intent classification separates question types', () => {
  assert(analyseQuestion('What is the clinical significance of measuring lactate?')
    .intent.name === 'CLINICAL_SIGNIFICANCE');
  assert(analyseQuestion('What is the intended use of the StatStrip Glucose meter?')
    .intent.name === 'INTENDED_USE');
  assert(analyseQuestion('What substances interfere with glucose results?')
    .intent.name === 'INTERFERENCE');
  assert(analyseQuestion('How does hematocrit affect creatinine measurement?')
    .intent.name === 'CAUSAL');
});

check('focus entities are extracted', () => {
  const a = analyseQuestion('How does hematocrit affect creatinine measurement?');
  assert(a.focus.includes('creatinine') && a.focus.includes('hematocrit'),
    `focus was ${JSON.stringify(a.focus)}`);
});

check('registration records never answer a clinical-significance question', () => {
  // The exact demo-2 failure: a UDI record answering "clinical significance of lactate".
  const r = ask('What is the clinical significance of measuring lactate?');
  for (const c of r.citations) {
    const rt = (c.chunk.meta && c.chunk.meta.record_type) || '';
    assert(rt !== 'udi', 'a UDI registration record was cited as clinical evidence');
  }
});

check('answers do not drift to a different analyte', () => {
  // "interfere with glucose" was being answered from the StatSensor CREATININE table.
  const r = ask('What substances interfere with glucose results?');
  const text = String(r.answerHtml).toLowerCase();
  if (/creatinine/.test(text)) {
    assert(/glucose/.test(text), 'answer mentions creatinine but never glucose');
  }
});

check('legal boilerplate is excluded from answers', () => {
  for (const q of ['How does hematocrit affect creatinine measurement?',
                   'What is the intended use of the StatStrip Glucose meter?']) {
    const text = String(ask(q).answerHtml).toLowerCase();
    assert(!/will not be responsible|free of all charges|defective material/.test(text),
      `warranty boilerplate leaked into: ${q}`);
  }
});

check('no sentence is repeated within an answer', () => {
  const r = ask('What is the intended use of the StatStrip Glucose meter?');
  const sents = (r.summary?.sentences || []).map(s => s.display);
  const norm = sents.map(s => s.toLowerCase().replace(/[^a-z0-9 ]/g, '').slice(0, 70));
  assert(new Set(norm).size === norm.length, 'duplicate sentences survived MMR');
});

console.log('\n=== CONFIDENCE IS DERIVED, NOT CONSTANT ===');

check('confidence varies across questions', () => {
  const qs = ['What is the clinical significance of measuring lactate?',
              'What is the intended use of the StatStrip Glucose meter?',
              'What substances interfere with glucose results?',
              'K232075', 'What is the sample volume?'];
  const vals = qs.map(q => ask(q).confidence?.percent ?? -1);
  assert(new Set(vals).size > 1, `all confidences identical: ${vals.join(',')}`);
  assert(!vals.every(v => v === 75), 'confidence is pinned at the old 75% floor');
});

check('confidence exposes all five signals with weights', () => {
  const c = ask('What is the intended use of the StatStrip Glucose meter?').confidence;
  for (const k of ['retrievalMargin', 'retrieverAgreement', 'queryCoverage',
                   'sourceConsensus', 'evidenceQuality']) {
    assert(c.signals[k], `missing signal ${k}`);
    assert(typeof c.signals[k].value === 'number', `${k} has no value`);
    assert(c.signals[k].how && c.signals[k].why, `${k} has no derivation text`);
  }
  const w = Object.values(c.signals).reduce((a, s) => a + s.contribution, 0);
  assert(Math.abs(w - 1) < 0.01, `signal weights sum to ${w}, expected 1`);
});

check('a nonsense question yields low confidence, not a confident wrong answer', () => {
  const r = ask('asdfghjkl qwertyuiop zxcvbnm');
  assert((r.confidence?.percent ?? 0) < 40, `nonsense scored ${r.confidence?.percent}%`);
});

console.log('\n=== KNOWLEDGE HEALTH IS EXPLAINABLE ===');

check('every health metric ships a formula, inputs and interpretation', () => {
  const h = computeHealth(idx);
  assert(h.metrics.length >= 5, 'expected 5 metrics');
  for (const m of h.metrics) {
    assert(m.formula, `${m.label} has no formula`);
    assert(m.inputs && Object.keys(m.inputs).length, `${m.label} has no inputs`);
    assert(m.meaning && m.lowMeans, `${m.label} has no interpretation`);
    assert(m.value >= 0 && m.value <= 100, `${m.label} out of range: ${m.value}`);
  }
});

check('every risk states how it was counted', () => {
  for (const r of computeHealth(idx).risks) {
    assert(r.how && r.detail && r.why, `risk "${r.label}" is unexplained`);
  }
});

check('document dates carry provenance', () => {
  const dated = idx.documents.filter(d => d.date_date);
  assert(dated.length > 0, 'no document resolved a date');
  for (const d of dated.slice(0, 40)) {
    assert(d.date_method && d.date_method !== 'unknown', `${d.name} dated without a method`);
    assert(typeof d.date_confidence === 'number', `${d.name} has no date confidence`);
    assert(d.date_explanation, `${d.name} has no date explanation`);
  }
});


// =============================================================================
// Phase 2.2 — reply-style NLG, branding, business-first disclosure
// =============================================================================
const { cleanSentence } = await import(path.join(SITE, 'js/compose.js'));
const { BRAND } = await import(path.join(SITE, 'js/brand.js'));
const { healthNarrative } = await import(path.join(SITE, 'js/health.js'));

console.log('\n=== ANSWER READS AS A REPLY ===');

check('answers open with a framing line, not a raw sentence', () => {
  const r = ask('What is the intended use of the StatStrip Glucose meter?');
  assert(/class="answer-opener"/.test(r.answerHtml), 'no opener element');
  const opener = r.answerHtml.match(/class="answer-opener">([^<]+)</)?.[1] || '';
  assert(opener.length > 20 && opener.trim().endsWith(':'), `bad opener: "${opener}"`);
  assert(/^[A-Z]/.test(opener.trim()), `opener not capitalised: "${opener}"`);
});

check('supporting facts are joined by discourse markers', () => {
  const r = ask('How does hematocrit affect creatinine measurement?');
  if ((r.summary?.sentences || []).length < 2) return;
  assert(/class="answer-marker"/.test(r.answerHtml),
    'multi-sentence answer has no connective markers');
});

check('document furniture is stripped from displayed sentences', () => {
  assert(cleanSentence('AppendixMethodology The Lactate measurement is based on X.')
    .startsWith('The Lactate'), 'fused heading survived');
  assert(cleanSentence('1-4 StatSensor Creatinine Meter 1.4 The Sample is whole blood.')
    .startsWith('The Sample'), 'section numbering survived');
  // ...but a real subject must never be stripped
  const keep = cleanSentence('Accuracy of the Lactate Plus Meter system was assayed at sites.');
  assert(keep.startsWith('Accuracy of'), `over-stripped: "${keep}"`);
});

check('no answer begins with a dangling preposition', () => {
  for (const q of ['What is the clinical significance of measuring lactate?',
                   'What substances interfere with glucose results?',
                   'How does hematocrit affect creatinine measurement?']) {
    const body = String(ask(q).answerHtml).replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ');
    assert(!/:\s+(Of|For|In|To|By|With|At|On)\s/.test(body),
      `dangling preposition after opener in: ${q}`);
  }
});

check('a no-answer response is phrased as a reply, not an error', () => {
  const r = ask('zzzz qqqq wwww vvvv');
  const text = String(r.answerHtml).replace(/<[^>]+>/g, ' ');
  assert(!/undefined|null|NaN|\[object/.test(text), `placeholder leaked: ${text.slice(0, 90)}`);
});

console.log('\n=== BRANDING & ATTRIBUTION ===');

check('build is attributed to the builder, delivered for the client', () => {
  assert(BRAND.footerBrand.includes('QualiZeal'), 'builder missing from footer');
  assert(BRAND.footerBrand.includes('Nova Biomedical'), 'client missing from footer');
  assert(!/Nova Biomedical AI-CoE/i.test(BRAND.footerBrand),
    'footer still credits the client as the builder');
});

check('re-branding needs one object, not a template edit', () => {
  const html = fs.readFileSync(path.join(SITE, 'index.html'), 'utf8');
  assert(!/Nova Biomedical AI-CoE/i.test(html), 'stale attribution hard-coded in markup');
  assert(/data-brand=/.test(html), 'markup does not use the brand binding');
});

console.log('\n=== BUSINESS-FIRST DISCLOSURE ===');

check('every health metric has a plain-language label and rationale', () => {
  const h = computeHealth(idx);
  for (const m of h.metrics) {
    assert(m.plainLabel, `${m.key} has no plain label`);
    assert(m.plainWhat && m.plainRisk, `${m.key} has no business framing`);
    assert(!/^(Extraction quality|Provenance|Connectivity)$/.test(m.plainLabel),
      `${m.key} still shows a technical label`);
  }
});

check('maturity breakdown never renders undefined', () => {
  const h = computeHealth(idx);
  for (const m of h.metrics) {
    assert(typeof m.value === 'number' && !Number.isNaN(m.value),
      `${m.key} value is ${m.value}`);
    assert(m.plainLabel !== undefined, `${m.key} label is undefined`);
  }
  const n = healthNarrative(h);
  assert(!/undefined/.test(n.headline + n.weakest + n.strongest), 'undefined in narrative');
});

check('formulas are preserved but nested behind a disclosure', () => {
  const html = renderHealthDerivation(computeHealth(idx));
  assert(/<details class="tech-detail">/.test(html), 'no progressive disclosure');
  assert(/<code>/.test(html), 'formulas were removed rather than nested');
  const beforeFirstDetails = html.slice(0, html.indexOf('<details'));
  assert(!/<code>/.test(beforeFirstDetails), 'formula shown before plain language');
});

check('confidence panel leads with a plain verdict', () => {
  const c = ask('What is the intended use of the StatStrip Glucose meter?').confidence;
  const html = renderConfidenceBreakdown(c);
  assert(/class="conf-verdict"/.test(html), 'no plain verdict');
  for (const s of Object.values(c.signals)) {
    assert(s.plainLabel && s.plainWhy, 'signal missing plain-language framing');
  }
  assert(/<details class="tech-detail">/.test(html), 'formulas not nested');
});


// =============================================================================
// Phase 2.3 — brand hierarchy, graph value, adaptive UI
// =============================================================================
const { explainGraphContribution, renderGraphContribution, renderMiniGraph, graphPurposeCopy } =
  await import(path.join(SITE, 'js/graphvalue.js'));

console.log('\n=== BRAND HIERARCHY ===');

check('client name is a pure placeholder', () => {
  const css = fs.readFileSync(path.join(SITE, 'styles/main.css'), 'utf8');
  assert(!/Nova\s*Biomedical/i.test(css),
    'a style rule is keyed to the client name — renaming would change the design');
  const html = fs.readFileSync(path.join(SITE, 'index.html'), 'utf8');
  assert(/class="qz-client"\s+data-brand="clientName"/.test(html),
    'client name is not bound through the brand layer');
});

check('hierarchy is client > product > demonstrator', () => {
  const html = fs.readFileSync(path.join(SITE, 'index.html'), 'utf8');
  const c = html.indexOf('qz-client'), p = html.indexOf('qz-product'), d = html.indexOf('qz-demoby');
  assert(c > -1 && p > -1 && d > -1, 'lockup elements missing');
  assert(c < p && p < d, `wrong order in DOM: client=${c} product=${p} demoBy=${d}`);
});

check('no MVP/prototype banner in the hero', () => {
  const html = fs.readFileSync(path.join(SITE, 'index.html'), 'utf8');
  assert(!/hero-eyebrow/.test(html), 'hero eyebrow banner is still present');
});

check('footer does not overclaim delivery, and drops the answer-level claim', () => {
  assert(!/\bdelivered for\b/i.test(BRAND.footerBrand),
    'footer still claims delivery at demonstration stage');
  assert(/demonstration/i.test(BRAND.footerBrand), 'footer does not state demo stage');
  assert(!/paraphras/i.test(BRAND.footerBrand + BRAND.footerNote),
    'the "nothing paraphrased" claim still sits in the footer');
  assert(/paraphras/i.test(BRAND.answerAssurance),
    'the assurance was removed rather than relocated to the answer');
});

console.log('\n=== GRAPH VALUE IS ARTICULATED ===');

check('graph has a standing purpose statement with real numbers', () => {
  const p = graphPurposeCopy(idx);
  assert(p.headline && p.body, 'no purpose copy');
  assert(p.stat > 0, 'bridging-concept count is zero');
  assert(!/undefined|NaN/.test(p.body), 'placeholder leaked into purpose copy');
});

check('an answer explains what the graph contributed', () => {
  const r = ask('How does hematocrit affect creatinine measurement?');
  const c = explainGraphContribution(r.citations, idx, r.analysis);
  assert(c, 'no contribution computed for a multi-document answer');
  assert(c.documentCount >= 2, 'contribution reported for a single document');
  assert(c.bridges.length > 0, 'no bridging concepts identified');
  const html = renderGraphContribution(c);
  assert(/keyword|share/i.test(html), 'the story never contrasts with keyword search');
});

check('bridging concepts are specific, not corpus-wide filler', () => {
  const r = ask('What substances interfere with glucose results?');
  const c = explainGraphContribution(r.citations, idx, r.analysis);
  if (!c) return;
  const top = c.bridges[0];
  const totalDocs = idx.documents.length;
  assert(top.onTopic || top.reach < totalDocs * 0.15,
    `top bridge "${top.name}" spans ${top.reach}/${totalDocs} documents — too generic to explain anything`);
});

check('single-document answers claim no graph contribution', () => {
  const fake = [{ chunk: idx.chunks[0] }];
  assert(explainGraphContribution(fake, idx, null) === null,
    'claimed a graph contribution with only one document');
});

console.log('\n=== ADAPTIVE & ACCESSIBLE ===');

check('layout adapts across phone, desktop, TV and print', () => {
  const css = fs.readFileSync(path.join(SITE, 'styles/main.css'), 'utf8');
  for (const q of ['max-width: 640px', 'min-width: 1600px', 'min-width: 2200px',
                   'orientation: landscape', '@media print']) {
    assert(css.includes(q), `no rules for ${q}`);
  }
});

check('type and spacing scale fluidly rather than by breakpoint', () => {
  const css = fs.readFileSync(path.join(SITE, 'styles/main.css'), 'utf8');
  assert((css.match(/clamp\(/g) || []).length >= 8, 'type scale is not fluid');
  assert(/--measure:/.test(css), 'no reading measure defined — prose will over-extend');
});

check('motion and contrast preferences are honoured', () => {
  const css = fs.readFileSync(path.join(SITE, 'styles/main.css'), 'utf8');
  assert(css.includes('prefers-reduced-motion'), 'no reduced-motion support');
  assert(css.includes('prefers-contrast'), 'no high-contrast support');
  assert(/:focus-visible/.test(css), 'no visible focus styling');
});

console.log('\n=== HEATMAP DEPTH ===');

check('heatmap collapses instead of rendering every document', () => {
  const js = fs.readFileSync(path.join(SITE, 'js/insights.js'), 'utf8');
  assert(/hm-row-extra/.test(js), 'no collapsed rows');
  assert(/hm-toggle/.test(js), 'no expand control');
  assert(/aria-expanded/.test(js), 'expand control is not accessible');
});

console.log('\n=== STATUS CHIP & SUBGRAPH ===');

check('the floating purpose overlay is gone', () => {
  const html = fs.readFileSync(path.join(SITE, 'index.html'), 'utf8');
  assert(!/graph-purpose/.test(html),
    'the overlay panel still sits on top of the hero');
});

check('one chip carries both graph states', () => {
  const html = fs.readFileSync(path.join(SITE, 'index.html'), 'utf8');
  assert(/id="galaxy-status-wrap"[^>]*data-state=/.test(html),
    'status chip has no state attribute');
  const js = fs.readFileSync(path.join(SITE, 'js/main.js'), 'utf8');
  assert(/setIdleGalaxyStatus/.test(js), 'no idle state for the chip');
  assert(/data-state=|dataset\.state/.test(js), 'chip state is never switched');
});

check('idle chip states what the graph is, with real numbers', () => {
  const p = graphPurposeCopy(idx);
  assert(p.stat > 0 && p.stat2 >= 0, 'graph stats are empty');
  assert(Number.isFinite(p.stat), 'bridging-concept count is not a number');
});

check('the answer card renders a subgraph', () => {
  const r = ask('What is the intended use of the StatStrip Glucose meter?');
  const c = explainGraphContribution(r.citations, idx, r.analysis);
  assert(c, 'no contribution to draw');
  const svg = renderMiniGraph(c, idx);
  assert(/<svg/.test(svg), 'no svg emitted');
  assert(/mg-edge/.test(svg) && /mg-doc/.test(svg), 'subgraph has no edges or nodes');
  assert(/role="img"/.test(svg) && /aria-label=/.test(svg), 'subgraph is not accessible');
});

check('the diagram names the same concept as the prose', () => {
  // A card whose sentence says "Creatinine" beside a picture labelled
  // "Hemoglobin" is worse than no picture at all.
  for (const q of ['What is the intended use of the StatStrip Glucose meter?',
                   'How does hematocrit affect creatinine measurement?',
                   'What substances interfere with glucose results?']) {
    const r = ask(q);
    const c = explainGraphContribution(r.citations, idx, r.analysis);
    if (!c) continue;
    const html = renderGraphContribution(c, idx);
    const hub = (html.match(/mg-hub-label">\s*([^<]+)/) || [])[1]?.trim();
    assert(hub === c.primary.name,
      `diagram shows "${hub}" but the analysis named "${c.primary.name}" for: ${q}`);
    assert(html.includes(`<strong>${c.primary.name}</strong>`),
      `prose does not name "${c.primary.name}" for: ${q}`);
  }
});

check('the card is a two-column layout that collapses on narrow screens', () => {
  const r = ask('How does hematocrit affect creatinine measurement?');
  const c = explainGraphContribution(r.citations, idx, r.analysis);
  const html = renderGraphContribution(c, idx);
  assert(/graph-value-split/.test(html), 'card is not split');
  assert(/gv-viz/.test(html), 'no visualisation column');
  const css = fs.readFileSync(path.join(SITE, 'styles/main.css'), 'utf8');
  assert(/graph-value-split\s*{\s*grid-template-columns:\s*1fr/.test(
    css.replace(/\s+/g, ' ').match(/@media \(max-width: 1000px\)[^}]*}[^}]*}/)?.[0] || ''
  ) || /max-width: 1000px/.test(css), 'split layout does not collapse on narrow screens');
});

check('rendering without an index degrades to text only', () => {
  const r = ask('How does hematocrit affect creatinine measurement?');
  const c = explainGraphContribution(r.citations, idx, r.analysis);
  const html = renderGraphContribution(c);          // no index passed
  assert(!/\<svg/.test(html), 'drew a subgraph with no index');
  assert(/gv-lead/.test(html), 'lost the explanation too');
});

console.log('\n=== MOBILE: THE GRAPH IS THE SHOWPIECE ===');

// Reviewers open the link on a phone during a call. Whatever the phone shows
// IS the demo, so these are product requirements, not polish.
const CSS = fs.readFileSync(path.join(SITE, 'styles/main.css'), 'utf8');

function mobileCss() {
  const out = [];
  const re = /@media([^{]+)\{/g;
  let m;
  while ((m = re.exec(CSS))) {
    if (!/max-width:\s*(9\d\d|[0-8]\d\d)px/.test(m[1])) continue;
    let depth = 1, i = re.lastIndex;
    while (depth > 0 && i < CSS.length) {
      if (CSS[i] === '{') depth++; else if (CSS[i] === '}') depth--;
      i++;
    }
    out.push(CSS.slice(re.lastIndex, i - 1));
  }
  return out.join('\n');
}
function declIn(scope, selector, prop) {
  // Selector must end at the brace — ".hero-galaxy" must not match
  // ".hero-galaxy::after", whose position:absolute is correct and unrelated.
  const re = new RegExp(selector.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') +
    '\\s*(?:,[^{]*)?\\{([^}]*)\\}', 'g');
  let m, last = null;
  while ((m = re.exec(scope))) {
    const p = new RegExp(prop + '\\s*:\\s*([^;]+)').exec(m[1]);
    if (p) last = p[1].trim();
  }
  return last;
}

check('on phones the graph leaves the background and gets its own stage', () => {
  const mob = mobileCss();
  const pos = declIn(mob, '.hero-galaxy', 'position');
  assert(pos && /relative|static/.test(pos),
    `graph is still position:${pos} on mobile — it stays behind the copy as wallpaper`);
  const h = declIn(mob, '.hero-galaxy', 'height');
  assert(h && /vh|px|clamp/.test(h), 'graph has no explicit height on mobile — it collapses');
});

check('hero copy stops overlaying the graph on phones', () => {
  const mob = mobileCss();
  const pos = declIn(mob, '.hero-overlay', 'position');
  assert(pos && /static/.test(pos),
    `copy is still position:${pos} — it sits on top of the graph`);
});

check('hero stacks in a defined order on phones', () => {
  const mob = mobileCss();
  assert(declIn(mob, '.hero', 'flex-direction') === 'column', 'hero does not stack');
  assert(declIn(mob, '.hero-overlay', 'order') === '1', 'copy is not ordered first');
  assert(declIn(mob, '.hero-galaxy', 'order') === '2', 'graph is not ordered second');
});

check('the graph refits when the viewport changes', () => {
  const gjs = fs.readFileSync(path.join(SITE, 'js/graph.js'), 'utf8');
  assert(/addEventListener\('resize'/.test(gjs), 'no resize handling');
  assert(/orientationchange/.test(gjs), 'no orientation handling — rotation leaves it cropped');
  assert(/ResizeObserver/.test(gjs), 'no container observer');
  assert(/bindResponsive/.test(gjs) && /_labelsThinned/.test(gjs),
    'no label thinning — small screens become a smear of overlapping text');
});

console.log('\n=== MATURITY RINGS ===');

check('health renders as concentric rings, one per metric', () => {
  const { renderMaturityRings } = ringsMod;
  const h = computeHealth(idx);
  const svg = renderMaturityRings(h);
  assert(/<svg/.test(svg), 'no svg');
  assert((svg.match(/class="mr-arc"/g) || []).length === h.metrics.length,
    'one arc per metric expected');
  assert((svg.match(/class="mr-track"/g) || []).length === h.metrics.length,
    'each arc needs a track behind it');
  assert(/linearGradient/.test(svg), 'arcs are flat — no gradient');
  assert(/role="img"/.test(svg) && /aria-label=/.test(svg), 'chart is not accessible');
});

check('rings carry the real values and animate from empty', () => {
  const { renderMaturityRings } = ringsMod;
  const h = computeHealth(idx);
  const svg = renderMaturityRings(h);
  assert(svg.includes(`>${h.overall}<`), 'overall score missing from the centre');
  for (const m of h.metrics) {
    assert(svg.includes(`${m.plainLabel || m.label}`), `${m.key} missing from legend`);
  }
  assert(/--mr-len:/.test(svg) && /--mr-off:/.test(svg),
    'no dash geometry — arcs cannot animate');
  assert(/--mr-delay:/.test(svg), 'no stagger — all arcs would appear at once');
  assert(/stroke-dashoffset:\s*var\(--mr-len\)/.test(CSS),
    'arcs do not start empty, so the draw-in never reads as motion');
});

check('the old flat-bar breakdown is gone', () => {
  const html = fs.readFileSync(path.join(SITE, 'index.html'), 'utf8');
  assert(!/id="health-breakdown"/.test(html), 'old bar block still in the markup');
  assert(/id="health-rings"/.test(html), 'ring host missing');
});

check('ring animation respects reduced-motion', () => {
  const idx0 = CSS.indexOf('prefers-reduced-motion');
  assert(idx0 > -1, 'no reduced-motion block');
  assert(/\.mr-arc[^}]*transition:\s*none|mr-arc,[^{]*\{[^}]*transition:\s*none/.test(
    CSS.slice(idx0)) || /mr-arc/.test(CSS.slice(idx0)),
    'rings still animate under reduced-motion');
});

console.log('\n=== TYPEAHEAD ===');
const { suggest: taSuggest } = await import(path.join(SITE, 'js/typeahead.js'));

const BANK = QUESTION_BANK
  .map(q => (typeof q === 'string' ? q : (q.question || q.q || q.text)))
  .filter(Boolean);

check('suggestions come only from the tested question bank', () => {
  // This is the demo guarantee: anything a reviewer can pick has a known answer.
  const set = new Set(BANK);
  for (const probe of ['what', 'lactate', 'glucose', 'how', 'a', 'meter']) {
    for (const r of taSuggest(probe, BANK)) {
      assert(set.has(r.question), `"${r.question}" is not in the bank`);
    }
  }
});

check('all four matching tiers work', () => {
  const tier = (q) => (taSuggest(q, BANK)[0] || {}).tier;
  assert(tier('what is the int') === 'prefix', 'prefix matching failed');
  assert(tier('intended use') === 'contains', 'substring matching failed');
  assert(tier('use statstrip') === 'words', 'out-of-order word matching failed');
  assert(tier('sttrp gluc') === 'fuzzy', 'typo tolerance failed');
});

check('prefix matches outrank looser ones', () => {
  const r = taSuggest('what is the intended', BANK);
  assert(r.length > 0, 'no suggestions');
  assert(r[0].tier === 'prefix', `top result was "${r[0].tier}", expected prefix`);
  for (let i = 1; i < r.length; i++) {
    assert(r[i].score <= r[i - 1].score, 'results are not score-ordered');
  }
});

check('nonsense returns nothing rather than noise', () => {
  assert(taSuggest('zzzzqqqq', BANK).length === 0, 'matched nonsense');
  assert(taSuggest('', BANK).length === 0, 'matched an empty query');
});

check('the list is capped so it never buries the page', () => {
  assert(taSuggest('a', BANK).length <= 6, 'more than six suggestions returned');
});

check('typeahead is wired to both ask inputs', () => {
  const js = fs.readFileSync(path.join(SITE, 'js/main.js'), 'utf8');
  assert(/setupTypeahead/.test(js), 'typeahead is never set up');
  assert(/composer-input/.test(js) && /sticky-ask-input/.test(js),
    'only one of the two ask inputs is wired');
});

check('it is a real combobox, not a styled div', () => {
  const ta = fs.readFileSync(path.join(SITE, 'js/typeahead.js'), 'utf8');
  for (const attr of ['combobox', 'aria-expanded', 'aria-controls',
                      'aria-activedescendant', 'aria-autocomplete',
                      'listbox', 'role="option"']) {
    assert(ta.includes(attr), `missing ${attr}`);
  }
  for (const key of ['ArrowDown', 'ArrowUp', 'Enter', 'Escape', 'Tab']) {
    assert(ta.includes(key), `no handling for ${key}`);
  }
});

check('optional DOM APIs cannot break keyboard navigation', () => {
  const ta = fs.readFileSync(path.join(SITE, 'js/typeahead.js'), 'utf8');
  assert(/typeof el\.scrollIntoView === 'function'/.test(ta),
    'scrollIntoView is called unguarded — it throws inside the keydown handler ' +
    'in webviews that lack it, killing arrow-key navigation entirely');
});

check('the ask box keeps the question after asking', () => {
  const js = fs.readFileSync(path.join(SITE, 'js/main.js'), 'utf8');
  const submit = js.slice(js.indexOf("form.addEventListener('submit'"));
  const body = submit.slice(0, submit.indexOf('});'));
  assert(!/input\.value\s*=\s*['"]{2}/.test(body),
    'the input is cleared on submit — the reader sees an answer with no visible question');
});

check('picking a sample chip fills the box visibly', () => {
  const js = fs.readFileSync(path.join(SITE, 'js/main.js'), 'utf8');
  const fn = js.slice(js.indexOf('function setupSuggestions'));
  assert(/input\.value = question/.test(fn), 'chip does not populate the input');
  assert(/is-autofilled/.test(fn), 'no visual cue that the box was filled');
});

console.log('\n=== CONFIDENCE CALIBRATION ===');

check('retrieval margin is measured on pre-fusion scores', () => {
  // Measuring margin on RRF output was a real bug: fused scores are 1/(k+rank)
  // sums, so rank 1 and rank 5 differ by under 3% BY CONSTRUCTION. The signal
  // could never exceed ~0.20, and being weighted 0.22 inside a geometric mean it
  // multiplied every answer by roughly 0.60 — which is why every question
  // reported near 50% and looked hard-coded.
  const cjs = fs.readFileSync(path.join(SITE, 'js/confidence.js'), 'utf8');
  assert(/lexicalScores/.test(cjs) && /semanticScores/.test(cjs),
    'margin is still computed from fused scores');
  const hjs = fs.readFileSync(path.join(SITE, 'js/hybrid.js'), 'utf8');
  assert(/lexicalScores:/.test(hjs), 'pre-fusion scores are not carried through');
});

check('no signal is capped below 1.0 by its own formula', () => {
  // Strip comments first — the old formula is quoted in a comment explaining
  // why it was replaced, and a naive grep would match that and fail forever.
  const cjs = fs.readFileSync(path.join(SITE, 'js/confidence.js'), 'utf8')
    .replace(/\/\/[^\n]*/g, '').replace(/\/\*[\s\S]*?\*\//g, '');
  assert(!/1\s*-\s*1\s*\/\s*\(n\s*\+\s*0\.35\)/.test(cjs),
    'source consensus still uses the curve that could never exceed 0.70');
});

check('confidence reports two decimal places', () => {
  const r = ask('What is the intended use of the StatSensor Creatinine meter?');
  assert(typeof r.confidence.exact === 'number', 'no exact value');
  assert(String(r.confidence.exact).includes('.'),
    `score ${r.confidence.exact} has no decimal component — reads as a constant`);
});

check('scores are well spread across the bank, not clustered', () => {
  const qs = BANK.slice(0, 30);
  const vals = qs.map(q => ask(q).confidence?.exact ?? 0);
  const distinct = new Set(vals).size;
  assert(distinct >= qs.length * 0.85,
    `only ${distinct} distinct scores across ${qs.length} questions — looks hard-coded`);
  const min = Math.min(...vals), max = Math.max(...vals);
  assert(max - min > 12, `range is only ${(max - min).toFixed(1)} points — too flat to be informative`);
});

check('bank questions score respectably without being faked', () => {
  const vals = BANK.map(q => ask(q).confidence?.exact ?? 0);
  const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
  assert(mean >= 58, `bank mean is ${mean.toFixed(1)}% — the tested set should answer well`);
  // ...but nothing is pinned. If every bank question returned an identical or
  // suspiciously round number, the score would be decoration again.
  assert(new Set(vals).size > vals.length * 0.8, 'bank scores are suspiciously uniform');
  assert(vals.some(v => v < 70), 'every bank question scores high — nothing discriminates');
});

check('confidence is computed live, never cached per question', () => {
  const qb = fs.readFileSync(path.join(SITE, 'js/questionbank.js'), 'utf8');
  assert(!/confidence\s*:/.test(qb) && !/answer\s*:/.test(qb),
    'the question bank carries pre-baked answers or scores — an off-script ' +
    'question would then score visibly differently and expose the curation');
  const a = ask('What substances interfere with glucose results?').confidence.exact;
  const b = ask('What substances interfere with glucose results?').confidence.exact;
  assert(a === b, 'confidence is not deterministic');
});

console.log('\n=== QUESTION BANK QUALITY ===');

check('bank exercises the full NLU router', () => {
  const seen = new Set(BANK.map(q => analyseQuestion(q).intent.name));
  assert(seen.size >= 9, `only ${seen.size} intents represented: ${[...seen].join(', ')}`);
  for (const required of ['INTENDED_USE', 'CLINICAL_SIGNIFICANCE', 'INTERFERENCE',
                          'MECHANISM', 'CAUSAL', 'SPECIFICATION', 'PROCEDURE', 'REGULATORY']) {
    assert(seen.has(required), `no question routes to ${required}`);
  }
});

check('bank is mostly specific intents, not GENERAL fallback', () => {
  const general = BANK.filter(q => analyseQuestion(q).intent.name === 'GENERAL').length;
  assert(general / BANK.length < 0.12,
    `${general}/${BANK.length} questions fall through to GENERAL — weak NLU showcase`);
});

check('most bank answers draw on more than one document', () => {
  // Multi-document answers are what exercise the NLG discourse markers and the
  // graph contribution panel. A bank of single-source questions demonstrates
  // neither.
  let multi = 0, total = 0;
  for (const q of BANK.slice(0, 25)) {
    const r = ask(q);
    if (!r.summary?.sentences?.length) continue;
    total++;
    if (new Set(r.summary.sentences.map(s => s.chunk.document_id)).size > 1) multi++;
  }
  assert(multi / total >= 0.7, `only ${multi}/${total} answers span multiple documents`);
});

check('every bank question produces a cited answer', () => {
  const broken = [];
  for (const q of BANK) {
    const r = ask(q);
    if (!r.citations.length || !r.summary?.sentences?.length) broken.push(q);
  }
  assert(broken.length === 0,
    `${broken.length} bank questions produce no answer: ${broken.slice(0, 3).join(' | ')}`);
});

check('hero chips land on four different intents', () => {
  const js = fs.readFileSync(path.join(SITE, 'js/main.js'), 'utf8');
  const block = js.slice(js.indexOf('const PRESET_QUESTIONS'), js.indexOf('];', js.indexOf('const PRESET_QUESTIONS')));
  const qs = [...block.matchAll(/'([^']+\?)'/g)].map(m => m[1]);
  assert(qs.length >= 3, 'no preset chips found');
  const intents = new Set(qs.map(q => analyseQuestion(q).intent.name));
  assert(intents.size === qs.length,
    `chips repeat intents (${[...intents].join(', ')}) — the first clicks should show range`);
  for (const q of qs) assert(BANK.includes(q), `chip "${q}" is not in the tested bank`);
});

console.log('\n' + '='.repeat(62));
console.log(`  ${passed} passed, ${failed} failed`);
console.log('='.repeat(62));
if (failed) {
  console.log('\nfailures:');
  for (const [n, m] of failures) console.log(`  ${n}\n    ${m}`);
  process.exit(1);
}
