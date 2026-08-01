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

console.log('\n' + '='.repeat(62));
console.log(`  ${passed} passed, ${failed} failed`);
console.log('='.repeat(62));
if (failed) {
  console.log('\nfailures:');
  for (const [n, m] of failures) console.log(`  ${n}\n    ${m}`);
  process.exit(1);
}
