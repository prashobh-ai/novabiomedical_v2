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
  const built = buildAnswer(q, cohesion.ranked, idx.chunks, cohesion);
  return { ranked: cohesion.ranked, cohesion, retrieval: { mode: h.mode, ...h.diagnostics }, ...built };
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

console.log('\n' + '='.repeat(62));
console.log(`  ${passed} passed, ${failed} failed`);
console.log('='.repeat(62));
if (failed) {
  console.log('\nfailures:');
  for (const [n, m] of failures) console.log(`  ${n}\n    ${m}`);
  process.exit(1);
}
