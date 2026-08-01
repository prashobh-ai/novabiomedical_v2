#!/usr/bin/env bash
# Post-copy verification. Run from repo root.
fail=0
chk(){ if eval "$2" >/dev/null 2>&1; then echo "  OK    $1"; else echo "  FAIL  $1"; fail=1; fi; }

echo "1. FILES PRESENT"
for f in site/js/semantic.js site/js/hybrid.js site/js/main.js \
         pipeline/build_index.py tests/integration.test.mjs \
         .github/workflows/ci.yml .github/workflows/deploy.yml; do
  chk "$f" "test -f $f"
done

echo; echo "2. FIXES ACTUALLY LANDED"
chk "semantic.js emits chunkIdx"        "grep -q 'chunkIdx: i, score: s' site/js/semantic.js"
chk "semantic.js has no stale 'id: i'"  "! grep -q 'scored\[i\] = { id: i' site/js/semantic.js"
chk "hybrid.js fuses on chunkIdx"       "grep -q 'const key = hit.chunkIdx' site/js/hybrid.js"
chk "hybrid.js takes chunks (not chunksById)" "grep -q 'bm25, semantic, chunks, filters' site/js/hybrid.js"
chk "main.js passes chunks"             "grep -q 'chunks: state.chunks' site/js/main.js"
chk "main.js maps chunkIdx"             "grep -q 'chunkIdx: r.chunkIdx' site/js/main.js"
chk "build_index emits mention_count"   "grep -q '\"mention_count\"' pipeline/build_index.py"
chk "build_index emits chunk_ids"       "grep -q '\"chunk_ids\"' pipeline/build_index.py"
chk "build_index emits document_ids"    "grep -q '\"document_ids\"' pipeline/build_index.py"

echo; echo "3. REBUILD + TEST"
python -m pipeline.build_index >/dev/null 2>&1 && echo "  OK    build" || { echo "  FAIL  build"; fail=1; }
chk "index has mention_count" "python -c \"import json;e=json.load(open('site/data/index.json'))['entities'][0];assert 'mention_count' in e and 'chunk_ids' in e and 'document_ids' in e\""
node tests/integration.test.mjs >/dev/null 2>&1 && echo "  OK    integration test (17 checks)" || { echo "  FAIL  integration test"; fail=1; }

echo
if [ $fail -eq 0 ]; then echo "ALL GREEN — safe to push."; else echo "SOMETHING FAILED — see above."; exit 1; fi
