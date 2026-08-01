"""Retrieval evaluation harness.

    python -m eval.evaluate                    # human-readable report
    python -m eval.evaluate --json out.json    # machine-readable
    python -m eval.evaluate --gate             # non-zero exit on regression (CI)

Measures retrieval, not generation. If the right evidence never enters the
context window, the answer is unfixable downstream — so this is the metric that
actually predicts whether the fabric works.

Reported per retrieval mode (lexical / semantic / hybrid) so the value of the
hybrid layer is demonstrated rather than asserted.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.semantic import embed_query, semantic_scores  # noqa: E402

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

# Mirrors site/js/hybrid.js::isDegenerateQuery — evaluation must measure the
# behaviour we actually ship, not an idealised variant of it.
FUNCTION_WORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "of", "to", "in", "on",
    "at", "by", "for", "with", "is", "are", "was", "were", "be", "been", "being",
    "it", "its", "this", "that", "these", "those", "as", "from", "so", "than",
    "too", "very", "can", "will", "just",
}


def is_degenerate(query: str) -> bool:
    tokens = [t.lower() for t in TOKEN_RE.findall(query) if len(t) > 1]
    return not tokens or all(t in FUNCTION_WORDS for t in tokens)

# Minimum acceptable quality. CI fails below these — a retrieval regression is
# a production incident, not a cosmetic one.
GATES = {
    "hybrid": {"recall@5": 0.70, "mrr": 0.55},
}


# ----------------------------------------------------------------- gold set
def load_gold(path: Path) -> list[dict]:
    """Minimal YAML subset reader — avoids adding PyYAML for one file."""
    try:
        import yaml
        return yaml.safe_load(path.read_text())["cases"]
    except ImportError:
        pass

    cases, cur, key, lst = [], None, None, None
    for raw in path.read_text().splitlines():
        if not raw.strip() or raw.strip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        line = raw.strip()
        if line == "cases:":
            continue
        if line.startswith("- id:"):
            if cur:
                cases.append(cur)
            cur = {"id": line.split(":", 1)[1].strip()}
            key, lst = None, None
        elif cur is not None and line.startswith("- ") and key:
            item = line[2:].strip()
            k, v = item.split(":", 1)
            v = v.strip()
            if v.startswith("["):
                v = [x.strip().strip('"\'') for x in v.strip("[]").split(",") if x.strip()]
            else:
                v = v.strip('"\'')
            lst.append({k.strip(): v})
        elif cur is not None and ":" in line:
            k, v = line.split(":", 1)
            k, v = k.strip(), v.strip()
            if not v:
                key, lst = k, []
                cur[k] = lst
            else:
                cur[k] = v.strip('"\'') if v not in ("true", "false") else (v == "true")
                key, lst = None, None
    if cur:
        cases.append(cur)
    return cases


# ----------------------------------------------------------------- matching
def chunk_field(chunk: dict, field: str):
    if field.startswith("meta."):
        return (chunk.get("meta") or {}).get(field[5:])
    return chunk.get(field)


def matches(chunk: dict, matcher: dict) -> bool:
    for field, want in matcher.items():
        if field == "text_contains":
            wants = want if isinstance(want, list) else [want]
            text = (chunk.get("text") or "").lower()
            if any(w.lower() in text for w in wants):
                return True
            return False
        got = chunk_field(chunk, field)
        if got is not None and str(got).lower() == str(want).lower():
            return True
    return False


def relevance(chunk: dict, case: dict) -> bool:
    for m in case.get("relevant_any", []):
        if matches(chunk, m):
            return True
    return False


# ----------------------------------------------------------------- retrievers
class BM25:
    """Mirrors site/js/search.js scoring so evaluation reflects production."""

    def __init__(self, payload: dict):
        # The pipeline ships vocab as a sorted list plus a term->id map.
        self.vocab = payload.get("term_id") or {t: i for i, t in enumerate(payload["vocab"])}
        self.idf_table = payload.get("idf") or []
        self.postings = payload["postings"]
        self.doc_len = payload["doc_len"]
        self.avgdl = payload["avgdl"] or 1.0
        self.k1 = payload.get("k1", 1.5)
        self.b = payload.get("b", 0.75)
        self.N = len(self.doc_len)

    def search(self, query: str, top_k: int = 20):
        tokens = [t.lower() for t in TOKEN_RE.findall(query) if len(t) > 1]
        scores: dict[int, float] = {}
        for tok in tokens:
            ti = self.vocab.get(tok)
            if ti is None:
                continue
            if isinstance(self.postings, dict):
                posting = self.postings.get(str(ti)) or self.postings.get(ti)
            else:
                posting = self.postings[ti]
            if not posting:
                continue
            # Prefer the precomputed IDF so evaluation matches the client exactly.
            if ti < len(self.idf_table):
                idf = self.idf_table[ti]
            else:
                df = len(posting)
                idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            for doc_id, tf in posting:
                dl = self.doc_len[doc_id]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * (tf * (self.k1 + 1)) / denom
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
        return [{"id": d, "score": s} for d, s in ranked]


def rrf(runs: list[list[dict]], k: int = 60, top_k: int = 20):
    fused: dict[int, float] = {}
    for run in runs:
        for rank, hit in enumerate(run, start=1):
            fused[hit["id"]] = fused.get(hit["id"], 0.0) + 1.0 / (k + rank)
    ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]
    return [{"id": d, "score": s} for d, s in ranked]


# ----------------------------------------------------------------- metrics
def dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def score_case(results: list[dict], chunks: list[dict], case: dict, k: int = 10) -> dict:
    top = results[:k]

    if case.get("expect_empty_or_low"):
        ok = len(top) <= 3
        return {"recall@5": float(ok), "recall@10": float(ok), "mrr": float(ok),
                "ndcg@10": float(ok), "hits": len(top), "robustness": ok}

    # Cross-source cases: every declared matcher must be met somewhere in top-k
    if case.get("relevant_all"):
        satisfied = []
        for m in case["relevant_all"]:
            satisfied.append(any(matches(chunks[r["id"]], m) for r in top))
        ok = all(satisfied)
        frac = sum(satisfied) / max(len(satisfied), 1)
        return {"recall@5": frac, "recall@10": frac, "mrr": float(ok),
                "ndcg@10": frac, "hits": len(top), "cross_source_satisfied": ok}

    rel_flags = [relevance(chunks[r["id"]], case) for r in top]
    first = next((i for i, f in enumerate(rel_flags) if f), None)

    return {
        "recall@5": float(any(rel_flags[:5])),
        "recall@10": float(any(rel_flags[:10])),
        "mrr": 1.0 / (first + 1) if first is not None else 0.0,
        "ndcg@10": (dcg([1.0 if f else 0.0 for f in rel_flags]) /
                    dcg([1.0] * max(sum(rel_flags), 1))) if any(rel_flags) else 0.0,
        "hits": len(top),
    }


# ----------------------------------------------------------------- runner
def evaluate(index_path: Path, semantic_path: Path, gold_path: Path) -> dict:
    index = json.loads(index_path.read_text())
    semantic = json.loads(semantic_path.read_text()) if semantic_path.exists() else {"enabled": False}
    cases = load_gold(gold_path)
    chunks = index["chunks"]
    bm25 = BM25(index["bm25"])

    modes = ["lexical"] + (["semantic", "hybrid"] if semantic.get("enabled") else [])
    report: dict = {
        "corpus": {
            "chunks": len(chunks),
            "documents": index["stats"]["document_count"],
            "sources": index["stats"].get("source_count", 1),
            "semantic_enabled": semantic.get("enabled", False),
        },
        "cases": len(cases),
        "modes": {},
        "per_case": [],
    }

    for mode in modes:
        agg: dict[str, list[float]] = {}
        latencies: list[float] = []

        for case in cases:
            q = case["question"]
            t0 = time.perf_counter()

            if is_degenerate(q):
                latencies.append((time.perf_counter() - t0) * 1000)
                m = score_case([], chunks, case)
                for k, v in m.items():
                    if isinstance(v, (int, float)):
                        agg.setdefault(k, []).append(float(v))
                if mode == modes[-1]:
                    report["per_case"].append({
                        "id": case["id"], "category": case.get("category", "-"),
                        "question": q,
                        **{k: round(v, 3) for k, v in m.items() if isinstance(v, (int, float))},
                    })
                continue

            lex = bm25.search(q, 40)
            sem: list[dict] = []
            if semantic.get("enabled"):
                qv = embed_query(semantic, q)
                if qv is not None:
                    scores = semantic_scores(semantic, qv)
                    order = scores.argsort()[::-1][:40]
                    sem = [{"id": int(i), "score": float(scores[i])} for i in order if scores[i] > 0.01]

            if mode == "lexical":
                results = lex
            elif mode == "semantic":
                results = sem
            else:
                runs = [r for r in (lex, sem) if r]
                results = rrf(runs) if runs else []

            latencies.append((time.perf_counter() - t0) * 1000)
            m = score_case(results, chunks, case)
            for k, v in m.items():
                if isinstance(v, (int, float)):
                    agg.setdefault(k, []).append(float(v))

            if mode == modes[-1]:
                report["per_case"].append({
                    "id": case["id"], "category": case.get("category", "-"),
                    "question": q, **{k: round(v, 3) for k, v in m.items() if isinstance(v, (int, float))},
                })

        report["modes"][mode] = {
            **{k: round(sum(v) / len(v), 4) for k, v in agg.items() if k != "hits"},
            "mean_latency_ms": round(sum(latencies) / len(latencies), 2),
        }

    # Per-category breakdown on the best mode
    best = modes[-1]
    by_cat: dict[str, list[float]] = {}
    for pc in report["per_case"]:
        by_cat.setdefault(pc["category"], []).append(pc.get("recall@10", 0.0))
    report["by_category"] = {c: round(sum(v) / len(v), 3) for c, v in sorted(by_cat.items())}
    report["best_mode"] = best
    return report


def print_report(r: dict):
    print("=" * 74)
    print("  KNOWLEDGE FABRIC — RETRIEVAL EVALUATION")
    print("=" * 74)
    c = r["corpus"]
    print(f"\ncorpus: {c['chunks']} chunks · {c['documents']} documents · "
          f"{c['sources']} sources · semantic={'on' if c['semantic_enabled'] else 'off'}")
    print(f"gold set: {r['cases']} cases\n")

    print(f"{'mode':<12} {'recall@5':>9} {'recall@10':>10} {'MRR':>8} {'nDCG@10':>9} {'latency':>10}")
    print("-" * 74)
    for mode, m in r["modes"].items():
        print(f"{mode:<12} {m.get('recall@5',0):>9.3f} {m.get('recall@10',0):>10.3f} "
              f"{m.get('mrr',0):>8.3f} {m.get('ndcg@10',0):>9.3f} {m['mean_latency_ms']:>8.1f}ms")

    if "lexical" in r["modes"] and "hybrid" in r["modes"]:
        lex, hyb = r["modes"]["lexical"], r["modes"]["hybrid"]
        d = hyb.get("recall@10", 0) - lex.get("recall@10", 0)
        pct = (d / lex["recall@10"] * 100) if lex.get("recall@10") else 0.0
        print(f"\nhybrid uplift over lexical: recall@10 {d:+.3f} ({pct:+.1f}%)")

    print(f"\nrecall@10 by category (mode: {r['best_mode']})")
    print("-" * 74)
    for cat, v in r["by_category"].items():
        bar = "#" * int(v * 34)
        print(f"  {cat:<16} {v:>5.2f}  {bar}")

    weak = [p for p in r["per_case"] if p.get("recall@10", 1) < 1.0]
    if weak:
        print(f"\ncases below full recall ({len(weak)}):")
        for p in weak:
            print(f"  [{p['id']}] {p['question'][:56]:<56} r@10={p.get('recall@10',0):.2f}")
    print("\n" + "=" * 74)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=Path("site/data/index.json"))
    ap.add_argument("--semantic", type=Path, default=Path("site/data/semantic.json"))
    ap.add_argument("--gold", type=Path, default=Path("eval/gold_set.yaml"))
    ap.add_argument("--json", type=Path)
    ap.add_argument("--gate", action="store_true", help="exit non-zero if below quality gates")
    args = ap.parse_args()

    if not args.index.exists():
        print(f"[!] {args.index} not found — run: python -m pipeline.build_index", file=sys.stderr)
        sys.exit(2)

    report = evaluate(args.index, args.semantic, args.gold)
    print_report(report)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2))
        print(f"[OK] wrote {args.json}")

    if args.gate:
        failures = []
        for mode, gates in GATES.items():
            got = report["modes"].get(mode)
            if not got:
                continue
            for metric, floor in gates.items():
                if got.get(metric, 0) < floor:
                    failures.append(f"{mode}.{metric} = {got.get(metric,0):.3f} < {floor}")
        if failures:
            print("\n[GATE FAILED]")
            for f in failures:
                print(f"  {f}")
            sys.exit(1)
        print("\n[GATE PASSED]")


if __name__ == "__main__":
    main()
