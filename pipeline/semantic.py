"""Semantic retrieval layer — dense vectors without a server, an API key, or a model download.

Why LSA and not a transformer:
    A sentence-transformer would mean a ~90 MB model at build time and either a
    serving process or a 20 MB+ ONNX payload in the browser. That breaks the
    zero-cost, static-hosting constraint this project is built on.

    Truncated SVD over TF-IDF (Latent Semantic Analysis) gives genuine distributional
    semantics — 'renal', 'kidney', 'creatinine', 'eGFR' land near each other because
    they co-occur — at roughly 1 MB total payload, deterministic across builds, and
    with a projection simple enough to run in JavaScript in under a millisecond.

    It is weaker than a transformer on paraphrase. It is dramatically better than
    BM25 alone on vocabulary mismatch, which is the actual failure mode in this
    corpus: clinicians say 'kidney', the IFU says 'creatinine clearance'.

The browser receives: term->vector table, IDF weights, and chunk vectors — all int8
quantised. Query embedding is an IDF-weighted mean of its term vectors, L2-normalised.
That is the same operation the pipeline performs, so client and server agree exactly.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

# Tuned for payload size vs. retrieval quality on a corpus of this scale.
MAX_TERMS = 6000
N_COMPONENTS = 96
MIN_DF = 2


def _quantize(mat: np.ndarray) -> tuple[list, float]:
    """int8 quantisation. Cuts payload 4x against float32 with negligible
    cosine-similarity error once vectors are L2-normalised."""
    scale = float(np.abs(mat).max()) or 1.0
    q = np.clip(np.round(mat / scale * 127.0), -127, 127).astype(np.int8)
    return q.tolist(), scale


def build_semantic_index(texts: list[str]) -> dict:
    """Fit LSA over the corpus and emit a browser-consumable semantic index."""
    if len(texts) < 3:
        return {"enabled": False, "reason": "corpus too small for LSA"}

    # Small corpora cannot support min_df=2 — every term looks rare and pruning
    # empties the vocabulary. Adapt rather than fail.
    min_df = 1 if len(texts) < 50 else MIN_DF

    vectorizer = TfidfVectorizer(
        max_features=MAX_TERMS,
        min_df=min_df,
        sublinear_tf=True,
        lowercase=True,
        token_pattern=r"[A-Za-z0-9]{2,}",
        stop_words="english",
    )
    try:
        tfidf = vectorizer.fit_transform(texts)
    except ValueError as exc:
        # Degenerate corpus (all stop words, empty docs). Retrieval falls back to
        # BM25 alone; the fabric stays up.
        return {"enabled": False, "reason": f"vectorizer: {exc}"}

    if tfidf.shape[1] < 4:
        return {"enabled": False, "reason": "vocabulary too small for LSA"}

    # Rank is bounded by both vocabulary and corpus size. Small corpora get a
    # low-rank projection rather than no semantics at all.
    n_components = min(N_COMPONENTS, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
    if n_components < 2:
        return {"enabled": False, "reason": "insufficient rank for LSA"}

    svd = TruncatedSVD(n_components=n_components, random_state=42, algorithm="randomized")
    try:
        doc_vectors = svd.fit_transform(tfidf)
    except ValueError as exc:
        return {"enabled": False, "reason": f"svd: {exc}"}

    # L2-normalise so cosine similarity reduces to a dot product in the client.
    norms = np.linalg.norm(doc_vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    doc_vectors = doc_vectors / norms

    # Term vectors: project each vocabulary term through the same space.
    # svd.components_ is (n_components, n_terms) — transpose gives per-term vectors.
    term_vectors = svd.components_.T                     # (n_terms, n_components)
    t_norms = np.linalg.norm(term_vectors, axis=1, keepdims=True)
    t_norms[t_norms == 0] = 1.0
    term_vectors = term_vectors / t_norms

    doc_q, doc_scale = _quantize(doc_vectors.astype(np.float32))
    term_q, term_scale = _quantize(term_vectors.astype(np.float32))

    vocab = vectorizer.vocabulary_
    terms_sorted = sorted(vocab.items(), key=lambda kv: kv[1])
    idf = vectorizer.idf_

    return {
        "enabled": True,
        "method": "tfidf+truncated_svd(lsa)",
        "dims": int(n_components),
        "explained_variance": round(float(svd.explained_variance_ratio_.sum()), 4),
        "terms": [t for t, _ in terms_sorted],
        "idf": [round(float(x), 4) for x in idf],
        "term_vectors": term_q,
        "term_scale": term_scale,
        "doc_vectors": doc_q,
        "doc_scale": doc_scale,
        "stats": {
            "vocabulary": len(vocab),
            "documents": int(tfidf.shape[0]),
            "payload_vectors": int(tfidf.shape[0] * n_components + len(vocab) * n_components),
        },
    }


def embed_query(semantic: dict, query: str) -> np.ndarray | None:
    """Reference implementation of the client-side query embedding.
    The evaluation harness uses this so we measure exactly what users experience."""
    if not semantic.get("enabled"):
        return None
    import re

    term_index = {t: i for i, t in enumerate(semantic["terms"])}
    tv = np.array(semantic["term_vectors"], dtype=np.float32) * semantic["term_scale"] / 127.0
    idf = np.array(semantic["idf"], dtype=np.float32)

    tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9]{2,}", query)]
    hits = [term_index[t] for t in tokens if t in term_index]
    if not hits:
        return None

    weights = idf[hits][:, None]
    vec = (tv[hits] * weights).sum(axis=0)
    norm = np.linalg.norm(vec)
    return vec / norm if norm else None


def semantic_scores(semantic: dict, query_vec: np.ndarray) -> np.ndarray:
    dv = np.array(semantic["doc_vectors"], dtype=np.float32) * semantic["doc_scale"] / 127.0
    return dv @ query_vec
