"""Text normalisation and sentence quality classification.

Everything downstream — retrieval, summarisation, confidence — is capped by the
quality of the text coming out of the PDFs. Right now that text contains:

    "point -of-care"        space injected before a hyphen
    "multiple- patient"     space injected after a hyphen
    "re - sponsible"        line-break hyphenation left in place
    "Mea surement"          intra-word space from PDF kerning
    "b lood", "w as"        single letters split off
    "EnzymesCreatinine"     missing space at a style boundary
    "Page 5 of 6"           running headers inlined into body text

A summariser cannot recover from this. It selects a sentence containing
"re - sponsible for any warranty" and presents it as an answer, because at the
token level it looks like ordinary prose.

The repair strategy avoids a dictionary dependency by using **the corpus's own
vocabulary as the lexicon**. Two fragments are merged only when the merged form
is attested elsewhere in the corpus. That is high precision on domain vocabulary
("Mea surement" -> "measurement") and conservative where it should be — it will
not invent a merge it has never seen.

Sentences are then classified, because "is this prose or a table row" determines
whether a sentence is answer material at all.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

# --------------------------------------------------------------------------- classes
PROSE = "prose"
TABLE = "table"
HEADER = "header"
LEGAL = "legal"
EQUATION = "equation"
FRAGMENT = "fragment"

# Sentences that are structurally fine but are never the answer to a clinical or
# technical question. Keeping them indexed is correct; offering them as an answer
# is not.
LEGAL_RE = re.compile(
    r"\b(will not be responsible|not be liable|no warranty|warrant(?:y|s|ies)\b|"
    r"disclaim|all rights reserved|copyright|trademark|"
    r"patent(?:s|ed)? (?:pending|no)|is a registered trademark|"
    r"shall not be liable|limitation of liability|terms and conditions|"
    r"exclusions and upon the conditions|free of all charges|"
    r"defective material or workmanship|authorized distributor)\b", re.I)

# Table-of-contents dot leaders: "Quick Reference Guide.........12"
TOC_RE = re.compile(r"\.{4,}|\s\.\s\.\s\.")

RUNNING_HEADER_RE = re.compile(
    r"^\s*(?:[A-Z]\d{6}\s*[-–]?\s*)?page\s+\d+\s+of\s+\d+\s*", re.I)

PAGE_NUM_PREFIX_RE = re.compile(r"^\s*\d{1,4}\s+(?=[A-Z][a-z])")

EQUATION_RE = re.compile(
    r"[→⟶⇌←]|—{2,}|\b\d\s*[A-Z]\d\b|\+\s*e\s*-|"
    r"\bEquation\s*\d|\b[A-Z][a-z]?\(\w+\)\s*\d|\bLOD(?:ox|red)\b")

TOKEN_RE = re.compile(r"[A-Za-z]{2,}")

# Sentence boundary: terminal punctuation not preceded by a known abbreviation.
ABBREV = {
    "no", "fig", "eq", "vs", "etc", "inc", "ltd", "co", "corp", "dr", "mr", "mrs",
    "ms", "prof", "st", "approx", "min", "max", "ref", "cf", "e.g", "i.e", "u.s",
    "ca", "mg", "ml", "dl", "hr", "sec", "temp", "conc", "avg", "std", "dev",
}


# =============================================================================
# Vocabulary — built from the corpus itself, used as the repair lexicon
# =============================================================================
def build_vocabulary(texts: list[str], min_count: int = 2) -> Counter:
    """Word frequencies over the corpus.

    Frequencies, not a set. A set is not enough, because a *recurring* extraction
    artifact becomes attested vocabulary in its own right: "Mea surement" appears
    often enough that both "mea" and "surement" look like real words. Only the
    relative counts disambiguate — "measurement" outnumbers "mea", so the merge
    is correct; "in" vastly outnumbers "intake", so "in take" must be left alone.
    """
    counts: Counter = Counter()
    for t in texts:
        counts.update(w.lower() for w in TOKEN_RE.findall(t))
    return Counter({w: n for w, n in counts.items() if n >= min_count})


# =============================================================================
# Artifact repair
# =============================================================================
def _repair_hyphens(text: str, vocab: Counter) -> str:
    # "re - sponsible" / "re- sponsible" -> "responsible" when attested
    def join_broken(m):
        a, b = m.group(1), m.group(2)
        if (a + b).lower() in vocab:
            return a + b
        return f"{a}-{b}"
    text = re.sub(r"\b([A-Za-z]{2,})\s*-\s+([a-z]{2,})\b", join_broken, text)

    # "point -of-care" -> "point-of-care"; "multiple- patient" -> "multiple-patient"
    text = re.sub(r"([A-Za-z])\s+-(?=[A-Za-z])", r"\1-", text)
    text = re.sub(r"([A-Za-z])-\s+(?=[A-Za-z])", r"\1-", text)
    return text


# PDF style boundaries fuse a heading to the following word with no space:
# "AppendixAccuracy", "EnzymesCreatinine", "ContentsQuick". Split on the
# lower->upper transition when both halves are attested, so retrieval sees two
# real words instead of one token that matches nothing.
# CamelCase brand names must never be split. "StatStrip" -> "Stat Strip" would
# break every product name in the corpus, and both halves are attested words.
PROTECTED_CAMEL = {
    "statstrip", "statsensor", "bioprofile", "novabiomedical", "lactateplus",
    "novamax", "statprofile", "biosensor", "healthcare", "wavesense",
}


def _split_fused_words(text: str, vocab: Counter) -> str:
    def split(m):
        whole, a, b = m.group(0), m.group(1), m.group(2)
        if whole.lower() in PROTECTED_CAMEL:
            return whole
        # A fused pair that is itself a frequent token is a real compound word,
        # not a PDF style-boundary artifact.
        if vocab.get(whole.lower(), 0) >= 5:
            return whole
        if vocab.get(a.lower(), 0) >= 2 and vocab.get(b.lower(), 0) >= 2:
            return f"{a} {b}"
        return whole
    return re.sub(r"\b([A-Z][a-z]{2,})([A-Z][a-z]{2,})\b", split, text)


def _repair_split_words(text: str, vocab: Counter) -> str:
    """Merge fragments split by PDF kerning: 'Mea surement', 'b lood', 'Intercep t'.

    Merge only when the joined form is *more frequent in this corpus* than either
    fragment standing alone. That single test handles both directions:

        "mea"(12) + "surement"(12) -> "measurement"(58)   merge   — artifact
        "in"(2100) + "take"(40)    -> "intake"(3)         keep    — real words

    A set-membership test cannot distinguish these, because a systematic PDF
    artifact makes its own fragments look like attested vocabulary.
    """
    def merge(m):
        a, b = m.group(1), m.group(2)
        joined = (a + b).lower()
        n_joined = vocab.get(joined, 0)
        if n_joined == 0:
            return m.group(0)
        if n_joined > max(vocab.get(a.lower(), 0), vocab.get(b.lower(), 0)):
            return a + b
        return m.group(0)

    # short fragment + word,  or  word + short fragment
    text = re.sub(r"\b([A-Za-z]{1,3}) ([A-Za-z]{2,})\b", merge, text)
    text = re.sub(r"\b([A-Za-z]{2,}) ([A-Za-z]{1,3})\b", merge, text)

    # Orphaned single-letter suffixes: "measurement s", "sample d". The
    # frequency rule above deliberately refuses these (the base word usually
    # outnumbers its plural), but a lone letter is never a word here — the only
    # real single-letter English words are "a" and "I".
    def merge_suffix(m):
        base, letter = m.group(1), m.group(2)
        if letter.lower() in {"a", "i"}:
            return m.group(0)
        return base + letter if (base + letter).lower() in vocab else m.group(0)
    text = re.sub(r"\b([A-Za-z]{3,}) ([A-Za-z])\b(?![.'])", merge_suffix, text)
    return text


def _strip_running_headers(text: str) -> str:
    text = RUNNING_HEADER_RE.sub("", text)
    text = re.sub(r"\s*[A-Z]\d{6}\s*[-–]\s*page\s+\d+\s+of\s+\d+\s*", " ", text, flags=re.I)
    text = PAGE_NUM_PREFIX_RE.sub("", text)
    return text


def normalize(text: str, vocab: Counter) -> str:
    if not text:
        return ""
    t = text.replace("\u00ad", "").replace("\ufb01", "fi").replace("\ufb02", "fl")
    t = t.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    t = _strip_running_headers(t)
    t = _split_fused_words(t, vocab)
    t = _repair_hyphens(t, vocab)
    t = _repair_split_words(t, vocab)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\s+([,.;:%])", r"\1", t)
    t = re.sub(r"\n{2,}", "\n\n", t)
    return t.strip()


# =============================================================================
# Sentence segmentation
# =============================================================================
# Layout blocks that carry no terminal punctuation: bullet lists, numbered
# specification lines, "Label: value" runs. Left unsplit they become one enormous
# pseudo-sentence that matches almost any query and swamps real prose — this is
# how a StatSensor Creatinine spec block came to answer a question about glucose.
BULLET_SPLIT_RE = re.compile(r"\s*[•·▪◦]\s*|\s{2,}(?=[A-Z][a-z]+:)|\s*(?<=[a-z])\s(?=\d+\.\d+\s+[A-Z])")
MAX_SENTENCE_CHARS = 320


# A period is only a sentence boundary when followed by whitespace and a capital
# (or end of string). "Sample size 1.2 uL" and "1.4 The Sample" are decimals and
# section numbers — testing for a bare "." treats those as prose and leaves the
# whole layout block unsplit.
TERMINAL_PUNCT_RE = re.compile(r"[.!?](?:\s|$)")


def _split_layout_blocks(part: str) -> list[str]:
    """Break a layout block (bullet list, spec run) into its constituent lines."""
    bullets = len(re.findall(r"[•·▪◦]", part))
    if bullets >= 2:
        pieces = [p.strip() for p in BULLET_SPLIT_RE.split(part) if p and p.strip()]
        if len(pieces) > 1:
            return pieces
    if len(part) <= MAX_SENTENCE_CHARS or TERMINAL_PUNCT_RE.search(part):
        return [part]
    pieces = [p.strip() for p in BULLET_SPLIT_RE.split(part) if p and p.strip()]
    return pieces if len(pieces) > 1 else [part]


def _hard_wrap(s: str) -> list[str]:
    """Last resort: a single run longer than the cap with no internal structure.
    Split on clause boundaries so we never emit a 2,000-character 'sentence'."""
    if len(s) <= MAX_SENTENCE_CHARS:
        return [s]
    out, cur = [], []
    length = 0
    for token in s.split(" "):
        cur.append(token)
        length += len(token) + 1
        if length >= MAX_SENTENCE_CHARS:
            out.append(" ".join(cur))
            cur, length = [], 0
    if cur:
        out.append(" ".join(cur))
    return out


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    text = re.sub(r"\s*\n+\s*", " ", text)
    out, buf = [], []
    for part in re.split(r"(?<=[.!?])\s+", text):
        buf.append(part)
        tail = part.rstrip()
        last_word = re.findall(r"([A-Za-z.]+)\.$", tail)
        if last_word and last_word[0].lower().rstrip(".") in ABBREV:
            continue                      # abbreviation, not a boundary
        if re.search(r"\b[A-Z]\.$", tail):
            continue                      # initial, e.g. "J. Smith"
        out.append(" ".join(buf).strip())
        buf = []
    if buf:
        out.append(" ".join(buf).strip())

    final: list[str] = []
    for part in out:
        for block in _split_layout_blocks(part):
            final.extend(_hard_wrap(block))
    return [s.strip() for s in final if s.strip()]


# =============================================================================
# Sentence quality classification
# =============================================================================
@dataclass
class SentenceInfo:
    text: str
    offset: int
    length: int
    kind: str
    quality: float          # 0..1 — usefulness as answer material


# A chunk boundary can cut a word in half: "...and neonata", "See Indications
# for Use belo". Presenting those as an answer looks like a bug to the reader.
TRUNCATED_TAIL_RE = re.compile(r"\b[a-z]{2,}$")
COMMON_SHORT_ENDINGS = {
    "be", "do", "go", "is", "as", "at", "in", "on", "of", "to", "up", "so", "no",
    "we", "he", "it", "or", "if", "by", "an",
}


def looks_truncated(s: str, vocab: Counter | None = None) -> bool:
    """A sentence cut off at a chunk boundary.

    The reliable test is the corpus vocabulary itself: a real word appears many
    times across the corpus, whereas a severed one ("neonata", "belo") appears
    only where the cut happened. A suffix-shape heuristic was tried first and
    rejected — it flagged "...diagnosis of renal disease" as truncated because
    "disease" matched no known suffix, which is exactly the kind of false
    positive that silently deletes good answers.
    """
    t = s.strip()
    if re.search(r"[.!?:;)\]]$", t):
        return False
    words = t.split()
    if not words:
        return False
    last = words[-1].strip(",;:").lower()
    if not last.isalpha() or last in COMMON_SHORT_ENDINGS:
        return False
    if vocab is None:
        return False
    # Attested several times elsewhere -> a real word that merely ends a passage.
    return vocab.get(last, 0) < 3


def classify_sentence(s: str, vocab: Counter | None = None) -> tuple[str, float]:
    stripped = s.strip()
    n = len(stripped)
    if n < 25:
        return FRAGMENT, 0.0

    if looks_truncated(stripped, vocab):
        return FRAGMENT, 0.0

    letters = sum(c.isalpha() for c in stripped)
    digits = sum(c.isdigit() for c in stripped)
    alpha_ratio = letters / max(n, 1)
    digit_ratio = digits / max(n, 1)
    words = stripped.split()
    n_words = len(words)

    if TOC_RE.search(stripped):
        return HEADER, 0.0            # table of contents, never answer material

    if LEGAL_RE.search(stripped):
        return LEGAL, 0.05

    if EQUATION_RE.search(stripped) or alpha_ratio < 0.45:
        return EQUATION, 0.05

    # Table rows. Digit density is the reliable signal. A high count of short
    # tokens alone is not — ordinary instructions ("Dispose of the cap in a
    # suitable container.") are full of short function words and were being
    # misfiled as tables, which removed genuinely useful procedural sentences
    # from the answer pool.
    short_tokens = sum(1 for w in words if len(w) <= 3)
    numeric_tokens = sum(1 for w in words if re.fullmatch(r"[\d.,%()+-]+", w))
    if digit_ratio > 0.18:
        return TABLE, 0.1
    if n_words > 8 and numeric_tokens / n_words > 0.30:
        return TABLE, 0.1
    if n_words > 10 and short_tokens / n_words > 0.72 and numeric_tokens > 0:
        return TABLE, 0.1

    # Headers: no terminal punctuation, short, mostly capitalised
    if not re.search(r"[.!?]$", stripped) and n_words <= 9:
        caps = sum(1 for w in words if w[:1].isupper())
        if caps / max(n_words, 1) > 0.6:
            return HEADER, 0.15

    # Prose quality: prefer complete, mid-length, verb-bearing sentences
    q = 0.55
    if re.search(r"[.!?]$", stripped):
        q += 0.15
    if 12 <= n_words <= 45:
        q += 0.15
    elif n_words > 70:
        q -= 0.15
    if re.search(r"\b(is|are|was|were|has|have|can|may|must|should|provides?|"
                 r"measures?|used|indicates?|requires?|allows?|detects?|"
                 r"determines?|intended)\b", stripped, re.I):
        q += 0.15
    if stripped[:1].isupper():
        q += 0.05
    if digit_ratio > 0.10:
        q -= 0.10
    return PROSE, max(0.0, min(1.0, q))


def analyse(text: str, vocab: Counter | None = None) -> list[SentenceInfo]:
    """Sentences with offsets into `text`, so the client can slice rather than
    carry a duplicate copy of the corpus."""
    infos: list[SentenceInfo] = []
    cursor = 0
    for s in split_sentences(text):
        idx = text.find(s[:40], cursor)
        if idx < 0:
            idx = cursor
        kind, quality = classify_sentence(s, vocab)
        infos.append(SentenceInfo(text=s, offset=idx, length=len(s), kind=kind, quality=round(quality, 3)))
        cursor = idx + max(len(s), 1)
    return infos


def chunk_quality(infos: list[SentenceInfo]) -> float:
    """Share of a chunk that is usable prose — a cheap, honest retrieval prior."""
    if not infos:
        return 0.0
    prose = [i for i in infos if i.kind == PROSE]
    if not prose:
        return 0.0
    return round(sum(i.quality for i in prose) / len(infos), 3)
