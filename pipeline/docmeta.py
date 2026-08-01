"""Document dating — with provenance, and with honest gaps.

"How do you get the dates?" is a fair challenge, because a date presented without
a source is indistinguishable from one that was invented. Every date this module
emits carries the method that produced it and a confidence band, and a document
with no recoverable date reports `null` rather than a plausible-looking guess.

Resolution order (first hit wins, highest authority first):

  1. REGULATORY   FDA decision date from the openFDA record. Authoritative — it is
                  the legal date of the clearance.
  2. PDF_METADATA /CreationDate or /ModDate in the PDF trailer. Reliable when the
                  file was produced by the publisher's own toolchain; unreliable if
                  the file was ever re-saved, so it is reported as medium confidence.
  3. DOC_CONTROL  A revision stamp inside the document body — Nova's own literature
                  part number carries one ("LPN 65965 D, 2023-04"). High confidence
                  when present because it is the publisher's stated revision date.
  4. TEXT_DATE    An ISO or long-form date in the first page of body text. Low
                  confidence: it may be a study date, not a publication date.
  5. UNKNOWN      No date recoverable. Reported as such.

The distinction that matters downstream: freshness scoring must not treat an
UNKNOWN as old, and must not treat a low-confidence TEXT_DATE as authoritative.
"""
from __future__ import annotations

import re
from datetime import datetime, date
from pathlib import Path

REGULATORY = "fda_decision_date"
PDF_METADATA = "pdf_metadata"
DOC_CONTROL = "document_revision_stamp"
TEXT_DATE = "date_in_body_text"
UNKNOWN = "unknown"

CONFIDENCE = {
    REGULATORY: 1.00,
    DOC_CONTROL: 0.85,
    PDF_METADATA: 0.60,
    TEXT_DATE: 0.35,
    UNKNOWN: 0.0,
}

# "D:20230412153000-04'00'"
PDF_DATE_RE = re.compile(r"D:(\d{4})(\d{2})(\d{2})")
ISO_RE = re.compile(r"\b(20[0-2]\d)-(\d{2})-(\d{2})\b")
LPN_REV_RE = re.compile(r"\bLPN[\s-]*(\d{5})[\s-]*([A-Z])\b", re.I)
REV_DATE_RE = re.compile(
    r"\b(?:rev(?:ision)?|revised|effective|issued)\b[^.\n]{0,24}?"
    r"(20[0-2]\d)[-/](\d{1,2})(?:[-/](\d{1,2}))?", re.I)
MONTH_RE = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(20[0-2]\d)\b", re.I)
MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}


def _safe_date(y, m=1, d=1) -> str | None:
    try:
        y, m, d = int(y), max(1, min(12, int(m))), max(1, min(28, int(d or 1)))
        if not (2000 <= y <= date.today().year + 1):
            return None
        return datetime(y, m, d).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def from_pdf_metadata(path: Path) -> str | None:
    try:
        from pypdf import PdfReader
        meta = PdfReader(str(path)).metadata or {}
    except Exception:
        return None
    for key in ("/ModDate", "/CreationDate"):
        raw = str(meta.get(key) or "")
        m = PDF_DATE_RE.search(raw)
        if m:
            got = _safe_date(*m.groups())
            if got:
                return got
    return None


def from_text(text: str) -> tuple[str | None, str]:
    """Body-text dating. Returns (iso_date, method)."""
    head = text[:4000]

    m = REV_DATE_RE.search(head)
    if m:
        got = _safe_date(m.group(1), m.group(2), m.group(3))
        if got:
            return got, DOC_CONTROL

    m = LPN_REV_RE.search(head)
    if m:
        # A revision letter is a version, not a date. Look for a nearby year to
        # anchor it; if there is none, this is deliberately not treated as a date.
        window = head[max(0, m.start() - 120): m.end() + 120]
        y = re.search(r"\b(20[0-2]\d)\b", window)
        if y:
            got = _safe_date(y.group(1))
            if got:
                return got, DOC_CONTROL

    m = ISO_RE.search(head)
    if m:
        got = _safe_date(*m.groups())
        if got:
            return got, TEXT_DATE

    m = MONTH_RE.search(head)
    if m:
        got = _safe_date(m.group(2), MONTHS[m.group(1).lower()[:3]])
        if got:
            return got, TEXT_DATE

    return None, UNKNOWN


def resolve(*, source_path: Path | None = None, text: str = "",
            regulatory_date: str | None = None) -> dict:
    """Resolve a document date with its provenance."""
    if regulatory_date:
        iso = regulatory_date if ISO_RE.fullmatch(regulatory_date) else _safe_date(
            *regulatory_date.split("-")[:3]) if "-" in regulatory_date else None
        if iso:
            return {"date": iso, "method": REGULATORY, "confidence": CONFIDENCE[REGULATORY],
                    "explanation": "FDA decision date from the openFDA clearance record."}

    if source_path and source_path.suffix.lower() == ".pdf":
        meta_date = from_pdf_metadata(source_path)
    else:
        meta_date = None

    text_date, text_method = from_text(text)

    # A revision stamp in the body outranks file metadata: it is what the
    # publisher asserts, whereas metadata reflects whoever last saved the file.
    if text_method == DOC_CONTROL and text_date:
        return {"date": text_date, "method": DOC_CONTROL, "confidence": CONFIDENCE[DOC_CONTROL],
                "explanation": "Revision stamp printed inside the document."}

    if meta_date:
        return {"date": meta_date, "method": PDF_METADATA, "confidence": CONFIDENCE[PDF_METADATA],
                "explanation": "PDF creation/modification timestamp. Reflects the last save, "
                               "which may post-date publication."}

    if text_date:
        return {"date": text_date, "method": TEXT_DATE, "confidence": CONFIDENCE[TEXT_DATE],
                "explanation": "A date found in the document body. May refer to a study or "
                               "reference period rather than publication."}

    return {"date": None, "method": UNKNOWN, "confidence": 0.0,
            "explanation": "No date recoverable from metadata, revision stamp, or body text."}


def age_days(iso: str | None) -> int | None:
    if not iso:
        return None
    try:
        return (date.today() - datetime.strptime(iso, "%Y-%m-%d").date()).days
    except ValueError:
        return None
