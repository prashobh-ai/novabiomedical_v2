"""FDA regulatory connector — structured public records become retrievable knowledge.

Design note: we *verbalise* each row into natural language rather than indexing raw
fields. A row like {k_number: K232075, device_name: StatStrip...} is invisible to
lexical and semantic search. The same row rendered as a sentence is retrievable by
the identical machinery that serves the PDF corpus — which is the whole point of a
fabric. Typed facets are preserved separately in metadata for filtering and graph
construction, so nothing is lost.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

from .base import KnowledgeRecord, sanitize

FDA_BASE = "https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfPMN/pmn.cfm?ID="


class FDARegulatoryConnector:
    name = "FDA openFDA"
    source_type = "fda_regulatory"

    def __init__(self, root: Path):
        self.root = Path(root)

    # ---------------------------------------------------------------- helpers
    def _rows(self, filename: str) -> Iterator[dict]:
        path = self.root / filename
        if not path.exists():
            return
        with open(path, newline="", encoding="utf-8") as f:
            yield from csv.DictReader(f)

    # ---------------------------------------------------------------- sources
    def _clearances(self) -> Iterator[KnowledgeRecord]:
        for r in self._rows("510k_clearances.csv"):
            k = sanitize(r.get("k_number"))
            if not k:
                continue
            device = sanitize(r.get("device_name"))
            code = sanitize(r.get("product_code"))
            decided = sanitize(r.get("decision_date"))
            received = sanitize(r.get("date_received"))
            committee = sanitize(r.get("advisory_committee_description"))
            decision = sanitize(r.get("decision_description"))
            disclosure = sanitize(r.get("statement_or_summary"))

            text = (
                f"FDA 510(k) premarket notification {k} covers the device "
                f"'{device}', submitted by Nova Biomedical. "
                f"The submission was received on {received or 'an unrecorded date'} and reached a "
                f"decision on {decided or 'an unrecorded date'}, with the outcome recorded as "
                f"'{decision or 'not recorded'}'. "
                f"It was reviewed by the {committee or 'unspecified'} advisory committee panel under "
                f"FDA product code {code or 'unassigned'}. "
                f"The applicant filed a 510(k) {disclosure or 'disclosure of unrecorded type'}, "
                f"which determines whether a public summary document exists for this clearance."
            )
            yield KnowledgeRecord(
                source_type=self.source_type,
                source_system=self.name,
                source_id=k,
                title=f"{k} — {device}",
                text=text,
                section_path=["FDA Regulatory", "510(k) Clearances", committee or "Uncategorised"],
                url=f"{FDA_BASE}{k}",
                metadata={
                    "record_type": "clearance",
                    "k_number": k,
                    "device_name": device,
                    "product_code": code,
                    "decision_date": decided,
                    "year": decided[:4] if decided else "",
                    "advisory_committee": committee,
                    "disclosure_type": disclosure,
                },
                entities=[("Clearance", k), ("Product", device), ("ProductCode", code)],
            )

    def _classifications(self) -> Iterator[KnowledgeRecord]:
        for r in self._rows("product_code_classifications.csv"):
            code = sanitize(r.get("product_code"))
            if not code:
                continue
            name = sanitize(r.get("device_name"))
            dev_class = sanitize(r.get("device_class"))
            reg = sanitize(r.get("regulation_number"))
            specialty = sanitize(r.get("medical_specialty_description"))
            definition = sanitize(r.get("definition"))

            text = (
                f"FDA product code {code} designates the device category '{name}'. "
                f"It is a Class {dev_class or 'unspecified'} device governed by "
                f"21 CFR {reg or 'an unrecorded regulation'}, within the "
                f"{specialty or 'unspecified'} medical specialty. "
                f"{definition}"
            ).strip()
            yield KnowledgeRecord(
                source_type=self.source_type,
                source_system=self.name,
                source_id=f"CODE-{code}",
                title=f"Product code {code} — {name}",
                text=text,
                section_path=["FDA Regulatory", "Device Classification", specialty or "Uncategorised"],
                url=f"https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfPCD/classification.cfm?ID={code}",
                metadata={
                    "record_type": "classification",
                    "product_code": code,
                    "device_class": dev_class,
                    "regulation_number": reg,
                    "specialty": specialty,
                },
                entities=[("ProductCode", code), ("Regulation", reg)],
            )

    def _recalls(self) -> Iterator[KnowledgeRecord]:
        for r in self._rows("device_recalls.csv"):
            res = sanitize(r.get("product_res_number"))
            initiated = sanitize(r.get("event_date_initiated"))
            product = sanitize(r.get("product_description"))
            reason = sanitize(r.get("reason_for_recall"))
            cause = sanitize(r.get("root_cause_description"))
            status = sanitize(r.get("recall_status"))
            code = sanitize(r.get("product_code"))

            text = (
                f"Nova Biomedical recall {res} was initiated on {initiated or 'an unrecorded date'} "
                f"affecting: {product}. "
                f"Reason for recall: {reason} "
                f"FDA recorded the root cause as '{cause or 'not classified'}'. "
                f"The recall status is {status or 'unrecorded'}."
            )
            yield KnowledgeRecord(
                source_type=self.source_type,
                source_system=self.name,
                source_id=res or f"RECALL-{initiated}",
                title=f"Recall {res} — {product[:60]}",
                text=text,
                section_path=["FDA Regulatory", "Recalls", cause or "Unclassified"],
                url="https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfRes/res.cfm",
                metadata={
                    "record_type": "recall",
                    "recall_number": res,
                    "date_initiated": initiated,
                    "year": initiated[:4] if initiated else "",
                    "root_cause": cause,
                    "status": status,
                    "product_code": code,
                    "severity_signal": "software" if "software" in cause.lower() else "other",
                },
                entities=[("Recall", res), ("Product", product[:60]), ("ProductCode", code)],
            )

    def _ifu_index(self) -> Iterator[KnowledgeRecord]:
        """The IFU catalogue itself is knowledge — it answers 'which manual covers X,
        and what revision are we on'. Revision drift is a real enterprise problem."""
        for r in self._rows("nova_ifu_document_index.csv"):
            title = sanitize(r.get("title"))
            if not title:
                continue
            product = sanitize(r.get("product"))
            family = sanitize(r.get("family"))
            doc_type = sanitize(r.get("doc_type"))
            lpn = sanitize(r.get("literature_part_number"))
            url = sanitize(r.get("url"))

            text = (
                f"The document '{title}' is the current published {doc_type} for the "
                f"{product} ({family} product family). "
                f"It carries Nova literature part number {lpn or 'unassigned'}, which encodes the "
                f"revision letter — a change in that letter signals a superseded document. "
                f"{'It is published at ' + url if url else 'No public file is published for this entry.'}"
            )
            yield KnowledgeRecord(
                source_type=self.source_type,
                source_system="Nova IFU Catalogue",
                source_id=f"IFU-{lpn or title[:30]}",
                title=title,
                text=text,
                section_path=["Document Control", f"{family} Documentation", product],
                url=url,
                metadata={
                    "record_type": "ifu_catalogue",
                    "product": product,
                    "family": family,
                    "doc_type": doc_type,
                    "literature_part_number": lpn,
                    "published": bool(url),
                },
                entities=[("Product", product), ("Document", title)],
            )

    def _udi(self) -> Iterator[KnowledgeRecord]:
        seen: set[str] = set()
        for r in self._rows("udi_device_identifiers.csv"):
            brand = sanitize(r.get("brand_name"))
            model = sanitize(r.get("version_or_model_number"))
            key = f"{brand}|{model}"
            if not brand or key in seen:
                continue
            seen.add(key)
            desc = sanitize(r.get("device_description"))
            codes = sanitize(r.get("product_codes"))
            status = sanitize(r.get("commercial_distribution_status"))

            text = (
                f"UDI registration for {brand}, model/version {model or 'unspecified'}. "
                f"{desc} "
                f"Associated FDA product codes: {codes or 'none recorded'}. "
                f"Commercial distribution status: {status or 'unrecorded'}."
            )
            yield KnowledgeRecord(
                source_type=self.source_type,
                source_system=self.name,
                source_id=f"UDI-{brand}-{model}"[:80],
                title=f"{brand} {model}".strip(),
                text=text,
                section_path=["FDA Regulatory", "UDI Registrations", brand],
                url="https://accessgudid.nlm.nih.gov/",
                metadata={
                    "record_type": "udi",
                    "brand_name": brand,
                    "model": model,
                    "product_codes": codes,
                    "distribution_status": status,
                    "publish_date": sanitize(r.get("publish_date")),
                },
                entities=[("Product", brand), ("ProductCode", codes.split(";")[0].strip() if codes else "")],
            )

    # ---------------------------------------------------------------- fetch
    def fetch(self) -> Iterator[KnowledgeRecord]:
        for source in (
            self._clearances,
            self._classifications,
            self._recalls,
            self._ifu_index,
            self._udi,
        ):
            yield from source()
