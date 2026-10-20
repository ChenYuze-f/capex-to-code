"""
SQLite & Baseline JSON Persistence Layer (`storage.py`).

Seeds the database from `baseline_universe.json` on initial boot, caches live
international company audits, and stores custom author thesis notes.
"""

from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional

from ai_audit_engine import run_deepseek_technical_audit, score_three_way_evidence
from analyst_calibration import get_author_thesis
from valuation_model import compute_rd_capitalization_and_valuation

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = BASE_DIR / "audit_cache.db"
BASELINE_JSON_PATH = BASE_DIR / "baseline_universe.json"


def _enrich_raw_company_record(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Compute full 3-way audit, Damodaran valuation, and deterministic synthesis for a raw record."""
    scores = score_three_way_evidence(raw)
    val = compute_rd_capitalization_and_valuation(raw, scores)
    synthesis = run_deepseek_technical_audit(
        company_name=raw["name"],
        ticker=raw["ticker"],
        raw_evidence=raw,
        base_scores=scores,
        api_key="",  # Seed fast with deterministic synthesis; live DeepSeek runs on demand
    )
    merged = dict(raw)
    merged.update(scores)
    merged.update(val)
    merged["audit_synthesis"] = synthesis
    merged["author_thesis"] = raw.get("author_thesis") or get_author_thesis(raw["ticker"])
    merged["last_updated"] = raw.get("last_updated") or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return merged


class AuditStorage:
    """SQLite persistence engine for global company R&D and AI reality audits."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path or DEFAULT_DB_PATH
        self._init_db()
        self._seed_if_empty()

    def _connect(self):
        return closing(sqlite3.connect(str(self.db_path)))

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS company_audits (
                    ticker TEXT PRIMARY KEY,
                    region TEXT NOT NULL,
                    name TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _seed_if_empty(self) -> None:
        if not BASELINE_JSON_PATH.exists():
            return
        with self._connect() as conn:
            cur = conn.execute("SELECT ticker FROM company_audits")
            existing_tickers = {r[0] for r in cur.fetchall()}

        raw_list = json.loads(BASELINE_JSON_PATH.read_text(encoding="utf-8"))
        for raw_item in raw_list:
            tk = str(raw_item.get("ticker", "")).strip().upper()
            if tk and tk not in existing_tickers:
                enriched = _enrich_raw_company_record(raw_item)
                self.upsert_company(enriched)

    def upsert_company(self, record: Dict[str, Any]) -> None:
        ticker = str(record["ticker"]).strip().upper()
        region = str(record.get("region", "US")).strip().upper()
        name = str(record.get("name", ticker))
        updated_at = str(
            record.get("last_updated")
            or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        )
        record["ticker"] = ticker
        record["last_updated"] = updated_at
        payload = json.dumps(record, ensure_ascii=False)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO company_audits (ticker, region, name, payload_json, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    region = excluded.region,
                    name = excluded.name,
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (ticker, region, name, payload, updated_at),
            )
            conn.commit()

    def get_company(self, ticker: str) -> Optional[Dict[str, Any]]:
        clean = (ticker or "").strip().upper()
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT payload_json FROM company_audits WHERE ticker = ?",
                (clean,),
            )
            row = cur.fetchone()
        if not row:
            return None
        return json.loads(row[0])

    def get_all_companies(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT payload_json FROM company_audits ORDER BY ticker ASC"
            )
            rows = cur.fetchall()
        return [json.loads(r[0]) for r in rows]

    def save_author_note(self, ticker: str, note: str) -> None:
        existing = self.get_company(ticker)
        if not existing:
            return
        existing["author_thesis"] = note.strip()
        self.upsert_company(existing)
