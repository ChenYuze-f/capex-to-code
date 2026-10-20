import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from data_collectors import normalize_ticker_and_region, run_full_company_audit
from storage import BASELINE_JSON_PATH, AuditStorage


class TestCollectorsAndStorage(unittest.TestCase):
    def test_normalize_ticker_and_region_explicit_suffixes(self):
        self.assertEqual(normalize_ticker_and_region(" nvda "), ("NVDA", "US", "USD"))
        self.assertEqual(normalize_ticker_and_region("sap.de"), ("SAP.DE", "EU", "EUR"))
        self.assertEqual(normalize_ticker_and_region("rel.l"), ("REL.L", "UK", "GBP"))
        self.assertEqual(normalize_ticker_and_region("asml.as"), ("ASML.AS", "EU", "EUR"))
        self.assertEqual(normalize_ticker_and_region("mc.pa"), ("MC.PA", "EU", "EUR"))

    def test_normalize_ticker_and_region_company_name_aliases(self):
        tk_meta, reg_meta, cur_meta = normalize_ticker_and_region("Meta Platforms")
        self.assertEqual((tk_meta, reg_meta, cur_meta), ("META", "US", "USD"))
        tk_vw, reg_vw, cur_vw = normalize_ticker_and_region("Volkswagen")
        self.assertEqual((tk_vw, reg_vw, cur_vw), ("VOW3.DE", "EU", "EUR"))
        tk_arm, reg_arm, _ = normalize_ticker_and_region("ARM")
        self.assertEqual((tk_arm, reg_arm), ("ARM", "UK"))

    def test_storage_seeds_all_14_baseline_companies(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = Path(tmpdir) / "test_audit.db"
            store = AuditStorage(db_path=db_file)
            companies = store.get_all_companies()
            tickers = {c["ticker"] for c in companies}
            self.assertGreaterEqual(len(companies), 14)
            expected = {
                "NVDA",
                "META",
                "MSFT",
                "GOOGL",
                "AAPL",
                "TSLA",
                "PLTR",
                "IBM",
                "ARM",
                "REL.L",
                "AZN",
                "SAP.DE",
                "IFX.DE",
                "VOW3.DE",
            }
            self.assertTrue(expected.issubset(tickers))

    def test_storage_idempotent_reinitialization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = Path(tmpdir) / "test_audit.db"
            store1 = AuditStorage(db_path=db_file)
            count1 = len(store1.get_all_companies())
            store2 = AuditStorage(db_path=db_file)
            count2 = len(store2.get_all_companies())
            self.assertEqual(count1, count2)

    def test_storage_upsert_and_retrieve_custom_company(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = Path(tmpdir) / "test_audit.db"
            store = AuditStorage(db_path=db_file)
            sample = store.get_company("NVDA")
            self.assertIsNotNone(sample)
            custom = dict(sample)
            custom["ticker"] = "AMD"
            custom["name"] = "Advanced Micro Devices, Inc."
            store.upsert_company(custom)
            loaded = store.get_company("amd")
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded["name"], "Advanced Micro Devices, Inc.")

    def test_custom_author_note_persists_across_connections(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = Path(tmpdir) / "test_audit.db"
            store = AuditStorage(db_path=db_file)
            store.save_author_note("NVDA", "Custom analyst note on Blackwell + TensorRT-LLM moat.")
            store_reopened = AuditStorage(db_path=db_file)
            nvda = store_reopened.get_company("NVDA")
            self.assertIsNotNone(nvda)
            self.assertEqual(nvda["author_thesis"], "Custom analyst note on Blackwell + TensorRT-LLM moat.")

    def test_run_full_company_audit_offline_resilience_with_mocked_collectors(self):
        with patch("data_collectors.collect_live_company_telemetry") as mock_col:
            mock_col.return_value = {
                "ticker": "ASML.AS",
                "name": "ASML Holding N.V.",
                "region": "EU",
                "country": "Netherlands",
                "sector": "Semiconductors",
                "currency": "EUR",
                "mkt_cap_b": 310.0,
                "revenue_b": 27.6,
                "rd_history_b": [4.3, 3.9, 3.3, 2.8, 2.5],
                "ebit_reported_b": 9.0,
                "invested_capital_reported_b": 28.0,
                "tax_rate": 0.18,
                "pe_market": 32.5,
                "hf_models_count": 2,
                "hf_downloads": 5000,
                "openalex_papers_count": 120,
                "openalex_citations": 3400,
                "github_repos_count": 35,
                "github_stars": 1200,
                "filing_text": "EUV lithography systems and computational metrology software.",
                "tech_texts": ["C++20 embedded lithography control and CUDA metrology acceleration."],
                "insider_texts": ["High precision physics and embedded C++ engineering."],
            }

            rec = run_full_company_audit(
                ticker="ASML.AS",
                use_deepseek=False,
            )
            self.assertEqual(rec["ticker"], "ASML.AS")
            self.assertEqual(rec["region"], "EU")
            self.assertIn("implied_fair_pe", rec)
            self.assertIn("proof_score", rec)
            self.assertIn("audit_synthesis", rec)

    def test_baseline_universe_json_schema_integrity(self):
        data = json.loads(Path(BASELINE_JSON_PATH).read_text(encoding="utf-8"))
        self.assertIsInstance(data, list)
        self.assertGreaterEqual(len(data), 14)
        required_keys = {
            "ticker",
            "name",
            "region",
            "sector",
            "rd_history_b",
            "ebit_reported_b",
            "filing_text",
            "tech_texts",
            "insider_texts",
        }
        for item in data:
            self.assertTrue(required_keys.issubset(item.keys()), f"Missing keys in {item.get('ticker')}")


if __name__ == "__main__":
    unittest.main()
