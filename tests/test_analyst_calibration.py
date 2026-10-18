import unittest

from analyst_calibration import (
    AI_WASHING_SIGNALS,
    AUTHOR_THESIS_NOTES,
    BUREAUCRACY_BLOAT_SIGNALS,
    CLAIM_HYPE_SIGNALS,
    CONTEMPORARY_AI_SIGNALS,
    DEFAULT_MODEL_WEIGHTS,
    SECTOR_AMORTIZATION_YEARS,
    SYSTEMS_ENGINEERING_SIGNALS,
    get_amortization_years,
    get_author_thesis,
)


class TestAnalystCalibration(unittest.TestCase):
    def test_default_weights_sum_and_bounds(self):
        self.assertAlmostEqual(
            DEFAULT_MODEL_WEIGHTS["ai_stack_weight"]
            + DEFAULT_MODEL_WEIGHTS["systems_stack_weight"],
            1.0,
            places=5,
        )
        self.assertAlmostEqual(
            DEFAULT_MODEL_WEIGHTS["proof_weight"]
            + DEFAULT_MODEL_WEIGHTS["culture_weight"],
            1.0,
            places=5,
        )
        self.assertGreater(DEFAULT_MODEL_WEIGHTS["awdi_penalty_weight"], 0.0)
        self.assertLessEqual(DEFAULT_MODEL_WEIGHTS["awdi_penalty_weight"], 1.0)

    def test_signal_dictionaries_non_empty_and_positive_weights(self):
        for signal_map in (
            CONTEMPORARY_AI_SIGNALS,
            SYSTEMS_ENGINEERING_SIGNALS,
            AI_WASHING_SIGNALS,
            BUREAUCRACY_BLOAT_SIGNALS,
            CLAIM_HYPE_SIGNALS,
        ):
            self.assertGreater(len(signal_map), 5)
            for term, weight in signal_map.items():
                self.assertIsInstance(term, str)
                self.assertTrue(term.strip())
                self.assertGreater(float(weight), 0.0)

    def test_frontier_ai_and_systems_keywords_present(self):
        for kw in ("vLLM", "TensorRT", "CUDA", "PyTorch", "FlashAttention", "Triton"):
            self.assertIn(kw, CONTEMPORARY_AI_SIGNALS)
        for kw in ("Rust", "C++20", "RISC-V", "Kubernetes", "Distributed Systems"):
            self.assertIn(kw, SYSTEMS_ENGINEERING_SIGNALS)

    def test_negative_bloat_and_washing_keywords_present(self):
        self.assertIn("API Wrapper", AI_WASHING_SIGNALS)
        self.assertIn("OpenAI Wrapper", AI_WASHING_SIGNALS)
        self.assertIn("Dienstleistersteuerung", BUREAUCRACY_BLOAT_SIGNALS)
        self.assertIn("SAFe Agile Coach", BUREAUCRACY_BLOAT_SIGNALS)
        self.assertIn("Autonomous Agents", CLAIM_HYPE_SIGNALS)

    def test_sector_amortization_lookup_known_sectors(self):
        self.assertEqual(get_amortization_years("Semiconductors"), 4)
        self.assertEqual(get_amortization_years("Software - Infrastructure"), 3)
        self.assertEqual(get_amortization_years("Automotive"), 5)
        self.assertEqual(get_amortization_years("Biotechnology"), 6)

    def test_sector_amortization_fallback_for_unknown_or_empty(self):
        self.assertEqual(get_amortization_years("Unknown Custom Sector"), 4)
        self.assertEqual(get_amortization_years(""), 4)
        for sector, yrs in SECTOR_AMORTIZATION_YEARS.items():
            self.assertTrue(3 <= yrs <= 6, f"Unexpected life {yrs} for {sector}")

    def test_author_thesis_lookup_case_insensitive_and_fallback(self):
        for tk in ("nvda", "NVDA", "ifx.de", "vow3.de"):
            thesis = get_author_thesis(tk)
            self.assertIsInstance(thesis, str)
            self.assertGreater(len(thesis), 25)
        fallback = get_author_thesis("UNLISTED_TICKER_XYZ")
        self.assertIsInstance(fallback, str)
        self.assertIn("UNLISTED_TICKER_XYZ", fallback)
        self.assertGreaterEqual(len(AUTHOR_THESIS_NOTES), 10)


if __name__ == "__main__":
    unittest.main()
