import unittest

from valuation_model import (
    REGIME_AI_WASHING_RISK,
    REGIME_DILUTION_TRAP,
    REGIME_QUALITY_MOAT,
    REGIME_VERIFIED_COMPOUNDER,
    classify_investment_regime,
    compute_rd_capitalization_and_valuation,
)


class TestValuationModel(unittest.TestCase):
    def test_damodaran_rd_capitalization_standard_case(self):
        financials = {
            "ticker": "IFX.DE",
            "sector": "Semiconductors",
            "currency": "EUR",
            "rd_history_b": [2.15, 2.00, 1.85, 1.70, 1.55],
            "ebit_reported_b": 3.40,
            "invested_capital_reported_b": 22.0,
            "tax_rate": 0.20,
            "pe_market": 16.4,
            "mkt_cap_b": 44.8,
            "revenue_b": 16.3,
        }
        audit = {
            "claim_score": 68.0,
            "proof_score": 84.0,
            "friction_score": 22.0,
            "awdi": 0.0,
            "conversion_rate": 0.82,
        }
        res = compute_rd_capitalization_and_valuation(financials, audit, amortization_years=4)
        self.assertEqual(res["amortization_years"], 4)
        self.assertEqual(res["productive_rd_b"], round(2.15 * 0.82, 2))
        self.assertEqual(res["deadweight_rd_b"], round(2.15 * (1 - 0.82), 2))
        self.assertGreater(res["rd_capital_asset_b"], 0)
        self.assertGreater(res["rd_amortization_b"], 0)
        self.assertGreater(res["ebit_adjusted_b"], res["ebit_reported_b"])
        self.assertGreater(res["roic_adjusted_pct"], 0)
        self.assertGreater(res["implied_fair_pe"], res["pe_market"])
        self.assertGreater(res["valuation_spread_pct"], 0)

    def test_linear_amortization_asset_and_expense_math_exact(self):
        # Uniform R&D of 10.0B/yr for L=4 years with eta=0.80 -> Productive = 8.0B/yr
        # Unamortized weights: k=0 (1.0), k=1 (0.75), k=2 (0.50), k=3 (0.25) -> sum = 2.5 * 8.0 = 20.0B
        # Annual amortization: 4 years * (8.0 / 4) = 8.0B -> Adjusted EBIT == Reported EBIT
        financials = {
            "ticker": "STEADY",
            "sector": "Semiconductors",
            "currency": "USD",
            "rd_history_b": [10.0, 10.0, 10.0, 10.0, 10.0],
            "ebit_reported_b": 20.0,
            "invested_capital_reported_b": 80.0,
            "tax_rate": 0.20,
            "pe_market": 20.0,
            "mkt_cap_b": 320.0,
            "revenue_b": 100.0,
        }
        audit = {
            "claim_score": 50.0,
            "proof_score": 80.0,
            "friction_score": 20.0,
            "awdi": 0.0,
            "conversion_rate": 0.80,
        }
        res = compute_rd_capitalization_and_valuation(financials, audit, amortization_years=4)
        self.assertAlmostEqual(res["rd_capital_asset_b"], 20.0, places=2)
        self.assertAlmostEqual(res["rd_amortization_b"], 8.0, places=2)
        self.assertAlmostEqual(res["ebit_adjusted_b"], 20.0, places=2)
        self.assertAlmostEqual(res["invested_capital_adjusted_b"], 100.0, places=2)

    def test_handles_short_rd_history_and_non_positive_ebit(self):
        financials = {
            "ticker": "NEWCO",
            "sector": "Software - Infrastructure",
            "currency": "USD",
            "rd_history_b": [1.20],
            "ebit_reported_b": -0.10,
            "invested_capital_reported_b": 5.0,
            "tax_rate": 0.21,
            "pe_market": 35.0,
            "mkt_cap_b": 18.0,
            "revenue_b": 3.5,
        }
        audit = {
            "claim_score": 80.0,
            "proof_score": 78.0,
            "friction_score": 25.0,
            "awdi": 2.5,
            "conversion_rate": 0.76,
        }
        res = compute_rd_capitalization_and_valuation(financials, audit, amortization_years=3)
        self.assertEqual(len(res["backcast_rd_series_b"]), 4)
        self.assertGreater(res["rd_capital_asset_b"], 0)
        self.assertGreater(res["implied_fair_pe"], 0)

    def test_handles_empty_rd_history_and_zero_invested_capital(self):
        financials = {
            "ticker": "EMPTY",
            "sector": "Unknown",
            "currency": "USD",
            "rd_history_b": [],
            "ebit_reported_b": 1.0,
            "invested_capital_reported_b": 0.0,
            "tax_rate": 0.25,
            "pe_market": 15.0,
            "mkt_cap_b": 10.0,
            "revenue_b": 5.0,
        }
        audit = {
            "claim_score": 40.0,
            "proof_score": 50.0,
            "friction_score": 30.0,
            "awdi": 0.0,
            "conversion_rate": 0.55,
        }
        res = compute_rd_capitalization_and_valuation(financials, audit, amortization_years=3)
        self.assertGreater(res["invested_capital_adjusted_b"], 0)
        self.assertTrue(4.5 <= res["implied_fair_pe"] <= 65.0)

    def test_tax_rate_clamping_and_after_tax_roic(self):
        financials = {
            "ticker": "TAX",
            "sector": "Software",
            "currency": "USD",
            "rd_history_b": [2.0, 1.8, 1.6, 1.4],
            "ebit_reported_b": 4.0,
            "invested_capital_reported_b": 10.0,
            "tax_rate": 0.95,  # Extreme outlier tax rate should be clamped
            "pe_market": 22.0,
            "mkt_cap_b": 50.0,
            "revenue_b": 12.0,
        }
        audit = {
            "claim_score": 60.0,
            "proof_score": 75.0,
            "friction_score": 20.0,
            "awdi": 0.0,
            "conversion_rate": 0.75,
        }
        res = compute_rd_capitalization_and_valuation(financials, audit)
        self.assertGreater(res["roic_adjusted_pct"], 0.0)

    def test_multiple_compression_under_severe_ai_washing(self):
        financials = {
            "ticker": "HYPE",
            "sector": "Software - Application",
            "currency": "USD",
            "rd_history_b": [0.5, 0.5, 0.5, 0.5],
            "ebit_reported_b": 0.8,
            "invested_capital_reported_b": 4.0,
            "tax_rate": 0.20,
            "pe_market": 72.0,
            "mkt_cap_b": 60.0,
            "revenue_b": 2.5,
        }
        audit = {
            "claim_score": 95.0,
            "proof_score": 38.0,
            "friction_score": 65.0,
            "awdi": 85.0,
            "conversion_rate": 0.25,
        }
        res = compute_rd_capitalization_and_valuation(financials, audit)
        self.assertLess(res["implied_fair_pe"], res["pe_market"])
        self.assertLess(res["valuation_spread_pct"], -20.0)
        self.assertEqual(res["regime"], REGIME_AI_WASHING_RISK)

    def test_classify_investment_regime_all_four_quadrants(self):
        self.assertEqual(
            classify_investment_regime(proof_score=85.0, awdi=0.0, valuation_spread_pct=15.0),
            REGIME_VERIFIED_COMPOUNDER,
        )
        self.assertEqual(
            classify_investment_regime(proof_score=80.0, awdi=10.0, valuation_spread_pct=-5.0),
            REGIME_QUALITY_MOAT,
        )
        self.assertEqual(
            classify_investment_regime(proof_score=40.0, awdi=55.0, valuation_spread_pct=-25.0),
            REGIME_AI_WASHING_RISK,
        )
        self.assertEqual(
            classify_investment_regime(proof_score=25.0, awdi=20.0, valuation_spread_pct=-15.0),
            REGIME_DILUTION_TRAP,
        )

    def test_amortization_override_changes_capitalized_asset(self):
        financials = {
            "ticker": "NVDA",
            "sector": "Semiconductors",
            "currency": "USD",
            "rd_history_b": [12.0, 10.0, 8.5, 7.0, 5.5, 4.5],
            "ebit_reported_b": 70.0,
            "invested_capital_reported_b": 65.0,
            "tax_rate": 0.15,
            "pe_market": 35.0,
            "mkt_cap_b": 3000.0,
            "revenue_b": 120.0,
        }
        audit = {
            "claim_score": 85.0,
            "proof_score": 91.0,
            "friction_score": 18.0,
            "awdi": 0.0,
            "conversion_rate": 0.82,
        }
        res_3yr = compute_rd_capitalization_and_valuation(financials, audit, amortization_years=3)
        res_5yr = compute_rd_capitalization_and_valuation(financials, audit, amortization_years=5)
        self.assertGreater(res_5yr["rd_capital_asset_b"], res_3yr["rd_capital_asset_b"])


if __name__ == "__main__":
    unittest.main()
