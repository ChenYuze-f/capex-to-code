"""
Damodaran Quality-Adjusted R&D Capitalization & Valuation Engine.

Implements multi-year R&D capitalization where only the productive share of R&D
(governed by the AI & Systems conversion rate `eta_conv`) forms an amortizable
research asset, while deadweight bureaucracy & AI-washing expenses remain written off.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from analyst_calibration import get_amortization_years

REGIME_VERIFIED_COMPOUNDER = "Verified AI Compounder (Undervalued)"
REGIME_QUALITY_MOAT = "Quality AI Moat (Fairly Priced)"
REGIME_AI_WASHING_RISK = "AI-Washing / Narrative Bubble Risk"
REGIME_DILUTION_TRAP = "R&D Dilution / Legacy Value Trap"


def _clamp(val: float, low: float, high: float) -> float:
    return max(low, min(high, val))


def classify_investment_regime(
    proof_score: float,
    awdi: float,
    valuation_spread_pct: float,
) -> str:
    """Classify an issuer into one of four institutional investment regimes."""
    if awdi >= 35.0:
        return REGIME_AI_WASHING_RISK
    if proof_score >= 68.0 and valuation_spread_pct >= 5.0:
        return REGIME_VERIFIED_COMPOUNDER
    if proof_score >= 68.0:
        return REGIME_QUALITY_MOAT
    return REGIME_DILUTION_TRAP


def compute_rd_capitalization_and_valuation(
    financials: Dict[str, Any],
    audit_metrics: Dict[str, Any],
    amortization_years: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute Damodaran Quality-Adjusted R&D Capitalization and Valuation Metrics.

    Equations:
      1. Productive R&D (t) = RD_t * eta_conv
      2. Deadweight R&D Waste (t) = RD_t * (1 - eta_conv)
      3. Research Capital Asset (RC_asset) = sum_{k=0..L-1} (RD_{t-k} * eta_conv) * (L - k) / L
      4. Annual R&D Amortization (A_RD) = sum_{k=1..L} (RD_{t-k} * eta_conv) / L
      5. Adjusted EBIT = EBIT_reported + Productive_RD_t - A_RD
      6. Adjusted Invested Capital = IC_reported + RC_asset
      7. Adjusted ROIC = EBIT_adj * (1 - tax_rate) / IC_adj
    """
    sector = str(financials.get("sector", ""))
    L = int(amortization_years or get_amortization_years(sector))
    L = max(2, min(7, L))

    raw_rd_list = [float(x) for x in financials.get("rd_history_b", []) if x is not None and float(x) >= 0]
    if not raw_rd_list:
        fallback_rd = float(financials.get("rd_current_b", 1.0))
        raw_rd_list = [max(0.01, fallback_rd)]

    # Back-cast historical R&D series to length L + 1 (t, t-1, ..., t-L) using 5% geometric taper
    rd_series: List[float] = list(raw_rd_list)
    while len(rd_series) < L + 1:
        rd_series.append(round(rd_series[-1] * 0.95, 4))
    rd_series = rd_series[: L + 1]

    rd_current_b = round(rd_series[0], 2)
    eta = _clamp(float(audit_metrics.get("conversion_rate", 0.60)), 0.15, 0.95)
    awdi = _clamp(float(audit_metrics.get("awdi", 0.0)), 0.0, 100.0)
    proof_score = float(audit_metrics.get("proof_score", 50.0))

    productive_rd_b = round(rd_current_b * eta, 2)
    deadweight_rd_b = round(rd_current_b * (1.0 - eta), 2)

    # Unamortized Research Capital Asset (k = 0 .. L-1)
    rc_asset = 0.0
    for k in range(L):
        weight_unamortized = (L - k) / float(L)
        rc_asset += (rd_series[k] * eta) * weight_unamortized
    rd_capital_asset_b = round(rc_asset, 2)

    # Current-year Amortization of past productive R&D (k = 1 .. L)
    amort = 0.0
    for k in range(1, L + 1):
        amort += (rd_series[k] * eta) / float(L)
    rd_amortization_b = round(amort, 2)

    ebit_reported_b = round(float(financials.get("ebit_reported_b", 1.0)), 2)
    ebit_adjusted_b = round(ebit_reported_b + productive_rd_b - rd_amortization_b, 2)

    ic_reported_b = max(0.5, round(float(financials.get("invested_capital_reported_b", 10.0)), 2))
    ic_adjusted_b = round(ic_reported_b + rd_capital_asset_b, 2)

    tax_rate = _clamp(float(financials.get("tax_rate", 0.21)), 0.05, 0.35)
    nopat_reported_b = ebit_reported_b * (1.0 - tax_rate)
    nopat_adjusted_b = ebit_adjusted_b * (1.0 - tax_rate)

    roic_reported_pct = round((nopat_reported_b / ic_reported_b) * 100.0, 1)
    roic_adjusted_pct = round((nopat_adjusted_b / ic_adjusted_b) * 100.0, 1)
    roic_delta_pct = round(roic_adjusted_pct - roic_reported_pct, 1)

    pe_market = max(2.0, round(float(financials.get("pe_market", 18.0)), 1))
    revenue_b = max(0.1, float(financials.get("revenue_b", 5.0)))

    # Safe denominator for EBIT accretion ratio when reported EBIT is negative or near zero
    safe_ebit_denom = max(ebit_reported_b, 0.12 * revenue_b, 0.05)
    if ebit_reported_b > 0:
        ebit_accretion_factor = _clamp(ebit_adjusted_b / safe_ebit_denom, 0.65, 1.45)
    else:
        ebit_accretion_factor = _clamp(1.0 + (productive_rd_b - rd_amortization_b) / safe_ebit_denom, 0.70, 1.35)

    # Incorporate both R&D quality accretion and valuation multiple discipline (penalizing extreme P/E > 40x)
    multiple_discipline_adj = _clamp((28.0 - pe_market) * 0.0045, -0.18, 0.12)
    roic_quality_adj = _clamp((roic_adjusted_pct - 15.0) * 0.003, -0.06, 0.10)
    quality_premium_factor = (
        1.0
        + 0.32 * (eta - 0.52)
        - 0.25 * (awdi / 100.0)
        + multiple_discipline_adj
        + roic_quality_adj
    )
    implied_fair_pe = round(
        _clamp(pe_market * ebit_accretion_factor * quality_premium_factor, 3.0, 95.0),
        1,
    )
    valuation_spread_pct = round(((implied_fair_pe - pe_market) / pe_market) * 100.0, 1)

    regime = classify_investment_regime(
        proof_score=proof_score,
        awdi=awdi,
        valuation_spread_pct=valuation_spread_pct,
    )

    return {
        "amortization_years": L,
        "rd_current_b": rd_current_b,
        "backcast_rd_series_b": [round(x, 2) for x in rd_series],
        "productive_rd_b": productive_rd_b,
        "deadweight_rd_b": deadweight_rd_b,
        "rd_capital_asset_b": rd_capital_asset_b,
        "rd_amortization_b": rd_amortization_b,
        "ebit_reported_b": ebit_reported_b,
        "ebit_adjusted_b": ebit_adjusted_b,
        "invested_capital_reported_b": ic_reported_b,
        "invested_capital_adjusted_b": ic_adjusted_b,
        "tax_rate": round(tax_rate, 2),
        "roic_reported_pct": roic_reported_pct,
        "roic_adjusted_pct": roic_adjusted_pct,
        "roic_delta_pct": roic_delta_pct,
        "pe_market": pe_market,
        "implied_fair_pe": implied_fair_pe,
        "valuation_spread_pct": valuation_spread_pct,
        "regime": regime,
    }
