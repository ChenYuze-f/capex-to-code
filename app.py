"""
CapEx-to-Code v2.0: Global R&D Capitalization & AI Reality Audit
Institutional Quantitative Equity & Technical Due Diligence Terminal
"""

from __future__ import annotations

import html
import math
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ai_audit_engine import (
    load_deepseek_api_key,
    run_deepseek_technical_audit,
    score_three_way_evidence,
)
from analyst_calibration import (
    AI_WASHING_SIGNALS,
    BUREAUCRACY_BLOAT_SIGNALS,
    CONTEMPORARY_AI_SIGNALS,
    DEFAULT_MODEL_WEIGHTS,
    SECTOR_AMORTIZATION_YEARS,
    SYSTEMS_ENGINEERING_SIGNALS,
)
from data_collectors import normalize_ticker_and_region, run_full_company_audit
from storage import AuditStorage
from valuation_model import (
    REGIME_AI_WASHING_RISK,
    REGIME_DILUTION_TRAP,
    REGIME_QUALITY_MOAT,
    REGIME_VERIFIED_COMPOUNDER,
    compute_rd_capitalization_and_valuation,
)

CURRENCY_SYMBOLS = {
    "USD": "$",
    "EUR": "€",
    "GBP": "£",
}


def _fmt_curr(val: float, currency: str) -> str:
    sym = CURRENCY_SYMBOLS.get(str(currency).upper(), "$")
    return f"{sym}{val:.2f}B"


def recompute_universe_dataframe(
    records: List[Dict[str, Any]],
    weights: Optional[Dict[str, float]] = None,
    amortization_override: Optional[int] = None,
) -> pd.DataFrame:
    """Recompute 3-way AI scores and Damodaran R&D valuation across all records."""
    rows: List[Dict[str, Any]] = []
    for rec in records:
        scores = score_three_way_evidence(rec, weights=weights)
        val = compute_rd_capitalization_and_valuation(
            rec,
            scores,
            amortization_years=amortization_override,
        )
        curr = rec.get("currency", "USD")
        sym = CURRENCY_SYMBOLS.get(curr, "$")
        synthesis = rec.get("audit_synthesis") or {}

        rows.append(
            {
                "Ticker": rec["ticker"],
                "Name": rec["name"],
                "Region": rec.get("region", "US"),
                "Country": rec.get("country", ""),
                "Sector": rec.get("sector", ""),
                "Currency": curr,
                "CurrSym": sym,
                "Claim_Score": scores["claim_score"],
                "AI_Stack_Score": scores["ai_stack_score"],
                "Systems_Stack_Score": scores["systems_stack_score"],
                "Proof_Score": scores["proof_score"],
                "Friction_Score": scores["friction_score"],
                "AWDI": scores["awdi"],
                "Conversion_Rate": scores["conversion_rate"],
                "Conversion_Pct": round(scores["conversion_rate"] * 100.0, 1),
                "Amortization_Years": val["amortization_years"],
                "RD_Current_B": val["rd_current_b"],
                "Productive_RD_B": val["productive_rd_b"],
                "Deadweight_RD_B": val["deadweight_rd_b"],
                "RD_Capital_Asset_B": val["rd_capital_asset_b"],
                "RD_Amortization_B": val["rd_amortization_b"],
                "EBIT_Reported_B": val["ebit_reported_b"],
                "EBIT_Adjusted_B": val["ebit_adjusted_b"],
                "IC_Reported_B": val["invested_capital_reported_b"],
                "IC_Adjusted_B": val["invested_capital_adjusted_b"],
                "ROIC_Reported_Pct": val["roic_reported_pct"],
                "ROIC_Adjusted_Pct": val["roic_adjusted_pct"],
                "ROIC_Delta_Pct": val["roic_delta_pct"],
                "PE_Market": val["pe_market"],
                "Implied_Fair_PE": val["implied_fair_pe"],
                "Valuation_Spread_Pct": val["valuation_spread_pct"],
                "Mkt_Cap_B": float(rec.get("mkt_cap_b", 10.0)),
                "Revenue_B": float(rec.get("revenue_b", 5.0)),
                "Regime": val["regime"],
                "HF_Models": int(rec.get("hf_models_count", 0)),
                "HF_Downloads": int(rec.get("hf_downloads", 0)),
                "OpenAlex_Papers": int(rec.get("openalex_papers_count", 0)),
                "OpenAlex_Citations": int(rec.get("openalex_citations", 0)),
                "GitHub_Repos": int(rec.get("github_repos_count", 0)),
                "GitHub_Stars": int(rec.get("github_stars", 0)),
                "Matched_AI": scores["matched_ai_signals"],
                "Matched_Systems": scores["matched_systems_signals"],
                "Matched_Claims": scores["matched_claims"],
                "Matched_Washing": scores["matched_washing_signals"],
                "Matched_Bloat": scores["matched_bloat_signals"],
                "Filing_Text": rec.get("filing_text", ""),
                "Tech_Texts": rec.get("tech_texts", []),
                "Insider_Texts": rec.get("insider_texts", []),
                "Audit_Mode": synthesis.get("mode", "Deterministic Quantitative Engine"),
                "Primary_Contradiction": synthesis.get("primary_contradiction", ""),
                "Executive_Verdict": synthesis.get("executive_verdict", ""),
                "Author_Thesis": rec.get("author_thesis", ""),
                "Last_Updated": rec.get("last_updated", ""),
                "Raw_Record": rec,
            }
        )
    return pd.DataFrame(rows)


def _chart_theme_tokens(theme_mode: str) -> Dict[str, str]:
    if theme_mode == "light":
        return {
            "text": "#0f172a",
            "muted": "#475569",
            "grid": "rgba(15, 23, 42, 0.09)",
            "line": "#64748b",
            "marker_border": "#0f172a",
        }
    return {
        "text": "#f8fafc",
        "muted": "#cbd5e1",
        "grid": "rgba(148, 163, 184, 0.15)",
        "line": "#94a3b8",
        "marker_border": "#ffffff",
    }


def _deoverlap_scatter_points(
    xs: List[float],
    ys: List[float],
    min_dx: float = 5.2,
    min_dy: float = 7.5,
    iterations: int = 40,
) -> Tuple[List[float], List[float], List[str]]:
    """
    Deterministic 2D repulsion solver that separates close scatter bubbles and
    assigns non-colliding label directions so text and circles never overlap.
    """
    n = len(xs)
    px = [float(x) for x in xs]
    py = [float(y) for y in ys]

    for _ in range(iterations):
        for i in range(n):
            for j in range(i + 1, n):
                dx = px[i] - px[j]
                dy = py[i] - py[j]
                if abs(dx) < min_dx and abs(dy) < min_dy:
                    # Compute normalized push vector
                    if abs(dx) < 0.01 and abs(dy) < 0.01:
                        angle = (i + 1) * 1.1
                        dx = math.cos(angle)
                        dy = math.sin(angle)
                    push_x = (min_dx - abs(dx)) * 0.32 * (1.0 if dx >= 0 else -1.0)
                    push_y = (min_dy - abs(dy)) * 0.32 * (1.0 if dy >= 0 else -1.0)
                    px[i] = max(4.0, min(97.5, px[i] + push_x))
                    px[j] = max(4.0, min(97.5, px[j] - push_x))
                    py[i] = max(-58.0, min(55.0, py[i] + push_y))
                    py[j] = max(-58.0, min(55.0, py[j] - push_y))

    # Assign label position based on nearest vertical neighbor to prevent label collision
    text_positions: List[str] = []
    for i in range(n):
        nearest_dy = 0.0
        min_dist = 1e9
        for j in range(n):
            if i == j:
                continue
            d = math.hypot((px[i] - px[j]) * 1.3, py[i] - py[j])
            if d < min_dist:
                min_dist = d
                nearest_dy = py[i] - py[j]
        if nearest_dy >= 0:
            text_positions.append("top center")
        else:
            text_positions.append("bottom center")

    return px, py, text_positions


def build_global_matrix_chart(
    df: pd.DataFrame,
    theme_mode: str = "light",
    highlight_ticker: Optional[str] = None,
) -> go.Figure:
    """
    Build a spacious, de-overlapped Global Valuation vs. Technical Proof Matrix.
    True unadjusted coordinates are shown in hover tooltips while display coordinates
    are gently separated so no two bubbles or labels ever collide.
    """
    tokens = _chart_theme_tokens(theme_mode)
    regime_colors = {
        REGIME_VERIFIED_COMPOUNDER: "#10b981",  # Emerald
        REGIME_QUALITY_MOAT: "#3b82f6",         # Royal Blue
        REGIME_AI_WASHING_RISK: "#ef4444",      # Crimson
        REGIME_DILUTION_TRAP: "#f59e0b",        # Amber
    }

    work_df = df.copy().reset_index(drop=True)
    disp_x, disp_y, disp_pos = _deoverlap_scatter_points(
        list(work_df["Proof_Score"].values),
        list(work_df["Valuation_Spread_Pct"].values),
    )
    work_df["Disp_X"] = disp_x
    work_df["Disp_Y"] = disp_y
    work_df["Text_Pos"] = disp_pos

    fig = go.Figure()

    y_min = min(-56.0, float(work_df["Disp_Y"].min()) - 12.0)
    y_max = max(44.0, float(work_df["Disp_Y"].max()) + 12.0)

    # Quadrant background shading
    fig.add_shape(
        type="rect",
        x0=68,
        x1=103,
        y0=0,
        y1=y_max,
        fillcolor="rgba(16, 185, 129, 0.06)",
        line=dict(width=0),
        layer="below",
    )
    fig.add_shape(
        type="rect",
        x0=0,
        x1=68,
        y0=y_min,
        y1=0,
        fillcolor="rgba(239, 68, 68, 0.05)",
        line=dict(width=0),
        layer="below",
    )

    fig.add_vline(x=68, line_width=1.4, line_dash="dash", line_color=tokens["line"])
    fig.add_hline(y=0, line_width=1.4, line_dash="dash", line_color=tokens["line"])

    fig.add_annotation(
        x=101,
        y=y_max - 2.0,
        xanchor="right",
        yanchor="top",
        text="<b>VERIFIED AI COMPOUNDER ZONE</b> (Proof ≥ 68, Fair P/E > Market P/E)",
        showarrow=False,
        font=dict(size=12, color="#10b981"),
    )
    fig.add_annotation(
        x=2,
        y=y_min + 2.0,
        xanchor="left",
        yanchor="bottom",
        text="<b>AI-WASHING & R&D DILUTION ZONE</b> (Proof < 68, Valuation Downside)",
        showarrow=False,
        font=dict(size=12, color="#ef4444"),
    )

    for regime_label in [
        REGIME_VERIFIED_COMPOUNDER,
        REGIME_QUALITY_MOAT,
        REGIME_AI_WASHING_RISK,
        REGIME_DILUTION_TRAP,
    ]:
        sub = work_df[work_df["Regime"] == regime_label]
        if sub.empty:
            continue

        # Moderate, well-proportioned bubble sizes (15px to 30px) to prevent giant overlaps
        sizes = [
            max(15.0, min(30.0, 12.0 + math.log1p(float(v)) * 4.6))
            for v in sub["RD_Current_B"]
        ]
        line_widths = [
            3.0 if (highlight_ticker and t == highlight_ticker) else 1.5
            for t in sub["Ticker"]
        ]

        custom = list(
            zip(
                sub["Name"],
                sub["Ticker"],
                sub["Region"],
                sub["AWDI"],
                sub["Conversion_Pct"],
                sub["ROIC_Adjusted_Pct"],
                sub["PE_Market"],
                sub["Implied_Fair_PE"],
                sub["CurrSym"],
                sub["RD_Current_B"],
                sub["Productive_RD_B"],
                sub["Proof_Score"],
                sub["Valuation_Spread_Pct"],
            )
        )

        fig.add_trace(
            go.Scatter(
                x=sub["Disp_X"],
                y=sub["Disp_Y"],
                mode="markers+text",
                name=regime_label,
                text=[f"<b>{t}</b>" for t in sub["Ticker"]],
                textposition=list(sub["Text_Pos"].values),
                textfont=dict(size=12.5, color=tokens["text"]),
                marker=dict(
                    size=sizes,
                    color=regime_colors.get(regime_label, "#64748b"),
                    opacity=0.90,
                    line=dict(width=line_widths, color=tokens["marker_border"]),
                ),
                customdata=custom,
                hovertemplate=(
                    "<b>%{customdata[0]} (%{customdata[1]} • %{customdata[2]})</b><br>"
                    "────────────────────────────────────<br>"
                    "Verifiable Technical Proof: <b>%{customdata[11]:.1f} / 100</b><br>"
                    "AI-Washing Divergence (AWDI): <b>%{customdata[3]:.1f} / 100</b><br>"
                    "Productive R&D Conversion (η): <b>%{customdata[4]:.1f}%</b><br>"
                    "Reported R&D → Productive Asset: <b>%{customdata[8]}%{customdata[9]:.2f}B → %{customdata[8]}%{customdata[10]:.2f}B</b><br>"
                    "Damodaran Adjusted ROIC: <b>%{customdata[5]:.1f}%</b><br>"
                    "Market P/E vs. Fair P/E: <b>%{customdata[6]:.1f}x → %{customdata[7]:.1f}x (%{customdata[12]:+.1f}%)</b>"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=dict(
            text="<b>Global R&D Technical Proof vs. Implied Valuation Spread</b> (De-Overlapped View • Hover for Exact Figures)",
            font=dict(size=16.5, color=tokens["text"]),
            x=0.0,
            y=0.96,
        ),
        font=dict(size=13, color=tokens["text"]),
        xaxis=dict(
            title=dict(
                text="<b>Verifiable Technical Proof Score (0–100)</b> — 65% Contemporary AI / 35% Computer Systems",
                font=dict(size=13.5, color=tokens["text"]),
            ),
            tickfont=dict(size=12.5, color=tokens["text"]),
            range=[0, 103],
            gridcolor=tokens["grid"],
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(
                text="<b>Implied Valuation Spread (%)</b> — (Fair P/E − Market P/E) / Market P/E",
                font=dict(size=13.5, color=tokens["text"]),
            ),
            tickfont=dict(size=12.5, color=tokens["text"]),
            range=[y_min, y_max],
            gridcolor=tokens["grid"],
            zeroline=False,
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.17,
            xanchor="center",
            x=0.5,
            font=dict(size=12.5, color=tokens["text"]),
        ),
        margin=dict(l=55, r=35, t=60, b=95),
        height=550,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def build_rd_capitalization_bridge(
    row: Dict[str, Any],
    theme_mode: str = "light",
) -> go.Figure:
    """Build a spacious Damodaran R&D Capitalization & EBIT Adjustment Bridge chart."""
    tokens = _chart_theme_tokens(theme_mode)
    sym = row.get("CurrSym", "$")
    rep_rd = float(row["RD_Current_B"])
    prod_rd = float(row["Productive_RD_B"])
    waste_rd = float(row["Deadweight_RD_B"])
    amort_rd = float(row["RD_Amortization_B"])
    rc_asset = float(row["RD_Capital_Asset_B"])

    categories = [
        "1. Reported Annual R&D Spend",
        "2. Productive AI/Tech Capitalized",
        "3. Deadweight Bureaucracy Write-Off",
        "4. Annual R&D Amortization (A_RD)",
        "5. Unamortized R&D Balance Sheet Asset",
    ]
    values = [rep_rd, prod_rd, waste_rd, amort_rd, rc_asset]
    colors = ["#64748b", "#10b981", "#ef4444", "#2563eb", "#0ea5e9"]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=categories,
            orientation="h",
            marker=dict(color=colors),
            text=[f"<b>{sym}{v:.2f}B</b>" for v in values],
            textposition="outside",
            textfont=dict(size=13.5, color=tokens["text"]),
            hovertemplate="<b>%{y}</b>: " + sym + "%{x:.2f}B<extra></extra>",
        )
    )
    max_x = max(values + [0.5]) * 1.28
    fig.update_layout(
        title=dict(
            text=f"<b>Damodaran R&D Capitalization Schedule — {row['Name']} ({row['Ticker']} • {row['Amortization_Years']}-Year Amortization)</b>",
            font=dict(size=16.5, color=tokens["text"]),
            x=0.0,
        ),
        font=dict(size=13.5, color=tokens["text"]),
        xaxis=dict(
            title=dict(text=f"Billions ({row.get('Currency', 'USD')})", font=dict(size=13.5, color=tokens["text"])),
            tickfont=dict(size=12.5, color=tokens["text"]),
            range=[0, max_x],
            gridcolor=tokens["grid"],
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(size=13.5, color=tokens["text"]),
        ),
        margin=dict(l=20, r=60, t=60, b=50),
        height=420,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def build_three_way_radar_bar(
    row: Dict[str, Any],
    theme_mode: str = "light",
) -> go.Figure:
    """Build a spacious horizontal bar chart comparing the 3-way sub-scores."""
    tokens = _chart_theme_tokens(theme_mode)
    labels = [
        "1. Executive Filing Claim Intensity (S_claim)",
        "2. Contemporary AI Execution (65% Priority Weight)",
        "3. Computer Systems Foundation (35% Weight)",
        "4. Composite Verifiable Proof (S_proof)",
        "5. Insider Friction & Wrapper Penalty (S_friction)",
        "6. AI-Washing Divergence Index (AWDI)",
    ]
    scores = [
        float(row["Claim_Score"]),
        float(row["AI_Stack_Score"]),
        float(row["Systems_Stack_Score"]),
        float(row["Proof_Score"]),
        float(row["Friction_Score"]),
        float(row["AWDI"]),
    ]
    colors = ["#0284c7", "#10b981", "#059669", "#2563eb", "#d97706", "#ef4444"]

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=labels,
            orientation="h",
            marker=dict(color=colors),
            text=[f"<b>{s:.1f} / 100</b>" for s in scores],
            textposition="outside",
            textfont=dict(size=13.5, color=tokens["text"]),
            hovertemplate="<b>%{y}</b>: %{x:.1f} / 100<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(
            text=f"<b>Three-Way AI Reality vs. Narrative Decomposition — {row['Name']} ({row['Ticker']})</b>",
            font=dict(size=16.5, color=tokens["text"]),
            x=0.0,
        ),
        font=dict(size=13.5, color=tokens["text"]),
        xaxis=dict(
            range=[0, 115],
            title=dict(text="Score (0–100 Scale)", font=dict(size=13.5, color=tokens["text"])),
            tickfont=dict(size=12.5, color=tokens["text"]),
            gridcolor=tokens["grid"],
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(size=13.5, color=tokens["text"]),
        ),
        margin=dict(l=20, r=60, t=60, b=50),
        height=420,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def _inject_theme_css(theme_mode: str) -> None:
    """Inject modern typography (`Plus Jakarta Sans` + `JetBrains Mono`) and Light/Dark theme variables."""
    if theme_mode == "light":
        bg_main = "#f8fafc"
        bg_sidebar = "#f1f5f9"
        bg_card = "#ffffff"
        bg_subtle = "#f1f5f9"
        text_main = "#0f172a"
        text_muted = "#475569"
        border_col = "#cbd5e1"
        accent_border = "#059669"
        hero_bg = "linear-gradient(135deg, #ffffff 0%, #f0fdf4 55%, #eff6ff 100%)"
        hero_border = "#bbf7d0"
    else:
        bg_main = "#0b0f19"
        bg_sidebar = "#111827"
        bg_card = "#161f30"
        bg_subtle = "#1e293b"
        text_main = "#f8fafc"
        text_muted = "#94a3b8"
        border_col = "#334155"
        accent_border = "#10b981"
        hero_bg = "linear-gradient(135deg, #0f172a 0%, #062c22 55%, #0f2547 100%)"
        hero_border = "#059669"

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;700&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');

        html, body, .stApp {{
            background-color: {bg_main} !important;
            color: {text_main} !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }}
        h1, h2, h3, .hero-headline, .summary-title {{
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }}
        code, [data-testid="stMetricValue"] {{
            font-family: 'JetBrains Mono', monospace !important;
        }}
        [data-testid="stHeader"] {{
            background-color: {bg_main} !important;
        }}
        [data-testid="stSidebar"] {{
            background-color: {bg_sidebar} !important;
            border-right: 1px solid {border_col} !important;
        }}
        [data-testid="stSidebar"] * {{
            color: {text_main} !important;
        }}
        .block-container {{
            padding-top: 3.4rem !important;
            padding-bottom: 3rem !important;
            max-width: 1280px !important;
        }}
        h1, h2, h3, h4, h5, h6, p, span, label, [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {{
            color: {text_main} !important;
        }}
        .hero-banner {{
            background: {hero_bg};
            border: 1px solid {hero_border};
            border-radius: 16px;
            padding: 2.6rem 2.8rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 10px 30px -10px rgba(16, 185, 129, 0.14);
        }}
        .hero-badge {{
            display: inline-block;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            padding: 5px 12px;
            border-radius: 999px;
            background: rgba(16, 185, 129, 0.16);
            color: #059669 !important;
            border: 1px solid rgba(16, 185, 129, 0.4);
            margin-bottom: 1rem;
        }}
        .gradient-accent {{
            background: linear-gradient(90deg, #059669 0%, #0284c7 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .clean-card {{
            background-color: {bg_card};
            border: 1px solid {border_col};
            border-radius: 12px;
            padding: 1.45rem 1.65rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
        }}
        .archetype-card-green {{
            background-color: {bg_card};
            border: 1px solid {border_col};
            border-top: 4px solid #10b981;
            border-radius: 12px;
            padding: 1.35rem 1.5rem;
            height: 100%;
        }}
        .archetype-card-red {{
            background-color: {bg_card};
            border: 1px solid {border_col};
            border-top: 4px solid #ef4444;
            border-radius: 12px;
            padding: 1.35rem 1.5rem;
            height: 100%;
        }}
        .archetype-card-amber {{
            background-color: {bg_card};
            border: 1px solid {border_col};
            border-top: 4px solid #f59e0b;
            border-radius: 12px;
            padding: 1.35rem 1.5rem;
            height: 100%;
        }}
        .summary-banner {{
            background-color: {bg_card};
            border: 1px solid {border_col};
            border-left: 5px solid {accent_border};
            border-radius: 12px;
            padding: 1.35rem 1.6rem;
            margin-bottom: 1.15rem;
        }}
        .summary-title {{
            font-size: 1.35rem;
            font-weight: 700;
            color: {text_main};
            margin-bottom: 0.35rem;
        }}
        .summary-meta {{
            font-size: 0.94rem;
            color: {text_muted};
            margin-bottom: 0.75rem;
        }}
        .summary-verdict {{
            font-size: 1.0rem;
            line-height: 1.6;
            color: {text_main};
            background-color: {bg_subtle};
            padding: 0.85rem 1.1rem;
            border-radius: 8px;
        }}
        .quote-box {{
            background-color: {bg_subtle};
            border-left: 3px solid {accent_border};
            padding: 0.8rem 1rem;
            border-radius: 6px;
            margin-bottom: 0.65rem;
            font-size: 0.93rem;
            line-height: 1.55;
            color: {text_main};
        }}
        .tag-pill {{
            display: inline-block;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.82rem;
            font-weight: 600;
            padding: 4px 10px;
            border-radius: 6px;
            margin-right: 6px;
            margin-bottom: 6px;
        }}
        .tag-pos {{
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.45);
            color: #059669 !important;
        }}
        .tag-neg {{
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid rgba(239, 68, 68, 0.45);
            color: #dc2626 !important;
        }}
        .tag-info {{
            background: rgba(37, 99, 235, 0.12);
            border: 1px solid rgba(37, 99, 235, 0.4);
            color: #2563eb !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_pills(items: List[str], css_class: str) -> str:
    if not items:
        return "<span style='font-size:0.9rem;opacity:0.65;'>None detected in sample</span>"
    return "".join(
        f"<span class='tag-pill {css_class}'>{html.escape(str(x))}</span>"
        for x in items
    )


def render_cover_view(store: AuditStorage, theme_mode: str) -> None:
    """Render Screen 0: Vibrant, high-impact Fintech & Quantitative Research Cover View."""
    top_l, top_r = st.columns([0.75, 0.25])
    with top_l:
        st.caption("CAPEX-TO-CODE // GLOBAL AI & R&D VALUATION ENGINE")
    with top_r:
        theme_btn = st.button(
            "🌙 Switch to Dark Mode" if theme_mode == "light" else "☀️ Switch to Light Mode",
            use_container_width=True,
        )
        if theme_btn:
            st.session_state["theme_mode"] = "dark" if theme_mode == "light" else "light"
            st.rerun()

    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-badge">GLOBAL AI REALITY &amp; R&amp;D VALUATION RADAR • US / UK / EU</div>
            <h1 class="hero-headline" style="font-size:2.45rem;font-weight:800;line-height:1.2;margin-bottom:0.9rem;">
                Separating <span class="gradient-accent">Frontier AI Moats</span> from
                Corporate <span style="color:#ef4444 !important;">AI-Washing &amp; R&amp;D Bloat</span>
            </h1>
            <p style="font-size:1.08rem;line-height:1.7;opacity:0.9;max-width:960px;margin-bottom:1.3rem;">
                Every year, public companies spend over <b>$500 Billion</b> on R&amp;D—yet standard accounting treats
                $1 spent on <b>CUDA kernels, Llama foundation models, and RISC-V silicon</b> identically to $1 wasted on
                <b>OpenAI API wrappers, external vendor bureaucracy, and slideware</b>. CapEx-to-Code cross-examines
                regulatory filings against live Hugging Face, OpenAlex, GitHub, and engineer forum telemetry to re-value
                global equities using Aswath Damodaran’s R&amp;D Capitalization framework.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Three Market Archetypes Exposed by Our Empirical Model")
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown(
            """
            <div class="archetype-card-green">
                <div style="font-family:monospace;font-size:0.8rem;font-weight:700;color:#10b981 !important;margin-bottom:0.3rem;">
                    ARCHETYPE 01 // VERIFIED AI COMPOUNDERS
                </div>
                <div style="font-size:1.15rem;font-weight:700;margin-bottom:0.4rem;">NVIDIA • Meta • Infineon</div>
                <div style="font-size:0.93rem;line-height:1.58;opacity:0.86;">
                    <b>High Proof (71–92) • Zero AI-Washing (AWDI = 0)</b><br>
                    Massive open-weight models (Llama), PyTorch/CUDA dominance, or RISC-V safety silicon. Over 78% of R&amp;D
                    converts into multi-year capital assets, unlocking substantial Adjusted ROIC and Fair P/E upside.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="archetype-card-red">
                <div style="font-family:monospace;font-size:0.8rem;font-weight:700;color:#ef4444 !important;margin-bottom:0.3rem;">
                    ARCHETYPE 02 // NARRATIVE OVERDRIVE &amp; WRAPPERS
                </div>
                <div style="font-size:1.15rem;font-weight:700;margin-bottom:0.4rem;">Palantir • IBM Consulting</div>
                <div style="font-size:0.93rem;line-height:1.58;opacity:0.86;">
                    <b>Extreme Filing Hype (82–96) • High AWDI Alert (44–88)</b><br>
                    Aggressive "AI-First / Autonomous Agent" filing rhetoric paired with sparse foundation-model output
                    or heavy integration consulting overhead. Triggers valuation multiple compression in our model.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="archetype-card-amber">
                <div style="font-family:monospace;font-size:0.8rem;font-weight:700;color:#f59e0b !important;margin-bottom:0.3rem;">
                    ARCHETYPE 03 // LEGACY R&amp;D DILUTION TRAP
                </div>
                <div style="font-size:1.15rem;font-weight:700;margin-bottom:0.4rem;">Volkswagen Group (€15.8B R&amp;D)</div>
                <div style="font-size:0.93rem;line-height:1.58;opacity:0.86;">
                    <b>Low Technical Proof (21.5) • Severe Friction (76+)</b><br>
                    Spends more on R&amp;D than Infineon and SAP combined, yet engineer telemetry reveals 80% vendor
                    coordination (<i>Dienstleistersteuerung</i>) and SAFe bloat—destroying R&amp;D capital formation.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown("#### Quick-Jump into a Live Company Audit or Enter the Full Terminal")
    q1, q2, q3, q4, q5, q6 = st.columns(6, gap="small")
    quick_picks = [
        (q1, "META", "🔥 Meta (US)"),
        (q2, "NVDA", "⚡ NVIDIA (US)"),
        (q3, "PLTR", "🚨 Palantir (US)"),
        (q4, "ARM", "🇬🇧 Arm (UK)"),
        (q5, "IFX.DE", "💎 Infineon (DE)"),
        (q6, "VOW3.DE", "⚠️ VW Group (DE)"),
    ]
    for col, tk, label in quick_picks:
        with col:
            if st.button(label, use_container_width=True):
                st.session_state["selected_ticker"] = tk
                st.session_state["view_mode"] = "terminal"
                st.rerun()

    st.write("")
    enter_l, enter_r = st.columns([0.65, 0.35], gap="medium")
    with enter_l:
        cover_search = st.text_input(
            "Or Type ANY Global Stock Ticker / Company Name to Audit Immediately:",
            value="",
            placeholder="e.g. AAPL, GOOGL, AMZN, AMD, ASML, BA.L, SAP.DE",
        )
    with enter_r:
        st.write("")
        if st.button("Launch Research Terminal →", type="primary", use_container_width=True):
            if cover_search.strip():
                st.session_state["pending_search_query"] = cover_search.strip()
            st.session_state["view_mode"] = "terminal"
            st.rerun()


def render_terminal_view(store: AuditStorage, theme_mode: str) -> None:
    """Render Screen 1: Clean, Spacious Progressive-Disclosure Research Terminal."""
    env_key = load_deepseek_api_key()

    # ──────────────────────────────────────────────────────────────────────────
    # SIDEBAR: THEME, MODEL SENSITIVITY & DEEPSEEK MODE
    # ──────────────────────────────────────────────────────────────────────────
    with st.sidebar:
        if st.button("← Back to Cover Page", use_container_width=True):
            st.session_state["view_mode"] = "cover"
            st.rerun()

        st.markdown("### Display Theme")
        sidebar_theme = st.radio(
            "Color Mode",
            options=["☀️ Light Mode", "🌙 Dark Mode"],
            index=0 if theme_mode == "light" else 1,
            horizontal=True,
            label_visibility="collapsed",
        )
        if ("Light" in sidebar_theme) != (theme_mode == "light"):
            st.session_state["theme_mode"] = "light" if "Light" in sidebar_theme else "dark"
            st.rerun()

        st.divider()
        st.markdown("### Model Calibration")
        ai_w = st.slider(
            "Contemporary AI Weight (S_AI)",
            min_value=0.40,
            max_value=0.85,
            value=float(DEFAULT_MODEL_WEIGHTS["ai_stack_weight"]),
            step=0.05,
        )
        sys_w = round(1.0 - ai_w, 2)
        st.caption(f"Computer Systems Weight: **{sys_w:.2f}**")

        awdi_w = st.slider(
            "AI-Washing Penalty Sensitivity",
            min_value=0.10,
            max_value=0.50,
            value=float(DEFAULT_MODEL_WEIGHTS["awdi_penalty_weight"]),
            step=0.05,
        )

        amort_override_opt = st.selectbox(
            "R&D Amortization Schedule",
            options=["Sector Default (3–6 yrs)", "Force 3 Years", "Force 4 Years", "Force 5 Years"],
            index=0,
        )
        amort_override: Optional[int] = None
        if "Force 3" in amort_override_opt:
            amort_override = 3
        elif "Force 4" in amort_override_opt:
            amort_override = 4
        elif "Force 5" in amort_override_opt:
            amort_override = 5

        st.divider()
        st.markdown("### DeepSeek Technical Mode")
        use_deepseek = st.toggle(
            "Enable DeepSeek Live Audit",
            value=bool(env_key),
        )
        st.caption(f"API Key: **{'Active (.env)' if env_key else 'Not Configured'}**")

    active_weights = {
        "ai_stack_weight": ai_w,
        "systems_stack_weight": sys_w,
        "proof_weight": 0.65,
        "culture_weight": 0.35,
        "awdi_penalty_weight": awdi_w,
    }

    # ──────────────────────────────────────────────────────────────────────────
    # UNIFIED COMPANY SEARCH & SELECTION BAR
    # ──────────────────────────────────────────────────────────────────────────
    head_l, head_r = st.columns([0.78, 0.22])
    with head_l:
        st.markdown("## CapEx-to-Code Research Terminal")
    with head_r:
        top_theme_btn = st.button(
            "🌙 Switch to Dark Mode" if theme_mode == "light" else "☀️ Switch to Light Mode",
            use_container_width=True,
        )
        if top_theme_btn:
            st.session_state["theme_mode"] = "dark" if theme_mode == "light" else "light"
            st.rerun()

    s_col1, s_col2, s_col3 = st.columns([0.52, 0.20, 0.28], gap="small")
    with s_col1:
        query_input = st.text_input(
            "Search or Analyze ANY Company (Type Ticker or Name, e.g., META, AAPL, AMZN, ASML, BA.L, SAP.DE):",
            value=st.session_state.pop("pending_search_query", ""),
            placeholder="Type any ticker or name (e.g. META, AMZN, AMD, ASML, ORCL) and click Analyze →",
        )
    with s_col2:
        st.write("")
        analyze_clicked = st.button("Analyze / Switch Company", type="primary", use_container_width=True)
    with s_col3:
        region_filter = st.selectbox(
            "Filter Saved Universe by Region:",
            options=["All Markets (Global)", "United States (US)", "United Kingdom (UK)", "Europe / Germany (EU)"],
            index=0,
        )

    raw_query = query_input.strip()
    if raw_query and (analyze_clicked or st.session_state.get("last_processed_query") != raw_query):
        st.session_state["last_processed_query"] = raw_query
        resolved_ticker, _, _ = normalize_ticker_and_region(raw_query)

        existing_all = store.get_all_companies()
        matched_existing = None
        for item in existing_all:
            if (
                item["ticker"].upper() == resolved_ticker
                or raw_query.lower() in item["name"].lower()
            ):
                matched_existing = item["ticker"]
                break

        if matched_existing:
            st.session_state["selected_ticker"] = matched_existing
        else:
            prog = st.progress(0, text=f"Fetching live financials & AI telemetry for {resolved_ticker}...")

            def _prog_cb(step: int, msg: str) -> None:
                prog.progress(min(95, step * 24), text=msg)

            new_rec = run_full_company_audit(
                ticker=resolved_ticker,
                use_deepseek=use_deepseek,
                api_key=env_key,
                weights=active_weights,
                progress_callback=_prog_cb,
            )
            store.upsert_company(new_rec)
            prog.progress(100, text=f"Loaded {new_rec['name']} ({new_rec['ticker']}) into database.")
            st.session_state["selected_ticker"] = new_rec["ticker"]

    all_records = store.get_all_companies()
    full_df = recompute_universe_dataframe(
        all_records,
        weights=active_weights,
        amortization_override=amort_override,
    )

    if "United States" in region_filter:
        df = full_df[full_df["Region"] == "US"].copy()
    elif "United Kingdom" in region_filter:
        df = full_df[full_df["Region"] == "UK"].copy()
    elif "Europe" in region_filter:
        df = full_df[full_df["Region"] == "EU"].copy()
    else:
        df = full_df.copy()

    if df.empty:
        df = full_df.copy()

    ticker_list = list(df["Ticker"].values)
    current_sel = st.session_state.get("selected_ticker", "NVDA")
    if current_sel not in ticker_list:
        if current_sel in list(full_df["Ticker"].values):
            df = full_df.copy()
            ticker_list = list(df["Ticker"].values)
        else:
            current_sel = ticker_list[0]

    selected_option = st.selectbox(
        "Currently Analyzed Companies (Click to switch, or use the search box above to add any new stock):",
        options=[
            f"{r['Ticker']} — {r['Name']} ({r['Region']} • {r['Regime']})"
            for _, r in df.iterrows()
        ],
        index=ticker_list.index(current_sel),
    )
    active_ticker = selected_option.split(" — ")[0].strip()
    st.session_state["selected_ticker"] = active_ticker
    focus_row = df[df["Ticker"] == active_ticker].iloc[0].to_dict()

    # ──────────────────────────────────────────────────────────────────────────
    # LAYER 1: SPACIOUS EXECUTIVE SUMMARY & 4 CORE KPI CARDS + INFO POPOVERS
    # ──────────────────────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="summary-banner">
            <div class="summary-title">{html.escape(focus_row['Name'])} ({html.escape(focus_row['Ticker'])}) — {html.escape(focus_row['Regime'])}</div>
            <div class="summary-meta">
                <b>Region:</b> {html.escape(focus_row['Country'])} ({html.escape(focus_row['Region'])}) &nbsp;|&nbsp;
                <b>Sector:</b> {html.escape(focus_row['Sector'])} &nbsp;|&nbsp;
                <b>Market Cap:</b> {_fmt_curr(focus_row['Mkt_Cap_B'], focus_row['Currency'])} &nbsp;|&nbsp;
                <b>R&amp;D Amortization Life:</b> {focus_row['Amortization_Years']} Years &nbsp;|&nbsp;
                <b>Audit Mode:</b> {html.escape(focus_row['Audit_Mode'])}
            </div>
            <div class="summary-verdict">
                <b>Executive Takeaway:</b> {html.escape(focus_row['Primary_Contradiction'])}
                <br><b>Valuation Verdict:</b> {html.escape(focus_row['Executive_Verdict'])}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4, gap="medium")
    with k1:
        st.metric(
            label="Productive R&D Conversion (η)",
            value=f"{focus_row['Conversion_Pct']:.1f}%",
            delta=f"{_fmt_curr(focus_row['Productive_RD_B'], focus_row['Currency'])} of {_fmt_curr(focus_row['RD_Current_B'], focus_row['Currency'])}",
            delta_color="normal" if focus_row["Conversion_Pct"] >= 60.0 else "inverse",
            help="Percentage of annual R&D that forms durable AI & systems assets instead of bureaucracy waste.",
        )
        with st.popover("ⓘ How is Productive R&D (η) defined?", use_container_width=True):
            st.markdown(
                """
                **Plain-English Definition**: The percentage of every 100 USD / EUR in reported R&D expenditure that converts into durable AI & computer-systems capital assets—rather than being diluted by external vendor coordination or narrative wrapper projects.
                
                **Mathematical Formula**:
                $$\\eta_{\\text{conv}} = \\text{clip}\\left(\\frac{0.65 \\cdot S_{\\text{proof}} + 0.35 \\cdot (100 - S_{\\text{friction}}) - 0.25 \\cdot \\text{AWDI}}{100},\\; 15\\%,\\; 95\\%\\right)$$
                
                **How to Interpret**:
                - **> 65% (Strong Capital Formation)**: High engineering output paired with low internal bureaucracy (`NVDA`, `META`, `IFX.DE`).
                - **< 45% (Severe Dilution Alert)**: Substantial R&D budget absorbed by external vendor coordination (`Dienstleistersteuerung`) or legacy maintenance (`VOW3.DE`).
                """
            )

    with k2:
        st.metric(
            label="AI-Washing Divergence (AWDI)",
            value=f"{focus_row['AWDI']:.1f} / 100",
            delta=f"Claim {focus_row['Claim_Score']:.0f} vs. Proof {focus_row['Proof_Score']:.0f}",
            delta_color="inverse" if focus_row["AWDI"] >= 35.0 else "normal",
            help="Measures the gap between management's AI hype in filings and verifiable technical proof.",
        )
        with st.popover("ⓘ How is AI-Washing (AWDI) defined?", use_container_width=True):
            st.markdown(
                """
                **Plain-English Definition**: Quantifies the divergence between executive AI narrative hype in regulatory filings ($S_{\\text{claim}}$) and verifiable open-weight models, research citations, and production systems code ($S_{\\text{proof}}$), amplified by internal engineering friction ($S_{\\text{friction}}$). Lower is better (`0` = zero hype gap).
                
                **Mathematical Formula**:
                $$\\text{AWDI} = \\max(0,\\; S_{\\text{claim}} - S_{\\text{proof}}) \\times \\left(1 + \\frac{S_{\\text{friction}}}{100}\\right)$$
                
                **How to Interpret**:
                - **0 – 20 (Grounded Execution)**: Verifiable engineering output matches or exceeds executive narrative claims.
                - **> 35 (AI-Washing Risk Alert)**: Aggressive filing rhetoric unsupported by primary model/systems artifacts and accompanied by elevated developer friction.
                """
            )

    with k3:
        st.metric(
            label="Damodaran Adjusted ROIC",
            value=f"{focus_row['ROIC_Adjusted_Pct']:.1f}%",
            delta=f"{focus_row['ROIC_Delta_Pct']:+.1f}% vs. GAAP ({focus_row['ROIC_Reported_Pct']:.1f}%)",
            delta_color="normal",
            help="After-tax Return on Invested Capital after capitalizing productive R&D assets.",
        )
        with st.popover("ⓘ How is Adjusted ROIC calculated?", use_container_width=True):
            st.markdown(
                """
                **Plain-English Definition**: Re-estimates Return on Invested Capital (ROIC) by treating **Productive R&D** as a multi-year intangible capital asset amortized over $L$ years (3–6 years by sector) following Professor Aswath Damodaran's capitalization framework.
                
                **Mathematical Formula**:
                $$\\text{EBIT}_{\\text{adj}} = \\text{EBIT}_{\\text{rep}} + R\\&D_t \\cdot \\eta_{\\text{conv}} - A_{\\text{RD}}$$
                $$\\text{ROIC}_{\\text{adj}} = \\frac{\\text{EBIT}_{\\text{adj}} \\cdot (1 - \\tau)}{\\text{IC}_{\\text{rep}} + RC_{\\text{asset}}}$$
                
                **How to Interpret**:
                - **Positive Delta vs. GAAP ROIC**: Indicates that immediate 100% R&D expensing under GAAP/IFRS understates the company's true economic profitability.
                """
            )

    with k4:
        st.metric(
            label="Implied Fair P/E vs. Market",
            value=f"{focus_row['Implied_Fair_PE']:.1f}x vs. {focus_row['PE_Market']:.1f}x",
            delta=f"{focus_row['Valuation_Spread_Pct']:+.1f}% Valuation Spread",
            delta_color="normal" if focus_row["Valuation_Spread_Pct"] >= 0 else "inverse",
            help="Model-implied fair Forward P/E multiple compared to current market Forward P/E.",
        )
        with st.popover("ⓘ How is Implied Fair P/E calculated?", use_container_width=True):
            st.markdown(
                """
                **Plain-English Definition**: Compares the model-implied fair Forward P/E multiple—adjusted for capitalized R&D earnings accretion, Adjusted ROIC quality, and AI-washing multiple compression—against the current market Forward P/E.
                
                **Mathematical Formula**:
                $$\\text{Valuation Spread (\\%)} = \\frac{\\text{Implied Fair P/E} - \\text{Market P/E}}{\\text{Market P/E}} \\times 100\\%$$
                
                **How to Interpret**:
                - **Spread > +10% (Undervalued Moat)**: Strong engineering capital formation not yet fully priced into consensus earnings multiples.
                - **Spread < 0% (Overvalued / Multiple Compression Risk)**: Market multiple reflects narrative premium or unadjusted R&D dilution.
                """
            )

    st.write("")

    # ──────────────────────────────────────────────────────────────────────────
    # SINGLE SPACIOUS FULL-WIDTH CHART WITH SEGMENTED SWITCHER
    # ──────────────────────────────────────────────────────────────────────────
    chart_hdr_l, chart_hdr_r = st.columns([0.74, 0.26])
    with chart_hdr_l:
        chart_mode = st.radio(
            "Select Primary Visual Chart:",
            options=[
                "1. Global Peer Matrix (Proof vs. Valuation Spread)",
                "2. Company R&D Capitalization Breakdown (Damodaran Schedule)",
                "3. Three-Way AI Score Decomposition (Claim vs. Proof vs. Friction)",
            ],
            horizontal=True,
        )
    with chart_hdr_r:
        st.write("")
        with st.popover("ⓘ How to Read This Chart", use_container_width=True):
            st.markdown(
                """
                **1. Global Peer Matrix (Proof vs. Valuation Spread)**:
                - **X-Axis (`Technical Proof Score`, 0–100)**: Measures verified AI & computer systems output across Hugging Face, OpenAlex, and GitHub.
                - **Y-Axis (`Implied Valuation Spread %`)**: Measures upside/downside between our Damodaran R&D-adjusted Fair P/E and current Market P/E.
                - **Top-Right Quadrant (`Verified AI Compounder Zone`)**: High verifiable engineering output paired with positive valuation upside.
                - *Note: Nearby bubbles apply a gentle 2D anti-collision layout offset for label legibility; hover over any bubble to inspect exact unadjusted figures.*
                """
            )

    if chart_mode.startswith("1."):
        st.plotly_chart(
            build_global_matrix_chart(df, theme_mode=theme_mode, highlight_ticker=focus_row["Ticker"]),
            use_container_width=True,
        )
    elif chart_mode.startswith("2."):
        st.plotly_chart(
            build_rd_capitalization_bridge(focus_row, theme_mode=theme_mode),
            use_container_width=True,
        )
    else:
        st.plotly_chart(
            build_three_way_radar_bar(focus_row, theme_mode=theme_mode),
            use_container_width=True,
        )

    st.divider()

    # ──────────────────────────────────────────────────────────────────────────
    # LAYER 2: CLICK-TO-EXPAND PROGRESSIVE DISCLOSURE BUTTONS
    # ──────────────────────────────────────────────────────────────────────────
    st.markdown("#### Explore Detailed Evidence & Financial Schedules (Click to Expand)")
    st.caption("Click any module below to expand or collapse detailed telemetry without cluttering the main view.")

    for key in ("show_evidence", "show_accounting", "show_workbench"):
        if key not in st.session_state:
            st.session_state[key] = False

    b1, b2, b3 = st.columns(3, gap="medium")
    with b1:
        lbl1 = "▼ Hide Three-Way Evidence & DeepSeek Audit" if st.session_state["show_evidence"] else "▶ 1. View Three-Way Evidence & DeepSeek Audit"
        if st.button(lbl1, use_container_width=True, type="primary" if st.session_state["show_evidence"] else "secondary"):
            st.session_state["show_evidence"] = not st.session_state["show_evidence"]
            st.rerun()
    with b2:
        lbl2 = "▼ Hide R&D Balance Sheet & Global Peer Table" if st.session_state["show_accounting"] else "▶ 2. View R&D Accounting & Global Peer Table"
        if st.button(lbl2, use_container_width=True, type="primary" if st.session_state["show_accounting"] else "secondary"):
            st.session_state["show_accounting"] = not st.session_state["show_accounting"]
            st.rerun()
    with b3:
        lbl3 = "▼ Hide Author Thesis & Parameter Calibration" if st.session_state["show_workbench"] else "▶ 3. View Author Thesis & Parameter Calibration"
        if st.button(lbl3, use_container_width=True, type="primary" if st.session_state["show_workbench"] else "secondary"):
            st.session_state["show_workbench"] = not st.session_state["show_workbench"]
            st.rerun()

    # --------------------------------------------------------------------------
    # EXPANDABLE PANEL 1: THREE-WAY EVIDENCE & DEEPSEEK TECHNICAL AUDIT
    # --------------------------------------------------------------------------
    if st.session_state["show_evidence"]:
        st.markdown("<div class='clean-card'>", unsafe_allow_html=True)
        ds_col1, ds_col2 = st.columns([0.65, 0.35])
        with ds_col1:
            st.markdown(f"### Three-Way Evidence Cross-Examination — {focus_row['Name']} ({focus_row['Ticker']})")
            st.caption(
                f"Hugging Face Models: **{focus_row['HF_Models']:,}** ({focus_row['HF_Downloads']:,} downloads)  |  "
                f"OpenAlex AI Papers: **{focus_row['OpenAlex_Papers']:,}** ({focus_row['OpenAlex_Citations']:,} citations)  |  "
                f"GitHub Repos: **{focus_row['GitHub_Repos']:,}** (★ {focus_row['GitHub_Stars']:,})"
            )
        with ds_col2:
            if st.button(
                f"⚡ Run DeepSeek Live Technical Audit ({focus_row['Ticker']})",
                type="primary",
                use_container_width=True,
            ):
                with st.spinner(f"Calling DeepSeek API (`deepseek-chat`) to cross-examine {focus_row['Name']}..."):
                    raw_rec = focus_row["Raw_Record"]
                    base_sc = score_three_way_evidence(raw_rec, weights=active_weights)
                    live_syn = run_deepseek_technical_audit(
                        company_name=focus_row["Name"],
                        ticker=focus_row["Ticker"],
                        raw_evidence=raw_rec,
                        base_scores=base_sc,
                        api_key=env_key,
                    )
                    raw_rec["audit_synthesis"] = live_syn
                    store.upsert_company(raw_rec)
                    st.rerun()

        ev1, ev2, ev3 = st.columns(3, gap="medium")
        with ev1:
            st.markdown(f"#### 1. Filing Claims (Score: {focus_row['Claim_Score']:.1f})")
            with st.popover("ⓘ How is Claim Score (S_claim) calculated?"):
                st.markdown(
                    "Measures the density of high-profile AI commitments (`GenAI Platform`, `Autonomous Agents`, `Foundation Model`, `AI-First`) across SEC 10-K/20-F filings and official annual reports on a `0–100` scale."
                )
            st.markdown(f"<div class='quote-box'>{html.escape(focus_row['Filing_Text'])}</div>", unsafe_allow_html=True)
            st.markdown("**Matched Claim Keywords:**")
            st.markdown(_render_pills(focus_row["Matched_Claims"], "tag-info"), unsafe_allow_html=True)

        with ev2:
            st.markdown(f"#### 2. Technical Proof (Score: {focus_row['Proof_Score']:.1f})")
            with st.popover("ⓘ How is Proof Score (S_proof) calculated?"):
                st.markdown(
                    "Synthesizes **65% Contemporary AI Proof** (Hugging Face open-weight models & downloads, OpenAlex research citations, and `CUDA/TensorRT/vLLM/PyTorch` stack signals) + **35% Computer Systems Proof** (GitHub engineering velocity and `Rust/C++20/RISC-V/Distributed Systems` stack density)."
                )
            for q in focus_row["Tech_Texts"]:
                st.markdown(f"<div class='quote-box'>{html.escape(str(q))}</div>", unsafe_allow_html=True)
            st.markdown("**Contemporary AI Signals (65% Weight):**")
            st.markdown(_render_pills(focus_row["Matched_AI"], "tag-pos"), unsafe_allow_html=True)
            st.markdown("**Computer Systems Signals (35% Weight):**")
            st.markdown(_render_pills(focus_row["Matched_Systems"], "tag-pos"), unsafe_allow_html=True)

        with ev3:
            st.markdown(f"#### 3. Community Telemetry (Score: {focus_row['Friction_Score']:.1f})")
            with st.popover("ⓘ How is Friction Score (S_friction) calculated?"):
                st.markdown(
                    "Captures anonymous developer telemetry from Hacker News and Reddit engineering forums, weighting **60% AI-Washing & Wrapper signals** (`API Wrapper`, `Rebranded BI`) and **40% Bureaucracy & Outsourcing friction** (`Dienstleistersteuerung`, `SAFe Agile Coach`)."
                )
            for q in focus_row["Insider_Texts"]:
                st.markdown(f"<div class='quote-box'>{html.escape(str(q))}</div>", unsafe_allow_html=True)
            st.markdown("**AI-Washing / Wrapper Signals:**")
            st.markdown(_render_pills(focus_row["Matched_Washing"], "tag-neg"), unsafe_allow_html=True)
            st.markdown("**Bureaucracy & Tech Debt Signals:**")
            st.markdown(_render_pills(focus_row["Matched_Bloat"], "tag-neg"), unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # EXPANDABLE PANEL 2: DAMODARAN ACCOUNTING SCHEDULE & GLOBAL PEER TABLE
    # --------------------------------------------------------------------------
    if st.session_state["show_accounting"]:
        st.markdown("<div class='clean-card'>", unsafe_allow_html=True)
        st.markdown(f"### Damodaran R&D Capitalization Accounting Bridge — {focus_row['Ticker']}")
        sym = focus_row["CurrSym"]
        acc_df = pd.DataFrame(
            [
                {
                    "Accounting Metric": "Annual R&D Expenditure (Current Year t)",
                    "GAAP / IFRS Reported": f"{sym}{focus_row['RD_Current_B']:.2f}B (100% Expensed)",
                    "Damodaran Quality-Adjusted": f"{sym}{focus_row['Productive_RD_B']:.2f}B Capitalized ({focus_row['Conversion_Pct']:.1f}%) / {sym}{focus_row['Deadweight_RD_B']:.2f}B Written Off",
                },
                {
                    "Accounting Metric": f"Unamortized Research Asset ({focus_row['Amortization_Years']}-Year Lookback)",
                    "GAAP / IFRS Reported": f"{sym}0.00B (Unrecognized on Balance Sheet)",
                    "Damodaran Quality-Adjusted": f"{sym}{focus_row['RD_Capital_Asset_B']:.2f}B Added to Invested Capital",
                },
                {
                    "Accounting Metric": "Operating Income (EBIT)",
                    "GAAP / IFRS Reported": f"{sym}{focus_row['EBIT_Reported_B']:.2f}B",
                    "Damodaran Quality-Adjusted": f"{sym}{focus_row['EBIT_Adjusted_B']:.2f}B (+Prod. R&D − {sym}{focus_row['RD_Amortization_B']:.2f}B Amortization)",
                },
                {
                    "Accounting Metric": "Invested Capital & After-Tax ROIC",
                    "GAAP / IFRS Reported": f"{sym}{focus_row['IC_Reported_B']:.2f}B → ROIC: {focus_row['ROIC_Reported_Pct']:.1f}%",
                    "Damodaran Quality-Adjusted": f"{sym}{focus_row['IC_Adjusted_B']:.2f}B → Adjusted ROIC: {focus_row['ROIC_Adjusted_Pct']:.1f}% ({focus_row['ROIC_Delta_Pct']:+.1f}%)",
                },
            ]
        )
        st.dataframe(acc_df, use_container_width=True, hide_index=True)

        st.markdown("### Cross-Sectional Global Peer Comparison Table")
        peer_df = df[
            [
                "Ticker",
                "Name",
                "Region",
                "Proof_Score",
                "Claim_Score",
                "Friction_Score",
                "AWDI",
                "Conversion_Pct",
                "RD_Current_B",
                "Productive_RD_B",
                "ROIC_Reported_Pct",
                "ROIC_Adjusted_Pct",
                "PE_Market",
                "Implied_Fair_PE",
                "Valuation_Spread_Pct",
                "Regime",
            ]
        ].sort_values(by="Proof_Score", ascending=False)
        st.dataframe(peer_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # EXPANDABLE PANEL 3: AUTHOR CONVICTION NOTE & CALIBRATION WORKBENCH
    # --------------------------------------------------------------------------
    if st.session_state["show_workbench"]:
        st.markdown("<div class='clean-card'>", unsafe_allow_html=True)
        st.markdown(f"### Personal Analyst Thesis & Calibration Workbench — {focus_row['Ticker']}")
        wb_l, wb_r = st.columns([0.55, 0.45], gap="large")
        with wb_l:
            edited_note = st.text_area(
                f"Your Personal Analyst Thesis for {focus_row['Name']} ({focus_row['Ticker']}):",
                value=focus_row["Author_Thesis"],
                height=140,
            )
            if st.button("Save Personal Thesis to SQLite Database", type="primary"):
                store.save_author_note(focus_row["Ticker"], edited_note)
                st.success(f"Saved personal thesis for {focus_row['Ticker']}.")
                st.rerun()
        with wb_r:
            st.markdown("**Sector R&D Economic Amortization Lives (`analyst_calibration.py`)**")
            amort_df = pd.DataFrame(
                [
                    {"Sector": k.title(), "Amortization Life": f"{v} Years", "Annual Rate": f"{100.0/v:.1f}%"}
                    for k, v in SECTOR_AMORTIZATION_YEARS.items()
                ]
            )
            st.dataframe(amort_df, use_container_width=True, hide_index=True, height=210)
        st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(
        page_title="CapEx-to-Code | Global R&D Capitalization & AI Audit",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if "theme_mode" not in st.session_state:
        st.session_state["theme_mode"] = "light"
    if "view_mode" not in st.session_state:
        st.session_state["view_mode"] = "cover"

    theme_mode = st.session_state["theme_mode"]
    _inject_theme_css(theme_mode)

    store = AuditStorage()
    if st.session_state["view_mode"] == "cover":
        render_cover_view(store, theme_mode=theme_mode)
    else:
        render_terminal_view(store, theme_mode=theme_mode)


if __name__ == "__main__":
    main()
