"""
CapEx-to-Code v2.0: Global R&D Capitalization & AI Reality Audit
Institutional Quantitative Equity & Multi-LLM Technical Due Diligence Terminal
"""

from __future__ import annotations

import html
import math
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ai_audit_engine import (
    LLM_PROVIDER_PRESETS,
    load_deepseek_api_key,
    load_provider_api_key,
    run_custom_llm_company_search,
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
            "text": "#0F172A",
            "muted": "#475569",
            "grid": "rgba(15, 23, 42, 0.07)",
            "line": "rgba(15, 23, 42, 0.24)",
            "marker_border": "#0F172A",
            "accent": "#059669",
            "secondary": "#2563EB",
            "neutral": "#64748B",
            "warn": "#D97706",
            "neg": "#E11D48",
            "quad_pos": "rgba(5, 150, 105, 0.05)",
            "quad_neg": "rgba(225, 29, 72, 0.04)",
        }
    return {
        "text": "#F8FAFC",
        "muted": "#94A3B8",
        "grid": "rgba(148, 163, 184, 0.10)",
        "line": "rgba(148, 163, 184, 0.28)",
        "marker_border": "#F8FAFC",
        "accent": "#10B981",
        "secondary": "#3B82F6",
        "neutral": "#64748B",
        "warn": "#F59E0B",
        "neg": "#F43F5E",
        "quad_pos": "rgba(16, 185, 129, 0.07)",
        "quad_neg": "rgba(244, 63, 94, 0.06)",
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
        REGIME_VERIFIED_COMPOUNDER: tokens["accent"],
        REGIME_QUALITY_MOAT: tokens["secondary"],
        REGIME_AI_WASHING_RISK: tokens["neg"],
        REGIME_DILUTION_TRAP: tokens["warn"],
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

    fig.add_shape(
        type="rect",
        x0=68,
        x1=103,
        y0=0,
        y1=y_max,
        fillcolor=tokens["quad_pos"],
        line=dict(width=0),
        layer="below",
    )
    fig.add_shape(
        type="rect",
        x0=0,
        x1=68,
        y0=y_min,
        y1=0,
        fillcolor=tokens["quad_neg"],
        line=dict(width=0),
        layer="below",
    )

    fig.add_vline(x=68, line_width=1.0, line_dash="dot", line_color=tokens["line"])
    fig.add_hline(y=0, line_width=1.0, line_dash="dot", line_color=tokens["line"])

    fig.add_annotation(
        x=101,
        y=y_max - 2.0,
        xanchor="right",
        yanchor="top",
        text="<b>Verified compounders</b> · Proof ≥ 68, Fair P/E above market",
        showarrow=False,
        font=dict(size=12, color=tokens["accent"]),
    )
    fig.add_annotation(
        x=2,
        y=y_min + 2.0,
        xanchor="left",
        yanchor="bottom",
        text="<b>Narrative &amp; dilution risk</b> · Proof &lt; 68, implied multiple downside",
        showarrow=False,
        font=dict(size=12, color=tokens["neg"]),
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

        sizes = [
            max(14.0, min(28.0, 11.5 + math.log1p(float(v)) * 4.4))
            for v in sub["RD_Current_B"]
        ]
        line_widths = [
            2.6 if (highlight_ticker and t == highlight_ticker) else 1.2
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
                textfont=dict(size=12, color=tokens["text"]),
                marker=dict(
                    size=sizes,
                    color=regime_colors.get(regime_label, tokens["neutral"]),
                    opacity=0.90,
                    line=dict(width=line_widths, color=tokens["marker_border"]),
                ),
                customdata=custom,
                hovertemplate=(
                    "<b>%{customdata[0]} (%{customdata[1]} · %{customdata[2]})</b><br>"
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
            text="<b>Technical Proof vs. Implied Valuation Spread</b> · Hover any security for unadjusted coordinates",
            font=dict(family="Space Grotesk, Plus Jakarta Sans, sans-serif", size=16, color=tokens["text"]),
            x=0.0,
            y=0.96,
        ),
        font=dict(family="Plus Jakarta Sans, -apple-system, BlinkMacSystemFont, sans-serif", size=12.5, color=tokens["text"]),
        xaxis=dict(
            title=dict(
                text="Verifiable Technical Proof Score (0–100) · 65% Contemporary AI, 35% Computer Systems",
                font=dict(size=13, color=tokens["muted"]),
            ),
            tickfont=dict(size=12, color=tokens["muted"]),
            range=[0, 103],
            gridcolor=tokens["grid"],
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(
                text="Implied Valuation Spread (%) · (Fair P/E − Market P/E) / Market P/E",
                font=dict(size=13, color=tokens["muted"]),
            ),
            tickfont=dict(size=12, color=tokens["muted"]),
            range=[y_min, y_max],
            gridcolor=tokens["grid"],
            zeroline=False,
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.16,
            xanchor="left",
            x=0.0,
            font=dict(size=12, color=tokens["text"]),
        ),
        margin=dict(l=50, r=30, t=55, b=90),
        height=540,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def build_rd_capitalization_bridge(
    row: Dict[str, Any],
    theme_mode: str = "light",
) -> go.Figure:
    """Build a high-contrast Damodaran R&D Capitalization & EBIT Adjustment Bridge chart."""
    tokens = _chart_theme_tokens(theme_mode)
    sym = row.get("CurrSym", "$")
    rep_rd = float(row["RD_Current_B"])
    prod_rd = float(row["Productive_RD_B"])
    waste_rd = float(row["Deadweight_RD_B"])
    amort_rd = float(row["RD_Amortization_B"])
    rc_asset = float(row["RD_Capital_Asset_B"])

    categories = [
        "Reported annual R&D expenditure",
        "Productive R&D capitalized",
        "Deadweight bureaucracy write-off",
        "Annual R&D amortization charge",
        "Unamortized R&D balance-sheet asset",
    ]
    values = [rep_rd, prod_rd, waste_rd, amort_rd, rc_asset]
    colors = [
        tokens["neutral"],
        tokens["accent"],
        tokens["neg"],
        tokens["secondary"],
        tokens["accent"],
    ]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=categories,
            orientation="h",
            marker=dict(color=colors),
            text=[f"<b>{sym}{v:.2f}B</b>" for v in values],
            textposition="outside",
            textfont=dict(size=13, color=tokens["text"]),
            hovertemplate="<b>%{y}</b>: " + sym + "%{x:.2f}B<extra></extra>",
        )
    )
    max_x = max(values + [0.5]) * 1.28
    fig.update_layout(
        title=dict(
            text=f"<b>Damodaran R&D Capitalization Schedule</b> · {row['Name']} ({row['Ticker']}, {row['Amortization_Years']}-year life)",
            font=dict(family="Space Grotesk, Plus Jakarta Sans, sans-serif", size=16, color=tokens["text"]),
            x=0.0,
        ),
        font=dict(family="Plus Jakarta Sans, -apple-system, BlinkMacSystemFont, sans-serif", size=13, color=tokens["text"]),
        xaxis=dict(
            title=dict(text=f"Billions ({row.get('Currency', 'USD')})", font=dict(size=12.5, color=tokens["muted"])),
            tickfont=dict(size=12, color=tokens["muted"]),
            range=[0, max_x],
            gridcolor=tokens["grid"],
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(size=13, color=tokens["text"]),
        ),
        margin=dict(l=20, r=60, t=55, b=45),
        height=400,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def build_three_way_radar_bar(
    row: Dict[str, Any],
    theme_mode: str = "light",
) -> go.Figure:
    """Build a high-contrast horizontal bar chart comparing the 3-way sub-scores."""
    tokens = _chart_theme_tokens(theme_mode)
    labels = [
        "Filing narrative claim intensity",
        "Contemporary AI execution (65% weight)",
        "Computer systems foundation (35% weight)",
        "Composite verifiable technical proof",
        "Developer friction & wrapper penalty",
        "AI-washing divergence index (AWDI)",
    ]
    scores = [
        float(row["Claim_Score"]),
        float(row["AI_Stack_Score"]),
        float(row["Systems_Stack_Score"]),
        float(row["Proof_Score"]),
        float(row["Friction_Score"]),
        float(row["AWDI"]),
    ]
    colors = [
        tokens["neutral"],
        tokens["accent"],
        tokens["accent"],
        tokens["secondary"],
        tokens["warn"],
        tokens["neg"],
    ]

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=labels,
            orientation="h",
            marker=dict(color=colors),
            text=[f"<b>{s:.1f}</b>" for s in scores],
            textposition="outside",
            textfont=dict(size=13, color=tokens["text"]),
            hovertemplate="<b>%{y}</b>: %{x:.1f} / 100<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(
            text=f"<b>Three-Way Evidence Decomposition</b> · {row['Name']} ({row['Ticker']})",
            font=dict(family="Space Grotesk, Plus Jakarta Sans, sans-serif", size=16, color=tokens["text"]),
            x=0.0,
        ),
        font=dict(family="Plus Jakarta Sans, -apple-system, BlinkMacSystemFont, sans-serif", size=13, color=tokens["text"]),
        xaxis=dict(
            range=[0, 115],
            title=dict(text="Score (0–100 scale)", font=dict(size=12.5, color=tokens["muted"])),
            tickfont=dict(size=12, color=tokens["muted"]),
            gridcolor=tokens["grid"],
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(size=13, color=tokens["text"]),
        ),
        margin=dict(l=20, r=60, t=55, b=45),
        height=400,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def _get_active_llm_config() -> Dict[str, Any]:
    """Retrieve the user's active BYOK Multi-Provider LLM configuration from session state."""
    if "byok_provider" not in st.session_state:
        st.session_state["byok_provider"] = "DeepSeek"
    if "byok_keys" not in st.session_state:
        st.session_state["byok_keys"] = {}
    if "byok_models" not in st.session_state:
        st.session_state["byok_models"] = {}
    if "byok_urls" not in st.session_state:
        st.session_state["byok_urls"] = {}
    if "byok_enabled" not in st.session_state:
        st.session_state["byok_enabled"] = True

    provider = st.session_state["byok_provider"]
    preset = LLM_PROVIDER_PRESETS.get(provider, LLM_PROVIDER_PRESETS["DeepSeek"])

    session_key = st.session_state["byok_keys"].get(provider, "").strip()
    env_key = load_provider_api_key(provider)
    effective_key = session_key or env_key

    model = st.session_state["byok_models"].get(provider, preset["default_model"]).strip() or preset["default_model"]
    base_url = st.session_state["byok_urls"].get(provider, preset["base_url"]).strip() or preset["base_url"]

    key_source = (
        "User Session Key"
        if session_key
        else ("Local .env Key" if env_key else "No Key (Deterministic Engine)")
    )

    return {
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "api_key": effective_key,
        "has_key": bool(effective_key),
        "key_source": key_source,
        "enabled": bool(st.session_state["byok_enabled"]),
        "badge": preset.get("badge", provider),
        "supports_web_search": bool(preset.get("supports_web_search", False)),
    }


def _render_byok_config_controls(location_prefix: str = "sidebar") -> Dict[str, Any]:
    """Render interactive BYOK Multi-LLM API Key & Endpoint configuration controls."""
    cfg = _get_active_llm_config()
    provider_names = list(LLM_PROVIDER_PRESETS.keys())

    selected_provider = st.selectbox(
        "LLM Provider / Search Engine",
        options=provider_names,
        index=provider_names.index(cfg["provider"]) if cfg["provider"] in provider_names else 0,
        key=f"{location_prefix}_provider_select",
    )
    if selected_provider != st.session_state["byok_provider"]:
        st.session_state["byok_provider"] = selected_provider
        st.rerun()

    preset = LLM_PROVIDER_PRESETS[selected_provider]
    current_session_key = st.session_state["byok_keys"].get(selected_provider, "")

    entered_key = st.text_input(
        f"{selected_provider} API Key (session-only, never saved to disk)",
        value=current_session_key,
        type="password",
        placeholder=preset.get("key_placeholder", "sk-..."),
        key=f"{location_prefix}_api_key_{selected_provider}",
    )
    if entered_key != current_session_key:
        st.session_state["byok_keys"][selected_provider] = entered_key.strip()
        st.rerun()

    model_options = list(preset["models"]) + ["Custom Model ID..."]
    stored_model = st.session_state["byok_models"].get(selected_provider, preset["default_model"])
    default_idx = model_options.index(stored_model) if stored_model in preset["models"] else 0

    chosen_model_opt = st.selectbox(
        "Model Name",
        options=model_options,
        index=default_idx,
        key=f"{location_prefix}_model_select_{selected_provider}",
    )

    if chosen_model_opt == "Custom Model ID..." or selected_provider == "OpenRouter / Custom Endpoint":
        custom_model = st.text_input(
            "Custom Model Identifier",
            value=stored_model,
            placeholder="e.g. deepseek-reasoner, gpt-4o, sonar-pro",
            key=f"{location_prefix}_custom_model_{selected_provider}",
        )
        st.session_state["byok_models"][selected_provider] = custom_model.strip() or preset["default_model"]
    else:
        st.session_state["byok_models"][selected_provider] = chosen_model_opt

    if selected_provider == "OpenRouter / Custom Endpoint":
        stored_url = st.session_state["byok_urls"].get(selected_provider, preset["base_url"])
        custom_url = st.text_input(
            "OpenAI-Compatible Chat Completions Endpoint URL",
            value=stored_url,
            placeholder="https://openrouter.ai/api/v1/chat/completions",
            key=f"{location_prefix}_custom_url_{selected_provider}",
        )
        st.session_state["byok_urls"][selected_provider] = custom_url.strip() or preset["base_url"]

    enabled_toggle = st.toggle(
        "Enable Live LLM Audit & Search",
        value=st.session_state["byok_enabled"],
        key=f"{location_prefix}_byok_enable_toggle",
    )
    st.session_state["byok_enabled"] = enabled_toggle

    updated_cfg = _get_active_llm_config()
    status_color = "#10B981" if updated_cfg["has_key"] else "#F59E0B"
    st.markdown(
        f"<div style='font-size:0.82rem;padding:0.5rem 0.7rem;border-radius:6px;"
        f"background:rgba(16,185,129,0.08);border-left:3px solid {status_color};margin-top:0.25rem;'>"
        f"<b>Gateway Status:</b> {html.escape(updated_cfg['key_source'])} · "
        f"<code>{html.escape(updated_cfg['model'])}</code></div>",
        unsafe_allow_html=True,
    )
    return updated_cfg


def _inject_theme_css(theme_mode: str) -> None:
    """
    Inject high-contrast modern quantitative fintech design system:
    `Space Grotesk` (bold geometric display & stats) + `Plus Jakarta Sans` (body & tables)
    + `JetBrains Mono` (tickers, metrics, API badges) with vibrant Emerald, Cobalt, and Rose accents.
    """
    if theme_mode == "light":
        bg_main = "#F3F6FA"
        bg_sidebar = "#FFFFFF"
        bg_card = "#FFFFFF"
        bg_subtle = "#EAF0F7"
        bg_obsidian = "#0B111E"
        text_main = "#0B111E"
        text_muted = "#475569"
        accent = "#059669"
        accent_bright = "#10B981"
        accent_soft = "rgba(5, 150, 105, 0.10)"
        cobalt = "#2563EB"
        cobalt_soft = "rgba(37, 99, 235, 0.10)"
        neg = "#E11D48"
        neg_soft = "rgba(225, 29, 72, 0.10)"
        warn = "#D97706"
        border_col = "#DCE4F0"
    else:
        bg_main = "#060911"
        bg_sidebar = "#0B101D"
        bg_card = "#0E1525"
        bg_subtle = "#151F34"
        bg_obsidian = "#090E1A"
        text_main = "#F8FAFC"
        text_muted = "#94A3B8"
        accent = "#10B981"
        accent_bright = "#34D399"
        accent_soft = "rgba(16, 185, 129, 0.15)"
        cobalt = "#3B82F6"
        cobalt_soft = "rgba(59, 130, 246, 0.16)"
        neg = "#F43F5E"
        neg_soft = "rgba(244, 63, 94, 0.16)"
        warn = "#F59E0B"
        border_col = "rgba(148, 163, 184, 0.15)"

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@600;700&display=swap');

        html, body, .stApp {{
            background-color: {bg_main} !important;
            color: {text_main} !important;
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
            font-variant-numeric: tabular-nums;
        }}
        h1, h2, h3, .display-h1, .display-h2, .bento-stat {{
            font-family: 'Space Grotesk', -apple-system, BlinkMacSystemFont, sans-serif !important;
            letter-spacing: -0.03em !important;
            color: {text_main} !important;
        }}
        [data-testid="stMetricValue"] {{
            font-family: 'Space Grotesk', sans-serif !important;
            font-variant-numeric: tabular-nums !important;
            font-weight: 700 !important;
            letter-spacing: -0.03em !important;
        }}
        /* Eliminate top header overlap and obstruction */
        header[data-testid="stHeader"] {{
            background: transparent !important;
            border-bottom: none !important;
            box-shadow: none !important;
            height: 2.25rem !important;
            pointer-events: none !important;
            z-index: 20 !important;
        }}
        header[data-testid="stHeader"] * {{
            pointer-events: auto !important;
        }}
        [data-testid="stDecoration"] {{
            display: none !important;
        }}
        [data-testid="stToolbar"] {{
            opacity: 0.65;
            transition: opacity 0.2s ease;
        }}
        [data-testid="stToolbar"]:hover {{
            opacity: 1;
        }}
        [data-testid="stSidebar"] {{
            background-color: {bg_sidebar} !important;
            border-right: 1px solid {border_col} !important;
        }}
        [data-testid="stSidebar"] * {{
            color: {text_main} !important;
        }}
        .block-container {{
            padding-top: 2.85rem !important;
            padding-bottom: 3.5rem !important;
            max-width: 1380px !important;
        }}
        p, span, label, [data-testid="stMetricLabel"] {{
            color: {text_main} !important;
        }}
        .stButton > button {{
            border-radius: 6px !important;
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            font-weight: 600 !important;
            letter-spacing: -0.01em !important;
            transition: all 140ms ease !important;
        }}
        .stButton > button[kind="primary"] {{
            background-color: {cobalt} !important;
            color: #FFFFFF !important;
            border: none !important;
        }}
        .stButton > button[kind="primary"]:hover {{
            background-color: {accent} !important;
            color: #FFFFFF !important;
        }}
        .stButton > button[kind="secondary"] {{
            background-color: {bg_card} !important;
            color: {text_main} !important;
            border: 1px solid {border_col} !important;
        }}
        .stTextInput > div > div > input, .stSelectbox > div > div, .stTextArea textarea {{
            border-radius: 6px !important;
            background-color: {bg_card} !important;
            color: {text_main} !important;
            border: 1px solid {border_col} !important;
        }}
        .top-nav-shell {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            background-color: {bg_card};
            border: 1px solid {border_col};
            border-radius: 10px;
            padding: 0.75rem 1.25rem;
            margin-bottom: 1.5rem;
        }}
        /* High-Impact Institutional Brand Header */
        .top-brand-box {{
            display: flex;
            align-items: center;
            padding: 0.15rem 0;
            user-select: none;
        }}
        .brand-text-col {{
            display: flex;
            flex-direction: column;
            justify-content: center;
            min-width: 0;
        }}
        .brand-title-line {{
            display: flex;
            align-items: center;
            gap: 0.65rem;
            flex-wrap: nowrap;
        }}
        .brand-main-title {{
            font-family: 'Space Grotesk', sans-serif !important;
            font-size: 1.68rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.035em !important;
            color: {text_main} !important;
            white-space: nowrap !important;
            line-height: 1.15 !important;
        }}
        .brand-to {{
            color: {accent} !important;
            font-weight: 800 !important;
            margin: 0 1px;
        }}
        .brand-pill-badge {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            font-weight: 700;
            padding: 2px 8px;
            border-radius: 6px;
            background-color: {cobalt_soft};
            color: {cobalt} !important;
            white-space: nowrap;
            letter-spacing: 0.04em;
        }}
        .brand-subtitle {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.73rem;
            font-weight: 600;
            color: {text_muted};
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 2px;
            white-space: nowrap;
        }}
        .byok-status-pill {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 6px;
            background-color: {accent_soft};
            color: {accent} !important;
            white-space: nowrap;
        }}
        .kicker-pill {{
            display: inline-block;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.76rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            padding: 5px 12px;
            border-radius: 999px;
            background-color: {cobalt_soft};
            color: {cobalt} !important;
            margin-bottom: 0.95rem;
        }}
        .display-h1 {{
            font-size: clamp(2.45rem, 3.8vw, 3.55rem);
            font-weight: 700 !important;
            line-height: 1.03;
            letter-spacing: -0.035em !important;
            margin: 0 0 1.15rem 0;
        }}
        .hl-emerald {{
            color: {accent} !important;
            border-bottom: 4px solid {accent_bright};
            padding-bottom: 2px;
        }}
        .hl-rose {{
            color: {neg} !important;
            font-weight: 700;
        }}
        .hl-cobalt {{
            color: {cobalt} !important;
            font-weight: 700;
        }}
        .hero-lead {{
            font-size: 1.04rem;
            line-height: 1.64;
            color: {text_muted};
            margin: 0 0 1.35rem 0;
            max-width: 60ch;
        }}
        .hero-stat-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.75rem;
            margin-bottom: 1.35rem;
        }}
        .hero-stat-card {{
            background-color: {bg_card};
            border: 1px solid {border_col};
            border-radius: 8px;
            padding: 0.85rem 1.0rem;
            border-top: 3px solid {cobalt};
        }}
        .hero-stat-card.emerald {{
            border-top-color: {accent_bright};
        }}
        .hero-stat-card.rose {{
            border-top-color: {neg};
        }}
        .hero-stat-num {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.55rem;
            font-weight: 700;
            line-height: 1.1;
            color: {text_main};
            margin-bottom: 0.2rem;
        }}
        .hero-stat-num.emerald {{
            color: {accent} !important;
        }}
        .hero-stat-num.rose {{
            color: {neg} !important;
        }}
        .hero-stat-num.cobalt {{
            color: {cobalt} !important;
        }}
        .hero-stat-label {{
            font-size: 0.78rem;
            font-weight: 600;
            color: {text_muted};
            line-height: 1.3;
        }}
        /* High-Contrast Dark Obsidian Telemetry Matrix (pops in both Light & Dark modes) */
        .obsidian-ledger {{
            background-color: {bg_obsidian};
            color: #F8FAFC !important;
            border-radius: 12px;
            padding: 1.45rem 1.55rem;
            border: 1px solid rgba(59, 130, 246, 0.28);
        }}
        .obsidian-ledger * {{
            color: #F8FAFC;
        }}
        .obsidian-header-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.3rem;
        }}
        .obsidian-title {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.08rem;
            font-weight: 700;
            color: #F8FAFC !important;
        }}
        .obsidian-tag {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            padding: 3px 8px;
            border-radius: 4px;
            background-color: rgba(16, 185, 129, 0.18);
            color: #34D399 !important;
        }}
        .obsidian-sub {{
            font-size: 0.82rem;
            color: #94A3B8 !important;
            margin-bottom: 1.0rem;
        }}
        .obsidian-table {{
            width: 100%;
            border-collapse: collapse;
            font-variant-numeric: tabular-nums;
            font-size: 0.87rem;
        }}
        .obsidian-table th {{
            text-align: left;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.73rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: #94A3B8 !important;
            padding: 0.5rem 0.45rem;
            border-bottom: 1px solid rgba(148, 163, 184, 0.18);
        }}
        .obsidian-table th.num, .obsidian-table td.num {{
            text-align: right;
        }}
        .obsidian-table td {{
            padding: 0.62rem 0.45rem;
            border-bottom: 1px solid rgba(148, 163, 184, 0.11);
            color: #F1F5F9 !important;
        }}
        .meter-wrap {{
            display: flex;
            align-items: center;
            justify-content: flex-end;
            gap: 0.45rem;
        }}
        .meter-track {{
            width: 54px;
            height: 6px;
            background-color: rgba(148, 163, 184, 0.2);
            border-radius: 99px;
            overflow: hidden;
        }}
        .meter-fill {{
            height: 100%;
            border-radius: 99px;
        }}
        .obs-pos {{
            color: #34D399 !important;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
        }}
        .obs-neg {{
            color: #FB7185 !important;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
        }}
        /* Bento Regime Grid */
        .bento-card {{
            background-color: {bg_card};
            border: 1px solid {border_col};
            border-radius: 10px;
            padding: 1.45rem 1.5rem;
            height: 100%;
            position: relative;
        }}
        .bento-card.emerald-top {{
            border-top: 4px solid {accent_bright};
        }}
        .bento-card.amber-top {{
            border-top: 4px solid {warn};
        }}
        .bento-card.rose-top {{
            border-top: 4px solid {neg};
        }}
        .bento-kicker {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.74rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.4rem;
        }}
        .bento-kicker.emerald {{ color: {accent} !important; }}
        .bento-kicker.amber {{ color: {warn} !important; }}
        .bento-kicker.rose {{ color: {neg} !important; }}
        .bento-stat {{
            font-size: 2.2rem;
            font-weight: 700;
            line-height: 1.05;
            margin-bottom: 0.25rem;
        }}
        .bento-title {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.15rem;
            font-weight: 700;
            color: {text_main};
            margin-bottom: 0.55rem;
        }}
        .bento-copy {{
            font-size: 0.91rem;
            line-height: 1.58;
            color: {text_muted};
            margin-bottom: 0.9rem;
        }}
        .bento-bars {{
            background-color: {bg_subtle};
            border-radius: 6px;
            padding: 0.65rem 0.8rem;
            font-size: 0.78rem;
            font-family: 'JetBrains Mono', monospace;
        }}
        .clean-card {{
            background-color: {bg_card};
            border: 1px solid {border_col};
            padding: 1.5rem 1.75rem;
            margin-bottom: 1.25rem;
            border-radius: 10px;
        }}
        .summary-banner {{
            background-color: {bg_card};
            border: 1px solid {border_col};
            border-left: 5px solid {cobalt};
            padding: 1.5rem 1.75rem;
            margin-bottom: 1.35rem;
            border-radius: 10px;
        }}
        .summary-title-row {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 0.6rem;
            margin-bottom: 0.45rem;
        }}
        .summary-title {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.55rem;
            font-weight: 700;
            color: {text_main};
        }}
        .summary-meta {{
            font-size: 0.9rem;
            color: {text_muted};
            margin-bottom: 0.95rem;
            font-variant-numeric: tabular-nums;
        }}
        .summary-verdict {{
            font-size: 0.95rem;
            line-height: 1.62;
            color: {text_main};
            background-color: {bg_subtle};
            padding: 1.0rem 1.2rem;
            border-radius: 8px;
            border-left: 3px solid {accent};
        }}
        .quote-box {{
            background-color: {bg_subtle};
            padding: 0.9rem 1.1rem;
            border-radius: 6px;
            margin-bottom: 0.7rem;
            font-size: 0.9rem;
            line-height: 1.56;
            color: {text_main};
            border-left: 3px solid {border_col};
        }}
        .tag-pill {{
            display: inline-block;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
            font-weight: 500;
            padding: 4px 10px;
            border-radius: 999px;
            margin-right: 6px;
            margin-bottom: 6px;
        }}
        .tag-pos {{
            background: {accent_soft};
            color: {accent} !important;
        }}
        .tag-neg {{
            background: {neg_soft};
            color: {neg} !important;
        }}
        .tag-info {{
            background: {cobalt_soft};
            color: {cobalt} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_pills(items: List[str], css_class: str) -> str:
    if not items:
        return "<span style='font-size:0.88rem;opacity:0.65;'>None detected in sample</span>"
    return "".join(
        f"<span class='tag-pill {css_class}'>{html.escape(str(x))}</span>"
        for x in items
    )


def render_cover_view(store: AuditStorage, theme_mode: str) -> None:
    """Render Screen 0: High-Contrast Fintech Command Center & Asymmetric Bento Grid."""
    llm_cfg = _get_active_llm_config()

    # Top Architectural Bar (Prominent Typographic Brand & Action Center)
    nav_brand, nav_byok, nav_term, nav_theme = st.columns([0.52, 0.19, 0.17, 0.12], gap="small")
    with nav_brand:
        engine_short = f"● {llm_cfg['provider']}"
        st.markdown(
            f"""
            <div class="top-brand-box">
                <div class="brand-text-col">
                    <div class="brand-title-line">
                        <span class="brand-main-title">CapEx<span class="brand-to">-to-</span>Code</span>
                        <span class="brand-pill-badge">v2.0 QUANT</span>
                        <span class="byok-status-pill">{html.escape(engine_short)}</span>
                    </div>
                    <div class="brand-subtitle">Empirical R&amp;D Capitalization &amp; Multi-LLM Due Diligence</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with nav_byok:
        st.write("")
        with st.popover("BYOK API Keys", width="stretch"):
            st.markdown("#### Multi-Provider LLM & Search Gateway")
            st.caption(
                "Plug in your own API key (OpenAI, DeepSeek, Perplexity Sonar Live Web Search, Gemini, Claude, or OpenRouter) to run live company intelligence searches and technical audits."
            )
            _render_byok_config_controls(location_prefix="cover_pop")
    with nav_term:
        st.write("")
        if st.button("Terminal View →", type="primary", width="stretch"):
            st.session_state["view_mode"] = "terminal"
            st.rerun()
    with nav_theme:
        st.write("")
        theme_btn = st.button(
            "Dark Mode" if theme_mode == "light" else "Light Mode",
            width="stretch",
        )
        if theme_btn:
            st.session_state["theme_mode"] = "dark" if theme_mode == "light" else "light"
            st.rerun()

    st.write("")

    all_records = store.get_all_companies()
    cover_df = recompute_universe_dataframe(all_records, weights=DEFAULT_MODEL_WEIGHTS)

    # Live hero callouts computed from the benchmark universe (no hardcoded figures)
    top_conv_row = cover_df.sort_values("Conversion_Pct", ascending=False).iloc[0]
    worst_spread_row = cover_df.sort_values("Valuation_Spread_Pct", ascending=True).iloc[0]
    hero_conv_val = f"{float(top_conv_row['Conversion_Pct']):.1f}%"
    hero_conv_lbl = f"Peak R&amp;D Asset Conversion ({html.escape(str(top_conv_row['Ticker']))})"
    hero_spread_val = f"{float(worst_spread_row['Valuation_Spread_Pct']):+.1f}%"
    hero_spread_lbl = f"Deepest Fair P/E Compression ({html.escape(str(worst_spread_row['Ticker']))})"
    n_engines = len(LLM_PROVIDER_PRESETS)

    hero_l, hero_r = st.columns([0.48, 0.52], gap="large")
    with hero_l:
        st.markdown(
            f"""
            <div class="kicker-pill">DAMODARAN R&amp;D CAPITALIZATION × MULTI-LLM TELEMETRY</div>
            <h1 class="display-h1">
                Capitalize <span class="hl-emerald">real code.</span><br>
                Write off <span class="hl-rose">AI narrative.</span>
            </h1>
            <p class="hero-lead">
                GAAP accounting treats <b>$1 billion</b> invested in CUDA kernels, open-weight foundation models,
                and RISC-V silicon identically to <b>$1 billion</b> burned on API wrappers and vendor coordination.
                Cross-examine SEC filings against live Hugging Face, OpenAlex, and GitHub telemetry—or connect
                your own <span class="hl-cobalt">OpenAI, Perplexity, DeepSeek, Gemini, or Claude API key</span>
                for live due diligence search.
            </p>
            <div class="hero-stat-grid">
                <div class="hero-stat-card emerald">
                    <div class="hero-stat-num emerald">{hero_conv_val}</div>
                    <div class="hero-stat-label">{hero_conv_lbl}</div>
                </div>
                <div class="hero-stat-card">
                    <div class="hero-stat-num cobalt">{n_engines} Engines</div>
                    <div class="hero-stat-label">BYOK LLM &amp; Live Web Search Gateway</div>
                </div>
                <div class="hero-stat-card rose">
                    <div class="hero-stat-num rose">{hero_spread_val}</div>
                    <div class="hero-stat-label">{hero_spread_lbl}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        search_col, btn_col = st.columns([0.66, 0.34], gap="small")
        with search_col:
            cover_search = st.text_input(
                "Audit any global public company",
                value="",
                placeholder="Enter Ticker or Company (e.g. NVDA, META, ASML, PLTR, SAP.DE)",
                label_visibility="collapsed",
            )
        with btn_col:
            if st.button("Run Live Audit →", type="primary", width="stretch"):
                if cover_search.strip():
                    st.session_state["pending_search_query"] = cover_search.strip()
                st.session_state["view_mode"] = "terminal"
                st.rerun()

    with hero_r:
        featured_tickers = ["NVDA", "META", "IFX.DE", "MSFT", "PLTR", "VOW3.DE"]
        sub_df = cover_df[cover_df["Ticker"].isin(featured_tickers)].copy()
        sub_df["sort_order"] = sub_df["Ticker"].apply(
            lambda x: featured_tickers.index(x) if x in featured_tickers else 99
        )
        sub_df = sub_df.sort_values("sort_order")

        rows_html = []
        for _, r in sub_df.iterrows():
            sym = r["CurrSym"]
            conv = float(r["Conversion_Pct"])
            awdi = float(r["AWDI"])
            spread = float(r["Valuation_Spread_Pct"])
            spread_cls = "obs-pos" if spread >= 0 else "obs-neg"
            bar_color = "#10B981" if conv >= 70 else ("#3B82F6" if conv >= 50 else "#F43F5E")
            awdi_badge_bg = "rgba(16,185,129,0.18)" if awdi < 20 else "rgba(244,63,94,0.22)"
            awdi_badge_col = "#34D399" if awdi < 20 else "#FB7185"

            rows_html.append(
                f"<tr>"
                f"<td><b style='font-family:JetBrains Mono,monospace;color:#FFFFFF;'>{html.escape(str(r['Ticker']))}</b> "
                f"<span style='color:#94A3B8;font-size:0.8rem;'>· {html.escape(str(r['Name'])[:15])}</span></td>"
                f"<td class='num' style='font-family:JetBrains Mono,monospace;'>{sym}{float(r['RD_Current_B']):.1f}B</td>"
                f"<td class='num'>"
                f"<div class='meter-wrap'>"
                f"<span style='font-family:JetBrains Mono,monospace;font-size:0.82rem;'>{conv:.0f}%</span>"
                f"<div class='meter-track'><div class='meter-fill' style='width:{min(100.0, conv):.0f}%;background:{bar_color};'></div></div>"
                f"</div>"
                f"</td>"
                f"<td class='num'><span style='font-family:JetBrains Mono,monospace;font-size:0.76rem;padding:2px 7px;border-radius:4px;background:{awdi_badge_bg};color:{awdi_badge_col};'>{awdi:.0f}</span></td>"
                f"<td class='num' style='font-family:JetBrains Mono,monospace;'>{float(r['ROIC_Adjusted_Pct']):.1f}%</td>"
                f"<td class='num {spread_cls}'>{spread:+.1f}%</td>"
                f"</tr>"
            )

        st.markdown(
            f"""
            <div class="obsidian-ledger">
                <div class="obsidian-header-row">
                    <div class="obsidian-title">Empirical R&amp;D Capitalization Telemetry</div>
                    <span class="obsidian-tag">LIVE 14-EQUITY BENCHMARK</span>
                </div>
                <div class="obsidian-sub">Damodaran multi-year R&amp;D amortization · Conversion η = Productive Asset / Reported R&amp;D</div>
                <table class="obsidian-table">
                    <thead>
                        <tr>
                            <th>Security</th>
                            <th class="num">R&amp;D Spend</th>
                            <th class="num">Productive η</th>
                            <th class="num">AWDI</th>
                            <th class="num">Adj. ROIC</th>
                            <th class="num">Fair P/E Spread</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(rows_html)}
                    </tbody>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

        q_cols = st.columns(6, gap="small")
        for col, tk in zip(q_cols, featured_tickers):
            with col:
                if st.button(tk, key=f"cover_quick_{tk}", width="stretch"):
                    st.session_state["selected_ticker"] = tk
                    st.session_state["view_mode"] = "terminal"
                    st.rerun()

    st.write("")
    st.markdown(
        """
        <div style="display:flex;align-items:baseline;justify-content:space-between;margin:1.2rem 0 0.85rem 0;">
            <h2 class="display-h2" style="font-size:1.65rem;margin:0;">
                Three Regimes of Corporate R&amp;D Capitalization
            </h2>
            <span style="font-family:'JetBrains Mono',monospace;font-size:0.8rem;opacity:0.75;">
                CROSS-EXAMINING 10-K CLAIMS × HUGGING FACE / GITHUB × ENGINEER FORUMS
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    b_col1, b_col2, b_col3 = st.columns(3, gap="medium")
    with b_col1:
        st.markdown(
            """
            <div class="bento-card emerald-top">
                <div class="bento-kicker emerald">REGIME I · VERIFIED COMPOUNDERS</div>
                <div class="bento-stat" style="color:#10B981 !important;">&gt; 78%</div>
                <div class="bento-title">Productive R&amp;D Asset Conversion</div>
                <p class="bento-copy">
                    <b>NVIDIA, Meta, and Infineon</b> convert the overwhelming majority of R&amp;D into open-weight foundation
                    models (Llama), CUDA/Triton kernels, and automotive RISC-V silicon. GAAP 100% immediate expensing
                    understates their true economic ROIC, unlocking positive Fair P/E valuation upside.
                </p>
                <div class="bento-bars">
                    Technical Proof: <b>71–92 / 100</b> &nbsp;·&nbsp; AWDI Hype Gap: <b>0.0</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with b_col2:
        st.markdown(
            """
            <div class="bento-card amber-top">
                <div class="bento-kicker amber">REGIME II · NARRATIVE OVERDRIVE</div>
                <div class="bento-stat" style="color:#F59E0B !important;">44–88</div>
                <div class="bento-title">AI-Washing Divergence Index (AWDI)</div>
                <p class="bento-copy">
                    <b>Palantir and IBM Consulting</b> exhibit intense executive filing claims (<i>Autonomous Agents</i>,
                    <i>AI-First Platform</i>) that outpace verifiable open-weight models or peer-reviewed systems artifacts,
                    triggering an institutional multiple compression discount in our valuation bridge.
                </p>
                <div class="bento-bars">
                    Filing Claim: <b>82–96 / 100</b> &nbsp;·&nbsp; <b>Multiple Compression Risk</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with b_col3:
        st.markdown(
            """
            <div class="bento-card rose-top">
                <div class="bento-kicker rose">REGIME III · BUREAUCRATIC DILUTION</div>
                <div class="bento-stat" style="color:#F43F5E !important;">€15.8B</div>
                <div class="bento-title">Headline R&amp;D with &lt;35% Conversion</div>
                <p class="bento-copy">
                    <b>Volkswagen Group (VOW3.DE)</b> outspends Infineon and SAP combined in nominal R&amp;D, yet
                    developer telemetry reveals heavy multi-tier supplier coordination (<i>Dienstleistersteuerung</i>)
                    and SAFe governance overhead, requiring large deadweight write-offs before capitalization.
                </p>
                <div class="bento-bars">
                    Technical Proof: <b>21.5 / 100</b> &nbsp;·&nbsp; Friction Penalty: <b>76+ / 100</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_terminal_view(store: AuditStorage, theme_mode: str) -> None:
    """Render Screen 1: Institutional Quantitative Research Terminal & BYOK Live Intelligence Search."""
    with st.sidebar:
        if st.button("← Back to Overview", width="stretch"):
            st.session_state["view_mode"] = "cover"
            st.rerun()

        st.markdown("### Appearance")
        sidebar_theme = st.radio(
            "Color Mode",
            options=["Light Mode", "Dark Mode"],
            index=0 if theme_mode == "light" else 1,
            horizontal=True,
            label_visibility="collapsed",
        )
        if ("Light" in sidebar_theme) != (theme_mode == "light"):
            st.session_state["theme_mode"] = "light" if "Light" in sidebar_theme else "dark"
            st.rerun()

        st.write("")
        st.markdown("### BYOK LLM & Search Gateway")
        st.caption("Connect your own API key to power live company search & technical due diligence.")
        llm_cfg = _render_byok_config_controls(location_prefix="sidebar")

        st.write("")
        st.markdown("### Quantitative Calibration")
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

    active_weights = {
        "ai_stack_weight": ai_w,
        "systems_stack_weight": sys_w,
        "proof_weight": 0.65,
        "culture_weight": 0.35,
        "awdi_penalty_weight": awdi_w,
    }

    head_brand, head_m, head_r = st.columns([0.62, 0.22, 0.16], gap="small")
    with head_brand:
        engine_short = f"● {llm_cfg['provider']}"
        st.markdown(
            f"""
            <div class="top-brand-box">
                <div class="brand-text-col">
                    <div class="brand-title-line">
                        <span class="brand-main-title">CapEx<span class="brand-to">-to-</span>Code</span>
                        <span class="brand-pill-badge">RESEARCH TERMINAL</span>
                        <span class="byok-status-pill">{html.escape(engine_short)}</span>
                    </div>
                    <div class="brand-subtitle">Damodaran Quality R&amp;D Capitalization &amp; Cross-Examination Matrix</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with head_m:
        st.write("")
        with st.popover("BYOK API Keys", width="stretch"):
            st.markdown("#### Multi-Provider LLM Gateway")
            _render_byok_config_controls(location_prefix="term_top")
    with head_r:
        st.write("")
        top_theme_btn = st.button(
            "Dark Mode" if theme_mode == "light" else "Light Mode",
            width="stretch",
        )
        if top_theme_btn:
            st.session_state["theme_mode"] = "dark" if theme_mode == "light" else "light"
            st.rerun()

    s_col1, s_col2, s_col3 = st.columns([0.52, 0.20, 0.28], gap="small")
    with s_col1:
        query_input = st.text_input(
            "Audit or switch security (Ticker or company name):",
            value=st.session_state.pop("pending_search_query", ""),
            placeholder="e.g. META, AAPL, AMZN, AMD, ASML, BA.L, SAP.DE",
        )
    with s_col2:
        st.write("")
        analyze_clicked = st.button("Audit Security", type="primary", width="stretch")
    with s_col3:
        region_filter = st.selectbox(
            "Universe Region:",
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
                use_deepseek=llm_cfg["enabled"] and llm_cfg["has_key"],
                api_key=llm_cfg["api_key"],
                weights=active_weights,
                progress_callback=_prog_cb,
                provider=llm_cfg["provider"],
                model=llm_cfg["model"],
                base_url=llm_cfg["base_url"],
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
        "Active Security in Universe:",
        options=[
            f"{r['Ticker']} — {r['Name']} ({r['Region']} · {r['Regime']})"
            for _, r in df.iterrows()
        ],
        index=ticker_list.index(current_sel),
    )
    active_ticker = selected_option.split(" — ")[0].strip()
    st.session_state["selected_ticker"] = active_ticker
    focus_row = df[df["Ticker"] == active_ticker].iloc[0].to_dict()

    st.markdown(
        f"""
        <div class="summary-banner">
            <div class="summary-title-row">
                <div class="summary-title">{html.escape(focus_row['Name'])} ({html.escape(focus_row['Ticker'])})</div>
                <span class="byok-live-pill">{html.escape(focus_row['Regime'])} · {html.escape(focus_row['Audit_Mode'])}</span>
            </div>
            <div class="summary-meta">
                Region: <b>{html.escape(focus_row['Country'])} ({html.escape(focus_row['Region'])})</b> &nbsp;·&nbsp;
                Sector: <b>{html.escape(focus_row['Sector'])}</b> &nbsp;·&nbsp;
                Market Cap: <b>{_fmt_curr(focus_row['Mkt_Cap_B'], focus_row['Currency'])}</b> &nbsp;·&nbsp;
                R&amp;D Amortization Life: <b>{focus_row['Amortization_Years']} Years</b>
            </div>
            <div class="summary-verdict">
                <b>Primary Finding:</b> {html.escape(focus_row['Primary_Contradiction'])}
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
        with st.popover("Methodology: Productive R&D (η)", width="stretch"):
            st.markdown(
                """
                **Definition**: The percentage of every 100 USD / EUR in reported R&D expenditure that converts into durable AI & computer-systems capital assets—rather than being diluted by external vendor coordination or narrative wrapper projects.
                
                **Formula**:
                $$\\eta_{\\text{conv}} = \\text{clip}\\left(\\frac{0.65 \\cdot S_{\\text{proof}} + 0.35 \\cdot (100 - S_{\\text{friction}}) - 0.25 \\cdot \\text{AWDI}}{100},\\; 15\\%,\\; 95\\%\\right)$$
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
        with st.popover("Methodology: AI-Washing (AWDI)", width="stretch"):
            st.markdown(
                """
                **Definition**: Quantifies the divergence between executive AI narrative hype in regulatory filings ($S_{\\text{claim}}$) and verifiable open-weight models, research citations, and production systems code ($S_{\\text{proof}}$), amplified by internal engineering friction ($S_{\\text{friction}}$).
                
                **Formula**:
                $$\\text{AWDI} = \\max(0,\\; S_{\\text{claim}} - S_{\\text{proof}}) \\times \\left(1 + \\frac{S_{\\text{friction}}}{100}\\right)$$
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
        with st.popover("Methodology: Adjusted ROIC", width="stretch"):
            st.markdown(
                """
                **Definition**: Re-estimates Return on Invested Capital (ROIC) by treating **Productive R&D** as a multi-year intangible capital asset amortized over $L$ years (3–6 years by sector) following Professor Aswath Damodaran's capitalization framework.
                
                **Formula**:
                $$\\text{EBIT}_{\\text{adj}} = \\text{EBIT}_{\\text{rep}} + RD_t \\cdot \\eta_{\\text{conv}} - A_{\\text{RD}}$$
                $$\\text{ROIC}_{\\text{adj}} = \\frac{\\text{EBIT}_{\\text{adj}} \\cdot (1 - \\tau)}{\\text{IC}_{\\text{rep}} + RC_{\\text{asset}}}$$
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
        with st.popover("Methodology: Implied Fair P/E", width="stretch"):
            st.markdown(
                """
                **Definition**: Compares the model-implied fair Forward P/E multiple—adjusted for capitalized R&D earnings accretion, Adjusted ROIC quality, and AI-washing multiple compression—against the current market Forward P/E.
                
                **Formula**:
                $$\\text{Valuation Spread (\\%)} = \\frac{\\text{Implied Fair P/E} - \\text{Market P/E}}{\\text{Market P/E}} \\times 100\\%$$
                """
            )

    st.write("")

    chart_hdr_l, chart_hdr_r = st.columns([0.76, 0.24])
    with chart_hdr_l:
        chart_mode = st.radio(
            "Primary Visualization:",
            options=[
                "Peer Valuation Matrix (Proof vs. Spread)",
                "Damodaran R&D Capitalization Schedule",
                "Three-Way Evidence Decomposition",
            ],
            horizontal=True,
        )
    with chart_hdr_r:
        st.write("")
        with st.popover("Chart Reading Guide", width="stretch"):
            st.markdown(
                """
                **Peer Valuation Matrix (Proof vs. Spread)**:
                - **Horizontal Axis (`Technical Proof Score`, 0–100)**: Measures verified AI & computer systems output across Hugging Face, OpenAlex, and GitHub.
                - **Vertical Axis (`Implied Valuation Spread %`)**: Measures upside/downside between our Damodaran R&D-adjusted Fair P/E and current Market P/E.
                - **Upper-Right Quadrant (`Verified Compounders`)**: High verifiable engineering output paired with positive valuation upside.
                """
            )

    if chart_mode.startswith("Peer"):
        st.plotly_chart(
            build_global_matrix_chart(df, theme_mode=theme_mode, highlight_ticker=focus_row["Ticker"]),
            width="stretch",
        )
    elif chart_mode.startswith("Damodaran"):
        st.plotly_chart(
            build_rd_capitalization_bridge(focus_row, theme_mode=theme_mode),
            width="stretch",
        )
    else:
        st.plotly_chart(
            build_three_way_radar_bar(focus_row, theme_mode=theme_mode),
            width="stretch",
        )

    st.write("")
    st.markdown("#### Deep-Dive Modules & BYOK Live LLM Intelligence Search")

    for key in ("show_llm_search", "show_evidence", "show_accounting", "show_workbench"):
        if key not in st.session_state:
            st.session_state[key] = (key == "show_llm_search")

    b0, b1, b2, b3 = st.columns(4, gap="small")
    with b0:
        lbl0 = "Hide Live LLM Search" if st.session_state["show_llm_search"] else "BYOK Live LLM Search"
        if st.button(lbl0, width="stretch", type="primary" if st.session_state["show_llm_search"] else "secondary"):
            st.session_state["show_llm_search"] = not st.session_state["show_llm_search"]
            st.rerun()
    with b1:
        lbl1 = "Hide Three-Way Evidence" if st.session_state["show_evidence"] else "Three-Way Evidence & Audit"
        if st.button(lbl1, width="stretch", type="primary" if st.session_state["show_evidence"] else "secondary"):
            st.session_state["show_evidence"] = not st.session_state["show_evidence"]
            st.rerun()
    with b2:
        lbl2 = "Hide R&D Balance Sheet" if st.session_state["show_accounting"] else "R&D Schedule & Peer Ledger"
        if st.button(lbl2, width="stretch", type="primary" if st.session_state["show_accounting"] else "secondary"):
            st.session_state["show_accounting"] = not st.session_state["show_accounting"]
            st.rerun()
    with b3:
        lbl3 = "Hide Analyst Workbench" if st.session_state["show_workbench"] else "Analyst Thesis & Parameters"
        if st.button(lbl3, width="stretch", type="primary" if st.session_state["show_workbench"] else "secondary"):
            st.session_state["show_workbench"] = not st.session_state["show_workbench"]
            st.rerun()

    # MODULE 0: BYOK LIVE LLM INTELLIGENCE SEARCH & CUSTOM QUERY CONSOLE
    if st.session_state["show_llm_search"]:
        st.markdown("<div class='clean-card'>", unsafe_allow_html=True)
        top_s1, top_s2 = st.columns([0.68, 0.32], gap="medium")
        with top_s1:
            st.markdown(
                f"### BYOK Multi-LLM Live Intelligence Search · {focus_row['Name']} ({focus_row['Ticker']})"
            )
            st.caption(
                f"Active Engine: **{llm_cfg['provider']}** (`{llm_cfg['model']}`) · Status: **{llm_cfg['key_source']}** · "
                "Enter your API key in the sidebar or top bar to query OpenAI, Perplexity Sonar (Live Web Search), DeepSeek, Gemini, or Claude."
            )
        with top_s2:
            if st.button(
                f"Re-Score {focus_row['Ticker']} with {llm_cfg['provider']} Audit",
                type="primary",
                width="stretch",
            ):
                with st.spinner(f"Running live {llm_cfg['provider']} ({llm_cfg['model']}) technical audit on {focus_row['Name']}..."):
                    raw_rec = focus_row["Raw_Record"]
                    base_sc = score_three_way_evidence(raw_rec, weights=active_weights)
                    live_syn = run_deepseek_technical_audit(
                        company_name=focus_row["Name"],
                        ticker=focus_row["Ticker"],
                        raw_evidence=raw_rec,
                        base_scores=base_sc,
                        api_key=llm_cfg["api_key"],
                        provider=llm_cfg["provider"],
                        model=llm_cfg["model"],
                        base_url=llm_cfg["base_url"],
                    )
                    raw_rec["audit_synthesis"] = live_syn
                    store.upsert_company(raw_rec)
                    st.rerun()

        preset_q1, preset_q2, preset_q3 = st.columns(3, gap="small")
        default_q_key = f"custom_llm_query_{focus_row['Ticker']}"
        if default_q_key not in st.session_state:
            st.session_state[default_q_key] = (
                f"Search and evaluate {focus_row['Name']} ({focus_row['Ticker']}) AI CapEx efficiency, "
                f"open-source/research moat vs. competitors, and key engineering bottlenecks."
            )

        with preset_q1:
            if st.button("Preset: AI CapEx vs. Real Code Moat", width="stretch"):
                st.session_state[default_q_key] = (
                    f"Analyze {focus_row['Name']} ({focus_row['Ticker']}) recent AI CapEx and R&D spending: "
                    f"how much converts into proprietary silicon/models vs. third-party API wrappers?"
                )
                st.rerun()
        with preset_q2:
            if st.button("Preset: Developer Friction & Tech Debt", width="stretch"):
                st.session_state[default_q_key] = (
                    f"Search engineering community sentiment and technical architecture for {focus_row['Name']} ({focus_row['Ticker']}): "
                    f"what are the primary software bottlenecks, outsourcing risks, or execution advantages?"
                )
                st.rerun()
        with preset_q3:
            if st.button("Preset: 2026 Valuation & ROIC Catalyst", width="stretch"):
                st.session_state[default_q_key] = (
                    f"What are the most critical technical and financial catalysts for {focus_row['Name']} ({focus_row['Ticker']}) "
                    f"that justify a higher or lower Fair P/E multiple under R&D capitalization?"
                )
                st.rerun()

        q_in_col, q_btn_col = st.columns([0.76, 0.24], gap="small")
        with q_in_col:
            custom_research_q = st.text_input(
                "Custom Research Prompt / Information Search Query:",
                value=st.session_state[default_q_key],
                label_visibility="collapsed",
            )
            st.session_state[default_q_key] = custom_research_q
        with q_btn_col:
            run_search_btn = st.button(
                f"Search with {llm_cfg['provider']} →",
                type="primary",
                width="stretch",
            )

        search_cache_key = f"llm_search_result_{focus_row['Ticker']}"
        if run_search_btn:
            with st.spinner(f"Querying {llm_cfg['provider']} ({llm_cfg['model']}) with grounded telemetry for {focus_row['Ticker']}..."):
                raw_rec = focus_row["Raw_Record"]
                base_sc = score_three_way_evidence(raw_rec, weights=active_weights)
                st.session_state[search_cache_key] = run_custom_llm_company_search(
                    company_name=focus_row["Name"],
                    ticker=focus_row["Ticker"],
                    query=custom_research_q,
                    raw_evidence=raw_rec,
                    base_scores=base_sc,
                    api_key=llm_cfg["api_key"],
                    provider=llm_cfg["provider"],
                    model=llm_cfg["model"],
                    base_url=llm_cfg["base_url"],
                )

        if search_cache_key in st.session_state:
            s_res = st.session_state[search_cache_key]
            st.markdown(
                f"""
                <div class="summary-verdict" style="margin-top:0.65rem;">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.4rem;">
                        <span style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;font-weight:600;">
                            {html.escape(str(s_res.get('mode', '')))}
                        </span>
                        <span style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;opacity:0.75;">
                            Query: {html.escape(str(s_res.get('query', ''))[:85])}
                        </span>
                    </div>
                    <div style="font-size:1.0rem;font-weight:600;margin-bottom:0.55rem;">
                        {html.escape(str(s_res.get('headline_answer', '')))}
                    </div>
                    <div style="font-size:0.9rem;">
                        <b>Capital Allocation Implication:</b> {html.escape(str(s_res.get('investment_implication', '')))}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            f_col1, f_col2 = st.columns([0.58, 0.42], gap="medium")
            with f_col1:
                st.markdown("**Key Analytical Findings**")
                for bullet in s_res.get("key_findings", []):
                    st.markdown(f"- {bullet}")
                if s_res.get("citations"):
                    st.markdown("**Live Web Search Citations**")
                    for cite_url in s_res["citations"]:
                        st.markdown(f"- [{cite_url}]({cite_url})")
            with f_col2:
                st.markdown("**Verified Technical Signals**")
                st.markdown(_render_pills(s_res.get("technical_signals", []), "tag-pos"), unsafe_allow_html=True)
                st.markdown("**Execution & Valuation Risk Signals**")
                st.markdown(_render_pills(s_res.get("risk_signals", []), "tag-neg"), unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state["show_evidence"]:
        st.markdown("<div class='clean-card'>", unsafe_allow_html=True)
        ds_col1, ds_col2 = st.columns([0.65, 0.35])
        with ds_col1:
            st.markdown(f"### Three-Way Evidence Cross-Examination · {focus_row['Name']} ({focus_row['Ticker']})")
            st.caption(
                f"Hugging Face Models: **{focus_row['HF_Models']:,}** ({focus_row['HF_Downloads']:,} downloads)  ·  "
                f"OpenAlex AI Papers: **{focus_row['OpenAlex_Papers']:,}** ({focus_row['OpenAlex_Citations']:,} citations)  ·  "
                f"GitHub Repositories: **{focus_row['GitHub_Repos']:,}** ({focus_row['GitHub_Stars']:,} stars)"
            )
        with ds_col2:
            if st.button(
                f"Run {llm_cfg['provider']} Live Technical Audit ({focus_row['Ticker']})",
                type="primary",
                width="stretch",
            ):
                with st.spinner(f"Calling {llm_cfg['provider']} (`{llm_cfg['model']}`) to cross-examine {focus_row['Name']}..."):
                    raw_rec = focus_row["Raw_Record"]
                    base_sc = score_three_way_evidence(raw_rec, weights=active_weights)
                    live_syn = run_deepseek_technical_audit(
                        company_name=focus_row["Name"],
                        ticker=focus_row["Ticker"],
                        raw_evidence=raw_rec,
                        base_scores=base_sc,
                        api_key=llm_cfg["api_key"],
                        provider=llm_cfg["provider"],
                        model=llm_cfg["model"],
                        base_url=llm_cfg["base_url"],
                    )
                    raw_rec["audit_synthesis"] = live_syn
                    store.upsert_company(raw_rec)
                    st.rerun()

        ev1, ev2, ev3 = st.columns(3, gap="medium")
        with ev1:
            st.markdown(f"#### Filing Claims (Score: {focus_row['Claim_Score']:.1f})")
            with st.popover("Methodology: Claim Score (S_claim)"):
                st.markdown(
                    "Measures the density of high-profile AI commitments (`GenAI Platform`, `Autonomous Agents`, `Foundation Model`, `AI-First`) across SEC 10-K/20-F filings and official annual reports on a `0–100` scale."
                )
            st.markdown(f"<div class='quote-box'>{html.escape(focus_row['Filing_Text'])}</div>", unsafe_allow_html=True)
            st.markdown("**Matched Claim Keywords**")
            st.markdown(_render_pills(focus_row["Matched_Claims"], "tag-info"), unsafe_allow_html=True)

        with ev2:
            st.markdown(f"#### Technical Proof (Score: {focus_row['Proof_Score']:.1f})")
            with st.popover("Methodology: Proof Score (S_proof)"):
                st.markdown(
                    "Synthesizes **65% Contemporary AI Proof** (Hugging Face open-weight models & downloads, OpenAlex research citations, and `CUDA/TensorRT/vLLM/PyTorch` stack signals) + **35% Computer Systems Proof** (GitHub engineering velocity and `Rust/C++20/RISC-V/Distributed Systems` stack density)."
                )
            for q in focus_row["Tech_Texts"]:
                st.markdown(f"<div class='quote-box'>{html.escape(str(q))}</div>", unsafe_allow_html=True)
            st.markdown("**Contemporary AI Signals (65% Weight)**")
            st.markdown(_render_pills(focus_row["Matched_AI"], "tag-pos"), unsafe_allow_html=True)
            st.markdown("**Computer Systems Signals (35% Weight)**")
            st.markdown(_render_pills(focus_row["Matched_Systems"], "tag-pos"), unsafe_allow_html=True)

        with ev3:
            st.markdown(f"#### Community Telemetry (Score: {focus_row['Friction_Score']:.1f})")
            with st.popover("Methodology: Friction Score (S_friction)"):
                st.markdown(
                    "Captures anonymous developer telemetry from Hacker News and Reddit engineering forums, weighting **60% AI-Washing & Wrapper signals** (`API Wrapper`, `Rebranded BI`) and **40% Bureaucracy & Outsourcing friction** (`Dienstleistersteuerung`, `SAFe Agile Coach`)."
                )
            for q in focus_row["Insider_Texts"]:
                st.markdown(f"<div class='quote-box'>{html.escape(str(q))}</div>", unsafe_allow_html=True)
            st.markdown("**AI-Washing & Wrapper Signals**")
            st.markdown(_render_pills(focus_row["Matched_Washing"], "tag-neg"), unsafe_allow_html=True)
            st.markdown("**Bureaucracy & Tech Debt Signals**")
            st.markdown(_render_pills(focus_row["Matched_Bloat"], "tag-neg"), unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state["show_accounting"]:
        st.markdown("<div class='clean-card'>", unsafe_allow_html=True)
        st.markdown(f"### Damodaran R&D Capitalization Accounting Bridge · {focus_row['Ticker']}")
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
        st.dataframe(acc_df, width="stretch", hide_index=True)

        st.markdown("### Cross-Sectional Global Peer Comparison Ledger")
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
        st.dataframe(peer_df, width="stretch", hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state["show_workbench"]:
        st.markdown("<div class='clean-card'>", unsafe_allow_html=True)
        st.markdown(f"### Personal Analyst Thesis & Calibration Schedule · {focus_row['Ticker']}")
        wb_l, wb_r = st.columns([0.55, 0.45], gap="large")
        with wb_l:
            edited_note = st.text_area(
                f"Analyst Thesis for {focus_row['Name']} ({focus_row['Ticker']}):",
                value=focus_row["Author_Thesis"],
                height=140,
            )
            if st.button("Save Thesis to SQLite Database", type="primary"):
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
            st.dataframe(amort_df, width="stretch", hide_index=True, height=210)
        st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(
        page_title="CapEx-to-Code | Global R&D Capitalization & Multi-LLM AI Audit",
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
