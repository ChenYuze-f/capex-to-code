"""
Three-Way AI Reality Cross-Examination & DeepSeek Technical Audit Engine.

Evaluates:
  1. Executive Filing Claim Score (S_claim)
  2. Verifiable Technical Proof Score (S_proof = 65% Contemporary AI + 35% Systems)
  3. Insider Friction & AI-Washing Penalty Score (S_friction)
  4. AI-Washing Divergence Index (AWDI) & Productive R&D Conversion Rate (eta_conv)
  5. Optional Live DeepSeek Technical Audit (via https://api.deepseek.com/chat/completions)
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

from analyst_calibration import (
    AI_WASHING_SIGNALS,
    BUREAUCRACY_BLOAT_SIGNALS,
    CLAIM_HYPE_SIGNALS,
    CONTEMPORARY_AI_SIGNALS,
    DEFAULT_MODEL_WEIGHTS,
    SYSTEMS_ENGINEERING_SIGNALS,
)


def _clamp(val: float, low: float, high: float) -> float:
    return max(low, min(high, val))


def _scan_dictionary(text: str, signal_dict: Dict[str, float]) -> Tuple[List[str], float]:
    """Scan text against a weighted signal dictionary (case-insensitive)."""
    matched: List[str] = []
    total_weight = 0.0
    for kw, weight in signal_dict.items():
        if re.search(re.escape(kw), text, re.IGNORECASE):
            matched.append(kw)
            total_weight += float(weight)
    return matched, round(total_weight, 2)


def _join_texts(val: Any) -> str:
    if isinstance(val, str):
        return val
    if isinstance(val, Iterable):
        return " \n ".join(str(x) for x in val if x)
    return ""


def load_deepseek_api_key() -> str:
    """
    Load DEEPSEEK_API_KEY from environment variables or local `.env` file.
    Never exposes or hardcodes credentials in source code.
    """
    env_val = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if env_val:
        return env_val

    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped.startswith("DEEPSEEK_API_KEY="):
                    return stripped.split("=", 1)[1].strip().strip('"').strip("'")
        except Exception:
            return ""
    return ""


def score_three_way_evidence(
    raw_evidence: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Compute the 3-way sub-scores (Claim, Proof [65% AI / 35% Systems], Friction),
    the AI-Washing Divergence Index (AWDI), and the R&D Conversion Rate (eta_conv).
    """
    w = dict(DEFAULT_MODEL_WEIGHTS)
    if weights:
        w.update(weights)

    ai_w = float(w.get("ai_stack_weight", 0.65))
    sys_w = float(w.get("systems_stack_weight", 0.35))
    total_stack_w = ai_w + sys_w
    if total_stack_w > 0:
        ai_w /= total_stack_w
        sys_w /= total_stack_w

    filing_text = _join_texts(raw_evidence.get("filing_text", ""))
    tech_text = _join_texts(raw_evidence.get("tech_texts", []))
    insider_text = _join_texts(raw_evidence.get("insider_texts", []))

    # 1. Executive Filing Claim Score (0 - 100)
    matched_claims, raw_claim_w = _scan_dictionary(filing_text, CLAIM_HYPE_SIGNALS)
    base_claim = float(raw_evidence.get("base_claim_score", 24.0))
    claim_score = _clamp(round(base_claim + raw_claim_w * 1.15, 1), 12.0, 96.0)

    # 2. Verifiable Technical Proof Score (0 - 100)
    # 2a. Contemporary AI Stack Score (65% default weight)
    matched_ai, raw_ai_w = _scan_dictionary(tech_text + " " + filing_text, CONTEMPORARY_AI_SIGNALS)
    hf_models = max(0, int(raw_evidence.get("hf_models_count", 0)))
    hf_downloads = max(0, int(raw_evidence.get("hf_downloads", 0)))
    oa_papers = max(0, int(raw_evidence.get("openalex_papers_count", 0)))
    oa_citations = max(0, int(raw_evidence.get("openalex_citations", 0)))

    hf_component = min(22.0, math.log1p(hf_models) * 2.4 + math.log1p(hf_downloads) * 0.55)
    paper_component = min(20.0, math.log1p(oa_papers) * 1.8 + math.log1p(oa_citations) * 0.65)
    keyword_ai_component = min(34.0, raw_ai_w * 0.52)
    base_ai_hint = float(raw_evidence.get("base_ai_proof_hint", 12.0)) * 0.65

    ai_stack_score = _clamp(
        round(base_ai_hint + hf_component + paper_component + keyword_ai_component, 1),
        8.0,
        96.0,
    )

    # 2b. Hardcore Computer Systems Score (35% default weight)
    matched_sys, raw_sys_w = _scan_dictionary(tech_text + " " + insider_text, SYSTEMS_ENGINEERING_SIGNALS)
    gh_repos = max(0, int(raw_evidence.get("github_repos_count", 0)))
    gh_stars = max(0, int(raw_evidence.get("github_stars", 0)))

    gh_component = min(26.0, math.log1p(gh_repos) * 2.1 + math.log1p(gh_stars) * 0.75)
    keyword_sys_component = min(38.0, raw_sys_w * 0.62)
    base_sys_hint = float(raw_evidence.get("base_systems_proof_hint", 14.0)) * 0.70

    systems_stack_score = _clamp(
        round(base_sys_hint + gh_component + keyword_sys_component, 1),
        8.0,
        96.0,
    )

    proof_score = _clamp(
        round(ai_w * ai_stack_score + sys_w * systems_stack_score, 1),
        8.0,
        96.0,
    )

    # 3. Insider Friction & AI-Washing Penalty Score (0 - 100)
    matched_washing, raw_wash_w = _scan_dictionary(insider_text, AI_WASHING_SIGNALS)
    matched_bloat, raw_bloat_w = _scan_dictionary(insider_text, BUREAUCRACY_BLOAT_SIGNALS)

    wash_penalty = min(100.0, 15.0 + raw_wash_w * 2.4)
    bloat_penalty = min(100.0, 15.0 + raw_bloat_w * 2.2)
    base_friction_hint = float(raw_evidence.get("base_friction_hint", 8.0))

    friction_score = _clamp(
        round(base_friction_hint + 0.60 * wash_penalty + 0.40 * bloat_penalty, 1),
        8.0,
        98.0,
    )

    # 4. AI-Washing Divergence Index (AWDI) & Productive R&D Conversion Rate (eta_conv)
    narrative_gap = max(0.0, claim_score - proof_score)
    awdi = _clamp(
        round(narrative_gap * (1.0 + friction_score / 100.0), 1),
        0.0,
        100.0,
    )

    w_proof = float(w.get("proof_weight", 0.65))
    w_culture = float(w.get("culture_weight", 0.35))
    w_awdi = float(w.get("awdi_penalty_weight", 0.25))

    raw_eta = (
        w_proof * proof_score
        + w_culture * (100.0 - friction_score)
        - w_awdi * awdi
    ) / 100.0
    conversion_rate = round(_clamp(raw_eta, 0.15, 0.95), 2)

    return {
        "claim_score": claim_score,
        "ai_stack_score": ai_stack_score,
        "systems_stack_score": systems_stack_score,
        "proof_score": proof_score,
        "friction_score": friction_score,
        "awdi": awdi,
        "conversion_rate": conversion_rate,
        "matched_claims": matched_claims,
        "matched_ai_signals": matched_ai,
        "matched_systems_signals": matched_sys,
        "matched_washing_signals": matched_washing,
        "matched_bloat_signals": matched_bloat,
    }


def _build_deterministic_synthesis(
    company_name: str,
    ticker: str,
    base_scores: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate a deterministic quantitative synthesis when DeepSeek API is off or unreachable."""
    claim = float(base_scores.get("claim_score", 50.0))
    proof = float(base_scores.get("proof_score", 50.0))
    friction = float(base_scores.get("friction_score", 35.0))
    awdi = float(base_scores.get("awdi", max(0.0, claim - proof)))

    if awdi >= 40.0:
        contradiction = (
            f"{company_name} ({ticker}) exhibits a pronounced narrative-to-execution gap "
            f"(Claim {claim:.1f} vs. Proof {proof:.1f}, AWDI {awdi:.1f}): high-profile AI "
            f"filing language contrasts with sparse open model/research artifacts and elevated "
            f"developer reports of wrapper/outsourcing friction ({friction:.1f})."
        )
        verdict = (
            "MATERIAL AI-WASHING / DILUTION WARNING: Discount management AI growth guidance; "
            "a substantial share of reported R&D is absorbed by integration and coordination overhead."
        )
    elif proof >= 70.0 and awdi < 20.0:
        contradiction = (
            f"Telemetry confirms strong alignment or conservative understatement for {company_name} "
            f"({ticker}): verifiable AI & systems output (Proof {proof:.1f}) matches or exceeds "
            f"executive filing claims (Claim {claim:.1f}), with low internal friction ({friction:.1f})."
        )
        verdict = (
            "VERIFIED TECHNICAL MOAT: R&D expenditure converts at a high rate into durable "
            "AI/systems intellectual property, supporting upward ROIC and multiple re-rating."
        )
    else:
        contradiction = (
            f"{company_name} ({ticker}) shows balanced execution (Claim {claim:.1f}, Proof {proof:.1f}, "
            f"Friction {friction:.1f}): core engineering teams deliver production systems, though "
            f"legacy maintenance moderates marginal R&D capital efficiency."
        )
        verdict = (
            "SELECTIVE COMPOUNDER: Monitor conversion of incremental AI CapEx into autonomous "
            "product revenue versus legacy maintenance."
        )

    return {
        "mode": "Deterministic Quantitative Engine",
        "claim_adjustment": 0.0,
        "proof_adjustment": 0.0,
        "friction_adjustment": 0.0,
        "primary_contradiction": contradiction,
        "verified_ai_capabilities": list(base_scores.get("matched_ai_signals", []))[:6],
        "ai_washing_red_flags": (
            list(base_scores.get("matched_washing_signals", []))
            + list(base_scores.get("matched_bloat_signals", []))
        )[:6],
        "executive_verdict": verdict,
    }


def run_deepseek_technical_audit(
    company_name: str,
    ticker: str,
    raw_evidence: Dict[str, Any],
    base_scores: Dict[str, Any],
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute a live DeepSeek Technical Audit (`deepseek-chat`) cross-examining
    management filing claims against technical telemetry and engineer forum quotes.
    """
    resolved_key = (api_key or load_deepseek_api_key()).strip()
    if not resolved_key:
        return _build_deterministic_synthesis(company_name, ticker, base_scores)

    prompt_payload = {
        "company": company_name,
        "ticker": ticker,
        "deterministic_scores": {
            "claim_score": base_scores.get("claim_score"),
            "proof_score": base_scores.get("proof_score"),
            "friction_score": base_scores.get("friction_score"),
            "awdi": base_scores.get("awdi"),
        },
        "telemetry_counts": {
            "huggingface_models": raw_evidence.get("hf_models_count", 0),
            "huggingface_downloads": raw_evidence.get("hf_downloads", 0),
            "openalex_ai_papers": raw_evidence.get("openalex_papers_count", 0),
            "github_repos": raw_evidence.get("github_repos_count", 0),
        },
        "filing_excerpt": _join_texts(raw_evidence.get("filing_text", ""))[:900],
        "tech_stack_evidence": _join_texts(raw_evidence.get("tech_texts", []))[:700],
        "insider_forum_quotes": _join_texts(raw_evidence.get("insider_texts", []))[:900],
    }

    system_msg = (
        "You are an institutional quantitative equity & AI technical due diligence auditor. "
        "Cross-examine the company's filing claims against its verifiable AI/systems telemetry "
        "and engineer community quotes. Respond ONLY with a valid JSON object containing keys: "
        "claim_adjustment (float between -10 and 10), proof_adjustment (float between -10 and 10), "
        "friction_adjustment (float between -10 and 10), primary_contradiction (concise 1-2 sentence string), "
        "verified_ai_capabilities (list of strings), ai_washing_red_flags (list of strings), "
        "and executive_verdict (concise 2 sentence institutional investment takeaway)."
    )

    try:
        response = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={
                "Authorization": f"Bearer {resolved_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=False)},
                ],
                "temperature": 0.1,
                "max_tokens": 500,
            },
            timeout=8.0,
        )
        if response.status_code != 200:
            return _build_deterministic_synthesis(company_name, ticker, base_scores)

        content = response.json()["choices"][0]["message"]["content"].strip()
        # Strip optional markdown code fences
        cleaned = re.sub(r"^```(?:json)?\s*", "", content)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        parsed = json.loads(cleaned)

        return {
            "mode": "DeepSeek Technical Mode (Live LLM)",
            "claim_adjustment": round(_clamp(float(parsed.get("claim_adjustment", 0.0)), -10.0, 10.0), 1),
            "proof_adjustment": round(_clamp(float(parsed.get("proof_adjustment", 0.0)), -10.0, 10.0), 1),
            "friction_adjustment": round(_clamp(float(parsed.get("friction_adjustment", 0.0)), -10.0, 10.0), 1),
            "primary_contradiction": str(parsed.get("primary_contradiction", "")),
            "verified_ai_capabilities": list(parsed.get("verified_ai_capabilities", []))[:8],
            "ai_washing_red_flags": list(parsed.get("ai_washing_red_flags", []))[:8],
            "executive_verdict": str(parsed.get("executive_verdict", "")),
        }
    except Exception:
        return _build_deterministic_synthesis(company_name, ticker, base_scores)
