"""
Three-Way AI Reality Cross-Examination & Multi-Provider BYOK LLM Audit Engine.

Evaluates:
  1. Executive Filing Claim Score (S_claim)
  2. Verifiable Technical Proof Score (S_proof = 65% Contemporary AI + 35% Systems)
  3. Insider Friction & AI-Washing Penalty Score (S_friction)
  4. AI-Washing Divergence Index (AWDI) & Productive R&D Conversion Rate (eta_conv)
  5. Extensible Multi-Provider BYOK LLM Technical Audit & Live Intelligence Search
     (Supports DeepSeek, OpenAI, Perplexity Sonar Web Search, Google Gemini,
      Anthropic Claude, and any OpenRouter / Custom OpenAI-Compatible Endpoint)
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


LLM_PROVIDER_PRESETS: Dict[str, Dict[str, Any]] = {
    "DeepSeek": {
        "base_url": "https://api.deepseek.com/chat/completions",
        "default_model": "deepseek-chat",
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "env_var": "DEEPSEEK_API_KEY",
        "protocol": "openai",
        "supports_web_search": False,
        "badge": "DeepSeek V3 / R1",
        "key_placeholder": "sk-...",
    },
    "OpenAI": {
        "base_url": "https://api.openai.com/v1/chat/completions",
        "default_model": "gpt-4o",
        "models": ["gpt-4o", "gpt-4o-mini", "o3-mini"],
        "env_var": "OPENAI_API_KEY",
        "protocol": "openai",
        "supports_web_search": False,
        "badge": "GPT-4o / o3-mini",
        "key_placeholder": "sk-proj-...",
    },
    "Perplexity Sonar (Web Search)": {
        "base_url": "https://api.perplexity.ai/chat/completions",
        "default_model": "sonar-pro",
        "models": ["sonar-pro", "sonar-reasoning-pro", "sonar"],
        "env_var": "PERPLEXITY_API_KEY",
        "protocol": "openai",
        "supports_web_search": True,
        "badge": "Live Web Search + Citations",
        "key_placeholder": "pplx-...",
    },
    "Google Gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "default_model": "gemini-2.5-flash",
        "models": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
        "env_var": "GEMINI_API_KEY",
        "protocol": "openai",
        "supports_web_search": False,
        "badge": "Gemini 2.5 Flash / Pro",
        "key_placeholder": "AIzaSy...",
    },
    "Anthropic Claude": {
        "base_url": "https://api.anthropic.com/v1/messages",
        "default_model": "claude-3-5-sonnet-latest",
        "models": ["claude-3-5-sonnet-latest", "claude-3-7-sonnet-latest", "claude-3-5-haiku-latest"],
        "env_var": "ANTHROPIC_API_KEY",
        "protocol": "anthropic",
        "supports_web_search": False,
        "badge": "Claude 3.5 / 3.7 Sonnet",
        "key_placeholder": "sk-ant-...",
    },
    "OpenRouter / Custom Endpoint": {
        "base_url": "https://openrouter.ai/api/v1/chat/completions",
        "default_model": "perplexity/sonar-pro",
        "models": [
            "perplexity/sonar-pro",
            "anthropic/claude-3.5-sonnet",
            "deepseek/deepseek-r1",
            "openai/gpt-4o",
        ],
        "env_var": "OPENROUTER_API_KEY",
        "protocol": "openai",
        "supports_web_search": True,
        "badge": "OpenAI-Compatible Gateway",
        "key_placeholder": "sk-or-v1-...",
    },
}


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


def load_provider_api_key(provider: str = "DeepSeek") -> str:
    """
    Load the API key for a given LLM provider from environment variables or local `.env` file.
    Never exposes or hardcodes credentials in source code.
    """
    preset = LLM_PROVIDER_PRESETS.get(provider, LLM_PROVIDER_PRESETS["DeepSeek"])
    env_var = str(preset.get("env_var", "DEEPSEEK_API_KEY"))

    env_val = os.environ.get(env_var, "").strip()
    if env_val:
        return env_val

    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped.startswith(f"{env_var}="):
                    return stripped.split("=", 1)[1].strip().strip('"').strip("'")
        except Exception:
            return ""
    return ""


def load_deepseek_api_key() -> str:
    """
    Load DEEPSEEK_API_KEY from environment variables or local `.env` file.
    Never exposes or hardcodes credentials in source code.
    """
    return load_provider_api_key("DeepSeek")


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


def _invoke_llm_json(
    system_msg: str,
    user_payload: str,
    api_key: str,
    provider: str = "DeepSeek",
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    max_tokens: int = 550,
    timeout: float = 10.0,
) -> Tuple[Optional[Dict[str, Any]], List[str], str, str]:
    """
    Unified multi-provider LLM JSON invocation supporting both OpenAI-compatible
    `/chat/completions` endpoints (DeepSeek, OpenAI, Perplexity Sonar, Gemini, OpenRouter)
    and Anthropic's `/v1/messages` protocol.
    Returns (parsed_json_or_none, citations_list, resolved_provider, resolved_model).
    """
    preset = LLM_PROVIDER_PRESETS.get(provider, LLM_PROVIDER_PRESETS["DeepSeek"])
    resolved_url = (base_url or preset["base_url"]).strip()
    resolved_model = (model or preset["default_model"]).strip()
    protocol = preset.get("protocol", "openai")
    if base_url and "anthropic.com" in resolved_url:
        protocol = "anthropic"

    citations: List[str] = []

    if protocol == "anthropic":
        response = requests.post(
            resolved_url,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": resolved_model,
                "max_tokens": max_tokens,
                "temperature": 0.1,
                "system": system_msg,
                "messages": [{"role": "user", "content": user_payload}],
            },
            timeout=timeout,
        )
        if response.status_code != 200:
            return None, [], provider, resolved_model
        data = response.json()
        blocks = data.get("content") or []
        content = ""
        for blk in blocks:
            if isinstance(blk, dict) and blk.get("type") == "text":
                content += str(blk.get("text", ""))
        content = content.strip()
    else:
        response = requests.post(
            resolved_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": resolved_model,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_payload},
                ],
                "temperature": 0.1,
                "max_tokens": max_tokens,
            },
            timeout=timeout,
        )
        if response.status_code != 200:
            return None, [], provider, resolved_model
        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()
        raw_citations = data.get("citations") or []
        if isinstance(raw_citations, list):
            citations = [str(c) for c in raw_citations[:6] if c]

    cleaned = re.sub(r"^```(?:json)?\s*", "", content)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    # Extract JSON object if reasoning models prepend <think> blocks
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    json_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if json_match:
        cleaned = json_match.group(0)

    parsed = json.loads(cleaned)
    return parsed, citations, provider, resolved_model


def run_deepseek_technical_audit(
    company_name: str,
    ticker: str,
    raw_evidence: Dict[str, Any],
    base_scores: Dict[str, Any],
    api_key: Optional[str] = None,
    provider: str = "DeepSeek",
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    custom_focus: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute a live Multi-Provider BYOK Technical Audit (defaulting to DeepSeek `deepseek-chat`)
    cross-examining management filing claims against technical telemetry and engineer forum quotes.
    """
    resolved_key = (api_key if api_key is not None else load_provider_api_key(provider)).strip()
    if not resolved_key:
        return _build_deterministic_synthesis(company_name, ticker, base_scores)

    prompt_payload: Dict[str, Any] = {
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
    if custom_focus and custom_focus.strip():
        prompt_payload["user_custom_audit_focus"] = custom_focus.strip()[:400]

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
        parsed, citations, used_provider, used_model = _invoke_llm_json(
            system_msg=system_msg,
            user_payload=json.dumps(prompt_payload, ensure_ascii=False),
            api_key=resolved_key,
            provider=provider,
            model=model,
            base_url=base_url,
            max_tokens=550,
            timeout=10.0,
        )
        if not parsed:
            return _build_deterministic_synthesis(company_name, ticker, base_scores)

        mode_label = (
            "DeepSeek Technical Mode (Live LLM)"
            if used_provider == "DeepSeek" and (not model or model == "deepseek-chat")
            else f"{used_provider} [{used_model}] (Live LLM)"
        )

        result = {
            "mode": mode_label,
            "provider": used_provider,
            "model": used_model,
            "claim_adjustment": round(_clamp(float(parsed.get("claim_adjustment", 0.0)), -10.0, 10.0), 1),
            "proof_adjustment": round(_clamp(float(parsed.get("proof_adjustment", 0.0)), -10.0, 10.0), 1),
            "friction_adjustment": round(_clamp(float(parsed.get("friction_adjustment", 0.0)), -10.0, 10.0), 1),
            "primary_contradiction": str(parsed.get("primary_contradiction", "")),
            "verified_ai_capabilities": list(parsed.get("verified_ai_capabilities", []))[:8],
            "ai_washing_red_flags": list(parsed.get("ai_washing_red_flags", []))[:8],
            "executive_verdict": str(parsed.get("executive_verdict", "")),
        }
        if citations:
            result["citations"] = citations
        return result
    except Exception:
        return _build_deterministic_synthesis(company_name, ticker, base_scores)


def run_custom_llm_company_search(
    company_name: str,
    ticker: str,
    query: str,
    raw_evidence: Dict[str, Any],
    base_scores: Dict[str, Any],
    api_key: Optional[str] = None,
    provider: str = "DeepSeek",
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute a user-driven Live Intelligence Search & Due Diligence Query on a company
    using the user's BYOK LLM provider/model and grounded multi-source telemetry.
    Falls back to a structured deterministic telemetry brief if no API key is configured.
    """
    clean_q = (query or "").strip() or (
        f"Evaluate {company_name} ({ticker}) AI CapEx efficiency, open-source/research moat, and execution bottlenecks."
    )
    resolved_key = (api_key if api_key is not None else load_provider_api_key(provider)).strip()
    preset = LLM_PROVIDER_PRESETS.get(provider, LLM_PROVIDER_PRESETS["DeepSeek"])
    resolved_model = (model or preset["default_model"]).strip()

    # Deterministic fallback when user has not entered an API key
    if not resolved_key:
        synth = _build_deterministic_synthesis(company_name, ticker, base_scores)
        return {
            "status": "deterministic_fallback",
            "mode": "Local Telemetry Synthesis (Add API Key for Live LLM Generation)",
            "provider": provider,
            "model": resolved_model,
            "query": clean_q,
            "headline_answer": synth["primary_contradiction"],
            "key_findings": [
                f"Filing Claim Intensity: {base_scores.get('claim_score', 0):.1f}/100 vs. Verifiable Technical Proof: {base_scores.get('proof_score', 0):.1f}/100.",
                f"Telemetry Footprint: {raw_evidence.get('hf_models_count', 0)} Hugging Face models, {raw_evidence.get('openalex_papers_count', 0)} OpenAlex papers, {raw_evidence.get('github_repos_count', 0)} GitHub repositories.",
                f"Productive R&D Conversion Rate (η_conv): {float(base_scores.get('conversion_rate', 0.5)) * 100:.0f}% (AWDI {base_scores.get('awdi', 0):.1f}).",
            ],
            "technical_signals": synth["verified_ai_capabilities"],
            "risk_signals": synth["ai_washing_red_flags"],
            "investment_implication": synth["executive_verdict"],
            "citations": [],
        }

    search_payload = {
        "target_company": f"{company_name} ({ticker})",
        "research_question": clean_q,
        "grounded_telemetry": {
            "scores": {
                "claim_score": base_scores.get("claim_score"),
                "proof_score": base_scores.get("proof_score"),
                "friction_score": base_scores.get("friction_score"),
                "awdi": base_scores.get("awdi"),
                "conversion_rate": base_scores.get("conversion_rate"),
            },
            "artifacts": {
                "hf_models_count": raw_evidence.get("hf_models_count", 0),
                "hf_downloads": raw_evidence.get("hf_downloads", 0),
                "openalex_papers_count": raw_evidence.get("openalex_papers_count", 0),
                "github_repos_count": raw_evidence.get("github_repos_count", 0),
            },
            "filing_context": _join_texts(raw_evidence.get("filing_text", ""))[:800],
            "tech_stack_context": _join_texts(raw_evidence.get("tech_texts", []))[:700],
            "engineer_community_context": _join_texts(raw_evidence.get("insider_texts", []))[:800],
        },
    }

    system_msg = (
        "You are an institutional AI & quantitative equity research intelligence engine. "
        "Answer the user's research question about the target company by combining your latest "
        "knowledge/web search with the grounded telemetry provided. Respond ONLY with a valid JSON "
        "object containing keys: headline_answer (2-3 sentence direct analytical answer), "
        "key_findings (list of 3-4 specific factual bullet strings with numbers/stack details), "
        "technical_signals (list of verified technical stack/model keywords), "
        "risk_signals (list of execution or valuation risk keywords), and "
        "investment_implication (1-2 sentence capital allocation conclusion)."
    )

    try:
        parsed, citations, used_provider, used_model = _invoke_llm_json(
            system_msg=system_msg,
            user_payload=json.dumps(search_payload, ensure_ascii=False),
            api_key=resolved_key,
            provider=provider,
            model=model,
            base_url=base_url,
            max_tokens=700,
            timeout=12.0,
        )
        if not parsed:
            raise ValueError("Empty LLM response")

        return {
            "status": "live_llm",
            "mode": f"Live BYOK Intelligence · {used_provider} ({used_model})",
            "provider": used_provider,
            "model": used_model,
            "query": clean_q,
            "headline_answer": str(parsed.get("headline_answer", "")),
            "key_findings": [str(x) for x in list(parsed.get("key_findings", []))[:5]],
            "technical_signals": [str(x) for x in list(parsed.get("technical_signals", []))[:8]],
            "risk_signals": [str(x) for x in list(parsed.get("risk_signals", []))[:8]],
            "investment_implication": str(parsed.get("investment_implication", "")),
            "citations": citations,
        }
    except Exception:
        synth = _build_deterministic_synthesis(company_name, ticker, base_scores)
        return {
            "status": "error_fallback",
            "mode": f"Telemetry Fallback ({provider} endpoint unreachable or invalid key)",
            "provider": provider,
            "model": resolved_model,
            "query": clean_q,
            "headline_answer": synth["primary_contradiction"],
            "key_findings": [
                f"Filing Claim Intensity: {base_scores.get('claim_score', 0):.1f}/100 vs. Verifiable Technical Proof: {base_scores.get('proof_score', 0):.1f}/100.",
                f"Telemetry Footprint: {raw_evidence.get('hf_models_count', 0)} Hugging Face models, {raw_evidence.get('openalex_papers_count', 0)} OpenAlex papers, {raw_evidence.get('github_repos_count', 0)} GitHub repositories.",
            ],
            "technical_signals": synth["verified_ai_capabilities"],
            "risk_signals": synth["ai_washing_red_flags"],
            "investment_implication": synth["executive_verdict"],
            "citations": [],
        }
