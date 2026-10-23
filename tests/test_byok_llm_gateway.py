import json
import unittest
from unittest.mock import MagicMock, patch

from ai_audit_engine import (
    LLM_PROVIDER_PRESETS,
    load_provider_api_key,
    run_custom_llm_company_search,
    run_deepseek_technical_audit,
)

AUDIT_JSON = json.dumps(
    {
        "claim_adjustment": 3.0,
        "proof_adjustment": -4.0,
        "friction_adjustment": 1.5,
        "primary_contradiction": "Narrative exceeds shipped artifacts.",
        "verified_ai_capabilities": ["PyTorch"],
        "ai_washing_red_flags": ["API Wrapper"],
        "executive_verdict": "Moderate AI-washing risk.",
    }
)

SEARCH_JSON = json.dumps(
    {
        "headline_answer": "CapEx converts mostly into proprietary silicon.",
        "key_findings": ["Finding A", "Finding B", "Finding C"],
        "technical_signals": ["CUDA", "Triton"],
        "risk_signals": ["Supply concentration"],
        "investment_implication": "Supports premium multiple.",
    }
)

EVIDENCE = {"filing_text": "AI platform", "insider_texts": ["wrapper"], "hf_models_count": 3}
SCORES = {"claim_score": 70.0, "proof_score": 45.0, "friction_score": 40.0, "awdi": 35.0, "conversion_rate": 0.52}


def _openai_resp(content, citations=None):
    resp = MagicMock()
    resp.status_code = 200
    body = {"choices": [{"message": {"content": content}}]}
    if citations:
        body["citations"] = citations
    resp.json.return_value = body
    return resp


def _anthropic_resp(content):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"content": [{"type": "text", "text": content}]}
    return resp


class TestBYOKLLMGateway(unittest.TestCase):
    def test_presets_are_complete(self):
        for name, preset in LLM_PROVIDER_PRESETS.items():
            for field in ("base_url", "default_model", "models", "env_var", "protocol"):
                self.assertIn(field, preset, f"{name} missing {field}")
            self.assertIn(preset["default_model"], preset["models"])
            self.assertIn(preset["protocol"], ("openai", "anthropic"))

    def test_openai_provider_routes_to_its_endpoint_and_model(self):
        with patch("requests.post", return_value=_openai_resp(AUDIT_JSON)) as mock_post:
            audit = run_deepseek_technical_audit(
                "TestCorp", "TEST", EVIDENCE, SCORES,
                api_key="sk-user", provider="OpenAI", model="gpt-4o-mini",
            )
        url = mock_post.call_args.args[0]
        payload = mock_post.call_args.kwargs["json"]
        headers = mock_post.call_args.kwargs["headers"]
        self.assertEqual(url, LLM_PROVIDER_PRESETS["OpenAI"]["base_url"])
        self.assertEqual(payload["model"], "gpt-4o-mini")
        self.assertEqual(headers["Authorization"], "Bearer sk-user")
        self.assertIn("OpenAI", audit["mode"])
        self.assertIn("Live LLM", audit["mode"])
        self.assertEqual(audit["proof_adjustment"], -4.0)

    def test_anthropic_provider_uses_messages_protocol(self):
        with patch("requests.post", return_value=_anthropic_resp(AUDIT_JSON)) as mock_post:
            audit = run_deepseek_technical_audit(
                "TestCorp", "TEST", EVIDENCE, SCORES,
                api_key="sk-ant-user", provider="Anthropic Claude",
            )
        headers = mock_post.call_args.kwargs["headers"]
        payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(headers["x-api-key"], "sk-ant-user")
        self.assertIn("anthropic-version", headers)
        self.assertIn("system", payload)
        self.assertEqual(payload["messages"][0]["role"], "user")
        self.assertIn("Anthropic Claude", audit["mode"])

    def test_custom_endpoint_url_is_respected(self):
        custom_url = "http://localhost:11434/v1/chat/completions"
        with patch("requests.post", return_value=_openai_resp(AUDIT_JSON)) as mock_post:
            run_deepseek_technical_audit(
                "TestCorp", "TEST", EVIDENCE, SCORES,
                api_key="local", provider="OpenRouter / Custom Endpoint",
                model="llama3.1", base_url=custom_url,
            )
        self.assertEqual(mock_post.call_args.args[0], custom_url)
        self.assertEqual(mock_post.call_args.kwargs["json"]["model"], "llama3.1")

    def test_reasoning_model_think_block_is_stripped(self):
        content = "<think>internal reasoning {not json}</think>\n" + AUDIT_JSON
        with patch("requests.post", return_value=_openai_resp(content)):
            audit = run_deepseek_technical_audit(
                "TestCorp", "TEST", EVIDENCE, SCORES,
                api_key="sk", provider="DeepSeek", model="deepseek-reasoner",
            )
        self.assertIn("Live LLM", audit["mode"])
        self.assertEqual(audit["claim_adjustment"], 3.0)

    def test_custom_search_live_with_citations(self):
        cites = ["https://example.com/a", "https://example.com/b"]
        with patch("requests.post", return_value=_openai_resp(SEARCH_JSON, citations=cites)):
            res = run_custom_llm_company_search(
                "TestCorp", "TEST", "How efficient is AI CapEx?", EVIDENCE, SCORES,
                api_key="pplx-user", provider="Perplexity Sonar (Web Search)",
            )
        self.assertEqual(res["status"], "live_llm")
        self.assertEqual(res["model"], "sonar-pro")
        self.assertEqual(len(res["key_findings"]), 3)
        self.assertEqual(res["citations"], cites)

    def test_custom_search_without_key_uses_deterministic_brief(self):
        with patch("requests.post") as mock_post:
            res = run_custom_llm_company_search(
                "TestCorp", "TEST", "", EVIDENCE, SCORES, api_key="", provider="OpenAI",
            )
        mock_post.assert_not_called()
        self.assertEqual(res["status"], "deterministic_fallback")
        self.assertTrue(res["query"])
        self.assertTrue(res["headline_answer"])

    def test_custom_search_http_error_falls_back(self):
        bad = MagicMock()
        bad.status_code = 401
        with patch("requests.post", return_value=bad):
            res = run_custom_llm_company_search(
                "TestCorp", "TEST", "q", EVIDENCE, SCORES, api_key="bad", provider="Google Gemini",
            )
        self.assertEqual(res["status"], "error_fallback")
        self.assertEqual(res["citations"], [])

    def test_load_provider_api_key_reads_env(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-env-test"}):
            self.assertEqual(load_provider_api_key("OpenAI"), "sk-env-test")


if __name__ == "__main__":
    unittest.main()
