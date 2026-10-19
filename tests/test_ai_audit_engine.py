import unittest
from unittest.mock import MagicMock, patch

from ai_audit_engine import (
    load_deepseek_api_key,
    run_deepseek_technical_audit,
    score_three_way_evidence,
)


class TestAIAuditEngine(unittest.TestCase):
    def test_low_claim_high_proof_clamps_awdi_to_zero(self):
        evidence = {
            "filing_text": "Industrial power semiconductors and microcontroller hardware.",
            "tech_texts": [
                "shipping CUDA, TensorRT, PyTorch, ONNX, Rust and C++20 RTOS kernels"
            ],
            "hf_models_count": 12,
            "hf_downloads": 150000,
            "openalex_papers_count": 45,
            "openalex_citations": 900,
            "github_repos_count": 85,
            "github_stars": 4200,
            "insider_texts": ["Strong low-level engineering team, very little bureaucracy."],
        }
        res = score_three_way_evidence(evidence)
        self.assertGreater(res["proof_score"], res["claim_score"])
        self.assertEqual(res["awdi"], 0.0)
        self.assertTrue(0.15 <= res["conversion_rate"] <= 0.95)
        self.assertGreater(res["ai_stack_score"], 0)
        self.assertGreater(res["systems_stack_score"], 0)

    def test_high_claim_low_proof_high_friction_triggers_high_awdi(self):
        evidence = {
            "filing_text": (
                "We are an AI-First GenAI Platform deploying Autonomous Agents, "
                "Foundation Models, and Proprietary LLM transformation everywhere. "
                "Generative AI and Agentic Workflow drive our AI Revolution."
            ),
            "tech_texts": ["Legacy maintenance and basic web dashboard"],
            "hf_models_count": 0,
            "hf_downloads": 0,
            "openalex_papers_count": 0,
            "openalex_citations": 0,
            "github_repos_count": 2,
            "github_stars": 5,
            "insider_texts": [
                "Our AI is just an OpenAI Wrapper and API Wrapper with Excel Automation and Vaporware.",
                "Heavy Outsourcing, Dienstleistersteuerung, SAFe Agile Coach, and PowerPoint Governance.",
            ],
        }
        res = score_three_way_evidence(evidence)
        self.assertGreaterEqual(res["claim_score"], 70.0)
        self.assertLessEqual(res["proof_score"], 35.0)
        self.assertGreaterEqual(res["friction_score"], 60.0)
        self.assertGreaterEqual(res["awdi"], 60.0)
        self.assertLessEqual(res["conversion_rate"], 0.45)

    def test_zero_telemetry_baseline_bounds(self):
        empty_evidence = {
            "filing_text": "",
            "tech_texts": [],
            "hf_models_count": 0,
            "hf_downloads": 0,
            "openalex_papers_count": 0,
            "openalex_citations": 0,
            "github_repos_count": 0,
            "github_stars": 0,
            "insider_texts": [],
        }
        res = score_three_way_evidence(empty_evidence)
        self.assertTrue(0.0 <= res["claim_score"] <= 100.0)
        self.assertTrue(0.0 <= res["proof_score"] <= 100.0)
        self.assertTrue(0.0 <= res["friction_score"] <= 100.0)
        self.assertTrue(0.0 <= res["awdi"] <= 100.0)
        self.assertTrue(0.15 <= res["conversion_rate"] <= 0.95)

    def test_log1p_monotonicity_across_orders_of_magnitude(self):
        small = {
            "filing_text": "AI research",
            "tech_texts": ["PyTorch"],
            "hf_models_count": 2,
            "hf_downloads": 500,
            "openalex_papers_count": 5,
            "openalex_citations": 50,
            "github_repos_count": 10,
            "github_stars": 100,
            "insider_texts": [],
        }
        large = {
            "filing_text": "AI research",
            "tech_texts": ["PyTorch", "CUDA", "vLLM", "Triton", "Rust"],
            "hf_models_count": 500,
            "hf_downloads": 25000000,
            "openalex_papers_count": 800,
            "openalex_citations": 95000,
            "github_repos_count": 1200,
            "github_stars": 350000,
            "insider_texts": [],
        }
        res_small = score_three_way_evidence(small)
        res_large = score_three_way_evidence(large)
        self.assertGreater(res_large["ai_stack_score"], res_small["ai_stack_score"])
        self.assertGreater(res_large["systems_stack_score"], res_small["systems_stack_score"])
        self.assertGreater(res_large["proof_score"], res_small["proof_score"])
        self.assertLessEqual(res_large["proof_score"], 99.0)

    def test_custom_weight_sensitivity_shifts_proof_score(self):
        evidence = {
            "filing_text": "Compute infrastructure",
            "tech_texts": ["CUDA, TensorRT, vLLM, FlashAttention, PyTorch"],
            "hf_models_count": 200,
            "hf_downloads": 5000000,
            "openalex_papers_count": 300,
            "openalex_citations": 40000,
            "github_repos_count": 1,
            "github_stars": 5,
            "insider_texts": [],
        }
        ai_heavy = score_three_way_evidence(
            evidence,
            weights={"ai_stack_weight": 0.85, "systems_stack_weight": 0.15},
        )
        sys_heavy = score_three_way_evidence(
            evidence,
            weights={"ai_stack_weight": 0.20, "systems_stack_weight": 0.80},
        )
        self.assertGreater(ai_heavy["proof_score"], sys_heavy["proof_score"])

    def test_deepseek_audit_parses_fenced_json_and_clamps_adjustments(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '```json\n{"claim_adjustment": 25.0, "proof_adjustment": -18.0, '
                            '"friction_adjustment": 5.0, "primary_contradiction": '
                            '"Claims proprietary LLM but uses API wrappers.", '
                            '"verified_ai_capabilities": ["PyTorch"], '
                            '"ai_washing_red_flags": ["API Wrapper"], '
                            '"executive_verdict": "High AI-washing risk."}\n```'
                        )
                    }
                }
            ]
        }
        with patch("requests.post", return_value=mock_resp):
            audit = run_deepseek_technical_audit(
                company_name="TestCorp",
                ticker="TEST",
                raw_evidence={"filing_text": "AI-First", "insider_texts": ["Wrapper"]},
                base_scores={"claim_score": 70.0, "proof_score": 40.0, "friction_score": 50.0},
                api_key="sk-test",
            )
        self.assertEqual(audit["mode"], "DeepSeek Technical Mode (Live LLM)")
        self.assertEqual(audit["claim_adjustment"], 10.0)
        self.assertEqual(audit["proof_adjustment"], -10.0)
        self.assertEqual(audit["friction_adjustment"], 5.0)
        self.assertIn("API wrappers", audit["primary_contradiction"])

    def test_deepseek_audit_handles_raw_json_without_fences(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"claim_adjustment": -2.0, "proof_adjustment": 6.5, '
                            '"friction_adjustment": -3.0, "primary_contradiction": '
                            '"Strong open-weight execution exceeds modest filings.", '
                            '"verified_ai_capabilities": ["CUDA", "Triton"], '
                            '"ai_washing_red_flags": [], '
                            '"executive_verdict": "Verified AI compounder."}'
                        )
                    }
                }
            ]
        }
        with patch("requests.post", return_value=mock_resp):
            audit = run_deepseek_technical_audit(
                company_name="HardTech",
                ticker="HTEC",
                raw_evidence={"filing_text": "Hardware", "insider_texts": []},
                base_scores={"claim_score": 50.0, "proof_score": 82.0, "friction_score": 20.0},
                api_key="sk-test",
            )
        self.assertEqual(audit["proof_adjustment"], 6.5)
        self.assertEqual(audit["claim_adjustment"], -2.0)

    def test_deepseek_audit_fallback_on_network_or_missing_key(self):
        audit_no_key = run_deepseek_technical_audit(
            company_name="TestCorp",
            ticker="TEST",
            raw_evidence={"filing_text": "AI", "insider_texts": []},
            base_scores={"claim_score": 60.0, "proof_score": 50.0, "friction_score": 30.0},
            api_key="",
        )
        self.assertIn("Deterministic", audit_no_key["mode"])

        with patch("requests.post", side_effect=TimeoutError("timeout")):
            audit_timeout = run_deepseek_technical_audit(
                company_name="TestCorp",
                ticker="TEST",
                raw_evidence={"filing_text": "AI", "insider_texts": []},
                base_scores={"claim_score": 60.0, "proof_score": 50.0, "friction_score": 30.0},
                api_key="sk-test",
            )
        self.assertIn("Deterministic", audit_timeout["mode"])
        self.assertTrue(len(audit_timeout["executive_verdict"]) > 10)
        self.assertIsInstance(load_deepseek_api_key(), str)


if __name__ == "__main__":
    unittest.main()
