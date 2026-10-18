"""
Author Calibration & Human-in-the-Loop Quantitative Parameters.

This module serves as the primary calibration workbench for the author.
You can directly customize:
  1. `SECTOR_AMORTIZATION_YEARS`: Economic useful life (years) of R&D capital
     assets across industries under the Damodaran R&D Capitalization framework.
  2. `CONTEMPORARY_AI_SIGNALS` (65% proof weight) & `SYSTEMS_ENGINEERING_SIGNALS`
     (35% proof weight): Keyword weights for evaluating verifiable technical output.
  3. `AI_WASHING_SIGNALS` & `BUREAUCRACY_BLOAT_SIGNALS`: Penalty weights for
     detecting fake AI wrapper hype and organizational R&D waste.
  4. `AUTHOR_THESIS_NOTES`: Your personal analyst conviction notes per company.
"""

from __future__ import annotations

from typing import Dict

# ──────────────────────────────────────────────────────────────────────────────
# 1. DEFAULT FACTOR WEIGHTS
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_MODEL_WEIGHTS: Dict[str, float] = {
    "ai_stack_weight": 0.65,         # Primary priority: Contemporary AI execution
    "systems_stack_weight": 0.35,    # Secondary priority: Computer systems foundation
    "proof_weight": 0.65,            # Weight of Technical Proof in R&D conversion rate
    "culture_weight": 0.35,          # Weight of low Insider Friction in R&D conversion rate
    "awdi_penalty_weight": 0.25,     # Penalty sensitivity to AI-Washing Divergence (AWDI)
}

# ──────────────────────────────────────────────────────────────────────────────
# 2. DAMODARAN SECTOR R&D AMORTIZATION LIVES (YEARS)
# ──────────────────────────────────────────────────────────────────────────────
# Fast-moving AI/cloud software has a shorter economic half-life (~3 years),
# whereas semiconductor architectures, industrial automation, and pharma IP
# amortize over 4–6 years.
SECTOR_AMORTIZATION_YEARS: Dict[str, int] = {
    "software": 3,
    "internet": 3,
    "cloud": 3,
    "information technology": 3,
    "semiconductor": 4,
    "silicon": 4,
    "hardware": 4,
    "industrial": 4,
    "automation": 4,
    "automotive": 5,
    "aerospace": 5,
    "biotechnology": 6,
    "pharmaceutical": 6,
    "healthcare": 5,
}

# ──────────────────────────────────────────────────────────────────────────────
# 3. CONTEMPORARY AI EXECUTION SIGNALS (65% OF TECHNICAL PROOF SCORE)
# ──────────────────────────────────────────────────────────────────────────────
CONTEMPORARY_AI_SIGNALS: Dict[str, float] = {
    "PyTorch": 9.5,
    "CUDA": 10.0,
    "TensorRT": 10.0,
    "vLLM": 10.0,
    "Triton": 9.5,
    "FlashAttention": 9.5,
    "Fine-Tuning": 8.5,
    "LoRA": 8.5,
    "RLHF": 9.0,
    "DPO": 8.5,
    "RAG": 8.0,
    "Vector Database": 7.5,
    "ONNX": 8.5,
    "Safetensors": 8.5,
    "Quantization": 9.0,
    "FP8": 9.5,
    "INT4": 9.0,
    "Multi-GPU": 9.0,
    "Distributed Training": 9.5,
    "NCCL": 9.5,
    "Megatron": 9.5,
    "DeepSpeed": 9.0,
    "Graph Neural Network": 8.5,
    "Molecular Dynamics AI": 9.0,
}

# ──────────────────────────────────────────────────────────────────────────────
# 4. COMPUTER SYSTEMS & SILICON SIGNALS (35% OF TECHNICAL PROOF SCORE)
# ──────────────────────────────────────────────────────────────────────────────
SYSTEMS_ENGINEERING_SIGNALS: Dict[str, float] = {
    "Rust": 10.0,
    "C++20": 9.5,
    "C++17": 8.5,
    "RISC-V": 9.5,
    "FPGA": 9.0,
    "RTOS": 8.5,
    "Embedded Linux": 8.5,
    "Kernel": 9.0,
    "Compiler": 9.5,
    "LLVM": 9.5,
    "Distributed Systems": 9.0,
    "Low-Latency": 9.0,
    "Zero-Copy": 9.0,
    "SIMD": 8.5,
    "Kubernetes": 7.5,
    "gRPC": 7.5,
    "eBPF": 9.0,
    "Custom Silicon": 9.5,
}

# ──────────────────────────────────────────────────────────────────────────────
# 5. EXECUTIVE FILING AI HYPE SIGNALS (USED TO MEASURE CLAIM INTENSITY)
# ──────────────────────────────────────────────────────────────────────────────
CLAIM_HYPE_SIGNALS: Dict[str, float] = {
    "AI-First": 9.5,
    "Autonomous Agents": 9.5,
    "Foundation Model": 9.0,
    "GenAI Platform": 9.0,
    "Proprietary LLM": 9.5,
    "Generative AI": 8.0,
    "Artificial Intelligence": 7.0,
    "Agentic Workflow": 9.0,
    "AI Transformation": 8.5,
    "Cognitive Computing": 8.5,
    "Next-Gen AI": 8.0,
    "AI Revolution": 8.5,
    "Software-Defined Vehicle": 8.0,
    "Digital Twin": 7.5,
}

# ──────────────────────────────────────────────────────────────────────────────
# 6. AI-WASHING & FAKE AI SIGNALS (60% OF INSIDER FRICTION PENALTY)
# ──────────────────────────────────────────────────────────────────────────────
AI_WASHING_SIGNALS: Dict[str, float] = {
    "API Wrapper": 10.0,
    "OpenAI Wrapper": 10.0,
    "Prompt Wrapper": 9.5,
    "Prompt Engineering Only": 9.0,
    "Rebranded BI": 9.0,
    "Excel Automation": 8.5,
    "If-Else Rules": 9.0,
    "AI Buzzword": 8.5,
    "Vaporware": 9.5,
    "No GPU Cluster": 9.0,
    "Marketing Stunt": 9.0,
    "Slideware": 8.5,
    "Outsourced Data Labeling": 8.0,
    "Consulting Rebrand": 8.5,
}

# ──────────────────────────────────────────────────────────────────────────────
# 7. ORGANIZATIONAL BLOAT & TECH DEBT SIGNALS (40% OF INSIDER FRICTION PENALTY)
# ──────────────────────────────────────────────────────────────────────────────
BUREAUCRACY_BLOAT_SIGNALS: Dict[str, float] = {
    "Dienstleistersteuerung": 9.5,
    "External Vendor Management": 9.0,
    "Outsourcing": 8.5,
    "SAFe Agile Coach": 9.0,
    "Scrum Master": 7.5,
    "Sub-supplier Coordination": 8.5,
    "Legacy Maintenance": 8.0,
    "Tech Debt": 8.0,
    "Historisch gewachsen": 9.0,
    "Slow Approval": 8.0,
    "Steering Committee": 7.5,
    "PowerPoint Governance": 9.0,
    "Excel-Reporting": 8.0,
}

# ──────────────────────────────────────────────────────────────────────────────
# 8. AUTHOR'S PERSONAL ANALYST CONVICTION NOTES
# ──────────────────────────────────────────────────────────────────────────────
AUTHOR_THESIS_NOTES: Dict[str, str] = {
    "NVDA": (
        "Author Note: NVIDIA's true moat is not merely H100/Blackwell silicon, but the "
        "CUDA, TensorRT-LLM, and NCCL software stack where >85% of R&D directly compounds "
        "into developer lock-in. Capitalizing R&D over 4 years reveals substantially "
        "higher economic ROIC than GAAP statements report."
    ),
    "MSFT": (
        "Author Note: Microsoft combines massive frontier AI infrastructure investment "
        "(DeepSpeed, ONNX Runtime, Azure AI kernels) with aggressive Copilot monetization. "
        "High technical proof backs management's filing claims, keeping AWDI near zero."
    ),
    "META": (
        "Author Note: Meta's open-weight Llama series, PyTorch stewardship, and FAIR research "
        "make it one of the highest-proof AI organizations globally. Capitalizing its $40B+ "
        "R&D over 3–4 years shows strong productive asset conversion despite past Reality Labs drag."
    ),
    "GOOGL": (
        "Author Note: Alphabet invented the Transformer and leads in custom TPU v5/Trillium silicon, "
        "JAX/XLA compilers, and DeepMind frontier research. High technical proof supports strong "
        "R&D asset formation at a reasonable forward multiple."
    ),
    "AAPL": (
        "Author Note: Apple's R&D excels in custom Apple Silicon (M-series/A-series Neural Engine), "
        "MLX on-device framework, and hardware-software vertical integration, though frontier "
        "cloud LLM execution lagged peers prior to Apple Intelligence."
    ),
    "TSLA": (
        "Author Note: Tesla deploys genuine end-to-end neural networks and FSD training clusters, "
        "yet executive filing/earnings claims around full autonomy and robotics carry extreme "
        "narrative intensity relative to open technical artifacts."
    ),
    "PLTR": (
        "Author Note: Palantir's AIP narrative in SEC filings is extraordinarily loud "
        "(Claim Score >90). While Foundry/Apollo distributed systems engineering is solid, "
        "sparse open foundation-model weights and forward-deployed consulting debates create "
        "a moderate narrative-to-artifact divergence that warrants valuation discipline."
    ),
    "IBM": (
        "Author Note: After years of 'Watson' AI over-promising, IBM's open-weight Granite "
        "models and Red Hat OpenShift/vLLM stack have partially rebuilt credibility, yet "
        "legacy consulting bloat and mainframes still dilute overall R&D conversion."
    ),
    "ARM": (
        "Author Note: UK semiconductor architecture leader whose R&D produces ultra-high-leverage "
        "instruction set IP (Armv9, KleidiAI kernels for LLM inference on edge/server CPUs). "
        "A classic long-duration R&D asset (4-year amortization) trading at a premium multiple."
    ),
    "REL.L": (
        "Author Note: RELX (London) is a quiet UK compounder: rather than hyping generic chatbots, "
        "it embeds domain-specific RAG and knowledge-graph retrieval over proprietary legal/scientific "
        "corpora (Lexis+ AI, Elsevier), achieving steady R&D conversion."
    ),
    "AZN": (
        "Author Note: AstraZeneca applies geometric deep learning, PyTorch molecular dynamics, "
        "and biological foundation models to target discovery. With a 6-year biopharma R&D "
        "amortization window, capitalizing productive R&D materially lifts adjusted ROIC."
    ),
    "SAP.DE": (
        "Author Note: SAP possesses elite C++20/SIMD systems engineering inside the HANA Cloud "
        "engine, but its high "
        "Generative AI filing claims ('Joule') face internal developer friction from legacy "
        "ABAP migration and agile ceremony layers."
    ),
    "IFX.DE": (
        "Author Note: Infineon is the archetypal understated deep-tech play: modest AI buzzword "
        "claims in filings, but dense RISC-V, Rust/C++20 safety microkernels, and edge TensorRT/ONNX "
        "compilers. Zero AI-washing and highly productive R&D conversion at an industrial multiple."
    ),
    "VOW3.DE": (
        "Author Note: Volkswagen's €15.8B annual R&D is the textbook case of accounting illusion: "
        "ambitious 'Software-Defined Vehicle' claims collide with severe CARIAD outsourcing "
        "(Dienstleistersteuerung) and multi-vendor integration bloat, destroying R&D capital formation."
    ),
}


def get_amortization_years(sector: str) -> int:
    """Return Damodaran R&D amortization life (in years) for a given sector string."""
    if not sector:
        return 4
    sec_lower = sector.lower()
    for key, years in SECTOR_AMORTIZATION_YEARS.items():
        if key in sec_lower:
            return years
    return 4


def get_author_thesis(ticker: str) -> str:
    """Return the author's personal conviction note for a ticker, or a default prompt."""
    clean = (ticker or "").strip().upper()
    if clean in AUTHOR_THESIS_NOTES:
        return AUTHOR_THESIS_NOTES[clean]
    return (
        f"Author Note ({clean}): Live-audited international issuer. Use the Calibration "
        f"Workbench in Tab 3 to record your custom fundamental & technical thesis for {clean}."
    )
