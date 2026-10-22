# CapEx-to-Code: System Architecture & Quantitative Methodology

This document details the mathematical formulation, data ingestion pipeline, and software architecture of **CapEx-to-Code**.

---

## 1. Analyst Calibration Layer (`analyst_calibration.py`)

The model exposes explicit domain parameters in `analyst_calibration.py`:
- **Sector R&D Economic Amortization Lives ($L \in [3, 6]$ years)**:
  - `Enterprise Software & SaaS`: **3 years** (rapid model and framework obsolescence).
  - `Hyperscalers & Cloud Platforms`: **4 years**.
  - `Automotive & Industrial Tech`: **4 years**.
  - `Semiconductors & AI Silicon`: **5 years** (long-cycle instruction-set and hardware IP moats).
  - `Biopharma & Computational Life Sciences`: **6 years**.
- **Signal Taxonomy & Weights**:
  - `DEFAULT_MODEL_WEIGHTS`: Configures the balance between Contemporary AI Signals (65%) and Computer Systems Signals (35%), alongside AI-Washing penalty sensitivity.
  - `ANALYST_CONVICTION_NOTES`: Stores persistent analyst due-diligence notes keyed by ticker.

---

## 2. Three-Way Evidence Scoring (`ai_audit_engine.py`)

### 2.1 Technical Proof Score ($S_{\text{proof}}$)
To avoid linear distortion caused by power-law distributions in open-source stars and academic citations, all raw telemetry counts undergo logarithmic normalization (`math.log1p`):

1. **Contemporary AI Stack Score ($S_{\text{AI}}$, 65% weight)**:
   - Aggregates Hugging Face Hub model counts and downloads, OpenAlex AI research publications and citations, and high-signal AI engineering keywords (`CUDA`, `TensorRT`, `vLLM`, `PyTorch`, `Triton`, `FlashAttention`).
2. **Computer Systems Stack Score ($S_{\text{Systems}}$, 35% weight)**:
   - Aggregates GitHub public repository counts, star velocity, and systems engineering keywords (`Rust`, `C++20`, `RISC-V`, `LLVM`, `Kubernetes`, `Distributed Systems`).

$$S_{\text{proof}} = 0.65 \cdot S_{\text{AI}} + 0.35 \cdot S_{\text{Systems}}$$

### 2.2 AI-Washing Divergence Index ($\text{AWDI}$)
Measures the positive gap between executive filing claims ($S_{\text{claim}}$) and verified engineering proof ($S_{\text{proof}}$), amplified by internal engineering friction ($S_{\text{friction}}$):

$$\text{AWDI} = \text{clip}\!\left(\max\big(0,\; S_{\text{claim}} - S_{\text{proof}}\big) \times \left(1 + \frac{S_{\text{friction}}}{100}\right),\; 0,\; 100\right)$$

- Companies with grounded technical execution where $S_{\text{proof}} \ge S_{\text{claim}}$ receive $\text{AWDI} = 0$.
- Companies with aggressive filing rhetoric but low verifiable output and high developer friction see $\text{AWDI}$ amplified up to `100`.

### 2.3 Productive R&D Conversion Factor ($\eta_{\text{conv}}$)
$$\eta_{\text{conv}} = \text{clip}\!\left(\frac{0.65 \cdot S_{\text{proof}} + 0.35 \cdot (100 - S_{\text{friction}}) - 0.25 \cdot \text{AWDI}}{100},\; 0.15,\; 0.95\right)$$

---

## 3. Quality-Adjusted Damodaran R&D Capitalization (`valuation_model.py`)

Standard IFRS and US GAAP expense 100% of R&D in the period incurred, depressing operating income ($\text{EBIT}$) and omitting intangible R&D capital from Invested Capital ($\text{IC}$). Conversely, unadjusted R&D capitalization treats unproductive bureaucratic R&D as an asset.

Given amortization life $L$ years and historical R&D series $\{R\&D_t, R\&D_{t-1}, \dots, R\&D_{t-L}\}$:

1. **Productive vs. Deadweight R&D Split**:
   $$R\&D_{\text{prod}, k} = R\&D_{t-k} \cdot \eta_{\text{conv}}, \qquad R\&D_{\text{waste}, k} = R\&D_{t-k} \cdot (1 - \eta_{\text{conv}})$$

2. **Unamortized Research Capital Asset ($RC_{\text{asset}}$)**:
   $$RC_{\text{asset}} = \sum_{k=0}^{L-1} R\&D_{\text{prod}, k} \cdot \frac{L - k}{L}$$

3. **Current-Year R&D Amortization ($A_{\text{RD}}$)**:
   $$A_{\text{RD}} = \sum_{k=1}^{L} \frac{R\&D_{\text{prod}, k}}{L}$$

4. **Adjusted Operating Income ($\text{EBIT}_{\text{adj}}$) & Adjusted ROIC**:
   $$\text{EBIT}_{\text{adj}} = \text{EBIT}_{\text{rep}} + R\&D_{\text{prod}, t} - A_{\text{RD}}$$
   $$\text{IC}_{\text{adj}} = \text{IC}_{\text{rep}} + RC_{\text{asset}}, \qquad \text{ROIC}_{\text{adj}} = \frac{\text{EBIT}_{\text{adj}} \cdot (1 - \tau)}{\text{IC}_{\text{adj}}}$$

---

## 4. Multi-Source Data Ingestion & Bounded LLM Audit (`data_collectors.py`)

When an analyst queries any global ticker (`NVDA`, `META`, `ARM`, `IFX.DE`, `VOW3.DE`, etc.):
1. **Financial Statements (`yfinance`)**: Retrieves multi-year R&D expenditure, operating income, invested capital, market capitalization, and forward P/E.
2. **Engineering & Research Telemetry**: Queries Hugging Face Hub API, OpenAlex API, GitHub REST API, Hacker News Algolia API, and Reddit JSON endpoints with automatic fallback to local SQLite persistence (`storage.py`).
3. **Bounded DeepSeek Technical Audit (`deepseek-chat`)**: Synthesizes qualitative three-way evidence into structured JSON (`primary_contradiction`, `executive_verdict`, and bounded `llm_conviction_delta` within `[-8.0, +8.0]`), ensuring full mathematical determinism in the core valuation engine.
