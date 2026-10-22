# CapEx-to-Code

**Quality-Adjusted Damodaran R&D Capitalization & AI Engineering Telemetry Audit (US • UK • EU Equities)**

[![Live Research Terminal](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://capex-to-code.streamlit.app/)

- **Live Web Application**: **[https://capex-to-code.streamlit.app/](https://capex-to-code.streamlit.app/)** *(Open directly in any desktop or mobile browser—no local installation required)*

Under US GAAP and IFRS, 100% of corporate R&D is expensed immediately on the income statement. Standard financial statements treat \$1 / €1 spent on CUDA kernels, custom silicon, and open-weight foundation models identically to \$1 / €1 diluted by third-party API wrappers, external vendor coordination (*Dienstleistersteuerung*), and agile overhead.

**CapEx-to-Code** cross-examines executive filing claims against verifiable open-source and academic engineering output (Hugging Face Hub, OpenAlex citation graphs, GitHub repositories, and developer forums) to estimate a **Productive R&D Conversion Rate ($\eta$)**, and feeds $\eta$ into **Aswath Damodaran's R&D Capitalization framework** to re-estimate Adjusted EBIT, Adjusted ROIC, and Implied Fair Forward P/E.

---

## Getting Started & Usage

> **Quick Access (Cloud Hosted)**: You can explore the full interactive terminal online at **[https://capex-to-code.streamlit.app/](https://capex-to-code.streamlit.app/)**, or run it locally using the instructions below.

### Prerequisites

- **Python 3.10+** (`3.10`, `3.11`, `3.12`, or `3.14`)
- **Git**
- *(Optional)* A `DEEPSEEK_API_KEY` if you want to trigger live LLM qualitative audit memos via `deepseek-chat`. The core scoring and valuation engine runs 100% deterministically offline and online without an API key.

---

### 1. Installation by Operating System

#### macOS & Linux (`zsh` / `bash`)

```bash
# 1. Clone the repository and enter the directory
git clone https://github.com/ChenYuze-f/capex-to-code.git
cd capex-to-code

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. (Optional) Configure DeepSeek API key for live qualitative memos
cp .env.example .env
# Edit .env and paste your key: DEEPSEEK_API_KEY=sk-...

# 5. Launch the interactive web terminal
streamlit run app.py
```

#### Windows (`PowerShell`)

```powershell
# 1. Clone the repository and enter the directory
git clone https://github.com/ChenYuze-f/capex-to-code.git
cd capex-to-code

# 2. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# Note: If PowerShell blocks script execution, run this first:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. (Optional) Configure DeepSeek API key
Copy-Item .env.example .env
# Open .env in Notepad and set DEEPSEEK_API_KEY=sk-...

# 5. Launch the interactive web terminal
streamlit run app.py
```

#### Windows (`Command Prompt / CMD`)

```cmd
git clone https://github.com/ChenYuze-f/capex-to-code.git
cd capex-to-code

python -m venv .venv
.venv\Scripts\activate.bat

pip install -r requirements.txt
copy .env.example .env

streamlit run app.py
```

Once launched, open **`http://localhost:8501`** in your browser.

---

### 2. How to Use the Web Terminal

1. **Cover View & Quick-Jump**:
   - On startup, you can toggle between **Light Mode** and **Dark Mode** in the top-right corner.
   - Click any of the 6 benchmark buttons (`META`, `NVDA`, `PLTR`, `ARM`, `IFX.DE`, `VOW3.DE`) to jump straight into a company audit, or type any ticker/company name into the search input and click **Launch Research Terminal →**.
2. **Searching Any Global Stock Ticker**:
   - Inside the terminal, use the top search bar to query any public ticker across US, UK, or European exchanges (e.g., `AAPL`, `GOOGL`, `AMD`, `ASML.AS`, `BA.L`, `SAP.DE`, `VOW3.DE`) and click **Analyze / Switch Company**.
   - If the company is not yet in the local SQLite cache (`audit_cache.db`), the collector fetches multi-year financials (`yfinance`) and public engineering telemetry on the fly and caches the result locally.
3. **Inspecting Metric Definitions (`ⓘ`) & Interactive Charts**:
   - Every KPI card includes a clickable **`ⓘ`** popover showing the exact formula and interpretation threshold.
   - Switch between the **Global Peer Matrix**, **Damodaran R&D Capitalization Bridge**, and **Three-Way Score Decomposition** using the chart selector.
4. **Expanding Detailed Panels & Calibrating Parameters**:
   - Use the left sidebar sliders to test sensitivity to **Contemporary AI vs. Systems Stack weights**, **AI-Washing penalty sensitivity**, and **R&D Amortization schedules (3–6 years)**.
   - Click the three bottom expansion buttons to inspect raw three-way evidence quotes, the full GAAP/IFRS vs. Damodaran accounting bridge table, or save your own analyst notes to SQLite.

---

### 3. Command-Line Interface (`cli.py`)

For headless workflows, batch inspection, or exporting data to CSV without opening a browser:

```bash
# Print the ranked cross-sectional peer table in the terminal
python cli.py --list

# Inspect or run a full audit for a single ticker
python cli.py --ticker NVDA
python cli.py --ticker VOW3.DE --amort-years 5

# Export the full recomputed universe to a CSV file
python cli.py --export-csv peer_valuation_matrix.csv
```

---

### 4. Running the Test Suite

The repository includes **38 unit and integration tests** covering telemetry normalization, `AWDI` boundary conditions, multi-year R&D amortization schedules, SQLite persistence, CLI output, and 2D scatter anti-collision layout:

```bash
# macOS / Linux / Windows (inside activated .venv)
python -m unittest discover -s tests -v
```

---

## Methodology & Mathematical Formulation

### 1. Three-Way Engineering Telemetry & AI-Washing Divergence

Because open-source repository stars, model downloads, and academic citations follow heavy power-law distributions, raw counts are transformed via `math.log1p` before combining with domain keyword density:

$$
S_{\text{proof}} = w_{\text{AI}} \cdot S_{\text{AI}} + w_{\text{Sys}} \cdot S_{\text{Systems}} \qquad (w_{\text{AI}} = 0.65,\; w_{\text{Sys}} = 0.35)
$$

The **AI-Washing Divergence Index ($\text{AWDI}$)** measures the positive gap between executive filing claims ($S_{\text{claim}}$) and verified engineering proof ($S_{\text{proof}}$), scaled by developer friction ($S_{\text{friction}}$):

$$
\text{AWDI} = \text{clip}\left(\max(0,\; S_{\text{claim}} - S_{\text{proof}}) \times \left(1 + \frac{S_{\text{friction}}}{100}\right),\; 0,\; 100\right)
$$

- If $S_{\text{proof}} \ge S_{\text{claim}}$ (grounded engineering execution), $\text{AWDI} = 0.0$.
- The **Productive R&D Conversion Factor ($\eta_{\text{conv}}$)** determines the share of annual R&D that forms durable intangible capital:

$$
\eta_{\text{conv}} = \text{clip}\left(\frac{0.65 \cdot S_{\text{proof}} + 0.35 \cdot (100 - S_{\text{friction}}) - 0.25 \cdot \text{AWDI}}{100},\; 0.15,\; 0.95\right)
$$

### 2. Quality-Adjusted Damodaran R&D Capitalization

Given sector economic amortization life $L \in [3, 6]$ years and historical R&D series $(RD_t, RD_{t-1}, \dots, RD_{t-L})$:

$$
RD_{\text{prod}, k} = RD_{t-k} \cdot \eta_{\text{conv}}, \qquad RD_{\text{waste}, k} = RD_{t-k} \cdot (1 - \eta_{\text{conv}})
$$

$$
RC_{\text{asset}} = \sum_{k=0}^{L-1} RD_{\text{prod}, k} \cdot \frac{L - k}{L}, \qquad A_{\text{RD}} = \sum_{k=1}^{L} \frac{RD_{\text{prod}, k}}{L}
$$

$$
\text{EBIT}_{\text{adj}} = \text{EBIT}_{\text{rep}} + RD_{\text{prod}, t} - A_{\text{RD}}, \qquad \text{ROIC}_{\text{adj}} = \frac{\text{EBIT}_{\text{adj}} \cdot (1 - \tau)}{\text{IC}_{\text{rep}} + RC_{\text{asset}}}
$$

---

## Project Layout

```text
capex-to-code/
├── app.py                        # Streamlit web application & 2D anti-collision Plotly visuals
├── cli.py                        # Headless CLI runner & CSV export utility
├── ai_audit_engine.py            # Log-scale 3-way telemetry scoring, AWDI, and DeepSeek audit client
├── valuation_model.py            # Damodaran R&D capitalization schedule, Adjusted ROIC & Fair P/E
├── data_collectors.py            # Multi-source collectors (yfinance, Hugging Face, OpenAlex, GitHub, HN/Reddit)
├── analyst_calibration.py        # Sector amortization lives, signal taxonomy & analyst conviction notes
├── storage.py                    # SQLite persistence layer (audit_cache.db)
├── baseline_universe.json        # Seed snapshot of 14 US/UK/EU benchmark companies
├── ARCHITECTURE_WALKTHROUGH.md   # Deep-dive mathematical and architectural documentation
├── requirements.txt              # Python package dependencies
├── pyproject.toml                # Package metadata and CLI entrypoint definition
├── Makefile                      # Developer shortcuts (make test, make run, make list)
├── .env.example                  # Environment variable template for optional DeepSeek API key
└── tests/                        # 38 unit and integration tests across 5 test modules
    ├── test_analyst_calibration.py
    ├── test_ai_audit_engine.py
    ├── test_valuation_model.py
    ├── test_collectors_and_storage.py
    └── test_app_components.py
```
