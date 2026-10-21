"""
Command-line interface for CapEx-to-Code.

Allows running single-ticker audits, listing the cached universe with quality-adjusted
Damodaran R&D valuation metrics, or exporting the cross-sectional table to CSV
without launching the browser UI.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from ai_audit_engine import load_deepseek_api_key
from analyst_calibration import DEFAULT_MODEL_WEIGHTS
from app import recompute_universe_dataframe
from data_collectors import normalize_ticker_and_region, run_full_company_audit
from storage import AuditStorage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="capex-to-code",
        description="CapEx-to-Code: R&D Capitalization & AI Engineering Telemetry Audit CLI",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the ranked cross-sectional peer table from the local SQLite database.",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Audit or inspect a specific ticker (e.g., NVDA, META, ARM, IFX.DE, VOW3.DE).",
    )
    parser.add_argument(
        "--live-deepseek",
        action="store_true",
        help="Enable live DeepSeek API qualitative synthesis when auditing a ticker.",
    )
    parser.add_argument(
        "--amort-years",
        type=int,
        choices=[3, 4, 5, 6],
        default=None,
        help="Override sector-default R&D amortization life (in years).",
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        default=None,
        help="Export the full recomputed universe table to the specified CSV path.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    store = AuditStorage()
    api_key = load_deepseek_api_key()

    if args.ticker:
        resolved_ticker, _, _ = normalize_ticker_and_region(args.ticker)
        existing = store.get_company(resolved_ticker)
        if existing is None or args.live_deepseek:
            rec = run_full_company_audit(
                ticker=resolved_ticker,
                use_deepseek=bool(args.live_deepseek and api_key),
                api_key=api_key,
                weights=DEFAULT_MODEL_WEIGHTS,
            )
            store.upsert_company(rec)

        records = store.get_all_companies()
        df = recompute_universe_dataframe(
            records,
            weights=DEFAULT_MODEL_WEIGHTS,
            amortization_override=args.amort_years,
        )
        sub = df[df["Ticker"] == resolved_ticker]
        if sub.empty:
            print(f"Ticker {resolved_ticker} could not be resolved.", file=sys.stderr)
            return 1

        row = sub.iloc[0]
        print(f"=== {row['Name']} ({row['Ticker']}) [{row['Region']} | {row['Sector']}] ===")
        print(f"Regime:                 {row['Regime']}")
        print(f"Technical Proof Score:  {row['Proof_Score']:.1f} / 100")
        print(f"Filing Claim Score:     {row['Claim_Score']:.1f} / 100")
        print(f"Developer Friction:     {row['Friction_Score']:.1f} / 100")
        print(f"AI-Washing (AWDI):      {row['AWDI']:.1f} / 100")
        print(f"Productive R&D (eta):   {row['Conversion_Pct']:.1f}% ({row['CurrSym']}{row['Productive_RD_B']:.2f}B of {row['CurrSym']}{row['RD_Current_B']:.2f}B)")
        print(f"Unamortized R&D Asset:  {row['CurrSym']}{row['RD_Capital_Asset_B']:.2f}B ({row['Amortization_Years']}-yr schedule)")
        print(f"ROIC (GAAP -> Adj):     {row['ROIC_Reported_Pct']:.1f}% -> {row['ROIC_Adjusted_Pct']:.1f}% ({row['ROIC_Delta_Pct']:+.1f}%)")
        print(f"P/E (Market -> Fair):   {row['PE_Market']:.1f}x -> {row['Implied_Fair_PE']:.1f}x ({row['Valuation_Spread_Pct']:+.1f}% spread)")
        print(f"Verdict:                {row['Executive_Verdict']}")
        return 0

    records = store.get_all_companies()
    df = recompute_universe_dataframe(
        records,
        weights=DEFAULT_MODEL_WEIGHTS,
        amortization_override=args.amort_years,
    ).sort_values(by="Proof_Score", ascending=False)

    if args.export_csv:
        out_path = Path(args.export_csv)
        cols = [
            "Ticker",
            "Name",
            "Region",
            "Sector",
            "Proof_Score",
            "Claim_Score",
            "Friction_Score",
            "AWDI",
            "Conversion_Pct",
            "RD_Current_B",
            "Productive_RD_B",
            "RD_Capital_Asset_B",
            "ROIC_Reported_Pct",
            "ROIC_Adjusted_Pct",
            "PE_Market",
            "Implied_Fair_PE",
            "Valuation_Spread_Pct",
            "Regime",
        ]
        df[cols].to_csv(out_path, index=False)
        print(f"Exported {len(df)} companies to {out_path}")
        return 0

    cols = [
        "Ticker",
        "Region",
        "Proof_Score",
        "AWDI",
        "Conversion_Pct",
        "ROIC_Adjusted_Pct",
        "PE_Market",
        "Implied_Fair_PE",
        "Valuation_Spread_Pct",
        "Regime",
    ]
    print(df[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
