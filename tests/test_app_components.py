import io
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from analyst_calibration import DEFAULT_MODEL_WEIGHTS
from app import (
    _deoverlap_scatter_points,
    build_global_matrix_chart,
    build_rd_capitalization_bridge,
    build_three_way_radar_bar,
    recompute_universe_dataframe,
)
from cli import main as cli_main
from storage import AuditStorage


class TestAppAndCLIComponents(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmpdir.name) / "ui_test.db"
        self.store = AuditStorage(db_path=self.db_path)
        self.records = self.store.get_all_companies()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_recompute_universe_dataframe_columns_and_types(self):
        df = recompute_universe_dataframe(self.records, DEFAULT_MODEL_WEIGHTS)
        self.assertGreaterEqual(len(df), 14)
        expected_cols = {
            "Ticker",
            "Name",
            "Region",
            "Proof_Score",
            "Claim_Score",
            "Friction_Score",
            "AWDI",
            "Conversion_Pct",
            "ROIC_Adjusted_Pct",
            "Implied_Fair_PE",
            "Valuation_Spread_Pct",
            "Regime",
        }
        self.assertTrue(expected_cols.issubset(set(df.columns)))

    def test_recompute_universe_dataframe_weight_and_amortization_overrides(self):
        df_3yr = recompute_universe_dataframe(
            self.records, DEFAULT_MODEL_WEIGHTS, amortization_override=3
        )
        df_5yr = recompute_universe_dataframe(
            self.records, DEFAULT_MODEL_WEIGHTS, amortization_override=5
        )
        self.assertTrue((df_3yr["Amortization_Years"] == 3).all())
        self.assertTrue((df_5yr["Amortization_Years"] == 5).all())
        nvda_3 = df_3yr[df_3yr["Ticker"] == "NVDA"].iloc[0]["RD_Capital_Asset_B"]
        nvda_5 = df_5yr[df_5yr["Ticker"] == "NVDA"].iloc[0]["RD_Capital_Asset_B"]
        self.assertGreater(nvda_5, nvda_3)

    def test_deoverlap_scatter_points_separates_colliding_coordinates(self):
        # 4 nearly identical coordinates should be pushed apart deterministically
        xs = [80.0, 80.1, 80.0, 79.9]
        ys = [15.0, 15.1, 14.9, 15.0]
        px, py, pos = _deoverlap_scatter_points(xs, ys)
        self.assertEqual(len(px), 4)
        self.assertEqual(len(py), 4)
        self.assertEqual(len(pos), 4)
        for i in range(4):
            for j in range(i + 1, 4):
                dist = math.hypot(px[i] - px[j], py[i] - py[j])
                self.assertGreater(dist, 2.0, f"Points {i} and {j} still overlap: {dist}")

    def test_deoverlap_scatter_points_handles_empty_and_single_point(self):
        self.assertEqual(_deoverlap_scatter_points([], []), ([], [], []))
        px, py, pos = _deoverlap_scatter_points([55.0], [12.0])
        self.assertEqual(px, [55.0])
        self.assertEqual(py, [12.0])
        self.assertEqual(pos, ["top center"])

    def test_build_global_matrix_chart_light_and_dark_themes(self):
        df = recompute_universe_dataframe(self.records, DEFAULT_MODEL_WEIGHTS)
        fig_light = build_global_matrix_chart(df, theme_mode="light", highlight_ticker="NVDA")
        fig_dark = build_global_matrix_chart(df, theme_mode="dark", highlight_ticker="META")
        self.assertGreaterEqual(len(fig_light.data), 2)
        self.assertGreaterEqual(len(fig_dark.data), 2)

    def test_build_rd_capitalization_bridge_and_radar_charts(self):
        df = recompute_universe_dataframe(self.records, DEFAULT_MODEL_WEIGHTS)
        row = df[df["Ticker"] == "NVDA"].iloc[0].to_dict()
        fig_bridge = build_rd_capitalization_bridge(row, theme_mode="light")
        fig_bar = build_three_way_radar_bar(row, theme_mode="dark")
        self.assertEqual(len(fig_bridge.data), 1)
        self.assertEqual(len(fig_bar.data), 1)

    def test_cli_list_ticker_and_csv_export(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = cli_main(["--ticker", "NVDA"])
        self.assertEqual(code, 0)
        self.assertIn("NVIDIA", buf.getvalue())

        csv_path = Path(self.tmpdir.name) / "export.csv"
        buf_csv = io.StringIO()
        with redirect_stdout(buf_csv):
            code_csv = cli_main(["--export-csv", str(csv_path)])
        self.assertEqual(code_csv, 0)
        self.assertTrue(csv_path.exists())
        self.assertIn("NVDA", csv_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
