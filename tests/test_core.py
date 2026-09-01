from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "analysis"))

from run_scan import build_output_paths, collect_tasks, format_float_for_path, write_macro_file
from transmutation_pipeline import _talys_channel_candidates, _talys_load_table, _talys_target_numbers


class ScanHelpersTest(unittest.TestCase):
    def test_format_and_paths(self) -> None:
        self.assertEqual(format_float_for_path(0.5), "0p5")
        macro, base = build_output_paths(Path("results"), "electron", "U", 200, 12, 20, 0.5)
        self.assertEqual(macro.name, "run.mac")
        self.assertIn("E200MeV", macro.as_posix())
        self.assertEqual(base.name, "run")

    def test_collect_tasks(self) -> None:
        tasks = collect_tasks(["electron", "proton"])
        self.assertIn(("electron", 200.0), tasks)
        self.assertIn(("proton", 800.0), tasks)
        self.assertEqual(len(tasks), 7)

    def test_macro_is_generated(self) -> None:
        macro = REPO_ROOT / "tests" / "generated.mac"
        with patch.object(Path, "write_text") as write_text:
            write_macro_file(macro, REPO_ROOT / "tests" / "run", "electron", 200, "U", 12, 20, 0.5, 10)
        text = write_text.call_args.args[0]
        self.assertIn("/detector/material natU", text)
        self.assertIn("/run/beamOn 10", text)


class TalysParserTest(unittest.TestCase):
    def test_target_and_channel_mapping(self) -> None:
        data_dir = REPO_ROOT / "analysis" / "data" / "talys" / "129I"
        self.assertEqual(_talys_target_numbers(data_dir), (53, 129))
        self.assertEqual(_talys_channel_candidates("n,gamma", 53, 129), [("rp053130", "tot")])

    def test_table_contains_finite_data(self) -> None:
        path = REPO_ROOT / "analysis" / "data" / "talys" / "135Cs" / "rp055136.tot"
        energy, sigma = _talys_load_table(path)
        self.assertGreater(energy.size, 10)
        self.assertEqual(energy.shape, sigma.shape)
        self.assertTrue((sigma >= 0).all())


if __name__ == "__main__":
    unittest.main()
