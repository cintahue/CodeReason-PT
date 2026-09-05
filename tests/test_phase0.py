from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Phase0PipelineTest(unittest.TestCase):
    def test_fixture_pipeline_passes_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_root = Path(tmpdir) / "phase0"
            raw_dir = ROOT / "tests" / "fixtures" / "raw"
            prepare = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "data.prepare",
                    "--config",
                    "configs/phase0.yaml",
                    "--raw-dir",
                    str(raw_dir),
                    "--artifact-root",
                    str(artifact_root),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn('"status": "prepared"', prepare.stdout)

            validate = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "data.validate",
                    "--config",
                    "configs/phase0.yaml",
                    "--artifact-root",
                    str(artifact_root),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn('"status": "passed"', validate.stdout)

            report_path = artifact_root / "reports" / "phase0_validation_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertTrue(report["gate"]["data_loadable"])
            self.assertEqual(report["gate"]["sft_pt_exact_id_overlap"], 0)
            self.assertEqual(report["gate"]["sft_pt_near_duplicate_overlap"], 0)
            self.assertTrue(report["gate"]["reward_heldout_split_fixed"])


if __name__ == "__main__":
    unittest.main()

