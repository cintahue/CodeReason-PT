from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from data.build_raw import _assemble_records
from data.schemas import make_problem_id, make_source_id


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
            self.assertEqual(report["gate"]["sft_dev_near_duplicate_overlap"], 0)
            self.assertEqual(report["gate"]["pt_dev_near_duplicate_overlap"], 0)

    def test_ocr_wrong_source_split_is_not_silently_mapped(self) -> None:
        train_source_id = make_source_id("train", 42)
        test_source_id = make_source_id("test", 42)
        train_source_record = {
            "source": "apps",
            "source_id": train_source_id,
            "difficulty": "easy",
            "prompt": "train split prompt",
            "reference_code": "print(1)",
            "tests": [{"input": "1\n", "output": "1\n"}],
            "metadata": {"source_split": "train", "source_index": 42},
        }
        test_candidate = {
            "problem_id": make_problem_id("apps", test_source_id),
            "dataset": "apps",
            "source_split": "test",
            "source_id": test_source_id,
            "index": 42,
            "reasoning": "Use the test split row.",
            "solution": "",
            "opencode_id": "ocr-test",
            "opencode_question_id": "question-test",
            "opencode_source_split": "test",
            "opencode_pass_rate": "1.0",
            "opencode_judgement": "right",
        }

        problems, reasonings, skip_counts, output_counts = _assemble_records(
            [test_candidate],
            {"apps": {("train", 42): train_source_record}},
            target=1,
        )

        self.assertEqual(problems, [])
        self.assertEqual(reasonings, [])
        self.assertEqual(output_counts, {})
        self.assertEqual(skip_counts["missing_source_problem"], 1)

        test_source_record = {
            **train_source_record,
            "source_id": test_source_id,
            "prompt": "test split prompt",
            "metadata": {"source_split": "test", "source_index": 42},
        }
        problems, reasonings, skip_counts, output_counts = _assemble_records(
            [test_candidate],
            {"apps": {("test", 42): test_source_record}},
            target=1,
        )

        self.assertEqual(len(problems), 1)
        self.assertEqual(len(reasonings), 1)
        self.assertEqual(problems[0]["source_id"], test_source_id)
        self.assertEqual(problems[0]["prompt"], "test split prompt")
        self.assertEqual(output_counts, {"apps/test": 1})
        self.assertEqual(skip_counts["missing_source_problem"], 0)

    def test_validator_rejects_manifest_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_root = Path(tmpdir) / "phase0"
            raw_dir = ROOT / "tests" / "fixtures" / "raw"
            subprocess.run(
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
            manifest_path = artifact_root / "reports" / "test_split_manifest.jsonl"
            manifest_records = [
                json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()
            ]
            manifest_records[0]["reward_test_ids"][0] = "tampered_test_id"
            manifest_path.write_text(
                "\n".join(json.dumps(record, sort_keys=True) for record in manifest_records) + "\n",
                encoding="utf-8",
            )

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
                check=False,
            )

            self.assertNotEqual(validate.returncode, 0)
            report_path = artifact_root / "reports" / "phase0_validation_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "failed")
            self.assertGreater(report["manifest_validation"]["mismatch_count"], 0)


if __name__ == "__main__":
    unittest.main()
