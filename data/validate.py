from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from data.config import ensure_phase0_dirs, load_config, resolve_phase0_paths
from data.deduplicate import detect_cross_split_overlaps
from data.logging_utils import setup_logging
from data.schemas import iter_jsonl, validate_problem
from data.split_tests import test_split_manifest_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Phase 0 artifacts.")
    parser.add_argument("--config", default="configs/phase0.yaml")
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--artifact-root", default=None)
    return parser.parse_args()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cross_split_report(
    by_split: dict[str, list[dict[str, Any]]],
    *,
    left_name: str,
    right_name: str,
    ngram_size: int,
    near_duplicate_threshold: float,
) -> dict[str, Any]:
    return detect_cross_split_overlaps(
        by_split[left_name],
        by_split[right_name],
        left_name=left_name,
        right_name=right_name,
        ngram_size=ngram_size,
        near_duplicate_threshold=near_duplicate_threshold,
    )


def _all_overlap_reports(
    by_split: dict[str, list[dict[str, Any]]],
    *,
    ngram_size: int,
    near_duplicate_threshold: float,
) -> dict[str, dict[str, Any]]:
    return {
        "sft_pt": _cross_split_report(
            by_split,
            left_name="sft",
            right_name="pt",
            ngram_size=ngram_size,
            near_duplicate_threshold=near_duplicate_threshold,
        ),
        "sft_dev": _cross_split_report(
            by_split,
            left_name="sft",
            right_name="dev",
            ngram_size=ngram_size,
            near_duplicate_threshold=near_duplicate_threshold,
        ),
        "pt_dev": _cross_split_report(
            by_split,
            left_name="pt",
            right_name="dev",
            ngram_size=ngram_size,
            near_duplicate_threshold=near_duplicate_threshold,
        ),
    }


def _validate_test_split_manifest(records: list[dict[str, Any]], manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.exists():
        return {
            "passed": False,
            "manifest_path": str(manifest_path),
            "reason": "missing_manifest",
            "processed_problem_count": len(records),
            "manifest_problem_count": 0,
        }

    manifest_records = list(iter_jsonl(manifest_path))
    expected_by_id = {record["problem_id"]: test_split_manifest_record(record) for record in records}
    manifest_by_id: dict[str, dict[str, Any]] = {}
    duplicate_problem_ids: list[str] = []
    for manifest_record in manifest_records:
        problem_id = manifest_record.get("problem_id")
        if not isinstance(problem_id, str) or not problem_id.strip():
            duplicate_problem_ids.append("<missing>")
            continue
        if problem_id in manifest_by_id:
            duplicate_problem_ids.append(problem_id)
        manifest_by_id[problem_id] = manifest_record

    expected_ids = set(expected_by_id)
    manifest_ids = set(manifest_by_id)
    missing_problem_ids = sorted(expected_ids - manifest_ids)
    extra_problem_ids = sorted(manifest_ids - expected_ids)
    required_fields = (
        "problem_id",
        "split",
        "reward_test_ids",
        "heldout_test_ids",
        "reward_fingerprints",
        "heldout_fingerprints",
    )
    mismatches: list[dict[str, Any]] = []
    for problem_id in sorted(expected_ids & manifest_ids):
        expected = expected_by_id[problem_id]
        actual = manifest_by_id[problem_id]
        mismatch_fields = [field for field in required_fields if actual.get(field) != expected[field]]
        if mismatch_fields:
            mismatches.append({"problem_id": problem_id, "fields": mismatch_fields})

    passed = not duplicate_problem_ids and not missing_problem_ids and not extra_problem_ids and not mismatches
    return {
        "passed": passed,
        "manifest_path": str(manifest_path),
        "processed_problem_count": len(records),
        "manifest_problem_count": len(manifest_records),
        "duplicate_problem_id_count": len(duplicate_problem_ids),
        "duplicate_problem_ids_sample": duplicate_problem_ids[:20],
        "missing_problem_id_count": len(missing_problem_ids),
        "missing_problem_ids_sample": missing_problem_ids[:20],
        "extra_problem_id_count": len(extra_problem_ids),
        "extra_problem_ids_sample": extra_problem_ids[:20],
        "mismatch_count": len(mismatches),
        "mismatches_sample": mismatches[:20],
    }


def _overlap_gate_passed(report: dict[str, Any]) -> bool:
    return (
        report["exact_id_overlap_count"] == 0
        and report["exact_text_overlap_count"] == 0
        and report["near_duplicate_count"] == 0
    )


def validate_phase0(config: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    ensure_phase0_dirs(paths)
    logger = setup_logging(paths["logs_dir"] / "phase0_validate.log")
    processed_dir = paths["processed_dir"]
    all_path = processed_dir / "problems.jsonl"
    if not all_path.exists():
        raise FileNotFoundError(f"Missing processed dataset: {all_path}")

    records = list(iter_jsonl(all_path))
    for record in records:
        validate_problem(record)
    by_split = {
        "sft": [record for record in records if record["split"] == "sft"],
        "pt": [record for record in records if record["split"] == "pt"],
        "dev": [record for record in records if record["split"] == "dev"],
    }
    for split_name, split_records in by_split.items():
        split_path = processed_dir / f"{split_name}.jsonl"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split dataset: {split_path}")
        loaded_split = list(iter_jsonl(split_path))
        if loaded_split != split_records:
            raise ValueError(f"{split_name} split file does not exactly match problems.jsonl records")

    dedup_config = config["dedup"]
    overlap_reports = _all_overlap_reports(
        by_split,
        ngram_size=int(dedup_config["ngram_size"]),
        near_duplicate_threshold=float(dedup_config["near_duplicate_threshold"]),
    )
    manifest_validation = _validate_test_split_manifest(records, paths["reports_dir"] / "test_split_manifest.jsonl")
    dedup_report = overlap_reports["sft_pt"]
    gate = {
        "data_loadable": bool(records),
        "schema_validation": "passed",
        "sft_pt_exact_id_overlap": dedup_report["exact_id_overlap_count"],
        "sft_pt_exact_text_overlap": dedup_report["exact_text_overlap_count"],
        "sft_pt_near_duplicate_overlap": dedup_report["near_duplicate_count"],
        "sft_pt_near_duplicate_report_generated": True,
        "sft_dev_exact_id_overlap": overlap_reports["sft_dev"]["exact_id_overlap_count"],
        "sft_dev_exact_text_overlap": overlap_reports["sft_dev"]["exact_text_overlap_count"],
        "sft_dev_near_duplicate_overlap": overlap_reports["sft_dev"]["near_duplicate_count"],
        "pt_dev_exact_id_overlap": overlap_reports["pt_dev"]["exact_id_overlap_count"],
        "pt_dev_exact_text_overlap": overlap_reports["pt_dev"]["exact_text_overlap_count"],
        "pt_dev_near_duplicate_overlap": overlap_reports["pt_dev"]["near_duplicate_count"],
        "dev_training_near_duplicate_report_generated": True,
        "reward_heldout_split_fixed": manifest_validation["passed"],
        "reward_heldout_manifest_problem_id_mismatch_count": (
            manifest_validation.get("missing_problem_id_count", 0) + manifest_validation.get("extra_problem_id_count", 0)
        ),
        "reward_heldout_manifest_content_mismatch_count": manifest_validation.get("mismatch_count", 0),
    }
    passed = (
        gate["data_loadable"]
        and gate["schema_validation"] == "passed"
        and all(_overlap_gate_passed(report) for report in overlap_reports.values())
        and manifest_validation["passed"]
    )
    report = {
        "phase": "phase0",
        "status": "passed" if passed else "failed",
        "artifact_root": str(paths["artifact_root"]),
        "record_counts": {key: len(value) for key, value in by_split.items()},
        "total_records": len(records),
        "gate": gate,
        "dedup": dedup_report,
        "overlap_reports": overlap_reports,
        "manifest_validation": manifest_validation,
    }
    _write_json(paths["reports_dir"] / "phase0_validation_report.json", report)
    logger.info("Phase 0 validation status: %s", report["status"])
    if not passed:
        raise SystemExit(1)
    return report


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = resolve_phase0_paths(config, raw_dir=args.raw_dir, artifact_root=args.artifact_root)
    report = validate_phase0(config, paths)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
