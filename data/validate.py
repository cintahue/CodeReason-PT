from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from data.config import ensure_phase0_dirs, load_config, resolve_phase0_paths
from data.deduplicate import detect_sft_pt_overlaps
from data.logging_utils import setup_logging
from data.schemas import iter_jsonl, validate_problem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Phase 0 artifacts.")
    parser.add_argument("--config", default="configs/phase0.yaml")
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--artifact-root", default=None)
    return parser.parse_args()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
        if len(loaded_split) != len(split_records):
            raise ValueError(f"{split_name} split file count does not match problems.jsonl")

    dedup_config = config["dedup"]
    dedup_report = detect_sft_pt_overlaps(
        by_split["sft"],
        by_split["pt"],
        ngram_size=int(dedup_config["ngram_size"]),
        near_duplicate_threshold=float(dedup_config["near_duplicate_threshold"]),
    )
    gate = {
        "data_loadable": bool(records),
        "schema_validation": "passed",
        "sft_pt_exact_id_overlap": dedup_report["exact_id_overlap_count"],
        "sft_pt_exact_text_overlap": dedup_report["exact_text_overlap_count"],
        "sft_pt_near_duplicate_overlap": dedup_report["near_duplicate_count"],
        "sft_pt_near_duplicate_report_generated": True,
        "reward_heldout_split_fixed": (paths["reports_dir"] / "test_split_manifest.jsonl").exists(),
    }
    passed = (
        gate["data_loadable"]
        and gate["schema_validation"] == "passed"
        and gate["sft_pt_exact_id_overlap"] == 0
        and gate["sft_pt_exact_text_overlap"] == 0
        and gate["sft_pt_near_duplicate_overlap"] == 0
        and gate["reward_heldout_split_fixed"]
    )
    report = {
        "phase": "phase0",
        "status": "passed" if passed else "failed",
        "artifact_root": str(paths["artifact_root"]),
        "record_counts": {key: len(value) for key, value in by_split.items()},
        "total_records": len(records),
        "gate": gate,
        "dedup": dedup_report,
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

