from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data.schemas import iter_jsonl, stable_hash
from eval.phase2_common import (
    config_hash,
    file_sha256,
    git_command,
    git_status_short,
    path_from_config,
    read_config,
    reports_dir,
    view_path,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize Phase 2 final SFT/PT/Dev views from frozen manifest.")
    parser.add_argument("--config", default="configs/base_eval.yaml")
    return parser.parse_args()


def _candidate_key(split: str) -> str:
    return f"{split}_candidate"


def _load_final_manifest(path: Path, expected_hash: str) -> dict[str, dict[str, Any]]:
    actual_hash = file_sha256(path)
    if actual_hash != expected_hash:
        raise ValueError(f"Phase 1 final manifest hash mismatch: expected {expected_hash}, got {actual_hash}")
    records = {record["problem_id"]: record for record in iter_jsonl(path)}
    return records


def _source_split_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(f"{record['source']}/{record['source_id'].split(':', 1)[0]}" for record in records)
    return dict(sorted(counts.items()))


def _validate_manifest_alignment(record: dict[str, Any], manifest_record: dict[str, Any]) -> None:
    problem_id = record["problem_id"]
    if manifest_record.get("dataset_split") != record.get("split"):
        raise ValueError(
            f"Manifest split mismatch for {problem_id}: "
            f"manifest={manifest_record.get('dataset_split')}, processed={record.get('split')}"
        )
    if manifest_record.get("source") != record.get("source"):
        raise ValueError(
            f"Manifest source mismatch for {problem_id}: "
            f"manifest={manifest_record.get('source')}, processed={record.get('source')}"
        )
    source_split = str(record.get("source_id", "")).split(":", 1)[0]
    if manifest_record.get("source_split") != source_split:
        raise ValueError(
            f"Manifest source split mismatch for {problem_id}: "
            f"manifest={manifest_record.get('source_split')}, processed={source_split}"
        )


def _problem_id_hash(records: list[dict[str, Any]]) -> str:
    return stable_hash([record["problem_id"] for record in records])


def materialize_views(config_path: str) -> dict[str, Any]:
    config = read_config(config_path)
    processed_path = path_from_config(config, "paths", "phase0_processed_path")
    manifest_path = path_from_config(config, "paths", "phase1_final_manifest_path")
    expected_manifest_hash = str(config["expected"]["phase1_final_manifest_hash"])
    expected_counts = {split: int(value) for split, value in config["expected"]["final_counts"].items()}
    manifest = _load_final_manifest(manifest_path, expected_manifest_hash)

    split_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in iter_jsonl(processed_path):
        manifest_record = manifest.get(record["problem_id"])
        if not manifest_record:
            continue
        _validate_manifest_alignment(record, manifest_record)
        split = str(record["split"])
        if manifest_record.get(_candidate_key(split)):
            split_records[split].append(record)

    views: dict[str, dict[str, Any]] = {}
    for split in ("sft", "pt", "dev"):
        records = split_records.get(split, [])
        expected_count = expected_counts[split]
        if len(records) != expected_count:
            raise ValueError(f"{split} count mismatch: expected {expected_count}, got {len(records)}")
        output_path = view_path(config, split)
        write_jsonl(output_path, records)
        views[split] = {
            "path": str(output_path),
            "count": len(records),
            "hash": file_sha256(output_path),
            "problem_id_hash": _problem_id_hash(records),
            "source_split_counts": _source_split_counts(records),
        }

    report = {
        "phase": "phase2_base_baseline",
        "step": "materialize_final_views",
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": git_command(["rev-parse", "HEAD"]),
            "dirty": bool(git_status_short()),
            "status_short": git_status_short(),
        },
        "config_file": config_path,
        "config_hash": config_hash(config_path),
        "phase0_processed_path": str(processed_path),
        "phase0_processed_hash": file_sha256(processed_path),
        "phase1_final_manifest_path": str(manifest_path),
        "phase1_final_manifest_hash": expected_manifest_hash,
        "views": views,
    }
    report_path = reports_dir(config) / "phase2_views_report.json"
    write_json(report_path, report)
    return report


def main() -> None:
    args = parse_args()
    report = materialize_views(args.config)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
