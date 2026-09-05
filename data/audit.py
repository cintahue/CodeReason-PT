from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data.config import load_config, resolve_phase0_paths
from data.schemas import iter_jsonl, split_source_id, stable_hash


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a lightweight Phase 0 provenance audit.")
    parser.add_argument("--config", default="configs/phase0.yaml")
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument("--output", default="phase0_audit.json")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _git_command(args: list[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""
    return completed.stdout.strip()


def _package_versions() -> dict[str, str]:
    packages = ("datasets", "huggingface-hub", "pandas", "pyarrow", "pyyaml")
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    return versions


def _source_split(record: dict[str, Any]) -> str:
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        value = metadata.get("source_split")
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    source_id = record.get("source_id")
    if isinstance(source_id, str):
        source_split, _ = split_source_id(source_id)
        if source_split:
            return source_split
    return "unknown"


def _source_split_distribution(records: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(f"{record.get('source', 'unknown')}/{_source_split(record)}" for record in records)
    return dict(sorted(counter.items()))


def _processed_distribution(paths: dict[str, Path]) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for split_name in ("sft", "pt", "dev"):
        split_path = paths["processed_dir"] / f"{split_name}.jsonl"
        if split_path.exists():
            output[split_name] = _source_split_distribution(list(iter_jsonl(split_path)))
        else:
            output[split_name] = {}
    return output


def _pair_overlap_counts(stage_report: dict[str, Any]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for pair_name, report in sorted(stage_report.items()):
        if not isinstance(report, dict):
            continue
        counts[pair_name] = {
            "exact_id_overlap_count": int(report.get("exact_id_overlap_count", 0)),
            "exact_text_overlap_count": int(report.get("exact_text_overlap_count", 0)),
            "near_duplicate_count": int(report.get("near_duplicate_count", 0)),
        }
    return counts


def _overlap_counts(overlap_report: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "blocked_pt_count": int(overlap_report.get("blocked_pt_count", 0)),
        "blocked_dev_count": int(overlap_report.get("blocked_dev_count", 0)),
    }
    for stage_name in ("initial", "dev_blocking", "post"):
        stage_report = overlap_report.get(stage_name, {})
        output[stage_name] = _pair_overlap_counts(stage_report) if isinstance(stage_report, dict) else {}
    return output


def _prepare_drop_summary(prepare_report: dict[str, Any]) -> dict[str, Any]:
    dropped = prepare_report.get("dropped_short_tests", [])
    if not isinstance(dropped, list):
        dropped = []
    by_split = Counter()
    by_reason_bucket = Counter()
    short_test_pattern = re.compile(r"has (?P<count>\d+) unique tests; min_total_tests=(?P<minimum>\d+)")
    for item in dropped:
        if not isinstance(item, dict):
            continue
        split = str(item.get("split", "unknown"))
        reason = str(item.get("reason", "unknown"))
        by_split[split] += 1
        match = short_test_pattern.search(reason)
        if match:
            bucket = f"{split}:below_min_total_tests:unique_tests={match.group('count')}:min={match.group('minimum')}"
        else:
            bucket = f"{split}:{reason}"
        by_reason_bucket[bucket] += 1
    return {
        "dropped_short_tests_count": len(dropped),
        "dropped_short_tests_by_split": dict(sorted(by_split.items())),
        "dropped_short_tests_by_reason_bucket": dict(sorted(by_reason_bucket.items())),
    }


def generate_audit(config: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    raw_report = _read_json(paths["raw_dir"] / "build_raw_report.json")
    prepare_report = _read_json(paths["reports_dir"] / "phase0_prepare_report.json")
    validation_report = _read_json(paths["reports_dir"] / "phase0_validation_report.json")
    overlap_report = _read_json(paths["reports_dir"] / "phase0_overlap_report.json")

    raw_problem_path = paths["raw_dir"] / "problems.jsonl"
    raw_reasoning_path = paths["raw_dir"] / "reasoning.jsonl"
    processed_path = paths["processed_dir"] / "problems.jsonl"
    processed_records = list(iter_jsonl(processed_path)) if processed_path.exists() else []
    status_short = _git_command(["status", "--short"])

    return {
        "phase": "phase0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": _git_command(["rev-parse", "HEAD"]),
            "dirty": bool(status_short),
            "status_short": status_short.splitlines(),
        },
        "config_hash": stable_hash(config),
        "dataset_hashes": {
            "raw_problems_sha256": _file_sha256(raw_problem_path),
            "raw_reasoning_sha256": _file_sha256(raw_reasoning_path),
            "processed_problems_sha256": _file_sha256(processed_path),
            "sft_sha256": _file_sha256(paths["processed_dir"] / "sft.jsonl"),
            "pt_sha256": _file_sha256(paths["processed_dir"] / "pt.jsonl"),
            "dev_sha256": _file_sha256(paths["processed_dir"] / "dev.jsonl"),
        },
        "record_counts": {
            "raw_problems": _jsonl_count(raw_problem_path),
            "raw_reasoning": _jsonl_count(raw_reasoning_path),
            "processed_total": _jsonl_count(processed_path),
            "sft": _jsonl_count(paths["processed_dir"] / "sft.jsonl"),
            "pt": _jsonl_count(paths["processed_dir"] / "pt.jsonl"),
            "dev": _jsonl_count(paths["processed_dir"] / "dev.jsonl"),
        },
        "source_split_distribution": {
            "raw_output": raw_report.get("output_source_split_counts", {}),
            "processed_all": _source_split_distribution(processed_records),
            "processed_by_split": _processed_distribution(paths),
        },
        "drop_reasons": {
            "candidate_skip_counts": raw_report.get("candidate_skip_counts", {}),
            "assembly_skip_counts": raw_report.get("assembly_skip_counts", {}),
            "prepare": _prepare_drop_summary(prepare_report),
        },
        "overlap_counts": _overlap_counts(overlap_report),
        "gate": validation_report.get("gate", {}),
        "phase0_status": validation_report.get("status"),
        "raw_build_status": raw_report.get("status"),
        "prepare_status": prepare_report.get("status"),
        "candidate_stop_reason": raw_report.get("candidate_stop_reason"),
        "scanned_full_available_opencode_stream": raw_report.get("scanned_full_available_opencode_stream"),
        "package_versions": _package_versions(),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = resolve_phase0_paths(config, raw_dir=args.raw_dir, artifact_root=args.artifact_root)
    audit = generate_audit(config, paths)
    output_path = Path(args.output)
    _write_json(output_path, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
