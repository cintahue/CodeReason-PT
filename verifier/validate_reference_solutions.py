from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data.config import load_config, resolve_phase0_paths
from data.schemas import iter_jsonl, split_source_id, stable_hash
from verifier.executor import verify_code
from verifier.judge import NORMALIZATION_POLICY
from verifier.result import SandboxConfig, VerificationResult, VerifierStatus


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Phase 0 reference solutions with the executable verifier.")
    parser.add_argument("--config", default="configs/phase0.yaml")
    parser.add_argument("--processed-path", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--audit-output", default="phase1_verifier_audit.json")
    parser.add_argument("--tests", choices=("all", "reward", "heldout"), default="all")
    parser.add_argument("--split", choices=("all", "sft", "pt", "dev"), default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--backend", choices=("local", "docker"), default="local")
    parser.add_argument("--python-executable", default=None)
    parser.add_argument("--docker-image", default="python:3.11-slim")
    parser.add_argument("--wall-time-seconds", type=float, default=2.0)
    parser.add_argument("--cpu-time-seconds", type=int, default=2)
    parser.add_argument("--memory-mb", type=int, default=1024)
    parser.add_argument("--process-limit", type=int, default=64)
    parser.add_argument("--output-limit-bytes", type=int, default=1_000_000)
    parser.add_argument("--failure-samples-per-status", type=int, default=5)
    return parser.parse_args()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git_command(args: list[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ""
    return completed.stdout.strip()


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


def _verifier_code_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted((ROOT / "verifier").glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _source_split(record: dict[str, Any]) -> str:
    source_id = str(record.get("source_id", ""))
    source_split, _ = split_source_id(source_id)
    return source_split or "unknown"


def _tests_for_record(record: dict[str, Any], test_scope: str) -> list[dict[str, Any]]:
    if test_scope == "reward":
        return list(record["reward_tests"])
    if test_scope == "heldout":
        return list(record["heldout_tests"])
    return list(record["reward_tests"]) + list(record["heldout_tests"])


def _load_records(processed_path: Path, *, split: str, limit: int | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in iter_jsonl(processed_path):
        if split != "all" and record.get("split") != split:
            continue
        records.append(record)
        if limit is not None and len(records) >= limit:
            break
    return records


def _first_failure(result: VerificationResult) -> dict[str, Any] | None:
    preferred = [item for item in result.test_results if item.status == result.status and not item.passed]
    fallback = [item for item in result.test_results if not item.passed]
    for item in preferred or fallback:
        return {
            "test_id": item.test_id,
            "status": item.status.value,
            "exit_code": item.exit_code,
            "runtime_ms": item.runtime_ms,
            "stdout_size": item.stdout_size,
            "stderr_size": item.stderr_size,
            "timeout": item.timeout,
            "error_type": item.error_type,
            "error_message": item.error_message,
        }
    return None


def _failure_key(result: dict[str, Any]) -> str:
    return f"{result['source']}/{result['source_split']}:{result['failure_category']}"


def _compact_failure(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "problem_id": result["problem_id"],
        "source": result["source"],
        "source_split": result["source_split"],
        "dataset_split": result["dataset_split"],
        "status": result["status"],
        "passed": result["passed"],
        "total": result["total"],
        "pass_rate": result["pass_rate"],
        "failure_category": result["failure_category"],
        "first_failure": result["first_failure"],
    }


def _classify_failure(result: VerificationResult) -> str:
    if result.status == VerifierStatus.AC:
        return "accepted"
    if result.status == VerifierStatus.CE:
        return "bad_reference_solution_or_extraction"
    failure = _first_failure(result)
    text = " ".join(
        str(value or "")
        for value in (
            failure.get("error_type") if failure else None,
            failure.get("error_message") if failure else None,
        )
    )
    if result.status == VerifierStatus.TLE:
        return "timeout_or_resource_limit"
    if result.status == VerifierStatus.WA:
        return "wrong_output_source_data_or_judging_mismatch"
    if "ModuleNotFoundError" in text or "ImportError" in text or "No module named" in text:
        return "unsupported_dependency"
    if "filesystem restricted" in text or "PermissionError" in text or "FileNotFoundError" in text:
        return "unsupported_filesystem_io"
    if "operation disabled" in text or "network" in text:
        return "unsupported_process_or_network_use"
    if any(token in text for token in ("EOFError", "invalid literal", "not enough values", "IndexError")):
        return "unsupported_io_format_or_bad_test_input"
    return "runtime_error_requires_inspection"


def _verify_record(payload: tuple[dict[str, Any], str, dict[str, Any]]) -> dict[str, Any]:
    record, test_scope, sandbox_config_dict = payload
    sandbox_config = SandboxConfig(**sandbox_config_dict)
    tests = _tests_for_record(record, test_scope)
    result = verify_code(
        str(record["reference_code"]),
        tests,
        sandbox_config=sandbox_config,
        extraction_strategy="reference_code",
    )
    failure = _first_failure(result)
    return {
        "problem_id": record["problem_id"],
        "source": record["source"],
        "source_split": _source_split(record),
        "dataset_split": record["split"],
        "status": result.status.value,
        "passed": result.passed,
        "total": result.total,
        "pass_rate": result.pass_rate,
        "compile_success": result.compile_success,
        "runtime_success": result.runtime_success,
        "timeout": result.timeout,
        "exit_code": result.exit_code,
        "runtime_ms": result.runtime_ms,
        "stdout_size": result.stdout_size,
        "stderr_size": result.stderr_size,
        "extraction_strategy": result.extraction_strategy,
        "sandbox_backend": result.sandbox_backend,
        "failure_category": _classify_failure(result),
        "first_failure": failure,
    }


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def _summarize_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    status_counts = Counter(record["status"] for record in records)
    accepted = status_counts.get(VerifierStatus.AC.value, 0)
    return {
        "total": total,
        "accepted": accepted,
        "acceptance_rate": _rate(accepted, total),
        "status_counts": dict(sorted(status_counts.items())),
    }


def _sample_failures(results: list[dict[str, Any]], per_status: int) -> dict[str, list[dict[str, Any]]]:
    samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        status = result["status"]
        if status == VerifierStatus.AC.value or len(samples[status]) >= per_status:
            continue
        samples[status].append(_compact_failure(result))
    return dict(sorted(samples.items()))


def _sample_failures_by_source_category(
    results: list[dict[str, Any]],
    per_group: int,
) -> dict[str, list[dict[str, Any]]]:
    samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        if result["status"] == VerifierStatus.AC.value:
            continue
        key = _failure_key(result)
        if len(samples[key]) >= per_group:
            continue
        samples[key].append(_compact_failure(result))
    return dict(sorted(samples.items()))


def _build_report(
    *,
    records: list[dict[str, Any]],
    results: list[dict[str, Any]],
    processed_path: Path,
    config_path: Path,
    test_scope: str,
    split: str,
    sandbox_config: SandboxConfig,
    failure_samples_per_status: int,
) -> dict[str, Any]:
    total = len(results)
    status_counts = Counter(result["status"] for result in results)
    failure_category_counts = Counter(result["failure_category"] for result in results if result["status"] != "AC")
    by_source_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_dataset_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_source_split[f"{result['source']}/{result['source_split']}"].append(result)
        by_dataset_split[str(result["dataset_split"])].append(result)

    return {
        "phase": "phase1_verifier",
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_file": str(config_path),
        "processed_path": str(processed_path),
        "processed_dataset_hash": _file_sha256(processed_path),
        "test_scope": test_scope,
        "dataset_split_filter": split,
        "input_problem_count": len(records),
        "verified_problem_count": total,
        "overall": _summarize_group(results),
        "by_source_split": {key: _summarize_group(value) for key, value in sorted(by_source_split.items())},
        "by_dataset_split": {key: _summarize_group(value) for key, value in sorted(by_dataset_split.items())},
        "failure_category_counts": dict(sorted(failure_category_counts.items())),
        "failure_samples": _sample_failures(results, failure_samples_per_status),
        "failure_samples_by_source_category": _sample_failures_by_source_category(
            results,
            max(1, min(3, failure_samples_per_status)),
        ),
        "normalization_policy": NORMALIZATION_POLICY,
        "sandbox_config": sandbox_config.to_dict(),
        "result_schema": [
            "status",
            "compile_success",
            "runtime_success",
            "timeout",
            "passed",
            "total",
            "pass_rate",
            "exit_code",
            "runtime_ms",
            "stdout_size",
            "stderr_size",
        ],
    }


def _build_audit(report: dict[str, Any], *, report_path: Path, sandbox_config: SandboxConfig) -> dict[str, Any]:
    status_short = _git_command(["status", "--short"])
    gate = {
        "regression_tests_required": True,
        "reference_validation_completed": report["status"] == "completed",
        "deterministic_seed_controls": {"PYTHONHASHSEED": "0"},
        "structured_result_fields_present": report["result_schema"],
        "normalization_policy_recorded": True,
        "classification_status_counts": report["overall"]["status_counts"],
    }
    return {
        "phase": "phase1_verifier",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": _git_command(["rev-parse", "HEAD"]),
            "dirty": bool(status_short),
            "status_short": status_short.splitlines(),
        },
        "verifier_code_hash": _verifier_code_hash(),
        "reference_validation_report": str(report_path),
        "reference_validation_report_hash": _file_sha256(report_path),
        "processed_dataset_hash": report["processed_dataset_hash"],
        "test_scope": report["test_scope"],
        "dataset_split_filter": report["dataset_split_filter"],
        "overall": report["overall"],
        "by_source_split": report["by_source_split"],
        "by_dataset_split": report["by_dataset_split"],
        "failure_category_counts": report["failure_category_counts"],
        "gate": gate,
        "normalization_policy_hash": stable_hash(NORMALIZATION_POLICY),
        "sandbox_config": sandbox_config.to_dict(),
    }


def validate_reference_solutions(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    paths = resolve_phase0_paths(config)
    processed_path = Path(args.processed_path) if args.processed_path else paths["processed_dir"] / "problems.jsonl"
    output_path = (
        Path(args.output)
        if args.output
        else paths["artifact_root"].parent / "phase1" / "reports" / "reference_validation_report.json"
    )
    sandbox_config = SandboxConfig(
        backend=args.backend,
        python_executable=args.python_executable,
        docker_image=args.docker_image,
        wall_time_seconds=args.wall_time_seconds,
        cpu_time_seconds=args.cpu_time_seconds,
        memory_mb=args.memory_mb,
        process_limit=args.process_limit,
        output_limit_bytes=args.output_limit_bytes,
    )
    records = _load_records(processed_path, split=args.split, limit=args.limit)
    payloads = [(record, args.tests, sandbox_config.to_dict()) for record in records]
    results: list[dict[str, Any]] = []
    completed = 0
    if args.workers <= 1:
        for payload in payloads:
            results.append(_verify_record(payload))
            completed += 1
            if completed % 250 == 0:
                print(f"Verified {completed}/{len(payloads)} reference solutions", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_verify_record, payload) for payload in payloads]
            for future in as_completed(futures):
                results.append(future.result())
                completed += 1
                if completed % 250 == 0:
                    print(f"Verified {completed}/{len(payloads)} reference solutions", flush=True)

    results.sort(key=lambda item: item["problem_id"])
    report = _build_report(
        records=records,
        results=results,
        processed_path=processed_path,
        config_path=Path(args.config),
        test_scope=args.tests,
        split=args.split,
        sandbox_config=sandbox_config,
        failure_samples_per_status=args.failure_samples_per_status,
    )
    _write_json(output_path, report)
    audit = _build_audit(report, report_path=output_path, sandbox_config=sandbox_config)
    _write_json(Path(args.audit_output), audit)
    return report


def main() -> None:
    args = parse_args()
    report = validate_reference_solutions(args)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
