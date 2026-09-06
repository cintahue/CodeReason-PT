from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from data.build_raw import _parse_json_maybe, _parse_solution_list
from data.config import load_config, resolve_phase0_paths
from data.schemas import iter_jsonl, split_source_id, stable_hash
from verifier.executor import verify, verify_code_with_raw_runs
from verifier.judge import NORMALIZATION_POLICY, judge_stdout, normalize_output
from verifier.result import SandboxConfig, VerificationResult, VerifierStatus
from verifier.validate_reference_solutions import (
    _classify_failure,
    _file_sha256,
    _first_failure,
    _git_status_short,
    _verifier_code_hash,
)


ROOT = Path(__file__).resolve().parents[1]
WA_DIAGNOSTIC_CATEGORIES = (
    "exact_normalized_mismatch",
    "whitespace_tokenization_mismatch",
    "suspected_source_testcase_issue",
    "unsupported_multiple_output_style_problem",
    "unknown",
)
MULTIPLE_OUTPUT_HINTS = (
    "any valid",
    "any one",
    "any of",
    "any order",
    "print any",
    "output any",
    "if there are multiple",
    "if multiple",
    "multiple answers",
    "multiple correct",
    "all possible",
    "in any order",
    "arbitrary",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qualify Phase 1 verifier eligibility using source solutions.")
    parser.add_argument("--config", default="configs/phase0.yaml")
    parser.add_argument("--processed-path", default=None)
    parser.add_argument("--raw-sources-dir", default=None)
    parser.add_argument("--manifest-output", default=None)
    parser.add_argument("--report-output", default=None)
    parser.add_argument("--audit-output", default="phase1_verifier_audit.json")
    parser.add_argument("--tests", choices=("all", "reward", "heldout"), default="all")
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
    parser.add_argument("--source-attempts", type=int, default=3)
    parser.add_argument("--tle-diagnostic-wall-time-seconds", type=float, default=10.0)
    parser.add_argument("--tle-diagnostic-cpu-time-seconds", type=int, default=10)
    parser.add_argument("--skip-docker-parity", action="store_true")
    parser.add_argument("--reuse-manifest", action="store_true")
    return parser.parse_args()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [record for record in iter_jsonl(path)]


def _git_command(args: list[str]) -> str:
    try:
        completed = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ""
    return completed.stdout.strip()


def _phase1_root(paths: dict[str, Path]) -> Path:
    return paths["artifact_root"].parent / "phase1"


def _raw_sources_dir(config: dict[str, Any], args: argparse.Namespace, paths: dict[str, Path]) -> Path:
    if args.raw_sources_dir:
        return Path(args.raw_sources_dir).expanduser()
    configured = config.get("paths", {}).get("raw_sources_dir")
    if configured:
        return Path(str(configured)).expanduser()
    return paths["raw_dir"].parent / "raw_sources"


def _is_empty_text(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip() == ""


def classify_protocol(source_row: dict[str, Any]) -> dict[str, Any]:
    parsed = _parse_json_maybe(source_row.get("input_output"))
    fn_name = parsed.get("fn_name") if isinstance(parsed, dict) else None
    starter_code = source_row.get("starter_code")
    fn_name_empty = _is_empty_text(fn_name)
    starter_code_empty = _is_empty_text(starter_code)
    protocol = "stdio" if fn_name_empty and starter_code_empty else "call_based"
    return {
        "protocol": protocol,
        "stdio_eligible": protocol == "stdio",
        "fn_name_present": not fn_name_empty,
        "starter_code_present": not starter_code_empty,
    }


def final_candidate_flags(record: dict[str, Any]) -> dict[str, bool]:
    source_ok = bool(record.get("source_reference_verified"))
    ocr_ok = bool(record.get("ocr_solution_verified"))
    stdio_ok = bool(record.get("stdio_eligible"))
    return {
        "sft_candidate": record["dataset_split"] == "sft" and stdio_ok and source_ok and ocr_ok,
        "pt_candidate": record["dataset_split"] == "pt" and stdio_ok and source_ok,
        "dev_candidate": record["dataset_split"] == "dev" and stdio_ok and source_ok,
    }


def _source_key(record: dict[str, Any]) -> tuple[str, str, int]:
    source_split, raw_index = split_source_id(str(record["source_id"]))
    if source_split is None:
        raise ValueError(f"Missing source split in source_id={record['source_id']}")
    return str(record["source"]), source_split, int(raw_index)


def _load_processed(processed_path: Path, limit: int | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in iter_jsonl(processed_path):
        records.append(record)
        if limit is not None and len(records) >= limit:
            break
    return records


def _wanted_indices(records: list[dict[str, Any]]) -> dict[str, dict[str, set[int]]]:
    wanted: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    for record in records:
        source, source_split, index = _source_key(record)
        wanted[source][source_split].add(index)
    return wanted


def _load_apps_rows(raw_sources_dir: Path, wanted_by_split: dict[str, set[int]]) -> dict[tuple[str, str, int], dict[str, Any]]:
    rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    for source_split, wanted_indices in sorted(wanted_by_split.items()):
        path = raw_sources_dir / "apps" / f"{source_split}.jsonl"
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle):
                if not line.strip():
                    continue
                row = json.loads(line)
                index = int(row.get("id", line_no))
                if index in wanted_indices:
                    rows[("apps", source_split, index)] = row
    return rows


def _load_taco_rows(raw_sources_dir: Path, wanted_by_split: dict[str, set[int]]) -> dict[tuple[str, str, int], dict[str, Any]]:
    rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    root = raw_sources_dir / "taco" / "ALL"
    for source_split, wanted_indices in sorted(wanted_by_split.items()):
        offset = 0
        for parquet_path in sorted(root.glob(f"{source_split}-*.parquet")):
            metadata = pd.read_parquet(parquet_path, columns=["question"])
            row_count = len(metadata)
            local_indices = sorted(index - offset for index in wanted_indices if offset <= index < offset + row_count)
            if local_indices:
                df = pd.read_parquet(parquet_path)
                for local_index in local_indices:
                    rows[("taco", source_split, offset + local_index)] = df.iloc[local_index].to_dict()
            offset += row_count
    return rows


def load_source_rows(
    records: list[dict[str, Any]],
    raw_sources_dir: Path,
) -> dict[tuple[str, str, int], dict[str, Any]]:
    wanted = _wanted_indices(records)
    rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    if "apps" in wanted:
        rows.update(_load_apps_rows(raw_sources_dir, wanted["apps"]))
    if "taco" in wanted:
        rows.update(_load_taco_rows(raw_sources_dir, wanted["taco"]))
    return rows


def _tests_for_record(record: dict[str, Any], test_scope: str) -> list[dict[str, Any]]:
    if test_scope == "reward":
        return list(record["reward_tests"])
    if test_scope == "heldout":
        return list(record["heldout_tests"])
    return list(record["reward_tests"]) + list(record["heldout_tests"])


def _compact_result(result: VerificationResult) -> dict[str, Any]:
    return {
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
        "failure_category": _classify_failure(result),
        "first_failure": _first_failure(result),
    }


def _looks_multiple_output(prompt: str) -> bool:
    lowered = prompt.lower()
    return any(hint in lowered for hint in MULTIPLE_OUTPUT_HINTS)


def _wa_diagnostic_from_runs(
    *,
    result: VerificationResult,
    raw_runs: list[dict[str, Any]],
    tests: list[dict[str, Any]],
    prompt: str,
) -> dict[str, Any] | None:
    if result.status != VerifierStatus.WA:
        return None
    mismatches = 0
    token_equal_mismatches = 0
    for index, run in enumerate(raw_runs):
        if index >= len(tests):
            break
        if run.get("timeout") or run.get("output_truncated") or run.get("error_type"):
            continue
        if run.get("exit_code") not in (0, None):
            continue
        actual = str(run.get("stdout") or "")
        expected = str(tests[index].get("output") or "")
        if judge_stdout(actual, expected):
            continue
        mismatches += 1
        if actual.split() == expected.split() and normalize_output(actual) != normalize_output(expected):
            token_equal_mismatches += 1

    if mismatches == 0:
        category = "unknown"
    elif token_equal_mismatches == mismatches:
        category = "whitespace_tokenization_mismatch"
    elif _looks_multiple_output(prompt):
        category = "unsupported_multiple_output_style_problem"
    else:
        category = "exact_normalized_mismatch"
    return {
        "category": category,
        "mismatch_count": mismatches,
        "token_equal_mismatch_count": token_equal_mismatches,
    }


def _best_attempt(attempts: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not attempts:
        return None
    for attempt in attempts:
        if attempt["status"] == VerifierStatus.AC.value:
            return attempt
    return max(attempts, key=lambda item: (float(item.get("pass_rate") or 0.0), -int(item.get("attempt_index") or 0)))


def _maybe_promote_source_wa_diagnostic(attempts: list[dict[str, Any]], best_attempt: dict[str, Any] | None) -> None:
    if not best_attempt or best_attempt.get("status") != VerifierStatus.WA.value:
        return
    diagnostic = best_attempt.get("wa_diagnostic")
    if not diagnostic or diagnostic.get("category") != "exact_normalized_mismatch":
        return
    wa_attempts = [attempt for attempt in attempts if attempt.get("status") == VerifierStatus.WA.value]
    if len(wa_attempts) < 2:
        return
    first_failure_ids = [
        attempt.get("first_failure", {}).get("test_id")
        for attempt in wa_attempts
        if isinstance(attempt.get("first_failure"), dict)
    ]
    shared_failure = first_failure_ids and Counter(first_failure_ids).most_common(1)[0][1] >= 2
    high_pass = max(float(attempt.get("pass_rate") or 0.0) for attempt in wa_attempts) >= 0.95
    if shared_failure and high_pass:
        diagnostic["category"] = "suspected_source_testcase_issue"


def _verify_solution(
    *,
    code: str,
    tests: list[dict[str, Any]],
    prompt: str,
    sandbox_config: SandboxConfig,
    extraction_strategy: str,
) -> tuple[dict[str, Any], VerificationResult]:
    result, raw_runs = verify_code_with_raw_runs(
        code,
        tests,
        sandbox_config=sandbox_config,
        extraction_strategy=extraction_strategy,
    )
    summary = _compact_result(result)
    summary["wa_diagnostic"] = _wa_diagnostic_from_runs(
        result=result,
        raw_runs=raw_runs,
        tests=tests,
        prompt=prompt,
    )
    return summary, result


def _rerun_tle_diagnostic(
    *,
    code: str,
    tests: list[dict[str, Any]],
    prompt: str,
    config: SandboxConfig,
) -> dict[str, Any]:
    summary, _ = _verify_solution(
        code=code,
        tests=tests,
        prompt=prompt,
        sandbox_config=config,
        extraction_strategy="source_solution_tle_10s_diagnostic",
    )
    return {
        "status": summary["status"],
        "passed": summary["passed"],
        "total": summary["total"],
        "pass_rate": summary["pass_rate"],
        "runtime_ms": summary["runtime_ms"],
    }


def _source_attempt_failure_category(attempt: dict[str, Any] | None) -> str:
    if attempt is None:
        return "no_source_solution"
    return str(attempt.get("failure_category") or "unknown")


def _record_failure(record: dict[str, Any], best_source: dict[str, Any] | None, ocr: dict[str, Any] | None) -> tuple[str | None, str]:
    if not record["stdio_eligible"]:
        return None, "skipped_call_based"
    if not record["source_reference_verified"]:
        status = best_source.get("status") if best_source else None
        return status, f"source_reference_not_verified:{_source_attempt_failure_category(best_source)}"
    if record["dataset_split"] == "sft" and not record["ocr_solution_verified"]:
        status = ocr.get("status") if ocr else None
        category = ocr.get("failure_category") if ocr else "unknown"
        return status, f"ocr_solution_not_ac:{category}"
    return None, "eligible"


def _qualify_record(payload: tuple[dict[str, Any], dict[str, Any], str, dict[str, Any], dict[str, Any], int]) -> dict[str, Any]:
    record, source_info, test_scope, sandbox_config_dict, tle_config_dict, source_attempt_limit = payload
    source, source_split, _ = _source_key(record)
    protocol_info = source_info["protocol_info"]
    tests = _tests_for_record(record, test_scope)
    manifest_record: dict[str, Any] = {
        "problem_id": record["problem_id"],
        "source": source,
        "source_split": source_split,
        "dataset_split": record["split"],
        "protocol": protocol_info["protocol"],
        "stdio_eligible": protocol_info["stdio_eligible"],
        "fn_name_present": protocol_info["fn_name_present"],
        "starter_code_present": protocol_info["starter_code_present"],
        "source_solution_count": source_info["source_solution_count"],
        "source_reference_attempts": [],
        "source_ref1_verified": False,
        "source_reference_verified": False,
        "ocr_solution_verified": False,
    }
    if not manifest_record["stdio_eligible"]:
        failure_status, failure_category = _record_failure(manifest_record, None, None)
        manifest_record.update(
            {
                "source_ref_any3_status": None,
                "source_ref_any3_pass_rate": 0.0,
                "ocr_solution": None,
                "failure_status": failure_status,
                "failure_category": failure_category,
            }
        )
        manifest_record.update(final_candidate_flags(manifest_record))
        return manifest_record

    sandbox_config = SandboxConfig(**sandbox_config_dict)
    tle_config = SandboxConfig(**tle_config_dict)
    source_solutions = source_info["source_solutions"][: max(0, source_attempt_limit)]
    prompt = source_info["prompt"]
    best_source: dict[str, Any] | None = None
    for attempt_index, code in enumerate(source_solutions, start=1):
        summary, _ = _verify_solution(
            code=code,
            tests=tests,
            prompt=prompt,
            sandbox_config=sandbox_config,
            extraction_strategy=f"source_solution_{attempt_index}",
        )
        summary["attempt_index"] = attempt_index
        if summary["status"] == VerifierStatus.TLE.value:
            summary["tle_10s_diagnostic"] = _rerun_tle_diagnostic(
                code=code,
                tests=tests,
                prompt=prompt,
                config=tle_config,
            )
        manifest_record["source_reference_attempts"].append(summary)
        if attempt_index == 1:
            manifest_record["source_ref1_verified"] = summary["status"] == VerifierStatus.AC.value
        if summary["status"] == VerifierStatus.AC.value:
            best_source = summary
            break

    if best_source is None:
        best_source = _best_attempt(manifest_record["source_reference_attempts"])
    _maybe_promote_source_wa_diagnostic(manifest_record["source_reference_attempts"], best_source)
    source_verified = any(attempt["status"] == VerifierStatus.AC.value for attempt in manifest_record["source_reference_attempts"])
    manifest_record["source_reference_verified"] = source_verified
    manifest_record["source_ref_any3_status"] = best_source.get("status") if best_source else None
    manifest_record["source_ref_any3_pass_rate"] = best_source.get("pass_rate", 0.0) if best_source else 0.0

    ocr_summary, _ = _verify_solution(
        code=str(record["reference_code"]),
        tests=tests,
        prompt=prompt,
        sandbox_config=sandbox_config,
        extraction_strategy="ocr_solution",
    )
    manifest_record["ocr_solution"] = ocr_summary
    manifest_record["ocr_solution_verified"] = ocr_summary["status"] == VerifierStatus.AC.value

    failure_status, failure_category = _record_failure(manifest_record, best_source, ocr_summary)
    manifest_record["failure_status"] = failure_status
    manifest_record["failure_category"] = failure_category
    manifest_record.update(final_candidate_flags(manifest_record))
    return manifest_record


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _status_summary(statuses: list[str | None]) -> dict[str, Any]:
    concrete = [status for status in statuses if status]
    status_counts = Counter(concrete)
    total = len(concrete)
    accepted = status_counts.get(VerifierStatus.AC.value, 0)
    return {
        "total": total,
        "accepted": accepted,
        "acceptance_rate": _rate(accepted, total),
        "status_counts": dict(sorted(status_counts.items())),
    }


def _split_groups(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[f"{record['source']}/{record['source_split']}"].append(record)
    return dict(sorted(groups.items()))


def _source_ref1_status(record: dict[str, Any]) -> str | None:
    attempts = record.get("source_reference_attempts") or []
    return attempts[0]["status"] if attempts else "NO_SOURCE"


def _source_ref_any3_status(record: dict[str, Any]) -> str | None:
    return record.get("source_ref_any3_status") or "NO_SOURCE"


def _ocr_status(record: dict[str, Any]) -> str | None:
    ocr = record.get("ocr_solution")
    return ocr.get("status") if isinstance(ocr, dict) else None


def _metric_bundle(records: list[dict[str, Any]]) -> dict[str, Any]:
    stdio = [record for record in records if record["stdio_eligible"]]
    return {
        "source_ref_at_1": _status_summary([_source_ref1_status(record) for record in stdio]),
        "source_ref_at_any3": _status_summary([_source_ref_any3_status(record) for record in stdio]),
        "ocr_solution_acceptance": _status_summary([_ocr_status(record) for record in stdio]),
    }


def _eligible_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"sft": 0, "pt": 0, "dev": 0}
    for record in records:
        if record.get("sft_candidate"):
            counts["sft"] += 1
        if record.get("pt_candidate"):
            counts["pt"] += 1
        if record.get("dev_candidate"):
            counts["dev"] += 1
    return counts


def _source_ref_wa_diagnostics(records: list[dict[str, Any]], status_fn) -> dict[str, Any]:
    counts = Counter()
    total = 0
    for record in records:
        if not record["stdio_eligible"]:
            continue
        target_status = status_fn(record)
        if target_status != VerifierStatus.WA.value:
            continue
        attempts = record.get("source_reference_attempts") or []
        if status_fn is _source_ref1_status:
            attempt = attempts[0] if attempts else None
        else:
            best = _best_attempt(attempts)
            attempt = best if best and best.get("status") == VerifierStatus.WA.value else None
        diagnostic = attempt.get("wa_diagnostic") if isinstance(attempt, dict) else None
        category = diagnostic.get("category") if isinstance(diagnostic, dict) else "unknown"
        counts[str(category)] += 1
        total += 1
    for category in WA_DIAGNOSTIC_CATEGORIES:
        counts.setdefault(category, 0)
    return {"total_wa": total, "counts": dict(sorted(counts.items()))}


def _re_error_types(records: list[dict[str, Any]], status_fn) -> dict[str, int]:
    counts = Counter()
    for record in records:
        if not record["stdio_eligible"] or status_fn(record) != VerifierStatus.RE.value:
            continue
        if status_fn is _source_ref1_status:
            attempts = record.get("source_reference_attempts") or []
            summary = attempts[0] if attempts else {}
        elif status_fn is _source_ref_any3_status:
            summary = _best_attempt(record.get("source_reference_attempts") or []) or {}
        else:
            summary = record.get("ocr_solution") or {}
        first_failure = summary.get("first_failure") if isinstance(summary, dict) else None
        error_type = first_failure.get("error_type") if isinstance(first_failure, dict) else None
        counts[str(error_type or "unknown")] += 1
    return dict(sorted(counts.items()))


def _tle_diagnostic_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    source_ref1_tle = 0
    source_ref1_tle_to_ac = 0
    source_attempt_tle = 0
    source_attempt_tle_to_ac = 0
    for record in records:
        if not record["stdio_eligible"]:
            continue
        attempts = record.get("source_reference_attempts") or []
        for attempt in attempts:
            if attempt.get("status") != VerifierStatus.TLE.value:
                continue
            source_attempt_tle += 1
            diagnostic = attempt.get("tle_10s_diagnostic") or {}
            if diagnostic.get("status") == VerifierStatus.AC.value:
                source_attempt_tle_to_ac += 1
            if attempt.get("attempt_index") == 1:
                source_ref1_tle += 1
                if diagnostic.get("status") == VerifierStatus.AC.value:
                    source_ref1_tle_to_ac += 1
    return {
        "source_ref1_tle": source_ref1_tle,
        "source_ref1_tle_to_ac_at_10s": source_ref1_tle_to_ac,
        "source_attempt_tle": source_attempt_tle,
        "source_attempt_tle_to_ac_at_10s": source_attempt_tle_to_ac,
    }


def _failure_category_counts(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts = Counter()
    for record in records:
        if not record["stdio_eligible"]:
            continue
        if key == "source_ref1":
            attempts = record.get("source_reference_attempts") or []
            summary = attempts[0] if attempts else None
        elif key == "source_ref_any3":
            summary = _best_attempt(record.get("source_reference_attempts") or [])
        else:
            summary = record.get("ocr_solution")
        if not isinstance(summary, dict):
            counts["no_source_solution"] += 1
            continue
        if summary.get("status") == VerifierStatus.AC.value:
            continue
        counts[str(summary.get("failure_category") or "unknown")] += 1
    return dict(sorted(counts.items()))


def _docker_image_available(image: str) -> tuple[bool, str | None]:
    try:
        completed = subprocess.run(
            ["docker", "image", "inspect", image],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return False, repr(exc)
    if completed.returncode != 0:
        return False, completed.stderr.strip() or f"docker image inspect exited {completed.returncode}"
    return True, None


def _run_backend_parity(args: argparse.Namespace) -> dict[str, Any]:
    if args.skip_docker_parity:
        return {"status": "skipped"}
    image_available, image_error = _docker_image_available(args.docker_image)
    if not image_available:
        return {
            "status": "docker_unavailable",
            "matched": False,
            "error": image_error,
            "policy": {
                "local_backend": "regression_and_fast_diagnostics_only",
                "docker_backend": "formal_rollout_and_reward_execution",
            },
        }
    cases = [
        ("AC", "print(sum(map(int, input().split())))\n", [{"input": "2 3\n", "output": "5\n"}]),
        ("WA", "print(6)\n", [{"input": "2 3\n", "output": "5\n"}]),
        ("RE", "raise RuntimeError('boom')\n", [{"input": "", "output": ""}]),
        ("TLE", "while True:\n    pass\n", [{"input": "", "output": ""}]),
    ]
    local_config = SandboxConfig(
        backend="local",
        wall_time_seconds=0.5,
        cpu_time_seconds=1,
        memory_mb=256,
        process_limit=16,
        output_limit_bytes=100_000,
    )
    docker_config = replace(local_config, backend="docker", docker_image=args.docker_image)
    results = []
    for expected, code, tests in cases:
        local_result = verify(code, tests, sandbox_config=local_config)
        docker_result = verify(code, tests, sandbox_config=docker_config)
        results.append(
            {
                "case": expected,
                "local_status": local_result.status.value,
                "docker_status": docker_result.status.value,
                "matched": local_result.status == docker_result.status,
            }
        )
    return {
        "status": "completed",
        "matched": all(item["matched"] for item in results),
        "cases": results,
        "policy": {
            "local_backend": "regression_and_fast_diagnostics_only",
            "docker_backend": "formal_rollout_and_reward_execution",
        },
    }


def _build_report(
    *,
    manifest_records: list[dict[str, Any]],
    processed_path: Path,
    raw_sources_dir: Path,
    config_path: Path,
    manifest_path: Path,
    test_scope: str,
    sandbox_config: SandboxConfig,
    tle_config: SandboxConfig,
    local_vs_docker: dict[str, Any],
) -> dict[str, Any]:
    protocol_counts = Counter(record["protocol"] for record in manifest_records)
    stdio_records = [record for record in manifest_records if record["stdio_eligible"]]
    by_source_split = {
        key: _metric_bundle(value)
        for key, value in _split_groups(manifest_records).items()
    }
    by_dataset_split = {
        key: _metric_bundle(value)
        for key, value in sorted(defaultdict(list, {
            split: [record for record in manifest_records if record["dataset_split"] == split]
            for split in ("sft", "pt", "dev")
        }).items())
    }
    return {
        "phase": "phase1.1_verifier_qualification",
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_file": str(config_path),
        "processed_path": str(processed_path),
        "raw_sources_dir": str(raw_sources_dir),
        "manifest_path": str(manifest_path),
        "manifest_hash": _file_sha256(manifest_path),
        "phase0_dataset_hash": _file_sha256(processed_path),
        "test_scope": test_scope,
        "input_problem_count": len(manifest_records),
        "stdio_problem_count": len(stdio_records),
        "protocol_counts": dict(sorted(protocol_counts.items())),
        "overall_stdio": _metric_bundle(stdio_records),
        "by_source_split": by_source_split,
        "by_dataset_split": by_dataset_split,
        "eligible_counts": _eligible_counts(manifest_records),
        "source_ref1_failure_categories": _failure_category_counts(manifest_records, "source_ref1"),
        "source_ref_any3_failure_categories": _failure_category_counts(manifest_records, "source_ref_any3"),
        "ocr_solution_failure_categories": _failure_category_counts(manifest_records, "ocr_solution"),
        "source_ref1_wa_diagnostics": _source_ref_wa_diagnostics(manifest_records, _source_ref1_status),
        "source_ref_any3_wa_diagnostics": _source_ref_wa_diagnostics(manifest_records, _source_ref_any3_status),
        "source_ref1_re_error_types": _re_error_types(manifest_records, _source_ref1_status),
        "source_ref_any3_re_error_types": _re_error_types(manifest_records, _source_ref_any3_status),
        "ocr_solution_re_error_types": _re_error_types(manifest_records, _ocr_status),
        "source_tle_10s_diagnostic": _tle_diagnostic_summary(manifest_records),
        "call_based_filtered": protocol_counts.get("call_based", 0),
        "normalization_policy": NORMALIZATION_POLICY,
        "sandbox_config": sandbox_config.to_dict(),
        "tle_diagnostic_sandbox_config": tle_config.to_dict(),
        "backend_policy": {
            "local": "regression_and_fast_diagnostics_only",
            "docker": "formal_rollout_and_reward_execution",
        },
        "local_vs_docker_basic_parity": local_vs_docker,
    }


def _build_audit(report: dict[str, Any], report_path: Path) -> dict[str, Any]:
    status_short = _git_status_short()
    return {
        "phase": "phase1.1_verifier_qualification",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": _git_command(["rev-parse", "HEAD"]),
            "dirty": bool(status_short),
            "status_short": status_short,
        },
        "verifier_code_hash": _verifier_code_hash(),
        "phase0_dataset_hash": report["phase0_dataset_hash"],
        "qualification_report": str(report_path),
        "qualification_report_hash": _file_sha256(report_path),
        "eligibility_manifest": report["manifest_path"],
        "eligibility_manifest_hash": report["manifest_hash"],
        "protocol_counts": report["protocol_counts"],
        "source_ref_at_1": report["overall_stdio"]["source_ref_at_1"],
        "source_ref_at_any3": report["overall_stdio"]["source_ref_at_any3"],
        "ocr_solution_acceptance": report["overall_stdio"]["ocr_solution_acceptance"],
        "eligible_counts": report["eligible_counts"],
        "source_ref1_failure_categories": report["source_ref1_failure_categories"],
        "source_ref_any3_failure_categories": report["source_ref_any3_failure_categories"],
        "ocr_solution_failure_categories": report["ocr_solution_failure_categories"],
        "source_ref1_wa_diagnostics": report["source_ref1_wa_diagnostics"],
        "source_ref_any3_wa_diagnostics": report["source_ref_any3_wa_diagnostics"],
        "source_ref1_re_error_types": report["source_ref1_re_error_types"],
        "source_ref_any3_re_error_types": report["source_ref_any3_re_error_types"],
        "source_tle_10s_diagnostic": report["source_tle_10s_diagnostic"],
        "local_vs_docker_basic_parity": report["local_vs_docker_basic_parity"],
        "normalization_policy_hash": stable_hash(NORMALIZATION_POLICY),
        "backend_policy": report["backend_policy"],
    }


def qualify_phase1(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    paths = resolve_phase0_paths(config)
    phase1_root = _phase1_root(paths)
    processed_path = Path(args.processed_path) if args.processed_path else paths["processed_dir"] / "problems.jsonl"
    raw_sources_dir = _raw_sources_dir(config, args, paths)
    manifest_path = (
        Path(args.manifest_output)
        if args.manifest_output
        else phase1_root / "manifests" / "phase1_eligibility_manifest.jsonl"
    )
    report_path = (
        Path(args.report_output)
        if args.report_output
        else phase1_root / "reports" / "phase1_qualification_report.json"
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
    tle_config = replace(
        sandbox_config,
        wall_time_seconds=args.tle_diagnostic_wall_time_seconds,
        cpu_time_seconds=args.tle_diagnostic_cpu_time_seconds,
    )

    if args.reuse_manifest:
        manifest_records = _read_jsonl(manifest_path)
        local_vs_docker = _run_backend_parity(args)
        report = _build_report(
            manifest_records=manifest_records,
            processed_path=processed_path,
            raw_sources_dir=raw_sources_dir,
            config_path=Path(args.config),
            manifest_path=manifest_path,
            test_scope=args.tests,
            sandbox_config=sandbox_config,
            tle_config=tle_config,
            local_vs_docker=local_vs_docker,
        )
        _write_json(report_path, report)
        audit = _build_audit(report, report_path)
        _write_json(Path(args.audit_output), audit)
        return report

    records = _load_processed(processed_path, args.limit)
    source_rows = load_source_rows(records, raw_sources_dir)
    source_infos: dict[str, dict[str, Any]] = {}
    for record in records:
        source, source_split, index = _source_key(record)
        row = source_rows.get((source, source_split, index))
        if row is None:
            source_infos[record["problem_id"]] = {
                "protocol_info": {
                    "protocol": "call_based",
                    "stdio_eligible": False,
                    "fn_name_present": False,
                    "starter_code_present": False,
                },
                "source_solution_count": 0,
                "source_solutions": [],
                "prompt": str(record.get("prompt") or ""),
            }
            continue
        source_solutions = _parse_solution_list(row.get("solutions"))
        source_infos[record["problem_id"]] = {
            "protocol_info": classify_protocol(row),
            "source_solution_count": len(source_solutions),
            "source_solutions": source_solutions[: max(0, int(args.source_attempts))],
            "prompt": str(row.get("question") or record.get("prompt") or ""),
        }

    payloads = [
        (
            record,
            source_infos[record["problem_id"]],
            args.tests,
            sandbox_config.to_dict(),
            tle_config.to_dict(),
            int(args.source_attempts),
        )
        for record in records
    ]
    manifest_records: list[dict[str, Any]] = []
    completed = 0
    if args.workers <= 1:
        for payload in payloads:
            manifest_records.append(_qualify_record(payload))
            completed += 1
            if completed % 250 == 0:
                print(f"Qualified {completed}/{len(payloads)} problems", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_qualify_record, payload) for payload in payloads]
            for future in as_completed(futures):
                manifest_records.append(future.result())
                completed += 1
                if completed % 250 == 0:
                    print(f"Qualified {completed}/{len(payloads)} problems", flush=True)

    manifest_records.sort(key=lambda item: item["problem_id"])
    _write_jsonl(manifest_path, manifest_records)
    local_vs_docker = _run_backend_parity(args)
    report = _build_report(
        manifest_records=manifest_records,
        processed_path=processed_path,
        raw_sources_dir=raw_sources_dir,
        config_path=Path(args.config),
        manifest_path=manifest_path,
        test_scope=args.tests,
        sandbox_config=sandbox_config,
        tle_config=tle_config,
        local_vs_docker=local_vs_docker,
    )
    _write_json(report_path, report)
    audit = _build_audit(report, report_path)
    _write_json(Path(args.audit_output), audit)
    return report


def main() -> None:
    args = parse_args()
    report = qualify_phase1(args)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
