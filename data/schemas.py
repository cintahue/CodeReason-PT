from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


SPLITS = {"sft", "pt", "dev"}


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def make_problem_id(source: str, source_id: str) -> str:
    if not source or not source_id:
        raise ValueError("source and source_id are required to derive problem_id")
    return f"{source}:{source_id}"


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    jsonl_path = Path(path)
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"{jsonl_path}:{line_no} is not a JSON object")
            yield value


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    jsonl_path = Path(path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(stable_json(record))
            handle.write("\n")


def require_string(record: dict[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def validate_testcase(testcase: dict[str, Any], *, require_test_id: bool) -> None:
    if not isinstance(testcase, dict):
        raise ValueError("testcase must be a JSON object")
    if require_test_id:
        require_string(testcase, "test_id")
    require_string(testcase, "input")
    require_string(testcase, "output")


def testcase_fingerprint(testcase: dict[str, Any]) -> str:
    return stable_hash(
        {
            "input": testcase.get("input", ""),
            "output": testcase.get("output", ""),
        }
    )


def validate_raw_problem(record: dict[str, Any]) -> None:
    require_string(record, "source")
    require_string(record, "source_id")
    require_string(record, "prompt")
    require_string(record, "reference_code")
    tests = record.get("tests")
    if not isinstance(tests, list) or not tests:
        raise ValueError("tests must be a non-empty list")
    for testcase in tests:
        validate_testcase(testcase, require_test_id=False)


def validate_reasoning_record(record: dict[str, Any]) -> None:
    has_problem_id = isinstance(record.get("problem_id"), str) and bool(record["problem_id"].strip())
    has_source_key = (
        isinstance(record.get("source"), str)
        and bool(record["source"].strip())
        and isinstance(record.get("source_id"), str)
        and bool(record["source_id"].strip())
    )
    if not has_problem_id and not has_source_key:
        raise ValueError("reasoning record needs problem_id or exact (source, source_id)")
    require_string(record, "reasoning")


def validate_problem(record: dict[str, Any]) -> None:
    for field in (
        "problem_id",
        "source",
        "source_id",
        "difficulty",
        "prompt",
        "reasoning",
        "reference_code",
        "split",
    ):
        require_string(record, field)
    if record["split"] not in SPLITS:
        raise ValueError(f"split must be one of {sorted(SPLITS)}")

    reward_tests = record.get("reward_tests")
    heldout_tests = record.get("heldout_tests")
    if not isinstance(reward_tests, list) or not reward_tests:
        raise ValueError("reward_tests must be a non-empty list")
    if not isinstance(heldout_tests, list) or not heldout_tests:
        raise ValueError("heldout_tests must be a non-empty list")
    for testcase in reward_tests:
        validate_testcase(testcase, require_test_id=True)
    for testcase in heldout_tests:
        validate_testcase(testcase, require_test_id=True)

    reward_ids = {testcase["test_id"] for testcase in reward_tests}
    heldout_ids = {testcase["test_id"] for testcase in heldout_tests}
    if reward_ids & heldout_ids:
        raise ValueError(f"reward/heldout test_id overlap for {record['problem_id']}")

    reward_fingerprints = {testcase_fingerprint(testcase) for testcase in reward_tests}
    heldout_fingerprints = {testcase_fingerprint(testcase) for testcase in heldout_tests}
    if reward_fingerprints & heldout_fingerprints:
        raise ValueError(f"reward/heldout testcase content overlap for {record['problem_id']}")


def problem_text_for_dedup(record: dict[str, Any]) -> str:
    return record.get("prompt", "")

