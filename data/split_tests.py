from __future__ import annotations

from typing import Any

from data.schemas import stable_hash, testcase_fingerprint


def _normalize_testcases(tests: list[dict[str, Any]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    normalized: list[dict[str, str]] = []
    for idx, testcase in enumerate(tests):
        output = {
            "input": str(testcase["input"]),
            "output": str(testcase["output"]),
        }
        fingerprint = testcase_fingerprint(output)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        supplied_id = testcase.get("test_id")
        if isinstance(supplied_id, str) and supplied_id.strip():
            output["test_id"] = supplied_id.strip()
        else:
            output["test_id"] = f"case_{idx:04d}_{fingerprint[:10]}"
        normalized.append(output)
    return normalized


def split_tests(
    *,
    problem_id: str,
    tests: list[dict[str, Any]],
    seed: int,
    reward_ratio: float,
    min_total_tests: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    normalized = _normalize_testcases(tests)
    if len(normalized) < min_total_tests:
        raise ValueError(f"{problem_id} has {len(normalized)} unique tests; min_total_tests={min_total_tests}")
    if not 0.0 < reward_ratio < 1.0:
        raise ValueError("reward_ratio must be between 0 and 1")

    ordered = sorted(
        normalized,
        key=lambda testcase: stable_hash(
            {
                "seed": seed,
                "problem_id": problem_id,
                "test_id": testcase["test_id"],
                "input": testcase["input"],
                "output": testcase["output"],
            }
        ),
    )
    reward_count = round(len(ordered) * reward_ratio)
    reward_count = min(max(1, reward_count), len(ordered) - 1)
    return ordered[:reward_count], ordered[reward_count:]


def attach_reward_heldout_split(
    record: dict[str, Any],
    *,
    split: str,
    seed: int,
    reward_ratio: float,
    min_total_tests: int,
) -> dict[str, Any]:
    reward_tests, heldout_tests = split_tests(
        problem_id=record["problem_id"],
        tests=record["tests"],
        seed=seed,
        reward_ratio=reward_ratio,
        min_total_tests=min_total_tests,
    )
    output = {
        "problem_id": record["problem_id"],
        "source": record["source"],
        "source_id": record["source_id"],
        "difficulty": record["difficulty"],
        "prompt": record["prompt"],
        "reasoning": record["reasoning"],
        "reference_code": record["reference_code"],
        "reward_tests": reward_tests,
        "heldout_tests": heldout_tests,
        "split": split,
    }
    return output


def test_split_manifest_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "problem_id": record["problem_id"],
        "split": record["split"],
        "reward_test_ids": [testcase["test_id"] for testcase in record["reward_tests"]],
        "heldout_test_ids": [testcase["test_id"] for testcase in record["heldout_tests"]],
        "reward_fingerprints": [testcase_fingerprint(testcase) for testcase in record["reward_tests"]],
        "heldout_fingerprints": [testcase_fingerprint(testcase) for testcase in record["heldout_tests"]],
    }

