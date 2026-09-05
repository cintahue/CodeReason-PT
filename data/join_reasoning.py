from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from data.schemas import make_problem_id, validate_raw_problem, validate_reasoning_record


@dataclass(frozen=True)
class JoinResult:
    records: list[dict[str, Any]]
    report: dict[str, Any]


def _append_index(
    index: dict[str, list[dict[str, Any]]],
    key: str | None,
    record: dict[str, Any],
) -> None:
    if key:
        index[key].append(record)


def _select_unique(index: dict[str, list[dict[str, Any]]], key: str) -> dict[str, Any] | None:
    matches = index.get(key, [])
    if len(matches) != 1:
        return None
    return matches[0]


def join_problem_and_reasoning(
    problems: list[dict[str, Any]],
    reasoning_records: list[dict[str, Any]],
) -> JoinResult:
    by_problem_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_source_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ambiguous_problem_ids: set[str] = set()
    ambiguous_source_keys: set[str] = set()

    for record in reasoning_records:
        validate_reasoning_record(record)
        problem_id = record.get("problem_id")
        if isinstance(problem_id, str) and problem_id.strip():
            _append_index(by_problem_id, problem_id.strip(), record)
        source = record.get("source")
        source_id = record.get("source_id")
        if isinstance(source, str) and isinstance(source_id, str) and source.strip() and source_id.strip():
            _append_index(by_source_key, make_problem_id(source.strip(), source_id.strip()), record)

    ambiguous_problem_ids = {key for key, values in by_problem_id.items() if len(values) > 1}
    ambiguous_source_keys = {key for key, values in by_source_key.items() if len(values) > 1}

    joined: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    dropped_no_reasoning = 0
    dropped_ambiguous = 0
    dropped_invalid_problem = 0

    for problem in problems:
        try:
            validate_raw_problem(problem)
        except ValueError:
            dropped_invalid_problem += 1
            continue

        source = problem["source"].strip()
        source_id = problem["source_id"].strip()
        problem_id = problem.get("problem_id") or make_problem_id(source, source_id)
        source_key = make_problem_id(source, source_id)

        if problem_id in ambiguous_problem_ids or source_key in ambiguous_source_keys:
            dropped_ambiguous += 1
            continue

        reasoning = _select_unique(by_problem_id, problem_id)
        mapping_key = "problem_id"
        if reasoning is None:
            reasoning = _select_unique(by_source_key, source_key)
            mapping_key = "source/source_id"
        if reasoning is None:
            dropped_no_reasoning += 1
            continue

        output = dict(problem)
        output["problem_id"] = problem_id
        output["source"] = source
        output["source_id"] = source_id
        output["difficulty"] = str(output.get("difficulty", "unknown"))
        output["reasoning"] = reasoning["reasoning"]
        output["mapping_key"] = mapping_key
        joined.append(output)
        source_counts[source] += 1

    report = {
        "raw_problems": len(problems),
        "raw_reasoning": len(reasoning_records),
        "joined": len(joined),
        "dropped_no_reasoning": dropped_no_reasoning,
        "dropped_ambiguous_reasoning": dropped_ambiguous,
        "dropped_invalid_problem": dropped_invalid_problem,
        "ambiguous_problem_id_keys": len(ambiguous_problem_ids),
        "ambiguous_source_id_keys": len(ambiguous_source_keys),
        "source_counts": dict(sorted(source_counts.items())),
        "join_policy": "exact problem_id or exact (source, source_id); no fuzzy or embedding join",
    }
    return JoinResult(records=joined, report=report)

