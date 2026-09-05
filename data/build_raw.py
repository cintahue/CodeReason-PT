from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download, snapshot_download

from data.config import load_config, resolve_phase0_paths
from data.schemas import stable_json, validate_raw_problem, validate_reasoning_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase 0 raw JSONL files from public sources.")
    parser.add_argument("--config", default="configs/phase0.yaml")
    parser.add_argument("--target-records", type=int, default=None)
    parser.add_argument("--max-opencode-records", type=int, default=None)
    return parser.parse_args()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(stable_json(record))
            handle.write("\n")


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value)


def _parse_json_maybe(value: Any) -> Any:
    if isinstance(value, str):
        if not value.strip():
            return None
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return None
    return value


def _parse_solution_list(value: Any) -> list[str]:
    if value is None:
        return []
    parsed = _parse_json_maybe(value)
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, str) and item.strip()]
    return []


def _parse_tests(value: Any) -> list[dict[str, str]]:
    parsed = _parse_json_maybe(value)
    if not isinstance(parsed, dict):
        return []
    inputs = parsed.get("inputs")
    outputs = parsed.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        return []
    if len(inputs) != len(outputs):
        return []

    tests: list[dict[str, str]] = []
    for raw_input, raw_output in zip(inputs, outputs):
        if not isinstance(raw_input, str) or not isinstance(raw_output, str):
            continue
        tests.append({"input": raw_input, "output": raw_output})
    return tests


def _contains_interactive(row: dict[str, Any]) -> bool:
    fields = [
        _as_text(row.get("question")),
        _as_text(row.get("raw_tags")),
        _as_text(row.get("tags")),
        _as_text(row.get("skill_types")),
    ]
    return any(re.search(r"\binteractive\b", field.lower()) for field in fields)


def _record_from_source_row(dataset_name: str, index: int, row: dict[str, Any]) -> dict[str, Any] | None:
    tests = _parse_tests(row.get("input_output"))
    if not tests:
        return None

    reference_solutions = _parse_solution_list(row.get("solutions"))
    if not reference_solutions:
        return None

    prompt = _as_text(row.get("question")).strip()
    if not prompt:
        return None

    return {
        "source": dataset_name,
        "source_id": str(index),
        "difficulty": _as_text(row.get("difficulty")).strip() or "unknown",
        "prompt": prompt,
        "reference_code": reference_solutions[0].strip(),
        "tests": tests,
        "metadata": {
            "url": _as_text(row.get("url")).strip(),
            "source_dataset": dataset_name,
            "source_split": "train",
            "source_index": index,
        },
    }


def _load_apps(raw_sources_dir: Path, wanted_indices: set[int]) -> dict[int, dict[str, Any]]:
    print(f"Loading APPS train rows for {len(wanted_indices)} candidate indices", flush=True)
    path = hf_hub_download(
        repo_id="codeparrot/apps",
        repo_type="dataset",
        filename="train.jsonl",
        local_dir=str(raw_sources_dir / "apps"),
    )
    records: dict[int, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            index = int(row.get("id", line_no))
            if index not in wanted_indices:
                continue
            converted = _record_from_source_row("apps", index, row)
            if converted is not None:
                records[index] = converted
    print(f"Loaded {len(records)} usable APPS rows", flush=True)
    return records


def _load_taco(raw_sources_dir: Path, wanted_indices: set[int], *, skip_interactive: bool) -> dict[int, dict[str, Any]]:
    print(f"Loading TACO train rows for {len(wanted_indices)} candidate indices", flush=True)
    root = Path(
        snapshot_download(
            repo_id="BAAI/TACO",
            repo_type="dataset",
            allow_patterns=["ALL/train-*.parquet"],
            local_dir=str(raw_sources_dir / "taco"),
        )
    )
    records: dict[int, dict[str, Any]] = {}
    offset = 0
    for parquet_path in sorted((root / "ALL").glob("train-*.parquet")):
        metadata = pd.read_parquet(parquet_path, columns=["question"])
        row_count = len(metadata)
        wanted_local_indices = sorted(index - offset for index in wanted_indices if offset <= index < offset + row_count)
        if not wanted_local_indices:
            offset += row_count
            continue
        df = pd.read_parquet(parquet_path)
        for local_index in wanted_local_indices:
            row = df.iloc[local_index].to_dict()
            global_index = offset + local_index
            if skip_interactive and _contains_interactive(row):
                continue
            converted = _record_from_source_row("taco", global_index, row)
            if converted is not None:
                records[global_index] = converted
        offset += row_count
        print(f"Loaded TACO shard {parquet_path.name}; usable rows so far: {len(records)}", flush=True)
    print(f"Loaded {len(records)} usable TACO rows", flush=True)
    return records


def _parse_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _select_reference_code(source_record: dict[str, Any], opencode_record: dict[str, Any]) -> str:
    solution = _as_text(opencode_record.get("solution")).strip()
    if solution:
        return solution
    return source_record["reference_code"]


def _collect_opencode_candidates(
    raw_build: dict[str, Any],
    *,
    datasets: set[str],
    target: int,
    max_seen: int,
    min_pass_rate: float,
    required_judgement: str,
) -> tuple[list[dict[str, Any]], dict[str, int], int]:
    candidate_goal = target * int(raw_build.get("candidate_multiplier", 4))
    print(
        f"Collecting OpenCodeReasoning-2 candidates: goal={candidate_goal}, max_seen={max_seen}",
        flush=True,
    )
    candidates: list[dict[str, Any]] = []
    seen_problem_ids: set[str] = set()
    skip_counts: dict[str, int] = {
        "dataset_not_selected": 0,
        "bad_index": 0,
        "duplicate_problem": 0,
        "bad_judgement": 0,
        "low_pass_rate": 0,
        "missing_reasoning": 0,
    }

    opencode = load_dataset(
        "nvidia/OpenCodeReasoning-2",
        split=raw_build.get("opencode_split", "python"),
        streaming=True,
    )
    seen_opencode = 0
    for seen_opencode, item in enumerate(opencode, start=1):
        if seen_opencode > max_seen or len(candidates) >= candidate_goal:
            break

        dataset_name = _as_text(item.get("dataset")).lower()
        if dataset_name not in datasets:
            skip_counts["dataset_not_selected"] += 1
            continue
        try:
            index = int(item.get("index"))
        except (TypeError, ValueError):
            skip_counts["bad_index"] += 1
            continue

        problem_id = f"{dataset_name}:{index}"
        if problem_id in seen_problem_ids:
            skip_counts["duplicate_problem"] += 1
            continue
        if _as_text(item.get("judgement")).lower() != required_judgement:
            skip_counts["bad_judgement"] += 1
            continue
        pass_rate = _parse_float(item.get("pass_rate"))
        if pass_rate is None or pass_rate < min_pass_rate:
            skip_counts["low_pass_rate"] += 1
            continue

        reasoning_text = _as_text(item.get("r1_generation")).strip()
        if not reasoning_text:
            skip_counts["missing_reasoning"] += 1
            continue

        candidates.append(
            {
                "problem_id": problem_id,
                "dataset": dataset_name,
                "index": index,
                "reasoning": reasoning_text,
                "solution": _as_text(item.get("solution")).strip(),
                "opencode_id": _as_text(item.get("id")),
                "opencode_question_id": _as_text(item.get("question_id")),
                "opencode_split": _as_text(item.get("split")),
                "opencode_pass_rate": item.get("pass_rate"),
                "opencode_judgement": item.get("judgement"),
            }
        )
        seen_problem_ids.add(problem_id)

        if len(candidates) % 2000 == 0:
            print(f"Collected {len(candidates)} candidates after {seen_opencode} OCR rows", flush=True)

    print(
        f"Collected {len(candidates)} candidates after scanning {seen_opencode} OpenCodeReasoning-2 rows",
        flush=True,
    )
    return candidates, skip_counts, seen_opencode


def build_raw(
    config: dict[str, Any],
    paths: dict[str, Path],
    *,
    target_records: int | None,
    max_opencode_records: int | None,
) -> dict[str, Any]:
    raw_build = config["raw_build"]
    hf_endpoint = raw_build.get("hf_endpoint")
    if hf_endpoint:
        os.environ.setdefault("HF_ENDPOINT", hf_endpoint)

    raw_sources_dir = Path(config["paths"].get("raw_sources_dir", paths["artifact_root"].parent / "raw_sources"))
    raw_dir = paths["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_sources_dir.mkdir(parents=True, exist_ok=True)

    target = int(target_records or raw_build["target_records"])
    max_seen = int(max_opencode_records or raw_build["max_opencode_records"])
    min_pass_rate = float(raw_build["min_pass_rate"])
    required_judgement = _as_text(raw_build["required_judgement"]).lower()
    datasets = set(raw_build["datasets"])
    skip_interactive = bool(raw_build.get("skip_interactive", True))

    candidates, candidate_skip_counts, seen_opencode = _collect_opencode_candidates(
        raw_build,
        datasets=datasets,
        target=target,
        max_seen=max_seen,
        min_pass_rate=min_pass_rate,
        required_judgement=required_judgement,
    )
    wanted_indices: dict[str, set[int]] = {dataset_name: set() for dataset_name in datasets}
    for candidate in candidates:
        wanted_indices[candidate["dataset"]].add(candidate["index"])

    source_tables: dict[str, dict[int, dict[str, Any]]] = {}
    if "apps" in datasets:
        source_tables["apps"] = _load_apps(raw_sources_dir, wanted_indices.get("apps", set()))
    if "taco" in datasets:
        source_tables["taco"] = _load_taco(
            raw_sources_dir,
            wanted_indices.get("taco", set()),
            skip_interactive=skip_interactive,
        )

    problems: list[dict[str, Any]] = []
    reasonings: list[dict[str, Any]] = []
    seen_problem_ids: set[str] = set()
    skip_counts: dict[str, int] = {
        "missing_source_problem": 0,
        "duplicate_problem": 0,
        "invalid_output": 0,
    }

    for item in candidates:
        if len(problems) >= target:
            break

        dataset_name = item["dataset"]
        index = item["index"]
        problem_id = item["problem_id"]
        if problem_id in seen_problem_ids:
            skip_counts["duplicate_problem"] += 1
            continue
        source_record = source_tables.get(dataset_name, {}).get(index)
        if source_record is None:
            skip_counts["missing_source_problem"] += 1
            continue

        reasoning_text = item["reasoning"]
        if not reasoning_text:
            skip_counts["invalid_output"] += 1
            continue

        problem = dict(source_record)
        problem["problem_id"] = problem_id
        problem["reference_code"] = _select_reference_code(source_record, item)
        problem["metadata"] = {
            **problem.get("metadata", {}),
            "opencode_id": item["opencode_id"],
            "opencode_question_id": item["opencode_question_id"],
            "opencode_dataset": dataset_name,
            "opencode_split": item["opencode_split"],
            "opencode_index": index,
            "opencode_pass_rate": item["opencode_pass_rate"],
            "opencode_judgement": item["opencode_judgement"],
            "mapping_policy": "exact (dataset, split, index) join",
        }
        reasoning = {
            "problem_id": problem_id,
            "source": dataset_name,
            "source_id": str(index),
            "reasoning": reasoning_text,
        }
        try:
            validate_raw_problem(problem)
            validate_reasoning_record(reasoning)
        except ValueError:
            skip_counts["invalid_output"] += 1
            continue

        problems.append(problem)
        reasonings.append(reasoning)
        seen_problem_ids.add(problem_id)
        if len(problems) % 2000 == 0:
            print(f"Assembled {len(problems)} raw records", flush=True)

    _write_jsonl(raw_dir / "problems.jsonl", problems)
    _write_jsonl(raw_dir / "reasoning.jsonl", reasonings)
    report = {
        "status": "built",
        "target_records": target,
        "max_opencode_records": max_seen,
        "seen_opencode_records": seen_opencode,
        "candidate_records": len(candidates),
        "output_problem_records": len(problems),
        "output_reasoning_records": len(reasonings),
        "source_counts": {name: len(records) for name, records in sorted(source_tables.items())},
        "candidate_skip_counts": candidate_skip_counts,
        "assembly_skip_counts": skip_counts,
        "raw_dir": str(raw_dir),
        "raw_sources_dir": str(raw_sources_dir),
        "mapping_policy": "OpenCodeReasoning-2 to APPS/TACO by exact (dataset, split, index); no fuzzy or embedding join",
    }
    _write_json(raw_dir / "build_raw_report.json", report)
    return report


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = resolve_phase0_paths(config)
    report = build_raw(
        config,
        paths,
        target_records=args.target_records,
        max_opencode_records=args.max_opencode_records,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
