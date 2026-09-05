from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from data.config import load_config, resolve_phase0_paths
from data.schemas import make_problem_id, make_source_id, stable_json, validate_raw_problem, validate_reasoning_record


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


def _configure_hf_endpoint(hf_endpoint: str | None) -> None:
    if not hf_endpoint:
        return
    os.environ["HF_ENDPOINT"] = hf_endpoint
    try:
        import datasets.config as datasets_config

        datasets_config.HF_ENDPOINT = hf_endpoint
    except Exception:
        pass
    try:
        import huggingface_hub.constants as hf_constants

        hf_constants.ENDPOINT = hf_endpoint
    except Exception:
        pass


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


def _record_from_source_row(
    dataset_name: str,
    source_split: str,
    index: int,
    row: dict[str, Any],
) -> dict[str, Any] | None:
    tests = _parse_tests(row.get("input_output"))
    if not tests:
        return None

    reference_solutions = _parse_solution_list(row.get("solutions"))
    if not reference_solutions:
        return None

    prompt = _as_text(row.get("question")).strip()
    if not prompt:
        return None

    source_id = make_source_id(source_split, index)
    return {
        "source": dataset_name,
        "source_id": source_id,
        "difficulty": _as_text(row.get("difficulty")).strip() or "unknown",
        "prompt": prompt,
        "reference_code": reference_solutions[0].strip(),
        "tests": tests,
        "metadata": {
            "url": _as_text(row.get("url")).strip(),
            "source_dataset": dataset_name,
            "source_split": source_split,
            "source_index": index,
        },
    }


def _load_apps(
    raw_sources_dir: Path,
    wanted_indices_by_split: dict[str, set[int]],
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, Any]]:
    records: dict[tuple[str, int], dict[str, Any]] = {}
    report: dict[str, Any] = {
        "requested_counts": {split: len(indices) for split, indices in sorted(wanted_indices_by_split.items())},
        "loaded_counts": {},
        "unavailable_splits": {},
    }
    for source_split, wanted_indices in sorted(wanted_indices_by_split.items()):
        if not wanted_indices:
            report["loaded_counts"][source_split] = 0
            continue
        print(f"Loading APPS {source_split} rows for {len(wanted_indices)} candidate indices", flush=True)
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(
                repo_id="codeparrot/apps",
                repo_type="dataset",
                filename=f"{source_split}.jsonl",
                local_dir=str(raw_sources_dir / "apps"),
            )
        except Exception as exc:  # noqa: BLE001 - split availability is reported, not hidden by train fallback.
            report["loaded_counts"][source_split] = 0
            report["unavailable_splits"][source_split] = repr(exc)
            print(f"APPS split {source_split} unavailable; no fallback will be used", flush=True)
            continue

        loaded_count = 0
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle):
                if not line.strip():
                    continue
                row = json.loads(line)
                index = int(row.get("id", line_no))
                if index not in wanted_indices:
                    continue
                converted = _record_from_source_row("apps", source_split, index, row)
                if converted is not None:
                    records[(source_split, index)] = converted
                    loaded_count += 1
        report["loaded_counts"][source_split] = loaded_count
        print(f"Loaded {loaded_count} usable APPS {source_split} rows", flush=True)
    report["total_loaded"] = len(records)
    return records, report


def _load_taco(
    raw_sources_dir: Path,
    wanted_indices_by_split: dict[str, set[int]],
    *,
    skip_interactive: bool,
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, Any]]:
    records: dict[tuple[str, int], dict[str, Any]] = {}
    report: dict[str, Any] = {
        "requested_counts": {split: len(indices) for split, indices in sorted(wanted_indices_by_split.items())},
        "loaded_counts": {},
        "unavailable_splits": {},
    }
    for source_split, wanted_indices in sorted(wanted_indices_by_split.items()):
        if not wanted_indices:
            report["loaded_counts"][source_split] = 0
            continue
        print(f"Loading TACO {source_split} rows for {len(wanted_indices)} candidate indices", flush=True)
        try:
            from huggingface_hub import snapshot_download

            root = Path(
                snapshot_download(
                    repo_id="BAAI/TACO",
                    repo_type="dataset",
                    allow_patterns=[f"ALL/{source_split}-*.parquet"],
                    local_dir=str(raw_sources_dir / "taco"),
                )
            )
        except Exception as exc:  # noqa: BLE001 - split availability is reported, not hidden by train fallback.
            report["loaded_counts"][source_split] = 0
            report["unavailable_splits"][source_split] = repr(exc)
            print(f"TACO split {source_split} unavailable; no fallback will be used", flush=True)
            continue

        parquet_paths = sorted((root / "ALL").glob(f"{source_split}-*.parquet"))
        if not parquet_paths:
            report["loaded_counts"][source_split] = 0
            report["unavailable_splits"][source_split] = "no matching parquet files"
            print(f"TACO split {source_split} has no local parquet files; no fallback will be used", flush=True)
            continue

        loaded_count = 0
        offset = 0
        for parquet_path in parquet_paths:
            metadata = pd.read_parquet(parquet_path, columns=["question"])
            row_count = len(metadata)
            wanted_local_indices = sorted(
                index - offset for index in wanted_indices if offset <= index < offset + row_count
            )
            if not wanted_local_indices:
                offset += row_count
                continue
            df = pd.read_parquet(parquet_path)
            for local_index in wanted_local_indices:
                row = df.iloc[local_index].to_dict()
                global_index = offset + local_index
                if skip_interactive and _contains_interactive(row):
                    continue
                converted = _record_from_source_row("taco", source_split, global_index, row)
                if converted is not None:
                    records[(source_split, global_index)] = converted
                    loaded_count += 1
            offset += row_count
            print(
                f"Loaded TACO {source_split} shard {parquet_path.name}; usable rows so far: {loaded_count}",
                flush=True,
            )
        report["loaded_counts"][source_split] = loaded_count
        print(f"Loaded {loaded_count} usable TACO {source_split} rows", flush=True)
    report["total_loaded"] = len(records)
    return records, report


def _parse_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_source_split(value: Any) -> str:
    return _as_text(value).strip().lower()


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
) -> tuple[list[dict[str, Any]], dict[str, int], int, str, dict[str, int]]:
    print(
        f"Collecting OpenCodeReasoning-2 candidates: target={target}, max_seen={max_seen}",
        flush=True,
    )
    candidates: list[dict[str, Any]] = []
    seen_problem_ids: set[str] = set()
    skip_counts: dict[str, int] = {
        "dataset_not_selected": 0,
        "bad_index": 0,
        "missing_source_split": 0,
        "duplicate_problem": 0,
        "bad_judgement": 0,
        "low_pass_rate": 0,
        "missing_reasoning": 0,
    }
    source_candidate_counts: Counter[str] = Counter()

    from datasets import load_dataset

    opencode = load_dataset(
        "nvidia/OpenCodeReasoning-2",
        split=raw_build.get("opencode_split", "python"),
        streaming=True,
    )
    seen_opencode = 0
    stop_reason = "dataset_exhausted"
    for item in opencode:
        if seen_opencode >= max_seen:
            stop_reason = "max_opencode_records_reached"
            break
        seen_opencode += 1

        dataset_name = _as_text(item.get("dataset")).lower()
        if dataset_name not in datasets:
            skip_counts["dataset_not_selected"] += 1
            continue
        try:
            index = int(item.get("index"))
        except (TypeError, ValueError):
            skip_counts["bad_index"] += 1
            continue

        source_split = _normalize_source_split(item.get("split"))
        if not source_split:
            skip_counts["missing_source_split"] += 1
            continue
        source_id = make_source_id(source_split, index)
        problem_id = make_problem_id(dataset_name, source_id)
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
                "source_split": source_split,
                "source_id": source_id,
                "index": index,
                "reasoning": reasoning_text,
                "solution": _as_text(item.get("solution")).strip(),
                "opencode_id": _as_text(item.get("id")),
                "opencode_question_id": _as_text(item.get("question_id")),
                "opencode_source_split": source_split,
                "opencode_pass_rate": item.get("pass_rate"),
                "opencode_judgement": item.get("judgement"),
            }
        )
        seen_problem_ids.add(problem_id)
        source_candidate_counts[f"{dataset_name}/{source_split}"] += 1

        if len(candidates) % 2000 == 0:
            print(f"Collected {len(candidates)} candidates after {seen_opencode} OCR rows", flush=True)

    print(
        f"Collected {len(candidates)} candidates after scanning {seen_opencode} OpenCodeReasoning-2 rows "
        f"({stop_reason})",
        flush=True,
    )
    return candidates, skip_counts, seen_opencode, stop_reason, dict(sorted(source_candidate_counts.items()))


def _assemble_records(
    candidates: list[dict[str, Any]],
    source_tables: dict[str, dict[tuple[str, int], dict[str, Any]]],
    *,
    target: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int], dict[str, int]]:
    problems: list[dict[str, Any]] = []
    reasonings: list[dict[str, Any]] = []
    seen_problem_ids: set[str] = set()
    output_source_split_counts: Counter[str] = Counter()
    skip_counts: dict[str, int] = {
        "missing_source_problem": 0,
        "duplicate_problem": 0,
        "source_split_mismatch": 0,
        "invalid_output": 0,
    }

    for item in candidates:
        if len(problems) >= target:
            break

        dataset_name = item["dataset"]
        source_split = item["source_split"]
        index = item["index"]
        source_id = item["source_id"]
        problem_id = item["problem_id"]
        if problem_id in seen_problem_ids:
            skip_counts["duplicate_problem"] += 1
            continue
        source_record = source_tables.get(dataset_name, {}).get((source_split, index))
        if source_record is None:
            skip_counts["missing_source_problem"] += 1
            continue
        source_metadata = source_record.get("metadata", {})
        if source_metadata.get("source_split") != source_split or source_record.get("source_id") != source_id:
            skip_counts["source_split_mismatch"] += 1
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
            "opencode_source_split": item["opencode_source_split"],
            "opencode_index": index,
            "opencode_pass_rate": item["opencode_pass_rate"],
            "opencode_judgement": item["opencode_judgement"],
            "mapping_policy": "exact (dataset, source split, index) join",
        }
        reasoning = {
            "problem_id": problem_id,
            "source": dataset_name,
            "source_id": source_id,
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
        output_source_split_counts[f"{dataset_name}/{source_split}"] += 1
        if len(problems) % 2000 == 0:
            print(f"Assembled {len(problems)} raw records", flush=True)

    return problems, reasonings, skip_counts, dict(sorted(output_source_split_counts.items()))


def build_raw(
    config: dict[str, Any],
    paths: dict[str, Path],
    *,
    target_records: int | None,
    max_opencode_records: int | None,
) -> dict[str, Any]:
    raw_build = config["raw_build"]
    hf_endpoint = raw_build.get("hf_endpoint")
    _configure_hf_endpoint(_as_text(hf_endpoint).strip() if hf_endpoint else None)

    raw_sources_dir = Path(config["paths"].get("raw_sources_dir", paths["artifact_root"].parent / "raw_sources"))
    raw_dir = paths["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_sources_dir.mkdir(parents=True, exist_ok=True)

    target = int(target_records or raw_build["target_records"])
    max_seen = int(max_opencode_records or raw_build["max_opencode_records"])
    min_pass_rate = float(raw_build["min_pass_rate"])
    required_judgement = _as_text(raw_build["required_judgement"]).lower()
    datasets = {_as_text(dataset_name).lower() for dataset_name in raw_build["datasets"]}
    skip_interactive = bool(raw_build.get("skip_interactive", True))

    (
        candidates,
        candidate_skip_counts,
        seen_opencode,
        candidate_stop_reason,
        source_candidate_counts,
    ) = _collect_opencode_candidates(
        raw_build,
        datasets=datasets,
        target=target,
        max_seen=max_seen,
        min_pass_rate=min_pass_rate,
        required_judgement=required_judgement,
    )
    wanted_indices: dict[str, dict[str, set[int]]] = {dataset_name: {} for dataset_name in datasets}
    for candidate in candidates:
        dataset_wanted = wanted_indices.setdefault(candidate["dataset"], {})
        dataset_wanted.setdefault(candidate["source_split"], set()).add(candidate["index"])

    source_tables: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    source_load_reports: dict[str, dict[str, Any]] = {}
    if "apps" in datasets:
        source_tables["apps"], source_load_reports["apps"] = _load_apps(
            raw_sources_dir,
            wanted_indices.get("apps", {}),
        )
    if "taco" in datasets:
        source_tables["taco"], source_load_reports["taco"] = _load_taco(
            raw_sources_dir,
            wanted_indices.get("taco", {}),
            skip_interactive=skip_interactive,
        )

    problems, reasonings, skip_counts, output_source_split_counts = _assemble_records(
        candidates,
        source_tables,
        target=target,
    )

    _write_jsonl(raw_dir / "problems.jsonl", problems)
    _write_jsonl(raw_dir / "reasoning.jsonl", reasonings)
    report = {
        "status": "built",
        "target_records": target,
        "max_opencode_records": max_seen,
        "seen_opencode_records": seen_opencode,
        "candidate_stop_reason": candidate_stop_reason,
        "scanned_full_available_opencode_stream": candidate_stop_reason == "dataset_exhausted",
        "candidate_collection_policy": (
            "scan OpenCodeReasoning-2 until max_opencode_records or dataset exhaustion; "
            "candidate_multiplier is not used for early stopping"
        ),
        "candidate_multiplier_config_value": raw_build.get("candidate_multiplier"),
        "candidate_records": len(candidates),
        "output_problem_records": len(problems),
        "output_reasoning_records": len(reasonings),
        "source_counts": {name: len(records) for name, records in sorted(source_tables.items())},
        "source_candidate_counts": source_candidate_counts,
        "source_loaded_counts": {
            f"{dataset_name}/{source_split}": count
            for dataset_name, load_report in sorted(source_load_reports.items())
            for source_split, count in sorted(load_report.get("loaded_counts", {}).items())
        },
        "output_source_split_counts": output_source_split_counts,
        "source_load_reports": source_load_reports,
        "candidate_skip_counts": candidate_skip_counts,
        "assembly_skip_counts": skip_counts,
        "raw_dir": str(raw_dir),
        "raw_sources_dir": str(raw_sources_dir),
        "mapping_policy": (
            "OpenCodeReasoning-2 to APPS/TACO by exact (dataset, source split, index); "
            "no fuzzy or embedding join"
        ),
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
