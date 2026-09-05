from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from data.config import ensure_phase0_dirs, load_config, resolve_phase0_paths
from data.deduplicate import detect_cross_split_overlaps
from data.join_reasoning import join_problem_and_reasoning
from data.logging_utils import setup_logging
from data.schemas import iter_jsonl, stable_hash, validate_problem, write_jsonl
from data.split_tests import attach_reward_heldout_split, test_split_manifest_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Phase 0 data artifacts.")
    parser.add_argument("--config", default="configs/phase0.yaml")
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--artifact-root", default=None)
    return parser.parse_args()


def _load_raw(raw_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    problems_path = raw_dir / "problems.jsonl"
    reasoning_path = raw_dir / "reasoning.jsonl"
    if not problems_path.exists():
        raise FileNotFoundError(f"Missing raw problem file: {problems_path}")
    if not reasoning_path.exists():
        raise FileNotFoundError(f"Missing raw reasoning file: {reasoning_path}")
    return list(iter_jsonl(problems_path)), list(iter_jsonl(reasoning_path))


def _split_records(
    records: list[dict[str, Any]],
    *,
    seed: int,
    ratios: dict[str, float],
    target_counts: dict[str, int] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    split_names = ("sft", "pt", "dev")
    if set(ratios) != set(split_names):
        raise ValueError(f"split.ratios must contain exactly {split_names}")
    if target_counts is not None and set(target_counts) != set(split_names):
        raise ValueError(f"split.target_counts must contain exactly {split_names}")
    total_ratio = sum(float(ratios[name]) for name in split_names)
    if total_ratio <= 0:
        raise ValueError("split ratios must sum to a positive value")

    ordered = sorted(
        records,
        key=lambda record: stable_hash({"seed": seed, "problem_id": record["problem_id"]}),
    )
    if target_counts is not None and len(ordered) >= sum(target_counts.values()):
        ordered = ordered[: sum(target_counts.values())]
        counts = {name: int(target_counts[name]) for name in split_names}
        return {
            "sft": ordered[: counts["sft"]],
            "pt": ordered[counts["sft"] : counts["sft"] + counts["pt"]],
            "dev": ordered[counts["sft"] + counts["pt"] :],
        }

    n_records = len(ordered)
    quotas = {name: n_records * float(ratios[name]) / total_ratio for name in split_names}
    counts = {name: int(quotas[name]) for name in split_names}
    remaining = n_records - sum(counts.values())
    for name in sorted(split_names, key=lambda item: quotas[item] - counts[item], reverse=True):
        if remaining <= 0:
            break
        counts[name] += 1
        remaining -= 1

    result: dict[str, list[dict[str, Any]]] = {"sft": [], "pt": [], "dev": []}
    cursor = 0
    for name in split_names:
        result[name] = ordered[cursor : cursor + counts[name]]
        cursor += counts[name]
    return result


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _min_total_tests_by_split(split_config: dict[str, Any]) -> dict[str, int]:
    split_names = ("sft", "pt", "dev")
    if "min_total_tests_by_split" in split_config:
        configured = split_config["min_total_tests_by_split"]
        if set(configured) != set(split_names):
            raise ValueError(f"split.min_total_tests_by_split must contain exactly {split_names}")
        return {name: int(configured[name]) for name in split_names}
    legacy_min_total_tests = int(split_config["min_total_tests"])
    return {name: legacy_min_total_tests for name in split_names}


def _cross_split_report(
    attached: dict[str, list[dict[str, Any]]],
    *,
    left_name: str,
    right_name: str,
    ngram_size: int,
    near_duplicate_threshold: float,
) -> dict[str, Any]:
    return detect_cross_split_overlaps(
        attached[left_name],
        attached[right_name],
        left_name=left_name,
        right_name=right_name,
        ngram_size=ngram_size,
        near_duplicate_threshold=near_duplicate_threshold,
    )


def _all_overlap_reports(
    attached: dict[str, list[dict[str, Any]]],
    *,
    ngram_size: int,
    near_duplicate_threshold: float,
) -> dict[str, dict[str, Any]]:
    return {
        "sft_pt": _cross_split_report(
            attached,
            left_name="sft",
            right_name="pt",
            ngram_size=ngram_size,
            near_duplicate_threshold=near_duplicate_threshold,
        ),
        "sft_dev": _cross_split_report(
            attached,
            left_name="sft",
            right_name="dev",
            ngram_size=ngram_size,
            near_duplicate_threshold=near_duplicate_threshold,
        ),
        "pt_dev": _cross_split_report(
            attached,
            left_name="pt",
            right_name="dev",
            ngram_size=ngram_size,
            near_duplicate_threshold=near_duplicate_threshold,
        ),
    }


def _drop_reason_counts(dropped_records: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter()
    short_test_pattern = re.compile(r"has (?P<count>\d+) unique tests; min_total_tests=(?P<minimum>\d+)")
    for item in dropped_records:
        split = str(item.get("split", "unknown"))
        reason = str(item.get("reason", "unknown"))
        match = short_test_pattern.search(reason)
        if match:
            key = f"{split}:below_min_total_tests:unique_tests={match.group('count')}:min={match.group('minimum')}"
        else:
            key = f"{split}:{reason}"
        counter[key] += 1
    return dict(sorted(counter.items()))


def prepare_phase0(config: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    ensure_phase0_dirs(paths)
    logger = setup_logging(paths["logs_dir"] / "phase0_prepare.log")
    logger.info("Starting Phase 0 data preparation")
    logger.info("Raw directory: %s", paths["raw_dir"])
    logger.info("Artifact root: %s", paths["artifact_root"])

    problems, reasoning_records = _load_raw(paths["raw_dir"])
    join_result = join_problem_and_reasoning(problems, reasoning_records)
    logger.info("Joined %s/%s raw problems", join_result.report["joined"], join_result.report["raw_problems"])

    split_config = config["split"]
    split_seed = int(split_config["seed"])
    min_total_tests_by_split = _min_total_tests_by_split(split_config)
    reward_ratio = float(split_config["reward_ratio"])
    raw_splits = _split_records(
        join_result.records,
        seed=split_seed,
        ratios={key: float(value) for key, value in split_config["ratios"].items()},
        target_counts=(
            {key: int(value) for key, value in split_config["target_counts"].items()}
            if "target_counts" in split_config
            else None
        ),
    )

    attached: dict[str, list[dict[str, Any]]] = {"sft": [], "pt": [], "dev": []}
    dropped_short_tests: list[dict[str, Any]] = []
    for split_name, split_records in raw_splits.items():
        for record in split_records:
            try:
                final_record = attach_reward_heldout_split(
                    record,
                    split=split_name,
                    seed=split_seed,
                    reward_ratio=reward_ratio,
                    min_total_tests=min_total_tests_by_split[split_name],
                )
            except ValueError as exc:
                dropped_short_tests.append({"problem_id": record["problem_id"], "split": split_name, "reason": str(exc)})
                continue
            validate_problem(final_record)
            attached[split_name].append(final_record)

    dedup_config = config["dedup"]
    ngram_size = int(dedup_config["ngram_size"])
    near_duplicate_threshold = float(dedup_config["near_duplicate_threshold"])
    initial_overlap_report = _all_overlap_reports(
        attached,
        ngram_size=ngram_size,
        near_duplicate_threshold=near_duplicate_threshold,
    )
    blocked_pt_ids = set(initial_overlap_report["sft_pt"]["blocked_pt_problem_ids"])
    if blocked_pt_ids:
        attached["pt"] = [record for record in attached["pt"] if record["problem_id"] not in blocked_pt_ids]

    dev_blocking_overlap_report = {
        "sft_dev": _cross_split_report(
            attached,
            left_name="sft",
            right_name="dev",
            ngram_size=ngram_size,
            near_duplicate_threshold=near_duplicate_threshold,
        ),
        "pt_dev": _cross_split_report(
            attached,
            left_name="pt",
            right_name="dev",
            ngram_size=ngram_size,
            near_duplicate_threshold=near_duplicate_threshold,
        ),
    }
    blocked_dev_ids = set(dev_blocking_overlap_report["sft_dev"]["blocked_dev_problem_ids"])
    blocked_dev_ids.update(dev_blocking_overlap_report["pt_dev"]["blocked_dev_problem_ids"])
    if blocked_dev_ids:
        attached["dev"] = [record for record in attached["dev"] if record["problem_id"] not in blocked_dev_ids]

    post_overlap_report = _all_overlap_reports(
        attached,
        ngram_size=int(dedup_config["ngram_size"]),
        near_duplicate_threshold=float(dedup_config["near_duplicate_threshold"]),
    )
    for split_name, records in attached.items():
        logger.info("Final %s records: %s", split_name, len(records))

    all_records = attached["sft"] + attached["pt"] + attached["dev"]
    split_counts = Counter(record["split"] for record in all_records)

    write_jsonl(paths["processed_dir"] / "problems.jsonl", all_records)
    for split_name, records in attached.items():
        write_jsonl(paths["processed_dir"] / f"{split_name}.jsonl", records)
    write_jsonl(
        paths["reports_dir"] / "test_split_manifest.jsonl",
        [test_split_manifest_record(record) for record in all_records],
    )
    overlap_report = {
        "initial": initial_overlap_report,
        "dev_blocking": dev_blocking_overlap_report,
        "post": post_overlap_report,
        "blocked_pt_count": len(blocked_pt_ids),
        "blocked_dev_count": len(blocked_dev_ids),
        "blocked_pt_problem_ids": sorted(blocked_pt_ids),
        "blocked_dev_problem_ids": sorted(blocked_dev_ids),
    }
    _write_json(paths["reports_dir"] / "phase0_overlap_report.json", overlap_report)

    report = {
        "phase": "phase0",
        "status": "prepared",
        "raw_dir": str(paths["raw_dir"]),
        "artifact_root": str(paths["artifact_root"]),
        "join": join_result.report,
        "split_seed": split_seed,
        "split_ratios": split_config["ratios"],
        "target_counts": split_config.get("target_counts"),
        "min_total_tests_by_split": min_total_tests_by_split,
        "reward_ratio": reward_ratio,
        "dropped_short_tests": dropped_short_tests,
        "dropped_short_tests_count": len(dropped_short_tests),
        "drop_reason_counts": _drop_reason_counts(dropped_short_tests),
        "initial_overlap_report": initial_overlap_report,
        "dev_blocking_overlap_report": dev_blocking_overlap_report,
        "post_overlap_report": post_overlap_report,
        "blocked_pt_count": len(blocked_pt_ids),
        "blocked_dev_count": len(blocked_dev_ids),
        "final_counts": dict(sorted(split_counts.items())),
        "outputs": {
            "all": str(paths["processed_dir"] / "problems.jsonl"),
            "sft": str(paths["processed_dir"] / "sft.jsonl"),
            "pt": str(paths["processed_dir"] / "pt.jsonl"),
            "dev": str(paths["processed_dir"] / "dev.jsonl"),
            "test_split_manifest": str(paths["reports_dir"] / "test_split_manifest.jsonl"),
            "overlap_report": str(paths["reports_dir"] / "phase0_overlap_report.json"),
        },
    }
    _write_json(paths["reports_dir"] / "phase0_prepare_report.json", report)
    _write_json(paths["reports_dir"] / "config_snapshot.json", config)

    raw_external_manifest = paths["raw_dir"] / "external_eval_manifest.jsonl"
    if raw_external_manifest.exists():
        logger.info("External eval manifest found; external decontamination will be implemented against it later")
    else:
        _write_json(
            paths["reports_dir"] / "external_decontamination_report.json",
            {
                "status": "skipped_missing_external_eval_manifest",
                "expected_path": str(raw_external_manifest),
            },
        )

    logger.info("Phase 0 preparation finished")
    return report


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = resolve_phase0_paths(config, raw_dir=args.raw_dir, artifact_root=args.artifact_root)
    report = prepare_phase0(config, paths)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
