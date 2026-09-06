from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data.schemas import stable_hash
from eval.phase2_common import (
    artifact_root,
    config_hash,
    file_sha256,
    generation_config_for_hash,
    git_command,
    git_status_short,
    load_jsonl,
    numeric_stats,
    package_versions,
    read_config,
    runtime_info,
    serialize_prompt,
    set_deterministic_seed,
    view_path,
    write_json,
    write_jsonl,
)
from eval.run_base_baseline import _generate, _load_model_and_tokenizer, _summarize_rollouts, _verify_generated_code
from verifier.extract_code import extract_python_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerun 2048-capped Phase 2 Dev samples with max_new_tokens=4096.")
    parser.add_argument("--config", default="configs/base_eval_v2.yaml")
    return parser.parse_args()


def _load_pilot_rollouts(config: dict[str, Any]) -> list[dict[str, Any]]:
    pilot = config["pilot_baseline"]
    rollouts_path = Path(str(pilot["rollouts_path"]))
    if not rollouts_path.exists():
        raise FileNotFoundError(f"Missing pilot baseline rollouts: {rollouts_path}")
    rollouts = load_jsonl(rollouts_path)
    expected = int(config["eval"]["expected_count"])
    if len(rollouts) != expected:
        raise ValueError(f"Pilot baseline rollout count mismatch: expected {expected}, got {len(rollouts)}")
    return rollouts


def _transition_matrix(old_rollouts: dict[str, dict[str, Any]], new_rollouts: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    matrix: dict[str, Counter] = defaultdict(Counter)
    for rollout in new_rollouts:
        old_status = str(old_rollouts[rollout["problem_id"]]["verifier_status"])
        new_status = str(rollout["verifier_status"])
        matrix[old_status][new_status] += 1
    return {old: dict(sorted(new.items())) for old, new in sorted(matrix.items())}


def _generation_cost(old_rollouts: dict[str, dict[str, Any]], new_rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    old_latencies = [int(old_rollouts[item["problem_id"]]["generation_latency_ms"]) for item in new_rollouts]
    new_latencies = [int(item["generation_latency_ms"]) for item in new_rollouts]
    old_tokens = [int(old_rollouts[item["problem_id"]]["response_token_count"]) for item in new_rollouts]
    new_tokens = [int(item["response_token_count"]) for item in new_rollouts]
    old_latency_sum = sum(old_latencies)
    new_latency_sum = sum(new_latencies)
    old_token_sum = sum(old_tokens)
    new_token_sum = sum(new_tokens)
    return {
        "old_generation_latency_ms": {"sum": old_latency_sum, "average": old_latency_sum / len(old_latencies), **numeric_stats(old_latencies, (50, 95))},
        "new_generation_latency_ms": {"sum": new_latency_sum, "average": new_latency_sum / len(new_latencies), **numeric_stats(new_latencies, (50, 95))},
        "latency_sum_ratio_new_over_old": new_latency_sum / old_latency_sum if old_latency_sum else None,
        "old_response_tokens": {"sum": old_token_sum, "average": old_token_sum / len(old_tokens), **numeric_stats(old_tokens, (50, 95))},
        "new_response_tokens": {"sum": new_token_sum, "average": new_token_sum / len(new_tokens), **numeric_stats(new_tokens, (50, 95))},
        "response_token_sum_ratio_new_over_old": new_token_sum / old_token_sum if old_token_sum else None,
    }


def diagnose_base_cap_extension(config_path: str) -> dict[str, Any]:
    config = read_config(config_path)
    if int(config["generation"]["max_new_tokens"]) != 4096:
        raise ValueError("This diagnostic requires configs/base_eval_v2.yaml with generation.max_new_tokens=4096")
    set_deterministic_seed(int(config["eval"]["deterministic_seed"]))

    pilot_rollouts = _load_pilot_rollouts(config)
    capped = [item for item in pilot_rollouts if int(item["response_token_count"]) == 2048]
    if len(capped) != 101:
        raise ValueError(f"Expected 101 pilot capped samples with response_token_count == 2048, got {len(capped)}")
    capped_ids = [item["problem_id"] for item in capped]
    capped_old_by_id = {item["problem_id"]: item for item in capped}

    dev_records = {record["problem_id"]: record for record in load_jsonl(view_path(config, "dev"))}
    missing = [problem_id for problem_id in capped_ids if problem_id not in dev_records]
    if missing:
        raise ValueError(f"Capped pilot IDs missing from final Dev view: {missing[:5]}")

    output_dir = artifact_root(config) / "length_policy"
    rollouts_path = output_dir / "base_dev_capped_2048_to_4096_rollouts.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_rollouts = load_jsonl(rollouts_path) if rollouts_path.exists() else []
    generation_config = generation_config_for_hash(config)
    generation_config_hash = stable_hash(generation_config)
    completed_by_id: dict[str, dict[str, Any]] = {}
    for rollout in existing_rollouts:
        problem_id = str(rollout["problem_id"])
        if problem_id not in capped_old_by_id:
            raise ValueError(f"Existing diagnostic rollout is not in capped pilot subset: {problem_id}")
        if rollout.get("generation_config_hash") != generation_config_hash:
            raise ValueError(f"Existing diagnostic rollout uses a different generation config: {problem_id}")
        completed_by_id[problem_id] = rollout

    model, tokenizer = _load_model_and_tokenizer(config)
    new_rollouts: list[dict[str, Any]] = [completed_by_id[problem_id] for problem_id in capped_ids if problem_id in completed_by_id]

    for index, problem_id in enumerate(capped_ids, start=1):
        if problem_id in completed_by_id:
            print(f"4096 diagnostic already had {index}/{len(capped_ids)} capped Dev records", flush=True)
            continue
        record = dev_records[problem_id]
        prompt = serialize_prompt(config, str(record["prompt"]))
        generated = _generate(config, model, tokenizer, prompt)
        extracted = extract_python_code(generated["response"])
        verifier = _verify_generated_code(config, extracted.code, record)
        rollout = {
            "problem_id": problem_id,
            "diagnostic": "base_dev_2048_capped_to_4096",
            "old_generation_config_hash": capped_old_by_id[problem_id]["generation_config_hash"],
            "old_verifier_status": capped_old_by_id[problem_id]["verifier_status"],
            "old_response_token_count": capped_old_by_id[problem_id]["response_token_count"],
            "old_generation_latency_ms": capped_old_by_id[problem_id]["generation_latency_ms"],
            "model_repo": config["model"]["repo_id"],
            "model_revision": config["model"]["model_revision"],
            "tokenizer_revision": config["model"]["tokenizer_revision"],
            "prompt_serialization_version": config["prompt"]["serialization_version"],
            "generation_config": generation_config,
            "generation_config_hash": generation_config_hash,
            "raw_response": generated["response"],
            "extracted_code": extracted.code,
            "extraction_strategy": extracted.strategy,
            "extraction_success": bool(extracted.code.strip()),
            "verifier_status": verifier["status"],
            "testcase_passed": verifier["passed"],
            "testcase_total": verifier["total"],
            "testcase_pass_rate": verifier["pass_rate"],
            "reward_status": verifier["reward"]["status"],
            "reward_passed": verifier["reward"]["passed"],
            "reward_total": verifier["reward"]["total"],
            "reward_pass_rate": verifier["reward"]["pass_rate"],
            "heldout_status": verifier["heldout"]["status"],
            "heldout_passed": verifier["heldout"]["passed"],
            "heldout_total": verifier["heldout"]["total"],
            "heldout_pass_rate": verifier["heldout"]["pass_rate"],
            "prompt_token_count": generated["prompt_tokens"],
            "response_token_count": generated["response_tokens"],
            "generation_latency_ms": generated["generation_latency_ms"],
            "hit_max_new_tokens": generated["hit_max_new_tokens"],
        }
        new_rollouts.append(rollout)
        with rollouts_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(rollout, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
        print(f"4096 diagnostic evaluated {index}/{len(capped_ids)} capped Dev records", flush=True)

    new_rollouts = [completed_by_id.get(problem_id) for problem_id in capped_ids if problem_id in completed_by_id]
    if len(new_rollouts) != len(capped_ids):
        new_rollouts = load_jsonl(rollouts_path)
    if len(new_rollouts) != len(capped_ids):
        raise RuntimeError(f"Diagnostic incomplete: expected {len(capped_ids)} rollouts, got {len(new_rollouts)}")
    write_jsonl(rollouts_path, new_rollouts)

    remaining_cap_count = sum(1 for item in new_rollouts if int(item["response_token_count"]) == 4096)
    full_dev_count = int(config["eval"]["expected_count"])
    remaining_cap_threshold = int(full_dev_count * 0.05)
    decision = "choose_4096" if remaining_cap_count <= remaining_cap_threshold else "stop_no_auto_increase"
    old_ac = sum(1 for item in capped if item["verifier_status"] == "AC")
    new_ac = sum(1 for item in new_rollouts if item["verifier_status"] == "AC")
    added_ac = sum(
        1
        for item in new_rollouts
        if item["verifier_status"] == "AC" and capped_old_by_id[item["problem_id"]]["verifier_status"] != "AC"
    )

    report = {
        "phase": "phase2_length_policy",
        "step": "base_eval_2048_to_4096_cap_diagnostic",
        "status": "completed",
        "decision": decision,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": git_command(["rev-parse", "HEAD"]),
            "dirty": bool(git_status_short()),
            "status_short": git_status_short(),
        },
        "config_file": config_path,
        "config_hash": config_hash(config_path),
        "pilot": {
            "config": config["pilot_baseline"]["config"],
            "max_new_tokens": 2048,
            "rollouts_path": config["pilot_baseline"]["rollouts_path"],
            "rollouts_hash": file_sha256(config["pilot_baseline"]["rollouts_path"]),
            "capped_count": len(capped),
        },
        "diagnostic": {
            "max_new_tokens": 4096,
            "rollouts_path": str(rollouts_path),
            "rollouts_hash": file_sha256(rollouts_path),
            "count": len(new_rollouts),
            "remaining_cap_count": remaining_cap_count,
            "remaining_cap_rate_subset": remaining_cap_count / len(new_rollouts) if new_rollouts else 0.0,
            "remaining_cap_rate_full_dev": remaining_cap_count / full_dev_count,
            "remaining_cap_threshold_count": remaining_cap_threshold,
            "transition_matrix": _transition_matrix(capped_old_by_id, new_rollouts),
            "old_ac_count": old_ac,
            "new_ac_count": new_ac,
            "added_ac_count": added_ac,
            "status_counts": dict(sorted(Counter(item["verifier_status"] for item in new_rollouts).items())),
            "response_tokens": numeric_stats([int(item["response_token_count"]) for item in new_rollouts], (50, 90, 95, 99)),
            "generation_cost": _generation_cost(capped_old_by_id, new_rollouts),
        },
        "metrics": _summarize_rollouts(new_rollouts),
        "runtime": runtime_info(),
        "package_versions": package_versions(),
    }
    report_path = artifact_root(config) / "reports" / "phase2_base_2048_to_4096_cap_diagnostic.json"
    write_json(report_path, report)
    return report


def main() -> None:
    args = parse_args()
    report = diagnose_base_cap_extension(args.config)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
