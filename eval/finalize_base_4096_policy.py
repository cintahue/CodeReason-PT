from __future__ import annotations

import argparse
import json
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
    package_versions,
    read_config,
    runtime_info,
    view_path,
    write_json,
    write_jsonl,
)
from eval.run_base_baseline import _audit, _summarize_rollouts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize Base Dev baseline under the frozen 4096 generation policy.")
    parser.add_argument("--config", default="configs/base_eval_v2.yaml")
    return parser.parse_args()


def _configured_output_path(config: dict[str, Any], key: str) -> Path:
    value = config.get("outputs", {}).get(key)
    if not value:
        raise KeyError(f"Missing outputs.{key} in config")
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else artifact_root(config) / path


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _mark_final_policy_rollout(
    rollout: dict[str, Any],
    *,
    generation_config: dict[str, Any],
    generation_config_hash: str,
    source: str,
) -> dict[str, Any]:
    output = dict(rollout)
    output["generation_config"] = generation_config
    output["generation_config_hash"] = generation_config_hash
    output["final_policy_generation_config"] = generation_config
    output["final_policy_generation_config_hash"] = generation_config_hash
    output["final_policy_rollout_source"] = source
    output["hit_max_new_tokens"] = int(output["response_token_count"]) >= int(generation_config["max_new_tokens"])
    return output


def finalize_base_4096_policy(config_path: str) -> dict[str, Any]:
    config = read_config(config_path)
    if int(config["generation"]["max_new_tokens"]) != 4096:
        raise ValueError("Final Base policy requires generation.max_new_tokens=4096")

    pilot_path = Path(str(config["pilot_baseline"]["rollouts_path"]))
    cap_path = artifact_root(config) / "length_policy" / "base_dev_capped_2048_to_4096_rollouts.jsonl"
    cap_report_path = artifact_root(config) / "reports" / "phase2_base_2048_to_4096_cap_diagnostic.json"
    if not pilot_path.exists():
        raise FileNotFoundError(f"Missing pilot rollouts: {pilot_path}")
    if not cap_path.exists():
        raise FileNotFoundError(f"Missing 4096 capped rerun rollouts: {cap_path}")

    pilot_rollouts = load_jsonl(pilot_path)
    capped_reruns = load_jsonl(cap_path)
    expected = int(config["eval"]["expected_count"])
    if len(pilot_rollouts) != expected:
        raise ValueError(f"Pilot rollout count mismatch: expected {expected}, got {len(pilot_rollouts)}")

    generation_config = generation_config_for_hash(config)
    generation_config_hash = stable_hash(generation_config)
    capped_by_id = {str(item["problem_id"]): item for item in capped_reruns}
    pilot_capped_ids = [str(item["problem_id"]) for item in pilot_rollouts if int(item["response_token_count"]) >= 2048]
    missing = [problem_id for problem_id in pilot_capped_ids if problem_id not in capped_by_id]
    if missing:
        raise ValueError(f"Missing 4096 reruns for capped pilot samples: {missing[:5]}")
    if len(capped_by_id) != len(pilot_capped_ids):
        raise ValueError(
            f"4096 rerun count mismatch: expected {len(pilot_capped_ids)}, got {len(capped_by_id)}"
        )

    combined: list[dict[str, Any]] = []
    reused_count = 0
    rerun_count = 0
    for pilot in pilot_rollouts:
        problem_id = str(pilot["problem_id"])
        if int(pilot["response_token_count"]) < 2048:
            combined.append(
                _mark_final_policy_rollout(
                    pilot,
                    generation_config=generation_config,
                    generation_config_hash=generation_config_hash,
                    source="reused_2048_natural_termination",
                )
            )
            reused_count += 1
            continue
        combined.append(
            _mark_final_policy_rollout(
                capped_by_id[problem_id],
                generation_config=generation_config,
                generation_config_hash=generation_config_hash,
                source="rerun_2048_capped_sample_at_4096",
            )
        )
        rerun_count += 1

    dev_ids = [str(item["problem_id"]) for item in load_jsonl(view_path(config, "dev"))]
    combined_ids = [str(item["problem_id"]) for item in combined]
    if combined_ids != dev_ids:
        raise ValueError("Combined rollout order or identity does not match final Dev view")

    rollouts_path = _configured_output_path(config, "dev_rollouts")
    report_path = _configured_output_path(config, "dev_report")
    write_jsonl(rollouts_path, combined)
    cap_report = _read_json(cap_report_path) if cap_report_path.exists() else None
    report = {
        "phase": "phase2_base_baseline",
        "mode": "dev",
        "status": "completed",
        "step": "finalize_4096_policy_baseline_from_existing_rollouts",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "dev",
        "evaluated": len(combined),
        "eval_config_version": config["eval_config_version"],
        "baseline_role": "official_base_baseline",
        "model": {
            "repo_id": config["model"]["repo_id"],
            "model_revision": config["model"]["model_revision"],
            "tokenizer_revision": config["model"]["tokenizer_revision"],
            "dtype": config["model"]["dtype"],
            "placement_policy": config["model"].get("placement_policy"),
        },
        "prompt": {
            "serialization_version": config["prompt"]["serialization_version"],
            "use_chat_template": config["prompt"]["use_chat_template"],
            "template": config["prompt"]["template"],
        },
        "generation_config": generation_config,
        "generation_config_hash": generation_config_hash,
        "docker_image": config["verifier"]["docker_image"],
        "rollouts_path": str(rollouts_path),
        "rollouts_hash": file_sha256(rollouts_path),
        "metrics": _summarize_rollouts(combined),
        "combination_rule": {
            "policy": "reuse_natural_2048_terminations_and_replace_2048_capped_outputs_with_4096_reruns",
            "pilot_2048_rollouts_path": str(pilot_path),
            "pilot_2048_rollouts_hash": file_sha256(pilot_path),
            "capped_4096_rollouts_path": str(cap_path),
            "capped_4096_rollouts_hash": file_sha256(cap_path),
            "reused_natural_termination_count": reused_count,
            "rerun_capped_at_4096_count": rerun_count,
            "old_2048_baseline_role": "pilot_baseline",
            "no_8192_eval_generation": True,
            "cap_diagnostic_decision": cap_report.get("decision") if cap_report else None,
        },
        "git": {
            "commit": git_command(["rev-parse", "HEAD"]),
            "dirty": bool(git_status_short()),
            "status_short": git_status_short(),
        },
        "config_file": config_path,
        "config_hash": config_hash(config_path),
        "runtime": runtime_info(),
        "package_versions": package_versions(),
        "report_path": str(report_path),
    }
    write_json(report_path, report)
    audit = _audit(config_path, config, report, rollouts_path)
    audit["official_4096_combination_rule"] = report["combination_rule"]
    audit["baseline_role"] = "official_base_baseline"
    audit_path = _configured_output_path(config, "base_audit")
    write_json(audit_path, audit)
    return report


def main() -> None:
    args = parse_args()
    report = finalize_base_4096_policy(args.config)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
