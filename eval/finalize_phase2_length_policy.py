from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data.schemas import stable_hash
from eval.phase2_common import (
    config_hash,
    file_sha256,
    generation_config_for_hash,
    git_command,
    git_status_short,
    read_config,
    reports_dir,
    runtime_info,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize Phase 2 length policy audit from existing artifacts.")
    parser.add_argument("--config", default="configs/base_eval_v2.yaml")
    return parser.parse_args()


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _existing_report(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _read_json(path)


def finalize_length_policy(config_path: str) -> dict[str, Any]:
    config = read_config(config_path)
    sft_report_path = reports_dir(config) / "phase2_sft_8192_length_diagnostic.json"
    cap_report_path = reports_dir(config) / "phase2_base_2048_to_4096_cap_diagnostic.json"
    official_report_path = Path(str(config["outputs"]["dev_report"]))
    if not official_report_path.is_absolute():
        official_report_path = reports_dir(config).parents[0] / official_report_path
    official_report = _existing_report(official_report_path)
    sft_report = _read_json(sft_report_path)
    cap_report = _read_json(cap_report_path)

    choose_4096 = cap_report.get("decision") == "choose_4096"
    official_complete = bool(
        official_report
        and official_report.get("mode") == "dev"
        and int(official_report.get("evaluated", -1)) == int(config["eval"]["expected_count"])
        and int(official_report.get("generation_config", {}).get("max_new_tokens", -1)) == 4096
    )
    sft_policy_confirmed = bool(
        sft_report.get("status") == "completed"
        and int(sft_report["counts"]["prompt_plus_full_code_plus_separator_eos_over_8192"]) == 0
    )
    final_eval_max_new_tokens = 4096 if choose_4096 and official_complete else None
    final_sft_max_sequence_length = int(config["sft_sequence"]["max_sequence_length"]) if sft_policy_confirmed else None

    serialization = {
        "prompt_serialization_version": config["prompt"]["serialization_version"],
        "prompt_template": config["prompt"]["template"],
        "sft_sequence_serialization_version": config["sft_sequence"]["serialization_version"],
        "sft_truncation_policy": config["sft_sequence"]["truncation_policy"],
        "eval_config_version": config["eval_config_version"],
        "generation_config": generation_config_for_hash(config),
    }
    status = "completed" if final_eval_max_new_tokens == 4096 and final_sft_max_sequence_length == 8192 else "incomplete"
    audit = {
        "phase": "phase2_length_policy",
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": git_command(["rev-parse", "HEAD"]),
            "dirty": bool(git_status_short()),
            "status_short": git_status_short(),
        },
        "phase1_final_eligibility_manifest_hash": config["expected"]["phase1_final_manifest_hash"],
        "config": {"path": config_path, "hash": config_hash(config_path)},
        "pilot_2048_baseline": {
            "role": "pilot_baseline",
            "config": config["pilot_baseline"]["config"],
            "metrics_path": config["pilot_baseline"]["metrics_path"],
            "rollouts_path": config["pilot_baseline"]["rollouts_path"],
            "metrics_hash": file_sha256(config["pilot_baseline"]["metrics_path"]),
            "rollouts_hash": file_sha256(config["pilot_baseline"]["rollouts_path"]),
        },
        "sft_8192_code_preserving_diagnostic": {
            "report_path": str(sft_report_path),
            "report_hash": file_sha256(sft_report_path),
            "status": sft_report["status"],
            "counts": sft_report["counts"],
            "original_reasoning_tokens": sft_report["original_reasoning_tokens"],
            "retained_reasoning_tokens": sft_report["retained_reasoning_tokens"],
            "retained_original_reasoning_ratio": sft_report["retained_original_reasoning_ratio"],
            "final_sequence_tokens": sft_report["final_sequence_tokens"],
            "final_sequence_token_max": sft_report["final_sequence_token_max"],
            "diagnostic_manifest": sft_report["diagnostic_manifest"],
        },
        "base_2048_to_4096_diagnostic": {
            "report_path": str(cap_report_path),
            "report_hash": file_sha256(cap_report_path),
            "decision": cap_report["decision"],
            "pilot": cap_report["pilot"],
            "diagnostic": cap_report["diagnostic"],
        },
        "official_4096_base_baseline": {
            "required": choose_4096,
            "completed": official_complete,
            "metrics_path": str(official_report_path),
            "metrics_hash": file_sha256(official_report_path) if official_report_path.exists() else None,
            "rollouts_path": official_report.get("rollouts_path") if official_report else None,
            "rollouts_hash": official_report.get("rollouts_hash") if official_report else None,
            "metrics": official_report.get("metrics") if official_report else None,
        },
        "final_decision": {
            "final_chosen_eval_max_new_tokens": final_eval_max_new_tokens,
            "final_chosen_sft_max_sequence_length": final_sft_max_sequence_length,
            "sft_length_policy": config["sft_sequence"],
            "eval_config_version": config["eval_config_version"] if final_eval_max_new_tokens == 4096 else None,
            "pilot_2048_marked_non_final": True,
        },
        "serialization_hashes": {
            "prompt_serialization_hash": stable_hash(
                {
                    "version": config["prompt"]["serialization_version"],
                    "use_chat_template": config["prompt"]["use_chat_template"],
                    "template": config["prompt"]["template"],
                }
            ),
            "sft_sequence_serialization_hash": stable_hash(config["sft_sequence"]),
            "eval_generation_config_hash": stable_hash(generation_config_for_hash(config)),
            "combined_length_policy_hash": stable_hash(serialization),
        },
        "runtime": runtime_info(),
    }
    write_json("phase2_length_policy_audit.json", audit)
    if status != "completed":
        raise SystemExit("Phase 2 length policy audit is incomplete; inspect phase2_length_policy_audit.json")
    return audit


def main() -> None:
    args = parse_args()
    audit = finalize_length_policy(args.config)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
