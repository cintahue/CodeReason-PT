from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data.config import load_config
from data.schemas import stable_hash
from eval.phase2_common import (
    artifact_root,
    config_hash,
    file_sha256,
    generation_config_for_hash,
    git_command,
    git_status_short,
    load_jsonl,
    read_config,
    runtime_info,
    serialize_prompt,
    set_deterministic_seed,
    stable_problem_sample,
    view_path,
    write_json,
    write_jsonl,
)
from eval.run_base_baseline import (
    _generate,
    _load_model_and_tokenizer,
    _summarize_rollouts,
    _verify_generated_code,
)
from eval.sft_v2_sequence import format_response_diagnostics
from verifier.extract_code import extract_python_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the Phase 3-v2 SFT adapter on frozen Dev.")
    parser.add_argument("--config", default="configs/sft_v2.yaml")
    parser.add_argument("--base-config", default=None)
    parser.add_argument("--mode", choices=("smoke", "dev"), default="dev")
    return parser.parse_args()


def _configured_output_path(config: dict[str, Any], key: str) -> Path:
    value = config.get("outputs", {}).get(key)
    if not value:
        raise KeyError(f"Missing outputs.{key} in config")
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else artifact_root(config) / path


def _base_output_path(config: dict[str, Any], key: str) -> Path:
    value = config.get("outputs", {}).get(key)
    if not value:
        raise KeyError(f"Missing base config outputs.{key}")
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else artifact_root(config) / path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _validate_configs(sft_config: dict[str, Any], base_config: dict[str, Any]) -> None:
    if sft_config.get("phase") != "phase3_reasoning_sft_v2":
        raise ValueError(f"Unexpected SFT-v2 config phase: {sft_config.get('phase')}")
    if not base_config.get("frozen"):
        raise ValueError("Base evaluation config must be frozen")
    if int(base_config["generation"]["max_new_tokens"]) != 4096:
        raise ValueError("SFT-v2 evaluation must use frozen max_new_tokens=4096")
    if int(sft_config["expected"]["eval_max_new_tokens"]) != 4096:
        raise ValueError("SFT-v2 expected eval budget must be 4096")
    if int(sft_config["sft_sequence"]["max_sequence_length"]) != 8192:
        raise ValueError("SFT-v2 training sequence budget must be 8192")
    if str(sft_config["prompt"]["serialization_version"]) != str(base_config["prompt"]["serialization_version"]):
        raise ValueError("SFT-v2 and Base prompt serialization versions differ")
    if str(sft_config["prompt"]["template"]) != str(base_config["prompt"]["template"]):
        raise ValueError("SFT-v2 and Base prompt templates differ")
    for key in ("repo_id", "model_revision", "tokenizer_revision"):
        if str(sft_config["model"][key]) != str(base_config["model"][key]):
            raise ValueError(f"SFT-v2 and Base model {key} differ")


def _load_records(base_config: dict[str, Any], mode: str) -> tuple[str, list[dict[str, Any]]]:
    split = str(base_config["eval"]["split"] if mode == "dev" else base_config["smoke"]["split"])
    records = load_jsonl(view_path(base_config, split))
    if mode == "smoke":
        return split, stable_problem_sample(
            records,
            count=int(base_config["smoke"]["sample_count"]),
            seed=int(base_config["smoke"]["sample_seed"]),
        )
    expected = int(base_config["eval"]["expected_count"])
    if len(records) != expected:
        raise ValueError(f"Final {split} view count mismatch: expected {expected}, got {len(records)}")
    token_profile_path = Path(str(base_config["paths"]["artifact_root"])) / "reports" / "phase2_token_profile.json"
    if not token_profile_path.exists():
        raise FileNotFoundError(f"Run token profiling first: {token_profile_path}")
    if _read_json(token_profile_path).get("status") != "completed":
        raise ValueError(f"Token profile is not completed: {token_profile_path}")
    return split, records


def _load_sft_model(base_config: dict[str, Any], sft_config: dict[str, Any]):
    from peft import PeftModel

    checkpoint = Path(str(sft_config["training"]["output_dir"])).expanduser()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing SFT-v2 checkpoint: {checkpoint}")
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        if not (checkpoint / filename).exists():
            raise FileNotFoundError(f"Missing SFT-v2 checkpoint file: {checkpoint / filename}")
    model, tokenizer = _load_model_and_tokenizer(base_config)
    model = PeftModel.from_pretrained(model, str(checkpoint), is_trainable=False)
    model.eval()
    return model, tokenizer, checkpoint


def _rollout(
    base_config: dict[str, Any],
    model: Any,
    tokenizer: Any,
    record: dict[str, Any],
    *,
    checkpoint: Path,
    checkpoint_hash: str,
    generation_config: dict[str, Any],
    generation_config_hash: str,
) -> dict[str, Any]:
    prompt = serialize_prompt(base_config, str(record["prompt"]))
    generated = _generate(base_config, model, tokenizer, prompt)
    extracted = extract_python_code(generated["response"])
    verifier = _verify_generated_code(base_config, extracted.code, record)
    format_info = format_response_diagnostics(generated["response"], bool(extracted.code.strip()))
    return {
        "problem_id": record["problem_id"],
        "model_role": "sft_v2",
        "base_model_repo": base_config["model"]["repo_id"],
        "base_model_revision": base_config["model"]["model_revision"],
        "sft_checkpoint": str(checkpoint),
        "sft_checkpoint_hash": checkpoint_hash,
        "prompt_serialization_version": base_config["prompt"]["serialization_version"],
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
        "format_diagnostics": format_info,
    }


def _summarize_v2(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = _summarize_rollouts(rollouts)
    total = len(rollouts)
    format_records = [record["format_diagnostics"] for record in rollouts]
    starts = sum(1 for item in format_records if item["starts_with_think"])
    closed = sum(1 for item in format_records if item["closed_think"])
    unclosed = sum(1 for item in format_records if item["unclosed_think"])
    fenced = sum(1 for item in format_records if item["fenced_code"])
    transitions = sum(1 for item in format_records if item["valid_reasoning_to_code_transition"])
    valid = sum(1 for item in format_records if item["format_valid"])
    format_classes = Counter(item["format_class"] for item in format_records)
    metrics.update(
        {
            "generation_cap_hit_count": metrics["hit_max_new_tokens_count"],
            "starts_with_think_count": starts,
            "starts_with_think_rate": starts / total if total else 0.0,
            "closed_think_count": closed,
            "closed_think_rate": closed / total if total else 0.0,
            "unclosed_think_count": unclosed,
            "unclosed_think_rate": unclosed / total if total else 0.0,
            "fenced_code_count": fenced,
            "fenced_code_rate": fenced / total if total else 0.0,
            "valid_reasoning_to_code_transition_count": transitions,
            "valid_reasoning_to_code_transition_rate": transitions / total if total else 0.0,
            "format_valid_count": valid,
            "format_valid_rate": valid / total if total else 0.0,
            "format_class_counts": dict(sorted(format_classes.items())),
            "G_V": metrics["reward_test_mean_pass_rate"] - metrics["heldout_test_mean_pass_rate"],
        }
    )
    return metrics


def _comparison(base_report: dict[str, Any], v1_report: dict[str, Any], v2_metrics: dict[str, Any]) -> dict[str, Any]:
    base_metrics = base_report["metrics"]
    v1_metrics = v1_report["metrics"]
    names = (
        "pass_at_1",
        "executable_rate",
        "mean_testcase_pass_rate",
        "reward_test_mean_pass_rate",
        "heldout_test_mean_pass_rate",
        "generation_cap_hit_rate",
        "G_V",
    )

    def normalized(source: dict[str, Any]) -> dict[str, float]:
        values = dict(source)
        values["G_V"] = float(values.get("reward_test_mean_pass_rate", 0.0)) - float(
            values.get("heldout_test_mean_pass_rate", 0.0)
        )
        return {name: float(values[name]) for name in names if name in values}

    base = normalized(base_metrics)
    v1 = normalized(v1_metrics)
    v2 = normalized(v2_metrics)
    return {
        "base_report_path": base_report.get("report_path"),
        "base_report_hash": file_sha256(base_report["report_path"]),
        "v1_report_path": v1_report.get("report_path"),
        "v1_report_hash": file_sha256(v1_report["report_path"]),
        "metrics": {"base": base, "sft_v1": v1, "sft_v2": v2},
        "delta_sft_v2_minus_base": {name: v2[name] - base[name] for name in v2.keys() & base.keys()},
        "delta_sft_v2_minus_sft_v1": {name: v2[name] - v1[name] for name in v2.keys() & v1.keys()},
    }


def evaluate_sft_v2(config_path: str, base_config_path: str | None, mode: str) -> dict[str, Any]:
    sft_config = load_config(config_path)
    base_path = Path(base_config_path or sft_config["paths"]["base_eval_config_path"])
    base_config = read_config(base_path)
    _validate_configs(sft_config, base_config)
    set_deterministic_seed(int(base_config["eval"]["deterministic_seed"]))

    artifact_root_v2 = Path(str(sft_config["paths"]["artifact_root"])).expanduser()
    train_report_path = artifact_root_v2 / "train" / "sft_v2_train_report.json"
    if not train_report_path.exists():
        raise FileNotFoundError(f"Missing completed v2 train report: {train_report_path}")
    train_report = _read_json(train_report_path)
    if train_report.get("status") != "completed" or train_report.get("mode") != "train":
        raise ValueError(f"v2 training report is not a completed formal run: {train_report_path}")
    if not _finite(train_report.get("train_metrics", {}).get("train_loss")):
        raise ValueError("v2 train loss is not finite")

    base_report_path = _base_output_path(base_config, "dev_report")
    if not base_report_path.exists():
        raise FileNotFoundError(f"Missing frozen Base Dev report: {base_report_path}")
    base_report = _read_json(base_report_path)
    expected = int(base_config["eval"]["expected_count"])
    if base_report.get("status") != "completed" or int(base_report.get("evaluated", -1)) != expected:
        raise ValueError(f"Frozen Base Dev report is incomplete: {base_report_path}")
    if int(base_report.get("generation_config", {}).get("max_new_tokens", -1)) != 4096:
        raise ValueError("Base report is not the frozen 4096-policy baseline")

    v1_report_path = Path(str(sft_config["paths"]["v1_dev_report_path"])).expanduser()
    if not v1_report_path.exists():
        raise FileNotFoundError(f"Missing preserved v1 Dev report: {v1_report_path}")
    v1_report = _read_json(v1_report_path)
    if v1_report.get("status") != "completed" or int(v1_report.get("evaluated", -1)) != expected:
        raise ValueError("Preserved v1 Dev report is incomplete")

    split, records = _load_records(base_config, mode)
    model, tokenizer, checkpoint = _load_sft_model(base_config, sft_config)
    checkpoint_hash = file_sha256(checkpoint / "adapter_model.safetensors")
    generation_config = generation_config_for_hash(base_config)
    generation_config_hash = stable_hash(generation_config)
    rollouts: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, record in enumerate(records, start=1):
        rollouts.append(
            _rollout(
                base_config,
                model,
                tokenizer,
                record,
                checkpoint=checkpoint,
                checkpoint_hash=checkpoint_hash,
                generation_config=generation_config,
                generation_config_hash=generation_config_hash,
            )
        )
        print(f"Evaluated {index}/{len(records)} SFT-v2 {mode} records", flush=True)
    elapsed_ms = int(round((time.perf_counter() - started) * 1000))

    metrics = _summarize_v2(rollouts)
    output_key = "smoke_report" if mode == "smoke" else "dev_report"
    rollout_key = "smoke_rollouts" if mode == "smoke" else "dev_rollouts"
    report_path = _configured_output_path(sft_config, output_key)
    rollouts_path = _configured_output_path(sft_config, rollout_key)
    if report_path.exists() or rollouts_path.exists():
        raise FileExistsError("Refusing to overwrite an existing Phase 3-v2 evaluation artifact")
    write_jsonl(rollouts_path, rollouts)
    report = {
        "phase": "phase3_reasoning_sft_v2",
        "step": "sft_v2_dev_evaluation",
        "mode": mode,
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "evaluated": len(rollouts),
        "model_role": "sft_v2",
        "base_model": {
            "repo_id": base_config["model"]["repo_id"],
            "model_revision": base_config["model"]["model_revision"],
            "tokenizer_revision": base_config["model"]["tokenizer_revision"],
        },
        "sft_checkpoint": {
            "path": str(checkpoint),
            "adapter_hash": checkpoint_hash,
            "train_report_path": str(train_report_path),
            "train_report_hash": file_sha256(train_report_path),
        },
        "prompt": {
            "serialization_version": base_config["prompt"]["serialization_version"],
            "template": base_config["prompt"]["template"],
            "use_chat_template": base_config["prompt"]["use_chat_template"],
        },
        "generation_config": generation_config,
        "generation_config_hash": generation_config_hash,
        "docker_image": base_config["verifier"]["docker_image"],
        "rollouts_path": str(rollouts_path),
        "rollouts_hash": file_sha256(rollouts_path),
        "metrics": metrics,
        "format_diagnostics": {
            "starts_with_think_count": metrics["starts_with_think_count"],
            "starts_with_think_rate": metrics["starts_with_think_rate"],
            "closed_think_count": metrics["closed_think_count"],
            "closed_think_rate": metrics["closed_think_rate"],
            "unclosed_think_count": metrics["unclosed_think_count"],
            "unclosed_think_rate": metrics["unclosed_think_rate"],
            "fenced_code_count": metrics["fenced_code_count"],
            "fenced_code_rate": metrics["fenced_code_rate"],
            "valid_reasoning_to_code_transition_count": metrics[
                "valid_reasoning_to_code_transition_count"
            ],
            "valid_reasoning_to_code_transition_rate": metrics[
                "valid_reasoning_to_code_transition_rate"
            ],
            "generation_cap_hit_count": metrics["generation_cap_hit_count"],
            "generation_cap_hit_rate": metrics["generation_cap_hit_rate"],
            "format_valid_count": metrics["format_valid_count"],
            "format_valid_rate": metrics["format_valid_rate"],
            "extraction_strategy_counts": metrics["extraction_strategy_counts"],
        },
        "comparison": None,
        "runtime": runtime_info(),
        "config": {
            "sft_config_path": str(Path(config_path).resolve()),
            "sft_config_hash": config_hash(config_path),
            "base_config_path": str(base_path),
            "base_config_hash": config_hash(base_path),
            "v1_report_path": str(v1_report_path),
            "v1_report_hash": file_sha256(v1_report_path),
        },
        "evaluation_elapsed_ms": elapsed_ms,
        "report_path": str(report_path),
    }
    if mode == "dev":
        report["comparison"] = _comparison(base_report, v1_report, metrics)
        report["phase3_v2_gate"] = {
            "formal_sft_training_completed": True,
            "train_loss_finite": True,
            "checkpoint_artifact_present": True,
            "checkpoint_adapter_hash_recorded": True,
            "all_dev_problems_evaluated": len(rollouts) == expected,
            "generation_policy_matches_frozen_base": generation_config_hash
            == str(base_report.get("generation_config_hash")),
            "code_extraction_success_rate": metrics["code_extraction_success_rate"],
            "format_valid_rate": metrics["format_valid_rate"],
            "unclosed_think_count": metrics["unclosed_think_count"],
            "response_format_observation": (
                "stable"
                if metrics["format_valid_rate"] == 1.0 and metrics["unclosed_think_count"] == 0
                else "needs_review"
            ),
            "behavioral_gate": "manual_review_required",
            "dpo_grpo_ready": False,
        }
    write_json(report_path, report)
    return report


def main() -> None:
    args = parse_args()
    report = evaluate_sft_v2(args.config, args.base_config, args.mode)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
