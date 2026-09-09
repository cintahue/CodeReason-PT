from __future__ import annotations

import argparse
import json
import math
import time
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
from verifier.extract_code import extract_python_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the Phase 3 Reasoning-SFT LoRA adapter on Dev.")
    parser.add_argument("--config", default="configs/sft.yaml")
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
    if sft_config.get("phase") != "phase3_reasoning_sft":
        raise ValueError(f"Unexpected SFT config phase: {sft_config.get('phase')}")
    if not base_config.get("frozen"):
        raise ValueError("Base evaluation config must be frozen")
    if int(base_config["generation"]["max_new_tokens"]) != 4096:
        raise ValueError("SFT evaluation must use the frozen 4096 generation budget")
    if int(sft_config["expected"]["eval_max_new_tokens"]) != 4096:
        raise ValueError("SFT config expected eval budget must be 4096")
    if str(sft_config["prompt"]["serialization_version"]) != str(base_config["prompt"]["serialization_version"]):
        raise ValueError("SFT and Base prompt serialization versions differ")
    if str(sft_config["prompt"]["template"]) != str(base_config["prompt"]["template"]):
        raise ValueError("SFT and Base prompt templates differ")
    for key in ("repo_id", "model_revision", "tokenizer_revision"):
        if str(sft_config["model"][key]) != str(base_config["model"][key]):
            raise ValueError(f"SFT and Base model {key} differ")


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
    token_profile = _read_json(token_profile_path)
    if token_profile.get("status") != "completed":
        raise ValueError(f"Token profile is not completed: {token_profile_path}")
    return split, records


def _load_sft_model(base_config: dict[str, Any], sft_config: dict[str, Any]):
    from peft import PeftModel

    checkpoint = Path(str(sft_config["training"]["output_dir"])).expanduser()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing SFT checkpoint: {checkpoint}")
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        if not (checkpoint / filename).exists():
            raise FileNotFoundError(f"Missing SFT checkpoint file: {checkpoint / filename}")
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
    generation_config: dict[str, Any],
    generation_config_hash: str,
) -> dict[str, Any]:
    prompt = serialize_prompt(base_config, str(record["prompt"]))
    generated = _generate(base_config, model, tokenizer, prompt)
    extracted = extract_python_code(generated["response"])
    verifier = _verify_generated_code(base_config, extracted.code, record)
    return {
        "problem_id": record["problem_id"],
        "model_role": "sft",
        "base_model_repo": base_config["model"]["repo_id"],
        "base_model_revision": base_config["model"]["model_revision"],
        "sft_checkpoint": str(checkpoint),
        "sft_checkpoint_hash": file_sha256(checkpoint / "adapter_model.safetensors"),
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
    }


def _comparison(base_report: dict[str, Any], sft_metrics: dict[str, Any]) -> dict[str, Any]:
    base_metrics = base_report["metrics"]
    metric_names = (
        "pass_at_1",
        "executable_rate",
        "mean_testcase_pass_rate",
        "reward_test_mean_pass_rate",
        "heldout_test_mean_pass_rate",
        "generation_cap_hit_rate",
    )
    deltas = {
        name: float(sft_metrics[name]) - float(base_metrics[name])
        for name in metric_names
        if name in base_metrics and name in sft_metrics
    }
    return {
        "base_report_path": base_report.get("report_path"),
        "base_report_hash": base_report.get("report_path") and file_sha256(base_report["report_path"]),
        "base_metrics": base_metrics,
        "sft_metrics": sft_metrics,
        "delta_sft_minus_base": deltas,
        "response_token_average_delta": (
            float(sft_metrics["response_tokens"]["average"])
            - float(base_metrics["response_tokens"]["average"])
        ),
    }


def evaluate_sft(config_path: str, base_config_path: str | None, mode: str) -> dict[str, Any]:
    sft_config = load_config(config_path)
    base_path = Path(base_config_path or sft_config["paths"]["base_eval_config_path"])
    base_config = read_config(base_path)
    _validate_configs(sft_config, base_config)
    set_deterministic_seed(int(base_config["eval"]["deterministic_seed"]))

    checkpoint = Path(str(sft_config["training"]["output_dir"])).expanduser()
    train_report_path = artifact_root(sft_config) / "train" / "sft_train_report.json"
    if not train_report_path.exists():
        raise FileNotFoundError(f"Missing completed SFT train report: {train_report_path}")
    train_report = _read_json(train_report_path)
    if train_report.get("status") != "completed" or train_report.get("mode") != "train":
        raise ValueError(f"SFT training report is not a completed formal run: {train_report_path}")
    train_loss = train_report.get("train_metrics", {}).get("train_loss")
    if not _finite(train_loss):
        raise ValueError(f"SFT train loss is not finite: {train_loss}")

    base_report_path = _base_output_path(base_config, "dev_report")
    if not base_report_path.exists():
        raise FileNotFoundError(f"Missing final Base Dev report: {base_report_path}")
    base_report = _read_json(base_report_path)
    expected = int(base_config["eval"]["expected_count"])
    if base_report.get("status") != "completed" or int(base_report.get("evaluated", -1)) != expected:
        raise ValueError(f"Final Base Dev report is incomplete: {base_report_path}")
    if int(base_report.get("generation_config", {}).get("max_new_tokens", -1)) != 4096:
        raise ValueError("Base report is not the frozen 4096-policy baseline")

    split, records = _load_records(base_config, mode)
    model, tokenizer, checkpoint = _load_sft_model(base_config, sft_config)
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
                generation_config=generation_config,
                generation_config_hash=generation_config_hash,
            )
        )
        print(f"Evaluated {index}/{len(records)} SFT {mode} records", flush=True)
    elapsed_ms = int(round((time.perf_counter() - started) * 1000))

    metrics = _summarize_rollouts(rollouts)
    output_key = "smoke_report" if mode == "smoke" else "dev_report"
    rollout_key = "smoke_rollouts" if mode == "smoke" else "dev_rollouts"
    report_path = _configured_output_path(sft_config, output_key)
    rollouts_path = _configured_output_path(sft_config, rollout_key)
    write_jsonl(rollouts_path, rollouts)
    report = {
        "phase": "phase3_reasoning_sft",
        "step": "sft_dev_evaluation",
        "mode": mode,
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "evaluated": len(rollouts),
        "model_role": "sft",
        "base_model": {
            "repo_id": base_config["model"]["repo_id"],
            "model_revision": base_config["model"]["model_revision"],
            "tokenizer_revision": base_config["model"]["tokenizer_revision"],
        },
        "sft_checkpoint": {
            "path": str(checkpoint),
            "adapter_hash": file_sha256(checkpoint / "adapter_model.safetensors"),
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
        "comparison": None,
        "runtime": runtime_info(),
        "config": {
            "sft_config_path": config_path,
            "sft_config_hash": config_hash(config_path),
            "base_config_path": str(base_path),
            "base_config_hash": config_hash(base_path),
        },
        "evaluation_elapsed_ms": elapsed_ms,
        "report_path": str(report_path),
    }
    if mode == "dev":
        report["comparison"] = _comparison(base_report, metrics)
        report["phase3_gate"] = {
            "formal_sft_training_completed": True,
            "train_loss_finite": True,
            "checkpoint_artifact_present": True,
            "checkpoint_adapter_hash_recorded": True,
            "all_dev_problems_evaluated": len(rollouts) == expected,
            "generation_policy_matches_frozen_base": generation_config_hash
            == str(base_report.get("generation_config_hash")),
            "code_extraction_success_rate": metrics["code_extraction_success_rate"],
            "response_format_observation": "stable" if metrics["code_extraction_success_rate"] == 1.0 else "needs_review",
            "behavioral_gate": "manual_review_required",
            "dpo_grpo_ready": False,
        }
    write_json(report_path, report)
    return report


def main() -> None:
    args = parse_args()
    report = evaluate_sft(args.config, args.base_config, args.mode)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
