from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

from data.config import load_config
from data.schemas import stable_hash
from eval.phase2_common import (
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
    _verify_generated_code,
)
from eval.run_sft_v2_dev import _summarize_v2
from eval.sft_v2_sequence import format_response_diagnostics
from verifier.extract_code import extract_python_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the Phase 3-v3 natural-length SFT adapter on frozen Dev.")
    parser.add_argument("--config", default="configs/sft_v3.yaml")
    parser.add_argument("--base-config", default=None)
    parser.add_argument("--mode", choices=("smoke", "dev"), default="dev")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _configured_path(config: dict[str, Any], key: str) -> Path:
    value = config["outputs"].get(key)
    if not value:
        raise KeyError(f"Missing outputs.{key}")
    return Path(str(value)).expanduser()


def _validate_configs(sft_config: dict[str, Any], base_config: dict[str, Any]) -> None:
    if sft_config.get("phase") != "phase3_reasoning_sft_v3":
        raise ValueError(f"Unexpected v3 config phase: {sft_config.get('phase')}")
    if not base_config.get("frozen"):
        raise ValueError("Base evaluation config must be frozen")
    if int(base_config["generation"]["max_new_tokens"]) != 4096:
        raise ValueError("v3 evaluation must use frozen max_new_tokens=4096")
    if int(sft_config["expected"]["eval_max_new_tokens"]) != 4096:
        raise ValueError("v3 expected eval budget must be 4096")
    if int(sft_config["sft_sequence"]["max_sequence_length"]) != 8192:
        raise ValueError("v3 max sequence length must be 8192")
    if sft_config["sft_sequence"].get("truncation_policy") != "none":
        raise ValueError("v3 evaluator cannot use a truncation policy")
    if str(sft_config["prompt"]["serialization_version"]) != str(base_config["prompt"]["serialization_version"]):
        raise ValueError("v3 and Base prompt serialization versions differ")
    if str(sft_config["prompt"]["template"]) != str(base_config["prompt"]["template"]):
        raise ValueError("v3 and Base prompt templates differ")
    for key in ("repo_id", "model_revision", "tokenizer_revision"):
        if str(sft_config["model"][key]) != str(base_config["model"][key]):
            raise ValueError(f"v3 and Base model {key} differ")


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
    token_profile = Path(str(base_config["paths"]["artifact_root"])) / "reports" / "phase2_token_profile.json"
    if not token_profile.exists() or _read_json(token_profile).get("status") != "completed":
        raise FileNotFoundError(f"Completed token profile required: {token_profile}")
    return split, records


def _load_sft_model(base_config: dict[str, Any], sft_config: dict[str, Any]):
    from peft import PeftModel

    checkpoint = Path(str(sft_config["training"]["output_dir"])).expanduser()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing v3 checkpoint: {checkpoint}")
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        if not (checkpoint / filename).exists():
            raise FileNotFoundError(f"Missing v3 checkpoint file: {checkpoint / filename}")
    model, tokenizer = _load_model_and_tokenizer(base_config)
    model = PeftModel.from_pretrained(model, str(checkpoint), is_trainable=False)
    model.eval()
    return model, tokenizer, checkpoint


def _rollout(
    base_config: dict[str, Any],
    model: Any,
    tokenizer: Any,
    record: dict[str, Any],
    checkpoint: Path,
    checkpoint_hash: str,
    generation_config: dict[str, Any],
    generation_config_hash: str,
) -> dict[str, Any]:
    prompt = serialize_prompt(base_config, str(record["prompt"]))
    generated = _generate(base_config, model, tokenizer, prompt)
    extracted = extract_python_code(generated["response"])
    verifier = _verify_generated_code(base_config, extracted.code, record)
    return {
        "problem_id": record["problem_id"],
        "model_role": "sft_v3",
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
        "format_diagnostics": format_response_diagnostics(
            generated["response"], bool(extracted.code.strip())
        ),
    }


def _comparison(
    base_report: dict[str, Any],
    v1_report: dict[str, Any],
    v2_report: dict[str, Any],
    v3_metrics: dict[str, Any],
) -> dict[str, Any]:
    names = (
        "pass_at_1",
        "executable_rate",
        "mean_testcase_pass_rate",
        "reward_test_mean_pass_rate",
        "heldout_test_mean_pass_rate",
        "generation_cap_hit_rate",
        "G_V",
    )

    def normalized(metrics: dict[str, Any]) -> dict[str, float]:
        values = dict(metrics)
        values["G_V"] = float(values.get("reward_test_mean_pass_rate", 0.0)) - float(
            values.get("heldout_test_mean_pass_rate", 0.0)
        )
        return {name: float(values[name]) for name in names if name in values}

    tables = {
        "base": normalized(base_report["metrics"]),
        "sft_v1": normalized(v1_report["metrics"]),
        "sft_v2": normalized(v2_report["metrics"]),
        "sft_v3": normalized(v3_metrics),
    }
    v3 = tables["sft_v3"]
    return {
        "base_report_path": base_report.get("report_path"),
        "base_report_hash": file_sha256(base_report["report_path"]),
        "v1_report_path": v1_report.get("report_path"),
        "v1_report_hash": file_sha256(v1_report["report_path"]),
        "v2_report_path": v2_report.get("report_path"),
        "v2_report_hash": file_sha256(v2_report["report_path"]),
        "metrics": tables,
        "delta_sft_v3_minus_base": {key: v3[key] - tables["base"][key] for key in v3.keys() & tables["base"].keys()},
        "delta_sft_v3_minus_sft_v1": {key: v3[key] - tables["sft_v1"][key] for key in v3.keys() & tables["sft_v1"].keys()},
        "delta_sft_v3_minus_sft_v2": {key: v3[key] - tables["sft_v2"][key] for key in v3.keys() & tables["sft_v2"].keys()},
    }


def evaluate_sft_v3(config_path: str, base_config_path: str | None, mode: str) -> dict[str, Any]:
    sft_config = load_config(config_path)
    base_path = Path(base_config_path or sft_config["paths"]["base_eval_config_path"])
    base_config = read_config(base_path)
    _validate_configs(sft_config, base_config)
    set_deterministic_seed(int(base_config["eval"]["deterministic_seed"]))

    artifact_root = Path(str(sft_config["paths"]["artifact_root"])).expanduser()
    train_report_path = Path(str(sft_config["outputs"]["train_report"])).expanduser()
    if not train_report_path.exists():
        raise FileNotFoundError(f"Missing v3 train report: {train_report_path}")
    train_report = _read_json(train_report_path)
    if train_report.get("status") != "completed" or train_report.get("mode") != "train":
        raise ValueError("v3 formal train report is incomplete")
    if not isinstance(train_report.get("train_metrics", {}).get("train_loss"), (int, float)) or not math.isfinite(
        float(train_report["train_metrics"]["train_loss"])
    ):
        raise ValueError("v3 train loss is not finite")

    base_report_path = Path(str(base_config["paths"]["artifact_root"])) / "reports" / "base_dev_metrics.json"
    configured_base_report = base_config.get("outputs", {}).get("dev_report")
    if configured_base_report:
        base_report_path = Path(str(configured_base_report)).expanduser()
        if not base_report_path.is_absolute():
            base_report_path = Path(str(base_config["paths"]["artifact_root"])).expanduser() / base_report_path
    base_report = _read_json(base_report_path)
    expected = int(base_config["eval"]["expected_count"])
    if base_report.get("status") != "completed" or int(base_report.get("evaluated", -1)) != expected:
        raise ValueError("Frozen Base Dev report is incomplete")
    if int(base_report.get("generation_config", {}).get("max_new_tokens", -1)) != 4096:
        raise ValueError("Base report is not frozen 4096 evaluation")

    reports = {
        "v1": _read_json(Path(str(sft_config["paths"]["v1_dev_report_path"])).expanduser()),
        "v2": _read_json(Path(str(sft_config["paths"]["v2_dev_report_path"])).expanduser()),
    }
    for name, report in reports.items():
        if report.get("status") != "completed" or int(report.get("evaluated", -1)) != expected:
            raise ValueError(f"Preserved {name} Dev report is incomplete")

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
                checkpoint,
                checkpoint_hash,
                generation_config,
                generation_config_hash,
            )
        )
        print(f"Evaluated {index}/{len(records)} SFT-v3 {mode} records", flush=True)
    elapsed_ms = int(round((time.perf_counter() - started) * 1000))

    metrics = _summarize_v2(rollouts)
    report_path = _configured_path(sft_config, "dev_report" if mode == "dev" else "smoke_report")
    rollouts_path = _configured_path(sft_config, "dev_rollouts") if mode == "dev" else artifact_root / "smoke" / "sft_v3_smoke_rollouts.jsonl"
    if report_path.exists() or rollouts_path.exists():
        raise FileExistsError("Refusing to overwrite existing v3 evaluation artifacts")
    write_jsonl(rollouts_path, rollouts)
    report = {
        "phase": "phase3_reasoning_sft_v3",
        "step": "sft_v3_dev_evaluation",
        "mode": mode,
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "evaluated": len(rollouts),
        "model_role": "sft_v3",
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
        "candidate_dataset": {
            "audit_path": str(Path(str(sft_config["paths"]["construction_audit_path"])).expanduser()),
            "audit_hash": file_sha256(sft_config["paths"]["construction_audit_path"]),
            "dataset_hash": train_report["candidate_audit"]["dataset_hash"],
            "dataset_size": train_report["candidate_audit"]["candidate_count"],
            "distribution": train_report["dataset_distribution"],
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
            key: metrics[key]
            for key in (
                "starts_with_think_count",
                "starts_with_think_rate",
                "closed_think_count",
                "closed_think_rate",
                "unclosed_think_count",
                "unclosed_think_rate",
                "fenced_code_count",
                "fenced_code_rate",
                "valid_reasoning_to_code_transition_count",
                "valid_reasoning_to_code_transition_rate",
                "generation_cap_hit_count",
                "generation_cap_hit_rate",
                "format_valid_count",
                "format_valid_rate",
                "extraction_strategy_counts",
            )
        },
        "comparison": None,
        "runtime": runtime_info(),
        "config": {
            "sft_config_path": str(Path(config_path).resolve()),
            "sft_config_hash": config_hash(config_path),
            "base_config_path": str(base_path),
            "base_config_hash": config_hash(base_path),
            "v1_report_path": str(sft_config["paths"]["v1_dev_report_path"]),
            "v2_report_path": str(sft_config["paths"]["v2_dev_report_path"]),
        },
        "evaluation_elapsed_ms": elapsed_ms,
        "report_path": str(report_path),
    }
    if mode == "dev":
        report["comparison"] = _comparison(base_report, reports["v1"], reports["v2"], metrics)
        report["phase3_v3_gate"] = {
            "formal_sft_training_completed": True,
            "train_loss_finite": True,
            "checkpoint_artifact_present": True,
            "checkpoint_adapter_hash_recorded": True,
            "all_dev_problems_evaluated": len(rollouts) == expected,
            "generation_policy_matches_frozen_base": generation_config_hash
            == str(base_report.get("generation_config_hash")),
            "dataset_is_natural_length_only": True,
            "artificial_truncation": False,
            "format_valid_rate": metrics["format_valid_rate"],
            "generation_cap_hit_rate": metrics["generation_cap_hit_rate"],
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
    print(json.dumps(evaluate_sft_v3(args.config, args.base_config, args.mode), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
