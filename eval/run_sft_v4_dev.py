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
from eval.phase2_common import config_hash, file_sha256, generation_config_for_hash, load_jsonl, read_config, runtime_info, serialize_prompt, set_deterministic_seed, stable_problem_sample, view_path, write_json, write_jsonl
from eval.run_base_baseline import _generate, _load_model_and_tokenizer, _verify_generated_code
from eval.run_sft_v2_dev import _summarize_v2
from eval.sft_v2_sequence import format_response_diagnostics
from sft.v4_data import validate_frozen_v3_control
from verifier.extract_code import extract_python_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the Phase 3-v4 low-LR SFT adapter on frozen Dev.")
    parser.add_argument("--config", default="configs/sft_v4.yaml")
    parser.add_argument("--base-config", default=None)
    parser.add_argument("--mode", choices=("smoke", "dev"), default="dev")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _configured_path(config: dict[str, Any], key: str) -> Path:
    return Path(str(config["outputs"][key])).expanduser()


def _validate_configs(sft_config: dict[str, Any], base_config: dict[str, Any]) -> None:
    if sft_config.get("phase") != "phase3_reasoning_sft_v4":
        raise ValueError("Unexpected v4 config phase")
    if not base_config.get("frozen") or int(base_config["generation"]["max_new_tokens"]) != 4096:
        raise ValueError("v4 must use frozen Base 4096 evaluation")
    if int(sft_config["expected"]["eval_max_new_tokens"]) != 4096:
        raise ValueError("v4 evaluation budget must be 4096")
    if int(sft_config["sft_sequence"]["max_sequence_length"]) != 8192 or sft_config["sft_sequence"].get("truncation_policy") != "none":
        raise ValueError("v4 sequence policy differs from frozen v3")
    if str(sft_config["prompt"]["template"]) != str(base_config["prompt"]["template"]):
        raise ValueError("v4 prompt differs from frozen Base")
    for key in ("repo_id", "model_revision", "tokenizer_revision"):
        if str(sft_config["model"][key]) != str(base_config["model"][key]):
            raise ValueError(f"v4 model {key} differs from Base")


def _load_records(base_config: dict[str, Any], mode: str) -> tuple[str, list[dict[str, Any]]]:
    split = str(base_config["eval"]["split"] if mode == "dev" else base_config["smoke"]["split"])
    records = load_jsonl(view_path(base_config, split))
    if mode == "smoke":
        return split, stable_problem_sample(records, count=int(base_config["smoke"]["sample_count"]), seed=int(base_config["smoke"]["sample_seed"]))
    if len(records) != int(base_config["eval"]["expected_count"]):
        raise ValueError("Frozen Dev count mismatch")
    return split, records


def _load_sft_model(base_config: dict[str, Any], sft_config: dict[str, Any]):
    from peft import PeftModel

    checkpoint = Path(str(sft_config["training"]["output_dir"])).expanduser()
    if not checkpoint.exists() or not (checkpoint / "adapter_config.json").exists() or not (checkpoint / "adapter_model.safetensors").exists():
        raise FileNotFoundError(f"Missing v4 checkpoint: {checkpoint}")
    model, tokenizer = _load_model_and_tokenizer(base_config)
    model = PeftModel.from_pretrained(model, str(checkpoint), is_trainable=False)
    model.eval()
    return model, tokenizer, checkpoint


def _rollout(base_config: dict[str, Any], model: Any, tokenizer: Any, record: dict[str, Any], checkpoint: Path, checkpoint_hash: str, generation_config: dict[str, Any], generation_config_hash: str) -> dict[str, Any]:
    prompt = serialize_prompt(base_config, str(record["prompt"]))
    generated = _generate(base_config, model, tokenizer, prompt)
    extracted = extract_python_code(generated["response"])
    verifier = _verify_generated_code(base_config, extracted.code, record)
    return {
        "problem_id": record["problem_id"],
        "model_role": "sft_v4",
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
        "format_diagnostics": format_response_diagnostics(generated["response"], bool(extracted.code.strip())),
    }


def _comparison(base: dict[str, Any], v1: dict[str, Any], v2: dict[str, Any], v3: dict[str, Any], v4: dict[str, Any]) -> dict[str, Any]:
    names = ("pass_at_1", "executable_rate", "mean_testcase_pass_rate", "reward_test_mean_pass_rate", "heldout_test_mean_pass_rate", "generation_cap_hit_rate", "G_V")
    def norm(report: dict[str, Any]) -> dict[str, float]:
        m = report["metrics"] if "metrics" in report else report
        m = dict(m)
        m["G_V"] = float(m.get("reward_test_mean_pass_rate", 0.0)) - float(m.get("heldout_test_mean_pass_rate", 0.0))
        return {name: float(m[name]) for name in names if name in m}
    tables = {"base": norm(base), "sft_v1": norm(v1), "sft_v2": norm(v2), "sft_v3": norm(v3), "sft_v4": norm(v4)}
    return {
        "base_report_path": base.get("report_path"),
        "base_report_hash": file_sha256(base["report_path"]),
        "v1_report_path": v1.get("report_path"),
        "v1_report_hash": file_sha256(v1["report_path"]),
        "v2_report_path": v2.get("report_path"),
        "v2_report_hash": file_sha256(v2["report_path"]),
        "v3_report_path": v3.get("report_path"),
        "v3_report_hash": file_sha256(v3["report_path"]),
        "metrics": tables,
        "delta_sft_v4_minus_base": {k: tables["sft_v4"][k] - tables["base"][k] for k in tables["sft_v4"].keys() & tables["base"].keys()},
        "delta_sft_v4_minus_sft_v3": {k: tables["sft_v4"][k] - tables["sft_v3"][k] for k in tables["sft_v4"].keys() & tables["sft_v3"].keys()},
    }


def evaluate_sft_v4(config_path: str, base_config_path: str | None, mode: str) -> dict[str, Any]:
    sft_config = load_config(config_path)
    base_path = Path(base_config_path or sft_config["paths"]["base_eval_config_path"])
    base_config = read_config(base_path)
    _validate_configs(sft_config, base_config)
    freeze = validate_frozen_v3_control(sft_config)
    set_deterministic_seed(int(base_config["eval"]["deterministic_seed"]))
    train_report_path = Path(str(sft_config["outputs"]["train_report"])).expanduser()
    if not train_report_path.exists():
        raise FileNotFoundError(train_report_path)
    train_report = _read_json(train_report_path)
    if train_report.get("status") != "completed" or train_report.get("mode") != "train" or not math.isfinite(float(train_report["train_metrics"]["train_loss"])):
        raise ValueError("v4 formal train report is incomplete or non-finite")
    expected = int(base_config["eval"]["expected_count"])
    configured_base_report = base_config.get("outputs", {}).get("dev_report")
    base_report_path = Path(str(configured_base_report)).expanduser() if configured_base_report else Path(str(base_config["paths"]["artifact_root"])) / "reports" / "base_dev_metrics.json"
    if not base_report_path.is_absolute():
        base_report_path = Path(str(base_config["paths"]["artifact_root"])).expanduser() / base_report_path
    base_report = _read_json(base_report_path)
    if base_report.get("status") != "completed" or int(base_report.get("evaluated", -1)) != expected:
        raise ValueError("Frozen Base report incomplete")
    reports = {name: _read_json(Path(str(sft_config["paths"][f"{name}_dev_report_path"])).expanduser()) for name in ("v1", "v2", "v3")}
    for name, report in reports.items():
        if report.get("status") != "completed" or int(report.get("evaluated", -1)) != expected:
            raise ValueError(f"Preserved {name} report incomplete")
    split, records = _load_records(base_config, mode)
    model, tokenizer, checkpoint = _load_sft_model(base_config, sft_config)
    checkpoint_hash = file_sha256(checkpoint / "adapter_model.safetensors")
    generation_config = generation_config_for_hash(base_config)
    generation_config_hash = stable_hash(generation_config)
    rollouts = []
    started = time.perf_counter()
    for index, record in enumerate(records, start=1):
        rollouts.append(_rollout(base_config, model, tokenizer, record, checkpoint, checkpoint_hash, generation_config, generation_config_hash))
        print(f"Evaluated {index}/{len(records)} SFT-v4 {mode} records", flush=True)
    elapsed_ms = int(round((time.perf_counter() - started) * 1000))
    metrics = _summarize_v2(rollouts)
    report_path = _configured_path(sft_config, "dev_report" if mode == "dev" else "smoke_report")
    rollouts_path = _configured_path(sft_config, "dev_rollouts") if mode == "dev" else Path(str(sft_config["paths"]["artifact_root"])).expanduser() / "smoke" / "sft_v4_smoke_rollouts.jsonl"
    if report_path.exists() or rollouts_path.exists():
        raise FileExistsError("Refusing to overwrite existing v4 evaluation artifacts")
    write_jsonl(rollouts_path, rollouts)
    report = {
        "phase": "phase3_reasoning_sft_v4", "step": "sft_v4_dev_evaluation", "mode": mode, "status": "completed", "generated_at_utc": datetime.now(timezone.utc).isoformat(), "split": split, "evaluated": len(rollouts), "model_role": "sft_v4", "training_subset": "natural_length_filtered", "learning_rate": float(sft_config["training"]["learning_rate"]),
        "base_model": {"repo_id": base_config["model"]["repo_id"], "model_revision": base_config["model"]["model_revision"], "tokenizer_revision": base_config["model"]["tokenizer_revision"]},
        "sft_checkpoint": {"path": str(checkpoint), "adapter_hash": checkpoint_hash, "train_report_path": str(train_report_path), "train_report_hash": file_sha256(train_report_path)},
        "training_hyperparameter_change": train_report["training_hyperparameter_change"],
        "training_diagnostics": train_report["training_diagnostics"],
        "candidate_dataset": {"count": freeze["candidate_count"], "dataset_hash": freeze["dataset_hash"], "candidate_ids_hash": freeze["candidate_ids_hash"], "manifest_hash": freeze["manifest_hash"], "distribution_shift_present": True},
        "dataset_distribution": freeze["dataset_distribution"],
        "prompt": {"serialization_version": base_config["prompt"]["serialization_version"], "template": base_config["prompt"]["template"], "use_chat_template": base_config["prompt"]["use_chat_template"]},
        "generation_config": generation_config, "generation_config_hash": generation_config_hash, "docker_image": base_config["verifier"]["docker_image"], "rollouts_path": str(rollouts_path), "rollouts_hash": file_sha256(rollouts_path), "metrics": metrics,
        "format_diagnostics": {key: metrics[key] for key in ("starts_with_think_count", "starts_with_think_rate", "closed_think_count", "closed_think_rate", "unclosed_think_count", "unclosed_think_rate", "fenced_code_count", "fenced_code_rate", "generation_cap_hit_count", "generation_cap_hit_rate", "format_valid_count", "format_valid_rate", "extraction_strategy_counts")},
        "comparison": None, "runtime": runtime_info(), "config": {"sft_config_path": str(Path(config_path).resolve()), "sft_config_hash": config_hash(config_path), "base_config_path": str(base_path), "base_config_hash": config_hash(base_path)}, "evaluation_elapsed_ms": elapsed_ms, "report_path": str(report_path),
    }
    if mode == "dev":
        report["comparison"] = _comparison(base_report, reports["v1"], reports["v2"], reports["v3"], report)
        report["phase3_v4_gate"] = {"formal_sft_training_completed": True, "train_loss_finite": True, "checkpoint_artifact_present": True, "checkpoint_adapter_hash_recorded": True, "all_dev_problems_evaluated": len(rollouts) == expected, "generation_policy_matches_frozen_base": generation_config_hash == str(base_report.get("generation_config_hash")), "same_frozen_v3_candidate_set": True, "only_training_change_learning_rate": True, "generation_cap_hit_rate": metrics["generation_cap_hit_rate"], "behavioral_gate": "manual_review_required", "dpo_grpo_ready": False}
    write_json(report_path, report)
    return report


def main() -> None:
    args = parse_args()
    print(json.dumps(evaluate_sft_v4(args.config, args.base_config, args.mode), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
