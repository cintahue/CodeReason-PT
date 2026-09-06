from __future__ import annotations

import argparse
import json
import time
from collections import Counter
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
    path_from_config,
    pretrained_load_reference,
    read_config,
    reports_dir,
    rollouts_dir,
    runtime_info,
    serialize_prompt,
    set_deterministic_seed,
    smoke_dir,
    stable_problem_sample,
    view_path,
    write_json,
    write_jsonl,
)
from verifier.executor import verify_code
from verifier.extract_code import extract_python_code
from verifier.result import SandboxConfig, VerificationResult, VerifierStatus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 2 Qwen2.5-Coder-3B Base baseline.")
    parser.add_argument("--config", default="configs/base_eval.yaml")
    parser.add_argument("--mode", choices=("smoke", "dev"), required=True)
    return parser.parse_args()


def _torch_dtype(dtype_name: str):
    import torch

    try:
        return getattr(torch, dtype_name)
    except AttributeError as exc:
        raise ValueError(f"Unsupported torch dtype in config: {dtype_name}") from exc


def _load_model_and_tokenizer(config: dict[str, Any]):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the fixed Qwen2.5-Coder-3B Base baseline")

    model_config = config["model"]
    tokenizer_source, tokenizer_kwargs = pretrained_load_reference(config, "tokenizer_revision")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)
    except OSError as exc:
        raise RuntimeError(
            "Failed to load the fixed Qwen2.5-Coder-3B tokenizer. "
            "Run bash scripts/03a_phase2_prefetch_base_model.sh before smoke/dev evaluation."
        ) from exc
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model_source, model_kwargs = pretrained_load_reference(config, "model_revision")
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            torch_dtype=_torch_dtype(str(model_config["dtype"])),
            device_map=model_config.get("device_map", "auto"),
            **model_kwargs,
        )
    except Exception as exc:
        raise RuntimeError("Qwen2.5-Coder-3B Base failed to load; do not fall back to another model") from exc
    model.eval()
    torch.set_grad_enabled(False)
    return model, tokenizer


def _tokenize_prompt(config: dict[str, Any], tokenizer, prompt: str):
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False, truncation=False)
    prompt_tokens = int(encoded["input_ids"].shape[-1])
    max_new_tokens = int(config["generation"]["max_new_tokens"])
    context_window = int(config["model"]["max_position_embeddings"])
    if prompt_tokens + max_new_tokens > context_window:
        raise ValueError(
            f"Prompt would exceed context window without truncation: prompt={prompt_tokens}, "
            f"max_new_tokens={max_new_tokens}, context={context_window}"
        )
    return encoded, prompt_tokens


def _model_device(model):
    return next(model.parameters()).device


def _generate(config: dict[str, Any], model, tokenizer, prompt: str) -> dict[str, Any]:
    import torch

    encoded, prompt_tokens = _tokenize_prompt(config, tokenizer, prompt)
    device = _model_device(model)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    generation = generation_config_for_hash(config)
    started = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            **encoded,
            do_sample=generation["do_sample"],
            num_return_sequences=generation["num_return_sequences"],
            max_new_tokens=generation["max_new_tokens"],
            use_cache=generation["use_cache"],
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    latency_ms = int(round((time.perf_counter() - started) * 1000))
    response_ids = outputs[0][prompt_tokens:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    response_token_count = int(response_ids.shape[-1])
    hit_max_new_tokens = response_token_count >= int(generation["max_new_tokens"])
    return {
        "response": response,
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_token_count,
        "generation_latency_ms": latency_ms,
        "hit_max_new_tokens": hit_max_new_tokens,
    }


def _sandbox_config(config: dict[str, Any]) -> SandboxConfig:
    verifier = config["verifier"]
    return SandboxConfig(
        backend=str(verifier["backend"]),
        docker_image=str(verifier["docker_image"]),
        wall_time_seconds=float(verifier["wall_time_seconds"]),
        cpu_time_seconds=int(verifier["cpu_time_seconds"]),
        memory_mb=int(verifier["memory_mb"]),
        process_limit=int(verifier["process_limit"]),
        output_limit_bytes=int(verifier["output_limit_bytes"]),
    )


def _scope_result_dict(result: VerificationResult) -> dict[str, Any]:
    return {
        "status": result.status.value,
        "passed": result.passed,
        "total": result.total,
        "pass_rate": result.pass_rate,
        "compile_success": result.compile_success,
        "runtime_success": result.runtime_success,
        "timeout": result.timeout,
        "exit_code": result.exit_code,
        "runtime_ms": result.runtime_ms,
        "stdout_size": result.stdout_size,
        "stderr_size": result.stderr_size,
    }


def _combined_status(reward: VerificationResult, heldout: VerificationResult) -> VerifierStatus:
    statuses = {reward.status, heldout.status}
    if VerifierStatus.CE in statuses:
        return VerifierStatus.CE
    if VerifierStatus.TLE in statuses:
        return VerifierStatus.TLE
    if VerifierStatus.RE in statuses:
        return VerifierStatus.RE
    if reward.status == VerifierStatus.AC and heldout.status == VerifierStatus.AC:
        return VerifierStatus.AC
    return VerifierStatus.WA


def _verify_generated_code(config: dict[str, Any], code: str, record: dict[str, Any]) -> dict[str, Any]:
    sandbox = _sandbox_config(config)
    reward = verify_code(code, list(record["reward_tests"]), sandbox_config=sandbox, extraction_strategy="phase2_extracted_code")
    heldout = verify_code(code, list(record["heldout_tests"]), sandbox_config=sandbox, extraction_strategy="phase2_extracted_code")
    status = _combined_status(reward, heldout)
    passed = reward.passed + heldout.passed
    total = reward.total + heldout.total
    return {
        "status": status.value,
        "passed": passed,
        "total": total,
        "pass_rate": passed / total if total else 0.0,
        "reward": _scope_result_dict(reward),
        "heldout": _scope_result_dict(heldout),
    }


def _load_records_for_mode(config: dict[str, Any], mode: str) -> tuple[str, list[dict[str, Any]]]:
    if mode == "smoke":
        split = str(config["smoke"]["split"])
        records = load_jsonl(view_path(config, split))
        selected = stable_problem_sample(
            records,
            count=int(config["smoke"]["sample_count"]),
            seed=int(config["smoke"]["sample_seed"]),
        )
        return split, selected
    split = str(config["eval"]["split"])
    records = load_jsonl(view_path(config, split))
    expected = int(config["eval"]["expected_count"])
    if len(records) != expected:
        raise ValueError(f"Final {split} view count mismatch: expected {expected}, got {len(records)}")
    token_profile_path = reports_dir(config) / "phase2_token_profile.json"
    if not token_profile_path.exists():
        raise FileNotFoundError(f"Run token profiling first: {token_profile_path}")
    token_profile = json.loads(token_profile_path.read_text(encoding="utf-8"))
    if token_profile.get("status") != "completed":
        raise ValueError(f"Token profile is not completed: {token_profile_path}")
    return split, records


def _rollout_path(config: dict[str, Any], mode: str) -> Path:
    configured = _configured_output_path(config, f"{mode}_rollouts")
    if configured is not None:
        return configured
    if mode == "smoke":
        return smoke_dir(config) / "base_sft_smoke_rollouts.jsonl"
    return rollouts_dir(config) / "base_dev_rollouts.jsonl"


def _report_path(config: dict[str, Any], mode: str) -> Path:
    configured = _configured_output_path(config, f"{mode}_report")
    if configured is not None:
        return configured
    if mode == "smoke":
        return smoke_dir(config) / "base_sft_smoke_report.json"
    return reports_dir(config) / "base_dev_metrics.json"


def _configured_output_path(config: dict[str, Any], key: str) -> Path | None:
    value = config.get("outputs", {}).get(key)
    if not value:
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return artifact_root(config) / path


def _summarize_rollouts(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rollouts)
    status_counts = Counter(record["verifier_status"] for record in rollouts)
    extraction_counts = Counter(record["extraction_strategy"] for record in rollouts)
    extraction_success = sum(1 for record in rollouts if record["extraction_success"])
    executable = sum(1 for record in rollouts if record["verifier_status"] in {"AC", "WA"})
    response_tokens = [int(record["response_token_count"]) for record in rollouts]
    generation_latencies = [int(record["generation_latency_ms"]) for record in rollouts]
    all_pass_rates = [float(record["testcase_pass_rate"]) for record in rollouts]
    reward_pass_rates = [float(record["reward_pass_rate"]) for record in rollouts]
    heldout_pass_rates = [float(record["heldout_pass_rate"]) for record in rollouts]
    return {
        "total": total,
        "pass_at_1": status_counts.get("AC", 0) / total if total else 0.0,
        "mean_testcase_pass_rate": sum(all_pass_rates) / total if total else 0.0,
        "executable_rate": executable / total if total else 0.0,
        "status_counts": dict(sorted(status_counts.items())),
        "code_extraction_success_rate": extraction_success / total if total else 0.0,
        "extraction_strategy_counts": dict(sorted(extraction_counts.items())),
        "response_tokens": {
            "average": sum(response_tokens) / total if total else 0.0,
            **numeric_stats(response_tokens, (50, 95)),
        },
        "generation_latency_ms": {
            "average": sum(generation_latencies) / total if total else 0.0,
            **numeric_stats(generation_latencies, (50, 95)),
        },
        "reward_test_mean_pass_rate": sum(reward_pass_rates) / total if total else 0.0,
        "heldout_test_mean_pass_rate": sum(heldout_pass_rates) / total if total else 0.0,
        "hit_max_new_tokens_count": sum(1 for record in rollouts if record["hit_max_new_tokens"]),
    }


def _audit(config_path: str, config: dict[str, Any], report: dict[str, Any], rollouts_path: Path) -> dict[str, Any]:
    token_profile_path = reports_dir(config) / "phase2_token_profile.json"
    views_report_path = reports_dir(config) / "phase2_views_report.json"
    token_profile = json.loads(token_profile_path.read_text(encoding="utf-8")) if token_profile_path.exists() else None
    return {
        "phase": "phase2_base_baseline",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": git_command(["rev-parse", "HEAD"]),
            "dirty": bool(git_status_short()),
            "status_short": git_status_short(),
        },
        "phase1_final_eligibility_manifest": str(path_from_config(config, "paths", "phase1_final_manifest_path")),
        "phase1_final_eligibility_manifest_hash": file_sha256(path_from_config(config, "paths", "phase1_final_manifest_path")),
        "final_dev_dataset_hash": file_sha256(view_path(config, "dev")),
        "model": {
            "repo_id": config["model"]["repo_id"],
            "model_revision": config["model"]["model_revision"],
            "tokenizer_revision": config["model"]["tokenizer_revision"],
            "dtype": config["model"]["dtype"],
        },
        "eval_config_version": config.get("eval_config_version", "causal_lm_eval_v1"),
        "baseline_role": config.get("baseline_role", "pilot_base_baseline"),
        "pilot_baseline": config.get("pilot_baseline"),
        "docker_image": config["verifier"]["docker_image"],
        "eval_config": {
            "path": config_path,
            "hash": config_hash(config_path),
            "prompt_serialization_version": config["prompt"]["serialization_version"],
            "generation_config_hash": stable_hash(generation_config_for_hash(config)),
            "generation_config": generation_config_for_hash(config),
        },
        "token_length_statistics": {
            "path": str(token_profile_path),
            "hash": file_sha256(token_profile_path) if token_profile_path.exists() else None,
            "splits": token_profile.get("splits") if isinstance(token_profile, dict) else None,
        },
        "views_report": {
            "path": str(views_report_path),
            "hash": file_sha256(views_report_path) if views_report_path.exists() else None,
        },
        "rollouts": {"path": str(rollouts_path), "hash": file_sha256(rollouts_path)},
        "metrics_report": {"path": report["report_path"], "hash": file_sha256(report["report_path"])},
        "base_metrics": report["metrics"],
        "runtime": report["runtime"],
        "phase2_gate": {
            "final_dev_identity_verified": report["split"] == "dev" and report["evaluated"] == int(config["eval"]["expected_count"]),
            "model_prompt_eval_config_frozen": bool(config.get("frozen")),
            "all_dev_problems_evaluated_successfully": report["split"] == "dev"
            and report["evaluated"] == int(config["eval"]["expected_count"]),
            "base_metrics_generated": bool(report["metrics"]),
            "result_audit_hashes_recorded": True,
        },
    }


def run_baseline(config_path: str, mode: str) -> dict[str, Any]:
    config = read_config(config_path)
    set_deterministic_seed(int(config["eval"]["deterministic_seed"]))
    split, records = _load_records_for_mode(config, mode)
    model, tokenizer = _load_model_and_tokenizer(config)
    generation_config = generation_config_for_hash(config)
    generation_config_hash = stable_hash(generation_config)
    rollouts: list[dict[str, Any]] = []

    for index, record in enumerate(records, start=1):
        prompt = serialize_prompt(config, str(record["prompt"]))
        generated = _generate(config, model, tokenizer, prompt)
        extracted = extract_python_code(generated["response"])
        verifier = _verify_generated_code(config, extracted.code, record)
        rollout = {
            "problem_id": record["problem_id"],
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
        rollouts.append(rollout)
        print(f"Evaluated {index}/{len(records)} {mode} records", flush=True)

    output_rollouts = _rollout_path(config, mode)
    write_jsonl(output_rollouts, rollouts)
    report_path = _report_path(config, mode)
    report = {
        "phase": "phase2_base_baseline",
        "mode": mode,
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "evaluated": len(rollouts),
        "model": {
            "repo_id": config["model"]["repo_id"],
            "model_revision": config["model"]["model_revision"],
            "tokenizer_revision": config["model"]["tokenizer_revision"],
            "dtype": config["model"]["dtype"],
        },
        "eval_config_version": config.get("eval_config_version", "causal_lm_eval_v1"),
        "baseline_role": config.get("baseline_role", "pilot_base_baseline"),
        "prompt": {
            "serialization_version": config["prompt"]["serialization_version"],
            "use_chat_template": config["prompt"]["use_chat_template"],
            "template": config["prompt"]["template"],
        },
        "generation_config": generation_config,
        "generation_config_hash": generation_config_hash,
        "docker_image": config["verifier"]["docker_image"],
        "rollouts_path": str(output_rollouts),
        "rollouts_hash": file_sha256(output_rollouts),
        "metrics": _summarize_rollouts(rollouts),
        "runtime": runtime_info(),
        "report_path": str(report_path),
    }
    write_json(report_path, report)
    if mode == "dev":
        audit = _audit(config_path, config, report, output_rollouts)
        audit_path = _configured_output_path(config, "base_audit") or Path("phase2_base_audit.json")
        write_json(audit_path, audit)
    return report


def main() -> None:
    args = parse_args()
    report = run_baseline(args.config, args.mode)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
