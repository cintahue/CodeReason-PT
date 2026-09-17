from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data.config import load_config
from eval.phase2_common import (
    file_sha256,
    git_command,
    git_status_short,
    load_jsonl,
    numeric_stats,
    pretrained_load_reference,
    runtime_info,
    set_deterministic_seed,
    write_json,
    write_jsonl,
)
from sft.v4_data import build_v4_examples


class SftTokenDatasetV4:
    def __init__(self, examples: list[Any]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        example = self.examples[index]
        return {"input_ids": example.input_ids, "labels": example.labels}


class DataCollatorForResponseOnlySftV4:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        max_len = max(len(item["input_ids"]) for item in features)
        input_ids, labels, attention_mask = [], [], []
        for item in features:
            pad = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * pad)
            labels.append(item["labels"] + [-100] * pad)
            attention_mask.append([1] * len(item["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 3-v4 controlled low-LR SFT.")
    parser.add_argument("--config", default="configs/sft_v4.yaml")
    parser.add_argument("--mode", choices=("smoke", "train"), required=True)
    return parser.parse_args()


def _torch_dtype(dtype_name: str):
    import torch

    try:
        return getattr(torch, dtype_name)
    except AttributeError as exc:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}") from exc


def _load_model_and_tokenizer(config: dict[str, Any]):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for Phase 3-v4 SFT")
    tokenizer_source, tokenizer_kwargs = pretrained_load_reference(config, "tokenizer_revision")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_source, model_kwargs = pretrained_load_reference(config, "model_revision")
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        dtype=_torch_dtype(str(config["model"]["dtype"])),
        device_map=config["model"].get("device_map", "auto"),
        **model_kwargs,
    )
    if bool(config["training"].get("gradient_checkpointing", False)):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.config.use_cache = False
    return model, tokenizer


def _apply_lora(config: dict[str, Any], model: Any) -> Any:
    from peft import LoraConfig, TaskType, get_peft_model

    lora = config["lora"]
    return get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(lora["r"]),
            lora_alpha=int(lora["alpha"]),
            lora_dropout=float(lora["dropout"]),
            target_modules=list(lora["target_modules"]),
        ),
    )


def _validate_smoke_before_formal(config: dict[str, Any], config_path: str) -> None:
    report_path = Path(str(config["outputs"]["smoke_report"])).expanduser()
    load_path = Path(str(config["paths"]["artifact_root"])).expanduser() / "smoke" / "checkpoint_load_report.json"
    if not report_path.exists() or not load_path.exists():
        raise ValueError("Run and load-check the v4 smoke checkpoint before formal training")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    load_report = json.loads(load_path.read_text(encoding="utf-8"))
    if report.get("status") != "completed" or report.get("mode") != "smoke":
        raise ValueError("v4 smoke training report is incomplete")
    if report.get("config_hash") != file_sha256(config_path):
        raise ValueError("v4 smoke report config hash does not match current config")
    if load_report.get("status") != "completed" or load_report.get("adapter_load_check") != "passed":
        raise ValueError("v4 smoke checkpoint load check is incomplete")


def _training_args(config: dict[str, Any], mode: str):
    from transformers import TrainingArguments

    training = config["training"]
    kwargs = {
        "output_dir": str(config["smoke"]["output_dir"] if mode == "smoke" else training["output_dir"]),
        "per_device_train_batch_size": int(training["per_device_train_batch_size"]),
        "gradient_accumulation_steps": int(training["gradient_accumulation_steps"]),
        "learning_rate": float(training["learning_rate"]),
        "warmup_steps": float(training["warmup_ratio"]),
        "weight_decay": float(training["weight_decay"]),
        "logging_steps": int(training["logging_steps"]),
        "save_strategy": str(training["save_strategy"]),
        "bf16": bool(training["bf16"]),
        "gradient_checkpointing": bool(training["gradient_checkpointing"]),
        "optim": str(training["optim"]),
        "report_to": [] if str(training.get("report_to", "none")) == "none" else training["report_to"],
        "remove_unused_columns": False,
        "seed": int(training["seed"]),
    }
    if mode == "smoke":
        kwargs.update(max_steps=int(config["smoke"]["max_steps"]), save_strategy="steps", save_steps=int(config["smoke"]["max_steps"]))
        # Five smoke steps are shorter than the formal logging interval. Keep
        # the formal v3 interval unchanged, but log every smoke step so the
        # smoke gate can inspect finite loss and gradient norms.
        kwargs["logging_steps"] = 1
    else:
        kwargs["num_train_epochs"] = float(training["num_train_epochs"])
    return TrainingArguments(**kwargs)


def _validate_saved_adapter(output_dir: Path, model: Any) -> dict[str, Any]:
    from peft import PeftConfig

    peft_config = PeftConfig.from_pretrained(str(output_dir))
    weights = output_dir / "adapter_model.safetensors"
    if not weights.exists() and not (output_dir / "adapter_model.bin").exists():
        raise FileNotFoundError(f"No PEFT adapter weights found in {output_dir}")
    return {
        "adapter_config_load_check": "passed",
        "adapter_weights_present": True,
        "base_model_name_or_path": str(getattr(peft_config, "base_model_name_or_path", "")),
        "adapter_hash": file_sha256(weights) if weights.exists() else None,
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }


def _training_diagnostics(trainer: Any, examples: list[Any], train_result: Any, args: Any) -> dict[str, Any]:
    log_history = list(getattr(trainer.state, "log_history", []))
    grad_norms = [float(row["grad_norm"]) for row in log_history if isinstance(row.get("grad_norm"), (int, float)) and math.isfinite(float(row["grad_norm"]))]
    learning_rates = [float(row["learning_rate"]) for row in log_history if isinstance(row.get("learning_rate"), (int, float))]
    losses = [float(row["loss"]) for row in log_history if isinstance(row.get("loss"), (int, float)) and math.isfinite(float(row["loss"]))]
    supervised_tokens = sum(sum(int(label != -100) for label in example.labels) for example in examples)
    optimizer_steps = int(trainer.state.global_step)
    return {
        "optimizer_step_count": optimizer_steps,
        "effective_training_tokens": supervised_tokens,
        "final_train_loss": float(train_result.metrics.get("train_loss", float("nan"))),
        "loss_log_stats": numeric_stats(losses, (50, 95)) if losses else {"count": 0},
        "grad_norm_distribution": numeric_stats(grad_norms, (50, 90, 95, 99)) if grad_norms else {"count": 0},
        "learning_rate_schedule": {
            "configured_learning_rate": float(args.learning_rate),
            "configured_warmup_ratio": float(args.warmup_steps),
            "actual_warmup_steps": int(args.get_warmup_steps(optimizer_steps)),
            "logged_count": len(learning_rates),
            "first_logged": learning_rates[0] if learning_rates else None,
            "last_logged": learning_rates[-1] if learning_rates else None,
            "min_logged": min(learning_rates) if learning_rates else None,
            "max_logged": max(learning_rates) if learning_rates else None,
        },
        "logged_history_count": len(log_history),
    }


def train_sft_v4(config_path: str, mode: str) -> dict[str, Any]:
    config = load_config(config_path)
    from sft.v4_data import validate_frozen_v3_control

    freeze = validate_frozen_v3_control(config)
    if mode == "train":
        _validate_smoke_before_formal(config, config_path)
    output_dir = Path(str(config["smoke"]["output_dir"] if mode == "smoke" else config["training"]["output_dir"])).expanduser()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing v4 checkpoint: {output_dir}")
    set_deterministic_seed(int(config["training"]["seed"]))
    model, tokenizer = _load_model_and_tokenizer(config)
    records = load_jsonl(config["paths"]["sft_view_path"])
    if len(records) != int(config["expected"]["source_sft_count"]):
        raise ValueError("v4 source SFT count mismatch")
    examples, manifest, dataset_stats, freeze = build_v4_examples(config, tokenizer, records)
    if mode == "smoke":
        count = int(config["smoke"]["sample_count"])
        examples, manifest = examples[:count], manifest[:count]
        dataset_stats = dict(dataset_stats)
        dataset_stats.update({"constructed_records": len(examples), "smoke_subset": True})
    if not examples:
        raise RuntimeError("No usable v4 examples")
    prompt_masked = sum(int(all(label == -100 for label in ex.labels[: ex.prompt_tokens])) for ex in examples)
    eos_supervised = sum(int(ex.labels[-1] == int(tokenizer.eos_token_id)) for ex in examples)
    model = _apply_lora(config, model)
    from transformers import Trainer

    args = _training_args(config, mode)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=SftTokenDatasetV4(examples),
        data_collator=DataCollatorForResponseOnlySftV4(int(tokenizer.pad_token_id)),
    )
    train_result = trainer.train()
    training_diagnostics = _training_diagnostics(trainer, examples, train_result, args)
    final_train_loss = training_diagnostics["final_train_loss"]
    if not math.isfinite(final_train_loss):
        raise ValueError("v4 training loss is non-finite")
    if mode == "smoke" and not training_diagnostics["grad_norm_distribution"].get("count"):
        raise ValueError("v4 smoke did not record any gradient norm")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    load_check = _validate_saved_adapter(Path(args.output_dir), trainer.model)
    artifact_root = Path(str(config["paths"]["artifact_root"])).expanduser()
    output_root = artifact_root / mode
    manifest_path = output_root / "sft_v4_sequence_manifest.jsonl"
    report_path = Path(str(config["outputs"]["smoke_report"] if mode == "smoke" else config["outputs"]["train_report"])).expanduser()
    write_jsonl(manifest_path, manifest)
    report = {
        "phase": "phase3_reasoning_sft_v4",
        "mode": mode,
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_file": str(Path(config_path).resolve()),
        "config_hash": file_sha256(config_path),
        "training_subset": "natural_length_filtered",
        "frozen_v3_candidate": freeze,
        "candidate_dataset": {
            "count": int(config["expected"]["candidate_count"]),
            "dataset_hash": freeze["dataset_hash"],
            "candidate_ids_hash": freeze["candidate_ids_hash"],
            "manifest_hash": freeze["manifest_hash"],
            "distribution_shift_present": True,
        },
        "dataset_distribution": freeze["dataset_distribution"],
        "model": {"repo_id": config["model"]["repo_id"], "model_revision": config["model"]["model_revision"], "tokenizer_revision": config["model"]["tokenizer_revision"], "dtype": config["model"]["dtype"]},
        "canonical_target": {"source_field": "reasoning", "source_semantics": "OpenCodeReasoning-2 r1_generation", "reference_code_appended": False, "artificial_truncation": False},
        "sft_sequence": dict(config["sft_sequence"]),
        "lora": dict(config["lora"]),
        "training_args": args.to_dict(),
        "training_hyperparameter_change": {"v3_learning_rate": 0.0001, "v4_learning_rate": 0.00005, "only_experimental_change": True},
        "dataset_stats": dataset_stats,
        "manifest": {"path": str(manifest_path), "hash": file_sha256(manifest_path), "count": len(manifest)},
        "label_invariant_audit": {"prompt_fully_masked_count": prompt_masked, "prompt_fully_masked_rate": prompt_masked / len(examples), "eos_supervised_count": eos_supervised, "eos_supervised_rate": eos_supervised / len(examples), "padding_policy": "collator masks all padding labels with -100"},
        "training_diagnostics": training_diagnostics,
        "provenance": {"code_commit_sha": git_command(["rev-parse", "HEAD"]), "working_tree_clean_at_train": not bool(git_status_short()), "v3_commit_sha": freeze["v3_commit_sha"]},
        "checkpoint": {"path": str(args.output_dir), **load_check},
        "train_metrics": train_result.metrics,
        "runtime": runtime_info(),
        "v1_v2_v3_artifacts_overwritten": False,
    }
    write_json(report_path, report)
    return report


def main() -> None:
    args = parse_args()
    print(json.dumps(train_sft_v4(args.config, args.mode), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
