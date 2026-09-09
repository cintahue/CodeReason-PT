from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data.config import load_config
from data.schemas import stable_hash
from eval.phase2_common import (
    file_sha256,
    git_command,
    git_status_short,
    load_jsonl,
    pretrained_load_reference,
    runtime_info,
    set_deterministic_seed,
    write_json,
    write_jsonl,
)
from eval.sft_sequence import SequenceExample, build_sft_sequence


class SftTokenDataset:
    def __init__(self, examples: list[SequenceExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        example = self.examples[index]
        return {"input_ids": example.input_ids, "labels": example.labels}


class DataCollatorForResponseOnlySft:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        max_len = max(len(item["input_ids"]) for item in features)
        input_ids: list[list[int]] = []
        labels: list[list[int]] = []
        attention_mask: list[list[int]] = []
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
    parser = argparse.ArgumentParser(description="Run Phase 3 canonical-target Reasoning SFT.")
    parser.add_argument("--config", default="configs/sft.yaml")
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
        raise RuntimeError("CUDA GPU is required for Phase 3 SFT")
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
        # The base model is frozen by LoRA. Checkpointed layers therefore
        # need input gradients enabled for the adapter path to receive grads.
        model.enable_input_require_grads()
        model.config.use_cache = False
    return model, tokenizer


def _validate_phase_gate(config: dict[str, Any]) -> None:
    if config.get("phase") != "phase3_reasoning_sft":
        raise ValueError(f"Unexpected SFT config phase: {config.get('phase')}")
    if int(config["sft_sequence"]["max_sequence_length"]) != 8192:
        raise ValueError("Phase 3 SFT requires max_sequence_length=8192")
    if int(config["expected"]["eval_max_new_tokens"]) != 4096:
        raise ValueError("Phase 3 SFT requires frozen eval max_new_tokens=4096")
    audit_path = Path(str(config["paths"]["phase2_length_policy_audit_path"]))
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        final = audit.get("final_decision", {})
        eval_budget = final.get("final_chosen_eval_max_new_tokens")
        sft_length = final.get("final_chosen_sft_max_sequence_length")
        if eval_budget not in {4096, None} or sft_length not in {8192, None}:
            raise ValueError(f"Phase 2 length policy audit is incompatible: {audit_path}")


def _select_records(config: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    records = load_jsonl(config["paths"]["sft_view_path"])
    expected = int(config["expected"]["source_sft_count"])
    if len(records) != expected:
        raise ValueError(f"SFT view count mismatch: expected {expected}, got {len(records)}")
    if mode == "smoke":
        return records[: int(config["smoke"]["sample_count"])]
    return records


def _build_dataset(config: dict[str, Any], tokenizer: Any, mode: str) -> tuple[SftTokenDataset, list[dict[str, Any]], dict[str, Any]]:
    records = _select_records(config, mode)
    max_len = int(config["sft_sequence"]["max_sequence_length"])
    examples: list[SequenceExample] = []
    manifest: list[dict[str, Any]] = []
    for record in records:
        prompt = str(config["prompt"]["template"]).format(problem=str(record["prompt"]))
        example = build_sft_sequence(
            tokenizer,
            prompt=prompt,
            r1_generation=str(record["reasoning"]),
            reference_code=str(record["reference_code"]),
            max_sequence_length=max_len,
        )
        manifest.append(
            {
                "problem_id": record["problem_id"],
                "dropped": example.dropped,
                "drop_reason": example.drop_reason,
                "prompt_tokens": example.prompt_tokens,
                "original_response_tokens": example.original_response_tokens,
                "original_reasoning_tokens_before_code": example.original_reasoning_tokens,
                "retained_reasoning_tokens": example.retained_reasoning_tokens,
                "retained_reasoning_ratio": example.retained_reasoning_ratio,
                "full_sequence_tokens": example.full_sequence_tokens,
                "final_sequence_tokens": example.final_sequence_tokens,
                "full_fit": example.full_fit,
                "reasoning_truncated": example.reasoning_truncated,
                "code_locatable": example.code_locatable,
                "code_location_method": example.code_location_method,
                "construction_hash": stable_hash(
                    {
                        "problem_id": record["problem_id"],
                        "target": "r1_generation",
                        "final_sequence_tokens": example.final_sequence_tokens,
                        "dropped": example.dropped,
                        "drop_reason": example.drop_reason,
                    }
                ),
            }
        )
        if not example.dropped:
            examples.append(example)
    stats = {
        "source_records": len(records),
        "usable_records": len(examples),
        "dropped_records": len(records) - len(examples),
        "dropped_unlocatable": sum(1 for item in manifest if item["drop_reason"] == "unlocatable_final_code"),
        "dropped_prompt_plus_preserved_code_overflow": sum(
            1 for item in manifest if item["drop_reason"] == "prompt_plus_preserved_code_overflow"
        ),
        "full_fit_count": sum(1 for item in manifest if item["full_fit"] and not item["dropped"]),
        "reasoning_truncated_count": sum(1 for item in manifest if item["reasoning_truncated"]),
    }
    if not examples:
        raise RuntimeError("No usable SFT examples after canonical sequence construction")
    return SftTokenDataset(examples), manifest, stats


def _training_args(config: dict[str, Any], mode: str):
    from transformers import TrainingArguments

    training = dict(config["training"])
    output_dir = str(config["smoke"]["output_dir"] if mode == "smoke" else training["output_dir"])
    kwargs = {
        "output_dir": output_dir,
        "per_device_train_batch_size": int(training["per_device_train_batch_size"]),
        "gradient_accumulation_steps": int(training["gradient_accumulation_steps"]),
        "learning_rate": float(training["learning_rate"]),
        # Transformers 5.x removed TrainingArguments.warmup_ratio. A
        # fractional warmup_steps value is interpreted as a ratio of the
        # eventual total training steps.
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
        kwargs["max_steps"] = int(config["smoke"]["max_steps"])
        kwargs["save_strategy"] = "steps"
        kwargs["save_steps"] = int(config["smoke"]["max_steps"])
    else:
        kwargs["num_train_epochs"] = float(training["num_train_epochs"])
    return TrainingArguments(**kwargs)


def _apply_lora(config: dict[str, Any], model: Any) -> Any:
    from peft import LoraConfig, TaskType, get_peft_model

    lora = config["lora"]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora["r"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        target_modules=list(lora["target_modules"]),
    )
    return get_peft_model(model, peft_config)


def _validate_saved_adapter(output_dir: Path, model: Any) -> dict[str, Any]:
    from peft import PeftConfig

    PeftConfig.from_pretrained(str(output_dir))
    has_weights = any((output_dir / name).exists() for name in ("adapter_model.safetensors", "adapter_model.bin"))
    if not has_weights:
        raise FileNotFoundError(f"No PEFT adapter weights found in {output_dir}")
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {"adapter_config_load_check": "passed", "adapter_weights_present": True, "trainable_parameters": int(trainable)}


def train_sft(config_path: str, mode: str) -> dict[str, Any]:
    config = load_config(config_path)
    _validate_phase_gate(config)
    set_deterministic_seed(int(config["training"]["seed"]))
    model, tokenizer = _load_model_and_tokenizer(config)
    dataset, manifest, dataset_stats = _build_dataset(config, tokenizer, mode)
    model = _apply_lora(config, model)

    from transformers import Trainer

    args = _training_args(config, mode)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollatorForResponseOnlySft(int(tokenizer.pad_token_id)),
    )
    train_result = trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    load_check = _validate_saved_adapter(Path(args.output_dir), trainer.model)

    output_root = Path(str(config["paths"]["artifact_root"])) / mode
    manifest_path = output_root / "sft_sequence_manifest.jsonl"
    report_path = output_root / "sft_train_report.json"
    write_jsonl(manifest_path, manifest)
    report = {
        "phase": "phase3_reasoning_sft",
        "mode": mode,
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_file": config_path,
        "config_hash": file_sha256(config_path),
        "git": {
            "commit": git_command(["rev-parse", "HEAD"]),
            "dirty": bool(git_status_short()),
            "status_short": git_status_short(),
        },
        "model": {
            "repo_id": config["model"]["repo_id"],
            "model_revision": config["model"]["model_revision"],
            "tokenizer_revision": config["model"]["tokenizer_revision"],
            "dtype": config["model"]["dtype"],
        },
        "canonical_target": {
            "source_field": "reasoning",
            "source_semantics": "OpenCodeReasoning-2 r1_generation",
            "reference_code_appended": False,
            "reference_code_role": "locate_and_verify_only",
        },
        "sft_sequence": dict(config["sft_sequence"]),
        "lora": dict(config["lora"]),
        "training_args": args.to_dict(),
        "dataset_stats": dataset_stats,
        "manifest": {"path": str(manifest_path), "hash": file_sha256(manifest_path), "count": len(manifest)},
        "checkpoint": {"path": str(args.output_dir), **load_check},
        "train_metrics": train_result.metrics,
        "runtime": runtime_info(),
    }
    write_json(report_path, report)
    return report


def main() -> None:
    args = parse_args()
    report = train_sft(args.config, args.mode)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
