from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data.config import load_config
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
from sft.v2_data import build_v2_examples_and_manifest, frozen_dataset_hash


class SftTokenDatasetV2:
    def __init__(self, examples: list[Any]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        example = self.examples[index]
        return {"input_ids": example.input_ids, "labels": example.labels}


class DataCollatorForResponseOnlySftV2:
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
    parser = argparse.ArgumentParser(description="Run Phase 3-v2 repaired Reasoning SFT.")
    parser.add_argument("--config", default="configs/sft_v2.yaml")
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
        raise RuntimeError("CUDA GPU is required for Phase 3-v2 SFT")
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


def _validate_phase_gate(config: dict[str, Any], config_path: str) -> dict[str, Any]:
    if config.get("phase") != "phase3_reasoning_sft_v2":
        raise ValueError(f"Unexpected v2 SFT config phase: {config.get('phase')}")
    if int(config["sft_sequence"]["max_sequence_length"]) != 8192:
        raise ValueError("Phase 3-v2 SFT requires max_sequence_length=8192")
    if int(config["sft_sequence"]["max_response_tokens_including_eos"]) != 4096:
        raise ValueError("Phase 3-v2 SFT requires response budget including EOS=4096")
    if int(config["expected"]["max_sequence_length"]) != 8192:
        raise ValueError("Phase 3-v2 expected max_sequence_length must be 8192")
    if int(config["expected"]["max_response_tokens_including_eos"]) != 4096:
        raise ValueError("Phase 3-v2 expected response budget including EOS must be 4096")
    if int(config["expected"]["eval_max_new_tokens"]) != 4096:
        raise ValueError("Phase 3-v2 SFT requires frozen eval max_new_tokens=4096")
    audit_path = Path(str(config["paths"]["construction_audit_path"])).expanduser()
    if not audit_path.exists():
        raise FileNotFoundError(f"Run the v2 construction diagnostic first: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "completed" or not audit.get("gate", {}).get("construction_gate_passed"):
        raise ValueError(f"Phase 3-v2 construction Gate is not passed: {audit_path}")
    if audit.get("config", {}).get("hash") != file_sha256(config_path):
        raise ValueError(f"Construction audit config hash does not match {config_path}")
    return audit


def _select_records(config: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    records = load_jsonl(config["paths"]["sft_view_path"])
    expected = int(config["expected"]["source_sft_count"])
    if len(records) != expected:
        raise ValueError(f"SFT view count mismatch: expected {expected}, got {len(records)}")
    if mode == "smoke":
        return records[: int(config["smoke"]["sample_count"])]
    return records


def _training_args(config: dict[str, Any], mode: str):
    from transformers import TrainingArguments

    training = dict(config["training"])
    output_dir = str(config["smoke"]["output_dir"] if mode == "smoke" else training["output_dir"])
    kwargs = {
        "output_dir": output_dir,
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

    peft_config = PeftConfig.from_pretrained(str(output_dir))
    has_weights = any((output_dir / name).exists() for name in ("adapter_model.safetensors", "adapter_model.bin"))
    if not has_weights:
        raise FileNotFoundError(f"No PEFT adapter weights found in {output_dir}")
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {
        "adapter_config_load_check": "passed",
        "adapter_weights_present": True,
        "base_model_name_or_path": str(getattr(peft_config, "base_model_name_or_path", "")),
        "trainable_parameters": int(trainable),
    }


def train_sft_v2(config_path: str, mode: str) -> dict[str, Any]:
    config = load_config(config_path)
    construction_audit = _validate_phase_gate(config, config_path)
    output_dir = Path(
        str(config["smoke"]["output_dir"] if mode == "smoke" else config["training"]["output_dir"])
    ).expanduser()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing Phase 3-v2 checkpoint: {output_dir}")
    set_deterministic_seed(int(config["training"]["seed"]))
    model, tokenizer = _load_model_and_tokenizer(config)
    records = _select_records(config, mode)
    examples, manifest, dataset_stats = build_v2_examples_and_manifest(config, tokenizer, records)
    if not examples:
        raise RuntimeError("No usable v2 SFT examples after construction")
    dataset_hash = frozen_dataset_hash(manifest)
    if mode == "train" and dataset_hash != construction_audit.get("dataset_hash"):
        raise ValueError("Formal v2 training dataset hash differs from the frozen construction audit")
    model = _apply_lora(config, model)

    from transformers import Trainer

    args = _training_args(config, mode)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=SftTokenDatasetV2(examples),
        data_collator=DataCollatorForResponseOnlySftV2(int(tokenizer.pad_token_id)),
    )
    train_result = trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    load_check = _validate_saved_adapter(Path(args.output_dir), trainer.model)

    output_root = Path(str(config["paths"]["artifact_root"])).expanduser() / mode
    manifest_path = output_root / "sft_v2_sequence_manifest.jsonl"
    report_path = output_root / "sft_v2_train_report.json"
    write_jsonl(manifest_path, manifest)
    report = {
        "phase": "phase3_reasoning_sft_v2",
        "mode": mode,
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_file": str(Path(config_path).resolve()),
        "config_hash": file_sha256(config_path),
        "construction_audit": {
            "path": str(Path(str(config["paths"]["construction_audit_path"])).expanduser()),
            "hash": file_sha256(config["paths"]["construction_audit_path"]),
            "dataset_hash": construction_audit.get("dataset_hash"),
        },
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
        "dataset_hash": dataset_hash,
        "manifest": {"path": str(manifest_path), "hash": file_sha256(manifest_path), "count": len(manifest)},
        "checkpoint": {"path": str(args.output_dir), **load_check},
        "train_metrics": train_result.metrics,
        "runtime": runtime_info(),
        "v1_artifacts_overwritten": False,
    }
    write_json(report_path, report)
    return report


def main() -> None:
    args = parse_args()
    report = train_sft_v2(args.config, args.mode)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
