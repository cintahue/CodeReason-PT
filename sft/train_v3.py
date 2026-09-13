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
from sft.v3_data import build_v3_examples


class SftTokenDatasetV3:
    def __init__(self, examples: list[Any]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        example = self.examples[index]
        return {"input_ids": example.input_ids, "labels": example.labels}


class DataCollatorForResponseOnlySftV3:
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
    parser = argparse.ArgumentParser(description="Run Phase 3-v3 natural-length Reasoning SFT.")
    parser.add_argument("--config", default="configs/sft_v3.yaml")
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
        raise RuntimeError("CUDA GPU is required for Phase 3-v3 SFT")
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
    if config.get("phase") != "phase3_reasoning_sft_v3":
        raise ValueError(f"Unexpected v3 SFT config phase: {config.get('phase')}")
    if int(config["sft_sequence"]["max_sequence_length"]) != 8192:
        raise ValueError("Phase 3-v3 requires max_sequence_length=8192")
    if int(config["sft_sequence"]["max_response_tokens_including_eos"]) != 4096:
        raise ValueError("Phase 3-v3 requires response budget including EOS=4096")
    if config["sft_sequence"].get("truncation_policy") != "none":
        raise ValueError("Phase 3-v3 must not use truncation")
    if int(config["expected"]["eval_max_new_tokens"]) != 4096:
        raise ValueError("Phase 3-v3 requires frozen eval max_new_tokens=4096")
    audit_path = Path(str(config["paths"]["construction_audit_path"])).expanduser()
    if not audit_path.exists():
        raise FileNotFoundError(f"Run v3 candidate diagnostic first: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "completed" or not audit.get("gate", {}).get("formal_training_allowed"):
        raise ValueError("v3 candidate diagnostic has not allowed formal training")
    if audit.get("config", {}).get("hash") != file_sha256(config_path):
        raise ValueError(f"v3 candidate audit config hash does not match {config_path}")
    if audit.get("default_candidate", {}).get("name") != "clean_joint_response_le_4096_code_start_le_3072":
        raise ValueError("v3 default candidate rule changed unexpectedly")
    return audit


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


def _validate_smoke_before_formal(config: dict[str, Any], config_path: str) -> None:
    smoke_report_path = Path(str(config["outputs"]["smoke_report"])).expanduser()
    smoke_load_path = Path(str(config["paths"]["artifact_root"])).expanduser() / "smoke" / "checkpoint_load_report.json"
    smoke_checkpoint = Path(str(config["smoke"]["output_dir"])).expanduser()
    if not smoke_report_path.exists() or not smoke_load_path.exists():
        raise ValueError("Run and load-check the v3 smoke checkpoint before formal training")
    smoke_report = json.loads(smoke_report_path.read_text(encoding="utf-8"))
    smoke_load = json.loads(smoke_load_path.read_text(encoding="utf-8"))
    if smoke_report.get("status") != "completed" or smoke_report.get("mode") != "smoke":
        raise ValueError("v3 smoke training report is incomplete")
    if smoke_report.get("config_hash") != file_sha256(config_path):
        raise ValueError("v3 smoke report config hash does not match the current config")
    if smoke_load.get("status") != "completed" or smoke_load.get("adapter_load_check") != "passed":
        raise ValueError("v3 smoke checkpoint load check is incomplete")
    if smoke_load.get("config_hash") != file_sha256(config_path):
        raise ValueError("v3 smoke checkpoint report config hash does not match the current config")
    if not smoke_checkpoint.exists():
        raise FileNotFoundError(f"Missing v3 smoke checkpoint: {smoke_checkpoint}")


def train_sft_v3(config_path: str, mode: str) -> dict[str, Any]:
    config = load_config(config_path)
    audit = _validate_phase_gate(config, config_path)
    if mode == "train":
        _validate_smoke_before_formal(config, config_path)
    output_dir = Path(
        str(config["smoke"]["output_dir"] if mode == "smoke" else config["training"]["output_dir"])
    ).expanduser()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing v3 checkpoint: {output_dir}")
    set_deterministic_seed(int(config["training"]["seed"]))
    model, tokenizer = _load_model_and_tokenizer(config)
    source_records = load_jsonl(config["paths"]["sft_view_path"])
    if len(source_records) != int(config["expected"]["source_sft_count"]):
        raise ValueError("v3 source SFT count mismatch")
    examples, manifest, dataset_stats = build_v3_examples(config, tokenizer, source_records)
    if mode == "smoke":
        sample_count = int(config["smoke"]["sample_count"])
        examples = examples[:sample_count]
        manifest = manifest[:sample_count]
        dataset_stats = dict(dataset_stats)
        dataset_stats["constructed_records"] = len(examples)
        dataset_stats["smoke_subset"] = True
    if not examples:
        raise RuntimeError("No usable v3 examples after natural-length selection")
    prompt_mask_ok_count = sum(
        int(all(label == -100 for label in example.labels[: example.prompt_tokens])) for example in examples
    )
    eos_supervised_count = sum(
        int(example.labels[-1] == int(tokenizer.eos_token_id)) for example in examples
    )
    model = _apply_lora(config, model)

    from transformers import Trainer

    args = _training_args(config, mode)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=SftTokenDatasetV3(examples),
        data_collator=DataCollatorForResponseOnlySftV3(int(tokenizer.pad_token_id)),
    )
    train_result = trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    load_check = _validate_saved_adapter(Path(args.output_dir), trainer.model)

    output_root = Path(str(config["paths"]["artifact_root"])).expanduser() / mode
    manifest_path = output_root / "sft_v3_sequence_manifest.jsonl"
    report_path = (
        Path(str(config["outputs"]["smoke_report"])).expanduser()
        if mode == "smoke"
        else Path(str(config["outputs"]["train_report"])).expanduser()
    )
    write_jsonl(manifest_path, manifest)
    report = {
        "phase": "phase3_reasoning_sft_v3",
        "mode": mode,
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_file": str(Path(config_path).resolve()),
        "config_hash": file_sha256(config_path),
        "candidate_audit": {
            "path": str(Path(str(config["paths"]["construction_audit_path"])).expanduser()),
            "hash": file_sha256(config["paths"]["construction_audit_path"]),
            "dataset_hash": audit["default_candidate"]["dataset_hash"],
            "candidate_count": audit["default_candidate"]["count"],
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
            "artificial_truncation": False,
        },
        "sft_sequence": dict(config["sft_sequence"]),
        "lora": dict(config["lora"]),
        "training_args": args.to_dict(),
        "dataset_stats": dataset_stats,
        "dataset_distribution": {
            "original_sft": audit["original_sft_distribution"],
            "selected_candidate": audit["candidate_subsets"][audit["default_candidate"]["name"]]["distribution"],
            "selected_shift_vs_original": audit["candidate_subsets"][audit["default_candidate"]["name"]][
                "distribution_shift_vs_original"
            ],
        },
        "dataset_hash": audit["default_candidate"]["dataset_hash"],
        "manifest": {"path": str(manifest_path), "hash": file_sha256(manifest_path), "count": len(manifest)},
        "label_invariant_audit": {
            "prompt_fully_masked_count": prompt_mask_ok_count,
            "prompt_fully_masked_rate": prompt_mask_ok_count / len(examples) if examples else 0.0,
            "eos_supervised_count": eos_supervised_count,
            "eos_supervised_rate": eos_supervised_count / len(examples) if examples else 0.0,
            "padding_policy": "collator masks all padding labels with -100",
        },
        "provenance": {
            "code_commit_sha": git_command(["rev-parse", "HEAD"]),
            "working_tree_clean_at_train": not bool(git_status_short()),
            "v2_provenance_commit_sha": "c0162a6d9910f48c69b2f6e32eb66432cb9e7d19",
        },
        "checkpoint": {"path": str(args.output_dir), **load_check},
        "train_metrics": train_result.metrics,
        "runtime": runtime_info(),
        "v1_v2_artifacts_overwritten": False,
    }
    write_json(report_path, report)
    return report


def main() -> None:
    args = parse_args()
    print(json.dumps(train_sft_v3(args.config, args.mode), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
