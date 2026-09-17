from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data.config import load_config
from data.schemas import stable_hash
from eval.phase2_common import file_sha256
from sft.v3_data import build_v3_examples


IMMUTABLE_CONFIG_SECTIONS = ("model", "prompt", "sft_sequence", "lora")
IMMUTABLE_TRAINING_KEYS = (
    "seed",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "num_train_epochs",
    "warmup_ratio",
    "weight_decay",
    "logging_steps",
    "save_strategy",
    "bf16",
    "gradient_checkpointing",
    "optim",
    "report_to",
)


def controlled_config_diff(v4_config: dict[str, Any], v3_config: dict[str, Any]) -> dict[str, Any]:
    section_matches = {
        section: v4_config.get(section) == v3_config.get(section) for section in IMMUTABLE_CONFIG_SECTIONS
    }
    training_matches = {
        key: v4_config.get("training", {}).get(key) == v3_config.get("training", {}).get(key)
        for key in IMMUTABLE_TRAINING_KEYS
    }
    changed_training = {
        "learning_rate": {
            "v3": v3_config.get("training", {}).get("learning_rate"),
            "v4": v4_config.get("training", {}).get("learning_rate"),
        },
        "output_dir": {
            "v3": v3_config.get("training", {}).get("output_dir"),
            "v4": v4_config.get("training", {}).get("output_dir"),
        },
    }
    return {
        "immutable_sections_match": section_matches,
        "immutable_training_keys_match": training_matches,
        "only_training_hyperparameter_change": all(section_matches.values())
        and all(training_matches.values())
        and changed_training["learning_rate"]["v3"] == 0.0001
        and changed_training["learning_rate"]["v4"] == 0.00005
        and changed_training["output_dir"]["v3"] != changed_training["output_dir"]["v4"],
        "changed_training": changed_training,
    }


def _load_manifest(config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    audit_path = Path(str(config["paths"]["construction_audit_path"])).expanduser()
    manifest_path = Path(str(config["paths"]["candidate_manifest_path"])).expanduser()
    frozen = config["frozen_v3_candidate"]
    if not audit_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("Frozen v3 construction artifacts are missing")
    if file_sha256(manifest_path) != str(frozen["manifest_hash"]):
        raise ValueError("v4 candidate manifest hash differs from frozen v3 manifest")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    manifest = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
    if audit.get("status") != "completed":
        raise ValueError("Frozen v3 candidate audit is incomplete")
    if audit.get("manifest", {}).get("hash") != str(frozen["manifest_hash"]):
        raise ValueError("Frozen v3 audit manifest hash differs from v4 expected hash")
    if audit.get("default_candidate", {}).get("dataset_hash") != str(frozen["dataset_hash"]):
        raise ValueError("Frozen v3 dataset hash differs from v4 expected hash")
    if int(audit.get("default_candidate", {}).get("count", -1)) != int(frozen["candidate_count"]):
        raise ValueError("Frozen v3 candidate count differs from v4 expected count")
    return audit, manifest


def validate_frozen_v3_control(config: dict[str, Any]) -> dict[str, Any]:
    v3_config_path = Path(str(config["paths"]["v3_config_path"])).expanduser()
    v3_config = load_config(v3_config_path)
    frozen = config["frozen_v3_candidate"]
    if file_sha256(v3_config_path) != str(frozen["v3_config_hash"]):
        raise ValueError("v3 config hash changed after construction freeze")
    diff = controlled_config_diff(config, v3_config)
    if not diff["only_training_hyperparameter_change"]:
        raise ValueError(f"v4 controlled comparison differs from v3 beyond learning rate: {diff}")
    audit, manifest = _load_manifest(config)
    candidate_name = str(frozen["candidate_name"])
    selected_ids = sorted(
        str(row["problem_id"])
        for row in manifest
        if bool(row.get("candidate_membership", {}).get(candidate_name, False))
    )
    if len(selected_ids) != int(frozen["candidate_count"]):
        raise ValueError("v4 selected candidate count differs from frozen v3")
    if stable_hash(selected_ids) != str(frozen["candidate_ids_hash"]):
        raise ValueError("v4 candidate problem-ID hash differs from frozen v3")
    return {
        "v3_config_path": str(v3_config_path),
        "v3_config_hash": file_sha256(v3_config_path),
        "v3_commit_sha": str(frozen["v3_commit_sha"]),
        "candidate_name": candidate_name,
        "candidate_count": len(selected_ids),
        "candidate_ids_hash": stable_hash(selected_ids),
        "dataset_hash": str(frozen["dataset_hash"]),
        "manifest_hash": str(frozen["manifest_hash"]),
        "audit_hash": file_sha256(config["paths"]["construction_audit_path"]),
        "controlled_config_diff": diff,
        "audit_status": audit.get("status"),
        "dataset_distribution": {
            "original_sft": audit["original_sft_distribution"],
            "selected_candidate": audit["candidate_subsets"][candidate_name]["distribution"],
            "selected_shift_vs_original": audit["candidate_subsets"][candidate_name][
                "distribution_shift_vs_original"
            ],
        },
    }


def build_v4_examples(
    config: dict[str, Any], tokenizer: Any, records: list[dict[str, Any]]
) -> tuple[list[Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    freeze = validate_frozen_v3_control(config)
    examples, manifest, stats = build_v3_examples(config, tokenizer, records)
    actual_ids = sorted(str(row["problem_id"]) for row in manifest)
    if len(actual_ids) != freeze["candidate_count"] or stable_hash(actual_ids) != freeze["candidate_ids_hash"]:
        raise ValueError("v4 constructed problem-ID set differs from frozen v3 candidate set")
    if stats["dataset_hash"] != freeze["dataset_hash"]:
        raise ValueError("v4 constructed dataset hash differs from frozen v3")
    stats = dict(stats)
    stats["training_subset"] = "natural_length_filtered"
    stats["candidate_ids_hash"] = freeze["candidate_ids_hash"]
    stats["manifest_hash"] = freeze["manifest_hash"]
    return examples, manifest, stats, freeze
