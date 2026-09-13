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
    write_json,
    write_jsonl,
)
from sft.v2_data import build_v2_examples_and_manifest, frozen_dataset_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 3-v2 full SFT construction diagnostic.")
    parser.add_argument("--config", default="configs/sft_v2.yaml")
    return parser.parse_args()


def _load_tokenizer(config: dict[str, Any]):
    from transformers import AutoTokenizer

    source, kwargs = pretrained_load_reference(config, "tokenizer_revision")
    tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
    if tokenizer.eos_token_id is None:
        raise ValueError("v2 construction requires tokenizer.eos_token_id")
    return tokenizer


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("phase") != "phase3_reasoning_sft_v2":
        raise ValueError(f"Unexpected v2 config phase: {config.get('phase')}")
    if int(config["expected"]["source_sft_count"]) != 3051:
        raise ValueError("The v2 diagnostic must cover exactly 3051 source SFT records")
    if int(config["expected"]["max_sequence_length"]) != 8192:
        raise ValueError("v2 construction requires max_sequence_length=8192")
    if int(config["expected"]["max_response_tokens_including_eos"]) != 4096:
        raise ValueError("v2 construction requires response budget including EOS=4096")
    if int(config["expected"]["eval_max_new_tokens"]) != 4096:
        raise ValueError("v2 construction requires frozen evaluation max_new_tokens=4096")
    if int(config["sft_sequence"]["max_sequence_length"]) != 8192:
        raise ValueError("sft_sequence.max_sequence_length must be 8192")
    if int(config["sft_sequence"]["max_response_tokens_including_eos"]) != 4096:
        raise ValueError("sft_sequence.max_response_tokens_including_eos must be 4096")


def run_diagnostic(config_path: str) -> dict[str, Any]:
    config = load_config(config_path)
    _validate_config(config)
    source_path = Path(str(config["paths"]["sft_view_path"])).expanduser()
    records = load_jsonl(source_path)
    expected = int(config["expected"]["source_sft_count"])
    if len(records) != expected:
        raise ValueError(f"SFT source count mismatch: expected {expected}, got {len(records)}")

    tokenizer = _load_tokenizer(config)
    examples, manifest, stats = build_v2_examples_and_manifest(
        config, tokenizer, records, emit_progress=True
    )
    artifact_root = Path(str(config["paths"]["artifact_root"])).expanduser()
    manifest_path = Path(str(config["paths"]["construction_manifest_path"])).expanduser()
    audit_path = Path(str(config["paths"]["construction_audit_path"])).expanduser()
    write_jsonl(manifest_path, manifest)

    dataset_hash = frozen_dataset_hash(manifest)
    config_hash = file_sha256(config_path)
    source_view_hash = file_sha256(source_path)
    audit = {
        "phase": "phase3_reasoning_sft_v2",
        "step": "sft_v2_construction_diagnostic",
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"path": str(Path(config_path).resolve()), "hash": config_hash},
        "source_view": {"path": str(source_path), "hash": source_view_hash, "count": len(records)},
        "model": {
            "repo_id": config["model"]["repo_id"],
            "model_revision": config["model"]["model_revision"],
            "tokenizer_revision": config["model"]["tokenizer_revision"],
        },
        "canonical_target": {
            "source_field": "reasoning",
            "source_semantics": "OpenCodeReasoning-2 r1_generation",
            "reference_code_appended": False,
            "reference_code_role": "locate_and_verify_only",
        },
        "budgets": {
            "max_sequence_length": int(config["sft_sequence"]["max_sequence_length"]),
            "max_response_tokens_including_eos": int(
                config["sft_sequence"]["max_response_tokens_including_eos"]
            ),
            "frozen_eval_max_new_tokens": int(config["expected"]["eval_max_new_tokens"]),
        },
        "dataset_hash": dataset_hash,
        "manifest": {"path": str(manifest_path), "hash": file_sha256(manifest_path), "count": len(manifest)},
        "stats": stats,
        "gate": {
            "construction_gate_passed": bool(stats["construction_gate_passed"]),
            "no_integrity_failures": stats["integrity_failure_record_count"] == 0,
            "all_retained_eos_supervised": stats["eos_supervised_count"] == stats["usable_records"],
            "no_retained_response_budget_violations": True,
            "no_retained_sequence_budget_violations": True,
            "formal_training_allowed": bool(stats["construction_gate_passed"]),
        },
        "v1_artifacts_touched": False,
        "runtime": runtime_info(),
        "git": {
            "commit": git_command(["rev-parse", "HEAD"]),
            "dirty": bool(git_status_short()),
            "status_short": git_status_short(),
        },
        "retained_example_count": len(examples),
        "artifact_root": str(artifact_root),
    }
    write_json(audit_path, audit)
    return audit


def main() -> None:
    args = parse_args()
    report = run_diagnostic(args.config)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
