from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data.schemas import stable_hash
from eval.phase2_common import file_sha256, load_jsonl
from eval.sft_v3_sequence import SequenceExampleV3, build_sft_sequence_v3


DEFAULT_CANDIDATE_NAME = "clean_joint_response_le_4096_code_start_le_3072"


def load_frozen_candidate_manifest(config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    audit_path = Path(str(config["paths"]["construction_audit_path"])).expanduser()
    manifest_path = Path(str(config["paths"]["candidate_manifest_path"])).expanduser()
    if not audit_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("Run the v3 candidate diagnostic before training")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    manifest = load_jsonl(manifest_path)
    if audit.get("status") != "completed":
        raise ValueError("v3 candidate audit is not completed")
    if audit.get("manifest", {}).get("count") != len(manifest):
        raise ValueError("v3 candidate manifest count does not match audit")
    expected_manifest_hash = audit.get("manifest", {}).get("hash")
    if expected_manifest_hash and file_sha256(manifest_path) != expected_manifest_hash:
        raise ValueError("v3 candidate manifest hash does not match the frozen audit")
    if audit.get("default_candidate", {}).get("name") != DEFAULT_CANDIDATE_NAME:
        raise ValueError("v3 default candidate name changed unexpectedly")
    if not audit.get("gate", {}).get("formal_training_allowed"):
        raise ValueError("v3 candidate audit does not allow formal training")
    return audit, manifest


def select_candidate_records(
    config: dict[str, Any],
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    audit, manifest = load_frozen_candidate_manifest(config)
    source_path = Path(str(config["paths"]["sft_view_path"])).expanduser()
    expected_source_hash = audit.get("source_view", {}).get("hash")
    if expected_source_hash and file_sha256(source_path) != expected_source_hash:
        raise ValueError("v3 source view hash does not match the frozen audit")
    manifest_by_id = {str(row["problem_id"]): row for row in manifest}
    candidate_name = DEFAULT_CANDIDATE_NAME
    selected: list[dict[str, Any]] = []
    selected_manifest: list[dict[str, Any]] = []
    for record in records:
        row = manifest_by_id.get(str(record["problem_id"]))
        if row is None:
            raise ValueError(f"Missing v3 manifest row for {record['problem_id']}")
        if row["source_response_hash"] != stable_hash(str(record["reasoning"])):
            raise ValueError(f"Source response hash mismatch for {record['problem_id']}")
        if bool(row["candidate_membership"].get(candidate_name, False)):
            selected.append(record)
            selected_manifest.append(row)
    dataset_hash = stable_hash([row["row_hash"] for row in selected_manifest])
    if dataset_hash != audit.get("default_candidate", {}).get("dataset_hash"):
        raise ValueError("Selected v3 candidate dataset hash differs from frozen audit")
    if len(selected) != int(audit["default_candidate"]["count"]):
        raise ValueError("Selected v3 candidate count differs from frozen audit")
    return selected, audit, selected_manifest


def build_v3_examples(
    config: dict[str, Any],
    tokenizer: Any,
    records: list[dict[str, Any]],
) -> tuple[list[SequenceExampleV3], list[dict[str, Any]], dict[str, Any]]:
    selected, audit, selected_manifest = select_candidate_records(config, records)
    examples: list[SequenceExampleV3] = []
    construction_manifest: list[dict[str, Any]] = []
    for record, candidate_row in zip(selected, selected_manifest):
        prompt = str(config["prompt"]["template"]).format(problem=str(record["prompt"]))
        example = build_sft_sequence_v3(
            tokenizer,
            prompt=prompt,
            r1_generation=str(record["reasoning"]),
            reference_code=str(record["reference_code"]),
            max_sequence_length=int(config["sft_sequence"]["max_sequence_length"]),
            max_response_tokens_including_eos=int(
                config["sft_sequence"]["max_response_tokens_including_eos"]
            ),
        )
        if example.dropped:
            raise ValueError(f"Frozen v3 candidate failed natural construction: {record['problem_id']}")
        if example.artificial_truncation or example.reference_code_appended or not example.eos_supervised:
            raise ValueError(f"v3 construction invariant failed: {record['problem_id']}")
        row = dict(candidate_row)
        row.update(
            {
                "final_response_tokens_including_eos": example.response_tokens_including_eos,
                "final_sequence_tokens": example.final_sequence_tokens,
                "artificial_truncation": example.artificial_truncation,
                "final_code_preserved": example.final_code_preserved,
                "eos_supervised": example.eos_supervised,
                "construction_hash": stable_hash(
                    {
                        "problem_id": record["problem_id"],
                        "input_ids": example.input_ids,
                        "labels": example.labels,
                        "source_row_hash": candidate_row["row_hash"],
                    }
                ),
            }
        )
        construction_manifest.append(row)
        examples.append(example)
    if len(examples) != len(selected):
        raise AssertionError("v3 candidate construction count changed")
    stats = {
        "source_records": len(records),
        "candidate_records": len(selected),
        "constructed_records": len(examples),
        "dataset_hash": audit["default_candidate"]["dataset_hash"],
        "candidate_rule": audit["default_candidate"]["rule"],
        "artificial_truncation_count": sum(int(item.artificial_truncation) for item in examples),
        "reference_code_appended_count": sum(int(item.reference_code_appended) for item in examples),
        "eos_supervised_count": sum(int(item.eos_supervised) for item in examples),
        "max_response_tokens_including_eos": max(item.response_tokens_including_eos for item in examples),
        "max_sequence_tokens": max(item.final_sequence_tokens for item in examples),
    }
    return examples, construction_manifest, stats
