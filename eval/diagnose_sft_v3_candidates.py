from __future__ import annotations

import argparse
import json
from collections import Counter
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
    numeric_stats,
    pretrained_load_reference,
    runtime_info,
    write_json,
    write_jsonl,
)
from eval.sft_v2_sequence import analyze_response_structure
from eval.sft_sequence import locate_reference_code


RESPONSE_THRESHOLDS = (1024, 2048, 3072, 4096)
CODE_START_THRESHOLDS = (1024, 2048, 3072, 4096)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose natural-length Phase 3-v3 SFT candidates.")
    parser.add_argument("--config", default="configs/sft_v3.yaml")
    return parser.parse_args()


def _load_tokenizer(config: dict[str, Any]):
    from transformers import AutoTokenizer

    source, kwargs = pretrained_load_reference(config, "tokenizer_revision")
    tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
    if tokenizer.eos_token_id is None:
        raise ValueError("v3 diagnostic requires tokenizer.eos_token_id")
    return tokenizer


def _tokenize(tokenizer: Any, text: str, *, offsets: bool = False) -> tuple[list[int], list[tuple[int, int]] | None]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=offsets,
    )
    ids = [int(value) for value in encoded["input_ids"]]
    raw_offsets = encoded.get("offset_mapping") if offsets else None
    mapped = [tuple(map(int, value)) for value in raw_offsets] if raw_offsets is not None else None
    return ids, mapped


def _first_code_token(offsets: list[tuple[int, int]] | None, start: int, end: int) -> int | None:
    if offsets is None:
        return None
    for index, (token_start, token_end) in enumerate(offsets):
        if token_end > start and token_start < end:
            return index
    return None


def _source_split(record: dict[str, Any]) -> str:
    source_id = str(record.get("source_id", ""))
    return source_id.split(":", 1)[0] if ":" in source_id else source_id or "unknown"


def _distribution(records: list[dict[str, Any]]) -> dict[str, Any]:
    source_split_counts = Counter(f"{record['source']}/{_source_split(record)}" for record in records)
    source_counts = Counter(str(record["source"]) for record in records)
    difficulty_counts = Counter(str(record["difficulty"]) for record in records)
    total = len(records)

    def rates(counter: Counter[str]) -> dict[str, float]:
        return {key: value / total if total else 0.0 for key, value in sorted(counter.items())}

    return {
        "count": total,
        "source_split_counts": dict(sorted(source_split_counts.items())),
        "source_split_rates": rates(source_split_counts),
        "source_counts": dict(sorted(source_counts.items())),
        "source_rates": rates(source_counts),
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "difficulty_rates": rates(difficulty_counts),
    }


def _distribution_shift(candidate: dict[str, Any], original: dict[str, Any]) -> dict[str, Any]:
    def delta(candidate_rates: dict[str, float], original_rates: dict[str, float]) -> dict[str, float]:
        keys = set(candidate_rates) | set(original_rates)
        return {key: candidate_rates.get(key, 0.0) - original_rates.get(key, 0.0) for key in sorted(keys)}

    return {
        "source_split_rate_delta": delta(candidate["source_split_rates"], original["source_split_rates"]),
        "source_rate_delta": delta(candidate["source_rates"], original["source_rates"]),
        "difficulty_rate_delta": delta(candidate["difficulty_rates"], original["difficulty_rates"]),
    }


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("phase") != "phase3_reasoning_sft_v3":
        raise ValueError(f"Unexpected v3 config phase: {config.get('phase')}")
    if int(config["expected"]["source_sft_count"]) != 3051:
        raise ValueError("v3 diagnostic must cover exactly 3051 source SFT records")
    if int(config["expected"]["max_sequence_length"]) != 8192:
        raise ValueError("v3 max sequence length must be 8192")
    if int(config["expected"]["max_response_tokens_including_eos"]) != 4096:
        raise ValueError("v3 response budget including EOS must be 4096")
    if int(config["expected"]["eval_max_new_tokens"]) != 4096:
        raise ValueError("v3 frozen evaluation budget must be 4096")
    if config["sft_sequence"].get("truncation_policy") != "none":
        raise ValueError("v3 must not use truncation")


def _threshold_stats(values: list[int], thresholds: tuple[int, ...]) -> dict[str, Any]:
    total = len(values)
    return {
        str(threshold): {
            "count": sum(value <= threshold for value in values),
            "rate": sum(value <= threshold for value in values) / total if total else 0.0,
        }
        for threshold in thresholds
    }


def _candidate_summary(
    name: str,
    records: list[dict[str, Any]],
    row_by_id: dict[str, dict[str, Any]],
    original_distribution: dict[str, Any],
) -> dict[str, Any]:
    selected = [record for record in records if row_by_id[record["problem_id"]]["candidate_membership"].get(name, False)]
    distribution = _distribution(selected)
    return {
        "name": name,
        "count": len(selected),
        "rate": len(selected) / len(records) if records else 0.0,
        "distribution": distribution,
        "distribution_shift_vs_original": _distribution_shift(distribution, original_distribution),
    }


def run_diagnostic(config_path: str) -> dict[str, Any]:
    config = load_config(config_path)
    _validate_config(config)
    source_path = Path(str(config["paths"]["sft_view_path"])).expanduser()
    records = load_jsonl(source_path)
    expected = int(config["expected"]["source_sft_count"])
    if len(records) != expected:
        raise ValueError(f"SFT source count mismatch: expected {expected}, got {len(records)}")

    tokenizer = _load_tokenizer(config)
    max_response = int(config["expected"]["max_response_tokens_including_eos"])
    max_sequence = int(config["expected"]["max_sequence_length"])
    manifest: list[dict[str, Any]] = []
    response_lengths: list[int] = []
    code_starts: list[int] = []
    code_lengths: list[int] = []
    row_by_id: dict[str, dict[str, Any]] = {}
    exact_count = 0
    normalized_count = 0
    unlocatable_count = 0
    structure_gate_count = 0
    total_sequence_ok_count = 0

    for index, record in enumerate(records, start=1):
        response = str(record["reasoning"])
        reference_code = str(record["reference_code"])
        prompt = str(config["prompt"]["template"]).format(problem=str(record["prompt"]))
        response_ids, response_offsets = _tokenize(tokenizer, response, offsets=True)
        prompt_ids, _ = _tokenize(tokenizer, prompt)
        response_including_eos = len(response_ids) + 1
        location = locate_reference_code(response, reference_code)
        if location is None:
            unlocatable_count += 1
        elif location.method == "exact":
            exact_count += 1
        elif location.method == "normalized_whitespace":
            normalized_count += 1
        code_start = _first_code_token(response_offsets, location.start_char, location.end_char) if location else None
        if code_start is None and location is not None:
            code_start = len(_tokenize(tokenizer, response[: location.start_char])[0])
        code_start = int(code_start) if code_start is not None else -1
        code_length = len(_tokenize(tokenizer, reference_code)[0])
        structure = analyze_response_structure(response, reference_code)
        structure_gate = bool(structure.valid_structure)
        sequence_ok = bool(len(prompt_ids) + response_including_eos <= max_sequence)
        natural_response_ok = response_including_eos <= max_response
        if structure_gate:
            structure_gate_count += 1
        if sequence_ok:
            total_sequence_ok_count += 1
        candidate_membership = {
            "clean_response_le_1024": bool(structure_gate and sequence_ok and response_including_eos <= 1024),
            "clean_response_le_2048": bool(structure_gate and sequence_ok and response_including_eos <= 2048),
            "clean_response_le_3072": bool(structure_gate and sequence_ok and response_including_eos <= 3072),
            "clean_response_le_4096": bool(structure_gate and sequence_ok and response_including_eos <= 4096),
            "clean_code_start_le_1024": bool(
                structure_gate and sequence_ok and natural_response_ok and 0 <= code_start <= 1024
            ),
            "clean_code_start_le_2048": bool(
                structure_gate and sequence_ok and natural_response_ok and 0 <= code_start <= 2048
            ),
            "clean_code_start_le_3072": bool(
                structure_gate and sequence_ok and natural_response_ok and 0 <= code_start <= 3072
            ),
            "clean_code_start_le_4096": bool(
                structure_gate and sequence_ok and natural_response_ok and 0 <= code_start <= 4096
            ),
            "clean_joint_response_le_4096_code_start_le_3072": bool(
                structure_gate and sequence_ok and natural_response_ok and 0 <= code_start <= 3072
            ),
            "clean_joint_response_le_4096_code_start_le_2048": bool(
                structure_gate and sequence_ok and natural_response_ok and 0 <= code_start <= 2048
            ),
            "clean_joint_response_le_3072_code_start_le_2048": bool(
                structure_gate and sequence_ok and response_including_eos <= 3072 and 0 <= code_start <= 2048
            ),
        }
        row = {
            "problem_id": record["problem_id"],
            "source": record["source"],
            "source_split": _source_split(record),
            "difficulty": record["difficulty"],
            "source_response_hash": stable_hash(response),
            "reference_code_hash": stable_hash(reference_code),
            "original_response_tokens_including_eos": response_including_eos,
            "final_code_start_token": code_start,
            "code_length_tokens": code_length,
            "prompt_tokens": len(prompt_ids),
            "total_sequence_tokens_including_eos": len(prompt_ids) + response_including_eos,
            "code_location_method": location.method if location else None,
            "starts_with_think": structure.starts_with_think,
            "has_matching_closed_think": structure.has_matching_closed_think,
            "code_fenced": structure.fenced_code,
            "structure_gate": structure_gate,
            "natural_response_budget_ok": natural_response_ok,
            "total_sequence_budget_ok": sequence_ok,
            "artificial_truncation": False,
            "reference_code_appended": False,
            "eos_supervised_if_selected": True,
            "candidate_membership": candidate_membership,
            "row_hash": stable_hash(
                {
                    "problem_id": record["problem_id"],
                    "response_hash": stable_hash(response),
                    "reference_code_hash": stable_hash(reference_code),
                    "candidate_membership": candidate_membership,
                }
            ),
        }
        manifest.append(row)
        row_by_id[record["problem_id"]] = row
        response_lengths.append(response_including_eos)
        code_starts.append(code_start)
        code_lengths.append(code_length)
        if index == 1 or index % 250 == 0 or index == len(records):
            print(f"Diagnosed {index}/{len(records)} v3 natural-length records", flush=True)

    original_distribution = _distribution(records)
    candidate_names = [
        "clean_response_le_1024",
        "clean_response_le_2048",
        "clean_response_le_3072",
        "clean_response_le_4096",
        "clean_code_start_le_1024",
        "clean_code_start_le_2048",
        "clean_code_start_le_3072",
        "clean_code_start_le_4096",
        "clean_joint_response_le_4096_code_start_le_3072",
        "clean_joint_response_le_4096_code_start_le_2048",
        "clean_joint_response_le_3072_code_start_le_2048",
    ]
    candidates = {
        name: _candidate_summary(name, records, row_by_id, original_distribution) for name in candidate_names
    }
    default_name = "clean_joint_response_le_4096_code_start_le_3072"
    default_count = candidates[default_name]["count"]
    artifact_root = Path(str(config["paths"]["artifact_root"])).expanduser()
    manifest_path = Path(str(config["paths"]["candidate_manifest_path"])).expanduser()
    audit_path = Path(str(config["paths"]["construction_audit_path"])).expanduser()
    write_jsonl(manifest_path, manifest)

    selected_rows = [row for row in manifest if row["candidate_membership"][default_name]]
    dataset_hash = stable_hash([row["row_hash"] for row in selected_rows])
    audit = {
        "phase": "phase3_reasoning_sft_v3",
        "step": "sft_v3_natural_length_candidate_diagnostic",
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"path": str(Path(config_path).resolve()), "hash": file_sha256(config_path)},
        "source_view": {"path": str(source_path), "hash": file_sha256(source_path), "count": len(records)},
        "model": {
            "repo_id": config["model"]["repo_id"],
            "model_revision": config["model"]["model_revision"],
            "tokenizer_revision": config["model"]["tokenizer_revision"],
        },
        "policy": {
            "max_sequence_length": max_sequence,
            "max_response_tokens_including_eos": max_response,
            "frozen_eval_max_new_tokens": int(config["expected"]["eval_max_new_tokens"]),
            "truncation": False,
            "reconstruction": False,
            "ocr_code_appended": False,
        },
        "raw_length_statistics": {
            "response_including_eos": numeric_stats(response_lengths, (50, 90, 95, 99)),
            "code_start_token": numeric_stats([value for value in code_starts if value >= 0], (50, 90, 95, 99)),
            "code_length_tokens": numeric_stats(code_lengths, (50, 90, 95, 99)),
            "response_thresholds": _threshold_stats(response_lengths, RESPONSE_THRESHOLDS),
            "code_start_thresholds": _threshold_stats(
                [value for value in code_starts if value >= 0], CODE_START_THRESHOLDS
            ),
        },
        "raw_location_counts": {
            "exact": exact_count,
            "normalized_whitespace": normalized_count,
            "unlocatable": unlocatable_count,
        },
        "raw_structure_counts": {
            "structure_gate_pass_count": structure_gate_count,
            "structure_gate_pass_rate": structure_gate_count / len(records) if records else 0.0,
            "total_sequence_budget_ok_count": total_sequence_ok_count,
            "total_sequence_budget_ok_rate": total_sequence_ok_count / len(records) if records else 0.0,
            "starts_with_think_count": sum(1 for row in manifest if row["starts_with_think"]),
            "closed_think_count": sum(1 for row in manifest if row["has_matching_closed_think"]),
            "fenced_code_count": sum(1 for row in manifest if row["code_fenced"]),
        },
        "original_sft_distribution": original_distribution,
        "candidate_subsets": candidates,
        "default_candidate": {
            "name": default_name,
            "rule": "structure_gate and response_including_eos<=4096 and code_start_token<=3072 and total_sequence<=8192",
            "count": default_count,
            "rate": default_count / len(records) if records else 0.0,
            "dataset_hash": dataset_hash,
            "selection_status": "available" if default_count else "stop_insufficient_candidates",
        },
        "gate": {
            "diagnostic_complete": True,
            "natural_length_only": True,
            "no_artificial_truncation": True,
            "no_fuzzy_reconstruction": True,
            "no_duplicate_code_append": True,
            "default_candidate_requires_manual_review": True,
            "formal_training_allowed": bool(default_count),
            "dpo_grpo_ready": False,
        },
        "manifest": {"path": str(manifest_path), "hash": file_sha256(manifest_path), "count": len(manifest)},
        "artifact_root": str(artifact_root),
        "v1_v2_artifacts_touched": False,
        "runtime": runtime_info(),
        "git": {
            "commit": git_command(["rev-parse", "HEAD"]),
            "dirty": bool(git_status_short()),
            "status_short": git_status_short(),
        },
    }
    write_json(audit_path, audit)
    return audit


def main() -> None:
    args = parse_args()
    print(json.dumps(run_diagnostic(args.config), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
