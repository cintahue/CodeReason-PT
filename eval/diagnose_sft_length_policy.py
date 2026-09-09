from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any

from data.schemas import stable_hash
from eval.phase2_common import (
    artifact_root,
    config_hash,
    file_sha256,
    git_command,
    git_status_short,
    load_jsonl,
    numeric_stats,
    package_versions,
    pretrained_load_reference,
    read_config,
    reports_dir,
    runtime_info,
    serialize_prompt,
    view_path,
    write_json,
    write_jsonl,
)
from eval.sft_sequence import build_sft_sequence, locate_reference_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose canonical-target, code-preserving SFT sequence construction at 8192 tokens."
    )
    parser.add_argument("--config", default="configs/base_eval_v2.yaml")
    return parser.parse_args()


def _load_tokenizer(config: dict[str, Any]):
    from transformers import AutoTokenizer

    source, kwargs = pretrained_load_reference(config, "tokenizer_revision")
    tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id for SFT sequence construction")
    return tokenizer


def _normalised_substring(response: str, solution: str) -> bool:
    from eval.sft_sequence import _mapped_substring_span

    return _mapped_substring_span(response, solution) is not None


def diagnose_sft_length_policy(config_path: str) -> dict[str, Any]:
    config = read_config(config_path)
    sft_config = config["sft_sequence"]
    max_sequence_length = int(sft_config["max_sequence_length"])
    if max_sequence_length != 8192:
        raise ValueError("Canonical Phase 3 SFT sequence diagnostic requires max_sequence_length=8192")
    tokenizer = _load_tokenizer(config)
    records = load_jsonl(view_path(config, "sft"))
    expected = int(config["expected"]["final_counts"]["sft"])
    if len(records) != expected:
        raise ValueError(f"Final SFT view count mismatch: expected {expected}, got {len(records)}")

    details: list[dict[str, Any]] = []
    exact_count = normalized_count = located_count = 0
    dropped_unlocatable = dropped_code_overflow = 0
    full_fit_count = truncated_count = prompt_code_overflow_count = 0
    code_char_positions: list[float] = []
    code_token_positions: list[float] = []
    unlocatable_examples: list[dict[str, str]] = []
    original_reasoning_tokens: list[int] = []
    retained_reasoning_tokens: list[int] = []
    retained_ratios: list[float] = []
    final_sequence_tokens: list[int] = []

    for record in records:
        problem = str(record["prompt"])
        response = str(record["reasoning"])
        solution = str(record["reference_code"])
        prompt = serialize_prompt(config, problem)
        location = locate_reference_code(response, solution)
        exact = bool(solution) and solution in response
        normalized = _normalised_substring(response, solution)
        exact_count += int(exact)
        normalized_count += int(normalized)
        located_count += int(location is not None)
        example = build_sft_sequence(
            tokenizer,
            prompt=prompt,
            r1_generation=response,
            reference_code=solution,
            max_sequence_length=max_sequence_length,
        )
        full_fit_count += int(example.full_fit)
        truncated_count += int(example.reasoning_truncated)
        dropped_unlocatable += int(example.drop_reason == "unlocatable_final_code")
        dropped_code_overflow += int(example.drop_reason == "prompt_plus_preserved_code_overflow")
        prompt_code_overflow_count += int(example.prompt_plus_preserved_code_overflow)
        if not example.code_locatable and len(unlocatable_examples) < 10:
            unlocatable_examples.append(
                {
                    "problem_id": str(record["problem_id"]),
                    "problem_id_hash": stable_hash({"problem_id": record["problem_id"]}),
                }
            )
        if location is not None and response:
            code_char_positions.append(location.start_char / len(response))
        if location is not None:
            code_token_positions.append(
                example.original_reasoning_tokens / max(example.original_response_tokens, 1)
            )
        original_reasoning_tokens.append(example.original_reasoning_tokens)
        retained_reasoning_tokens.append(example.retained_reasoning_tokens)
        retained_ratios.append(example.retained_reasoning_ratio)
        if not example.dropped:
            final_sequence_tokens.append(example.final_sequence_tokens)
        details.append(
            {
                "problem_id": record["problem_id"],
                "target_source": "reasoning/r1_generation",
                "reference_code_used_only_for_location": True,
                "exact_substring": exact,
                "normalized_substring": normalized,
                "code_locatable": example.code_locatable,
                "code_location_method": example.code_location_method,
                "code_start_char": example.code_start_char,
                "code_end_char": example.code_end_char,
                "code_start_char_ratio": (
                    example.code_start_char / len(response)
                    if example.code_start_char is not None and response
                    else None
                ),
                "prompt_tokens": example.prompt_tokens,
                "original_response_tokens": example.original_response_tokens,
                "original_reasoning_tokens_before_code": example.original_reasoning_tokens,
                "retained_reasoning_tokens": example.retained_reasoning_tokens,
                "retained_reasoning_ratio": example.retained_reasoning_ratio,
                "full_sequence_tokens": example.full_sequence_tokens,
                "final_sequence_tokens": example.final_sequence_tokens,
                "full_fit": example.full_fit,
                "reasoning_truncated": example.reasoning_truncated,
                "dropped": example.dropped,
                "drop_reason": example.drop_reason,
                "prompt_plus_preserved_code_overflow": example.prompt_plus_preserved_code_overflow,
                "construction_hash": stable_hash(
                    {
                        "problem_id": record["problem_id"],
                        "target": "r1_generation",
                        "policy": sft_config["serialization_version"],
                        "max_sequence_length": max_sequence_length,
                        "location_method": example.code_location_method,
                        "final_sequence_tokens": example.final_sequence_tokens,
                    }
                ),
            }
        )

    total = len(records)
    usable = total - dropped_unlocatable - dropped_code_overflow
    output_dir = artifact_root(config) / "length_policy"
    detail_path = output_dir / "sft_8192_sequence_diagnostic.jsonl"
    write_jsonl(detail_path, details)
    report = {
        "phase": "phase2_length_policy",
        "step": "canonical_target_and_sft_code_preserving_sequence_diagnostic",
        "status": "completed" if prompt_code_overflow_count == 0 else "failed_prompt_plus_preserved_code_overflow",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": git_command(["rev-parse", "HEAD"]),
            "dirty": bool(git_status_short()),
            "status_short": git_status_short(),
        },
        "config_file": config_path,
        "config_hash": config_hash(config_path),
        "sft_view": {"path": str(view_path(config, "sft")), "hash": file_sha256(view_path(config, "sft"))},
        "canonical_target": {
            "source_field": "reasoning",
            "source_semantics": "OpenCodeReasoning-2 r1_generation",
            "serialization": "### Problem\\n{problem}\\n\\n### Solution\\n{r1_generation}",
            "reference_code_appended": False,
            "reference_code_role": "locate_and_verify_only",
        },
        "tokenizer": {
            "repo_id": config["model"]["repo_id"],
            "tokenizer_revision": config["model"]["tokenizer_revision"],
            "class": tokenizer.__class__.__name__,
            "eos_token_id": int(tokenizer.eos_token_id),
        },
        "serialization": {
            "prompt_serialization_version": config["prompt"]["serialization_version"],
            "sft_sequence_serialization_version": sft_config["serialization_version"],
            "max_sequence_length": max_sequence_length,
            "eos_policy": sft_config["eos_policy"],
            "truncation_policy": sft_config["truncation_policy"],
        },
        "canonical_target_diagnostic": {
            "total": total,
            "exact_substring_count": exact_count,
            "exact_substring_rate": exact_count / total if total else 0.0,
            "normalized_substring_count": normalized_count,
            "normalized_substring_rate": normalized_count / total if total else 0.0,
            "response_contains_final_code_count": located_count,
            "response_contains_final_code_rate": located_count / total if total else 0.0,
            "unlocatable_final_code_count": sum(1 for d in details if not d["code_locatable"]),
            "unlocatable_example_count": len(unlocatable_examples),
            "unlocatable_examples_no_raw_text": unlocatable_examples,
            "code_start_character_ratio": numeric_stats(code_char_positions, (5, 50, 95, 99)),
            "code_start_token_ratio": numeric_stats(code_token_positions, (5, 50, 95, 99)),
        },
        "counts": {
            "total": total,
            "final_usable_sft_count": usable,
            "full_fit_count": full_fit_count,
            "full_fit_rate": full_fit_count / total if total else 0.0,
            "reasoning_truncated_count": truncated_count,
            "reasoning_truncated_rate": truncated_count / total if total else 0.0,
            "dropped_unlocatable_count": dropped_unlocatable,
            "dropped_code_overflow_count": dropped_code_overflow,
            "prompt_plus_preserved_code_over_8192_count": prompt_code_overflow_count,
            "prompt_plus_full_code_plus_separator_eos_over_8192": prompt_code_overflow_count,
        },
        "original_reasoning_tokens": numeric_stats(original_reasoning_tokens, (50, 90, 95, 99)),
        "retained_reasoning_tokens": numeric_stats(retained_reasoning_tokens, (50, 90, 95, 99)),
        "retained_reasoning_ratio": numeric_stats(retained_ratios, (5, 50, 95)),
        "final_sequence_tokens": numeric_stats(final_sequence_tokens, (50, 90, 95, 99)),
        "final_sequence_token_max": max(final_sequence_tokens) if final_sequence_tokens else 0,
        "diagnostic_manifest": {"path": str(detail_path), "hash": file_sha256(detail_path), "count": len(details)},
        "runtime": runtime_info(),
        "package_versions": package_versions(),
    }
    report_path = reports_dir(config) / "phase2_sft_8192_length_diagnostic.json"
    write_json(report_path, report)
    return report


def main() -> None:
    args = parse_args()
    report = diagnose_sft_length_policy(args.config)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
