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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose code-preserving SFT sequence construction at 8192 tokens.")
    parser.add_argument("--config", default="configs/base_eval_v2.yaml")
    return parser.parse_args()


def _load_tokenizer(config: dict[str, Any]):
    from transformers import AutoTokenizer

    source, kwargs = pretrained_load_reference(config, "tokenizer_revision")
    tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id for SFT sequence construction")
    return tokenizer


def _token_ids(tokenizer, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"])


def diagnose_sft_length_policy(config_path: str) -> dict[str, Any]:
    config = read_config(config_path)
    sft_config = config["sft_sequence"]
    max_sequence_length = int(sft_config["max_sequence_length"])
    separator = str(sft_config["reasoning_code_separator"])
    tokenizer = _load_tokenizer(config)
    eos_ids = [int(tokenizer.eos_token_id)]
    records = load_jsonl(view_path(config, "sft"))
    expected = int(config["expected"]["final_counts"]["sft"])
    if len(records) != expected:
        raise ValueError(f"Final SFT view count mismatch: expected {expected}, got {len(records)}")

    details: list[dict[str, Any]] = []
    original_reasoning_tokens: list[int] = []
    retained_reasoning_tokens: list[int] = []
    retained_ratios: list[float] = []
    final_sequence_tokens: list[int] = []
    full_fit_count = 0
    truncated_count = 0
    prompt_code_overflow_count = 0

    for record in records:
        prompt_ids = _token_ids(tokenizer, serialize_prompt(config, str(record["prompt"])))
        reasoning_ids = _token_ids(tokenizer, str(record["reasoning"]))
        separator_ids = _token_ids(tokenizer, separator)
        code_ids = _token_ids(tokenizer, str(record["reference_code"]))

        fixed_tokens = len(prompt_ids) + len(separator_ids) + len(code_ids) + len(eos_ids)
        reasoning_budget = max_sequence_length - fixed_tokens
        prompt_code_overflow = reasoning_budget < 0
        if prompt_code_overflow:
            retained = 0
            prompt_code_overflow_count += 1
        else:
            retained = min(len(reasoning_ids), reasoning_budget)

        full_sequence = fixed_tokens + len(reasoning_ids)
        final_tokens = fixed_tokens + retained
        full_fit = full_sequence <= max_sequence_length
        reasoning_truncated = not full_fit
        if full_fit:
            full_fit_count += 1
        else:
            truncated_count += 1

        ratio = 1.0 if not reasoning_ids else retained / len(reasoning_ids)
        original_reasoning_tokens.append(len(reasoning_ids))
        retained_reasoning_tokens.append(retained)
        retained_ratios.append(ratio)
        final_sequence_tokens.append(final_tokens)
        details.append(
            {
                "problem_id": record["problem_id"],
                "prompt_tokens": len(prompt_ids),
                "original_reasoning_tokens": len(reasoning_ids),
                "retained_reasoning_tokens": retained,
                "ocr_code_tokens": len(code_ids),
                "separator_tokens": len(separator_ids),
                "eos_tokens": len(eos_ids),
                "fixed_prompt_code_eos_tokens": fixed_tokens,
                "full_sequence_tokens": full_sequence,
                "final_sequence_tokens": final_tokens,
                "full_fit": full_fit,
                "reasoning_truncated": reasoning_truncated,
                "prompt_code_overflow": prompt_code_overflow,
                "retained_original_reasoning_ratio": ratio,
                "construction_hash": stable_hash(
                    {
                        "problem_id": record["problem_id"],
                        "policy": sft_config["serialization_version"],
                        "prompt_tokens": len(prompt_ids),
                        "retained_reasoning_tokens": retained,
                        "code_tokens": len(code_ids),
                        "separator_tokens": len(separator_ids),
                        "eos_tokens": len(eos_ids),
                    }
                ),
            }
        )

    output_dir = artifact_root(config) / "length_policy"
    detail_path = output_dir / "sft_8192_sequence_diagnostic.jsonl"
    write_jsonl(detail_path, details)
    total = len(records)
    report = {
        "phase": "phase2_length_policy",
        "step": "sft_code_preserving_sequence_diagnostic",
        "status": "completed" if prompt_code_overflow_count == 0 else "failed_prompt_code_overflow",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": git_command(["rev-parse", "HEAD"]),
            "dirty": bool(git_status_short()),
            "status_short": git_status_short(),
        },
        "config_file": config_path,
        "config_hash": config_hash(config_path),
        "sft_view": {"path": str(view_path(config, "sft")), "hash": file_sha256(view_path(config, "sft"))},
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
            "reasoning_code_separator": separator,
            "eos_policy": sft_config["eos_policy"],
            "truncation_policy": sft_config["truncation_policy"],
        },
        "counts": {
            "total": total,
            "full_fit_8192": full_fit_count,
            "full_fit_8192_rate": full_fit_count / total if total else 0.0,
            "reasoning_truncated": truncated_count,
            "reasoning_truncated_rate": truncated_count / total if total else 0.0,
            "prompt_plus_full_code_plus_separator_eos_over_8192": prompt_code_overflow_count,
        },
        "original_reasoning_tokens": numeric_stats(original_reasoning_tokens, (50, 90, 95, 99)),
        "retained_reasoning_tokens": numeric_stats(retained_reasoning_tokens, (50, 90, 95, 99)),
        "retained_original_reasoning_ratio": numeric_stats(retained_ratios, (5, 50, 95)),
        "final_sequence_tokens": numeric_stats(final_sequence_tokens, (50, 90, 95, 99)),
        "final_sequence_token_max": max(final_sequence_tokens) if final_sequence_tokens else 0,
        "diagnostic_manifest": {"path": str(detail_path), "hash": file_sha256(detail_path), "count": len(details)},
        "runtime": runtime_info(),
        "package_versions": package_versions(),
    }
    report_path = reports_dir(config) / "phase2_sft_8192_length_diagnostic.json"
    write_json(report_path, report)
    if prompt_code_overflow_count:
        raise SystemExit(f"SFT prompt+code overflow detected; see {report_path}")
    return report


def main() -> None:
    args = parse_args()
    report = diagnose_sft_length_policy(args.config)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
