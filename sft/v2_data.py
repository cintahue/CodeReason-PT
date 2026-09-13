from __future__ import annotations

from collections import Counter
from typing import Any

from data.schemas import stable_hash
from eval.phase2_common import numeric_stats
from eval.sft_v2_sequence import (
    SequenceExampleV2,
    analyze_response_structure,
    build_sft_sequence_v2,
    locate_reference_code,
    shifted_loss_positions,
)


def build_v2_examples_and_manifest(
    config: dict[str, Any],
    tokenizer: Any,
    records: list[dict[str, Any]],
    *,
    emit_progress: bool = False,
) -> tuple[list[SequenceExampleV2], list[dict[str, Any]], dict[str, Any]]:
    max_sequence_length = int(config["sft_sequence"]["max_sequence_length"])
    max_response_tokens = int(config["sft_sequence"]["max_response_tokens_including_eos"])
    examples: list[SequenceExampleV2] = []
    manifest: list[dict[str, Any]] = []
    integrity_failures = Counter()

    for record_index, record in enumerate(records, start=1):
        prompt = str(config["prompt"]["template"]).format(problem=str(record["prompt"]))
        response = str(record["reasoning"])
        reference_code = str(record["reference_code"])
        source_structure = analyze_response_structure(response, reference_code)
        example = build_sft_sequence_v2(
            tokenizer,
            prompt=prompt,
            r1_generation=response,
            reference_code=reference_code,
            max_sequence_length=max_sequence_length,
            max_response_tokens_including_eos=max_response_tokens,
        )

        final_location = None
        final_structure = None
        prompt_masked = False
        response_supervised = False
        eos_in_shifted_loss = False
        code_complete = False
        think_structure_preserved = False
        fence_structure_preserved = False
        no_duplicate_code_append = not example.reference_code_appended
        sequence_within_budget = False
        response_within_budget = False
        eos_supervised = False
        integrity_errors: list[str] = []

        if not example.dropped:
            final_location = locate_reference_code(example.final_response_text, reference_code)
            final_structure = analyze_response_structure(example.final_response_text, reference_code)
            prompt_masked = all(label == -100 for label in example.labels[: example.prompt_tokens])
            response_supervised = all(label != -100 for label in example.labels[example.prompt_tokens :])
            eos_in_shifted_loss = len(example.labels) - 1 in shifted_loss_positions(example.labels)
            code_complete = final_location is not None
            think_structure_preserved = (
                not source_structure.starts_with_think
                or (
                    example.final_response_text.lstrip().startswith("<think>")
                    and final_structure.has_matching_closed_think
                )
            )
            fence_structure_preserved = not source_structure.fenced_code or final_structure.fenced_code
            sequence_within_budget = example.final_sequence_tokens <= max_sequence_length
            response_within_budget = example.response_tokens_including_eos <= max_response_tokens
            eos_supervised = bool(
                example.eos_supervised
                and example.input_ids
                and example.labels
                and example.input_ids[-1] == int(tokenizer.eos_token_id)
                and example.labels[-1] == int(tokenizer.eos_token_id)
            )

            checks = {
                "prompt_not_fully_masked": prompt_masked,
                "response_not_fully_supervised": response_supervised,
                "eos_missing_from_shifted_loss": eos_in_shifted_loss,
                "final_code_not_preserved": code_complete,
                "think_structure_not_preserved": think_structure_preserved,
                "fence_structure_not_preserved": fence_structure_preserved,
                "duplicate_code_appended": no_duplicate_code_append,
                "sequence_budget_exceeded": sequence_within_budget,
                "response_budget_exceeded": response_within_budget,
                "eos_not_supervised": eos_supervised,
            }
            integrity_errors = [name for name, passed in checks.items() if not passed]
            integrity_failures.update(integrity_errors)

        construction_hash = stable_hash(
            {
                "problem_id": record["problem_id"],
                "source_response_hash": stable_hash(response),
                "reference_code_hash": stable_hash(reference_code),
                "input_ids_hash": stable_hash(example.input_ids),
                "labels_hash": stable_hash(example.labels),
                "dropped": example.dropped,
                "drop_reason": example.drop_reason,
            }
        )
        manifest.append(
            {
                "problem_id": record["problem_id"],
                "target_source": "reasoning/r1_generation",
                "reference_code_used_only_for_location": True,
                "reference_code_appended": example.reference_code_appended,
                "source_response_hash": stable_hash(response),
                "reference_code_hash": stable_hash(reference_code),
                "starts_with_think": source_structure.starts_with_think,
                "has_matching_closed_think": source_structure.has_matching_closed_think,
                "code_fenced": source_structure.fenced_code,
                "code_unfenced": bool(source_structure.code_location and not source_structure.fenced_code),
                "source_code_location_method": (
                    source_structure.code_location.method if source_structure.code_location else None
                ),
                "source_code_start_char": (
                    source_structure.code_location.start_char if source_structure.code_location else None
                ),
                "source_code_end_char": (
                    source_structure.code_location.end_char if source_structure.code_location else None
                ),
                "mandatory_structural_suffix_kind": example.mandatory_suffix_kind,
                "mandatory_structural_suffix_start_char": example.mandatory_suffix_start_char,
                "mandatory_structural_suffix_tokens": example.mandatory_suffix_tokens,
                "full_response_tokens": example.original_response_tokens,
                "full_response_tokens_including_eos": example.original_response_tokens + 1,
                "truncated": example.reasoning_truncated,
                "non_truncated": bool(example.full_fit and not example.dropped),
                "final_response_tokens": example.response_tokens,
                "final_response_tokens_including_eos": example.response_tokens_including_eos,
                "full_sequence_tokens": example.full_sequence_tokens,
                "final_sequence_tokens": example.final_sequence_tokens,
                "retained_reasoning_tokens": example.retained_reasoning_tokens,
                "retained_reasoning_ratio": example.retained_reasoning_ratio,
                "final_code_location_method": final_location.method if final_location else None,
                "final_code_start_char": final_location.start_char if final_location else None,
                "final_code_end_char": final_location.end_char if final_location else None,
                "final_code_complete": code_complete,
                "think_structure_preserved": think_structure_preserved,
                "fence_structure_preserved": fence_structure_preserved,
                "prompt_fully_masked": prompt_masked,
                "response_including_eos_fully_supervised": response_supervised,
                "eos_supervised": eos_supervised,
                "eos_in_shifted_causal_loss": eos_in_shifted_loss,
                "response_budget_ok": response_within_budget,
                "sequence_budget_ok": sequence_within_budget,
                "no_duplicate_code_append": no_duplicate_code_append,
                "integrity_errors": integrity_errors,
                "dropped": example.dropped,
                "drop_reason": example.drop_reason,
                "construction_hash": construction_hash,
            }
        )
        if not example.dropped:
            examples.append(example)
        if emit_progress and (record_index == 1 or record_index % 250 == 0 or record_index == len(records)):
            print(f"Constructed {record_index}/{len(records)} v2 SFT records", flush=True)

    total = len(manifest)
    usable = len(examples)
    drop_counts = Counter(str(item["drop_reason"]) for item in manifest if item["dropped"])
    full_fit_count = sum(1 for item in manifest if item["non_truncated"])
    truncated_count = sum(1 for item in manifest if item["truncated"] and not item["dropped"])
    retained = [item for item in manifest if not item["dropped"]]
    stats = {
        "source_records": total,
        "usable_records": usable,
        "usable_rate": usable / total if total else 0.0,
        "dropped_records": total - usable,
        "drop_reason_counts": dict(sorted(drop_counts.items())),
        "starts_with_think_count": sum(1 for item in manifest if item["starts_with_think"]),
        "starts_with_think_rate": sum(1 for item in manifest if item["starts_with_think"]) / total if total else 0.0,
        "has_matching_closed_think_count": sum(1 for item in manifest if item["has_matching_closed_think"]),
        "has_matching_closed_think_rate": (
            sum(1 for item in manifest if item["has_matching_closed_think"]) / total if total else 0.0
        ),
        "code_fenced_count": sum(1 for item in manifest if item["code_fenced"]),
        "code_fenced_rate": sum(1 for item in manifest if item["code_fenced"]) / total if total else 0.0,
        "code_unfenced_count": sum(1 for item in manifest if item["code_unfenced"]),
        "code_unfenced_rate": sum(1 for item in manifest if item["code_unfenced"]) / total if total else 0.0,
        "exact_code_location_count": sum(
            1 for item in manifest if item["source_code_location_method"] == "exact"
        ),
        "normalized_code_location_count": sum(
            1 for item in manifest if item["source_code_location_method"] == "normalized_whitespace"
        ),
        "unlocatable_code_count": sum(1 for item in manifest if item["source_code_location_method"] is None),
        "full_fit_count": full_fit_count,
        "full_fit_rate": full_fit_count / total if total else 0.0,
        "reasoning_truncated_count": truncated_count,
        "reasoning_truncated_rate": truncated_count / total if total else 0.0,
        "mandatory_suffix_over_response_budget_count": drop_counts.get(
            "mandatory_structural_suffix_over_response_budget", 0
        ),
        "eos_supervised_count": sum(1 for item in retained if item["eos_supervised"]),
        "eos_supervised_rate": (
            sum(1 for item in retained if item["eos_supervised"]) / usable if usable else 0.0
        ),
        "integrity_failure_count": sum(len(item["integrity_errors"]) for item in retained),
        "integrity_failure_record_count": sum(1 for item in retained if item["integrity_errors"]),
        "integrity_failure_counts": dict(sorted(integrity_failures.items())),
        "mandatory_structural_suffix_tokens": numeric_stats(
            [int(item["mandatory_structural_suffix_tokens"]) for item in manifest], (50, 90, 95, 99)
        ),
        "full_response_tokens": numeric_stats(
            [int(item["full_response_tokens"]) for item in manifest], (50, 90, 95, 99)
        ),
        "final_response_tokens_including_eos": numeric_stats(
            [int(item["final_response_tokens_including_eos"]) for item in retained], (50, 90, 95, 99)
        ),
        "final_sequence_tokens": numeric_stats(
            [int(item["final_sequence_tokens"]) for item in retained], (50, 90, 95, 99)
        ),
        "retained_reasoning_ratio": numeric_stats(
            [float(item["retained_reasoning_ratio"]) for item in retained], (5, 50, 95)
        ),
    }
    stats["construction_gate_passed"] = bool(
        total == int(config["expected"]["source_sft_count"])
        and usable > 0
        and stats["integrity_failure_record_count"] == 0
        and stats["eos_supervised_count"] == usable
        and all(int(item["final_response_tokens_including_eos"]) <= max_response_tokens for item in retained)
        and all(int(item["final_sequence_tokens"]) <= max_sequence_length for item in retained)
    )
    return examples, manifest, stats


def frozen_dataset_hash(manifest: list[dict[str, Any]]) -> str:
    return stable_hash(
        [item["construction_hash"] for item in manifest if not bool(item["dropped"])]
    )
