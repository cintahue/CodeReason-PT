from __future__ import annotations

"""Phase 3-v3 natural-length SFT construction.

The constructor either keeps the complete original ``r1_generation`` or
drops the record. It never truncates, repairs, summarizes, or appends OCR
code.
"""

from dataclasses import dataclass
from typing import Any, Sequence

from eval.sft_v2_sequence import StructuralAnalysis, analyze_response_structure
from eval.sft_sequence import CodeLocation, locate_reference_code


@dataclass(frozen=True)
class SequenceExampleV3:
    input_ids: list[int]
    labels: list[int]
    final_response_text: str
    prompt_tokens: int
    response_tokens: int
    response_tokens_including_eos: int
    original_response_tokens: int
    original_reasoning_tokens: int
    full_sequence_tokens: int
    final_sequence_tokens: int
    code_locatable: bool
    code_location_method: str | None
    code_start_char: int | None
    code_end_char: int | None
    starts_with_think: bool
    has_matching_closed_think: bool
    fenced_code: bool
    final_code_preserved: bool
    artificial_truncation: bool
    reference_code_appended: bool
    eos_supervised: bool
    dropped: bool
    drop_reason: str | None


def canonical_sft_target(config: dict[str, Any], problem: str, r1_generation: str) -> str:
    template = str(config["prompt"]["template"])
    return template.format(problem=problem) + str(r1_generation)


def response_only_labels(
    input_ids: Sequence[int],
    *,
    prompt_tokens: int,
    response_tokens_including_eos: int,
    pad_token_id: int | None,
    ignore_index: int = -100,
) -> list[int]:
    if prompt_tokens < 0 or response_tokens_including_eos < 0:
        raise ValueError("Invalid prompt/response token boundaries")
    end = prompt_tokens + response_tokens_including_eos
    if end > len(input_ids):
        raise ValueError("Response boundary exceeds input length")
    labels = [ignore_index] * len(input_ids)
    for index in range(prompt_tokens, end):
        token = int(input_ids[index])
        labels[index] = ignore_index if pad_token_id is not None and token == pad_token_id else token
    return labels


def shifted_loss_positions(labels: Sequence[int], ignore_index: int = -100) -> list[int]:
    return [index for index, label in enumerate(labels[1:], start=1) if int(label) != ignore_index]


def _drop(
    *,
    prompt_tokens: int,
    response_tokens: int,
    full_sequence_tokens: int,
    analysis: StructuralAnalysis,
    reason: str,
) -> SequenceExampleV3:
    location = analysis.code_location
    return SequenceExampleV3(
        input_ids=[],
        labels=[],
        final_response_text="",
        prompt_tokens=prompt_tokens,
        response_tokens=0,
        response_tokens_including_eos=0,
        original_response_tokens=response_tokens,
        original_reasoning_tokens=response_tokens,
        full_sequence_tokens=full_sequence_tokens,
        final_sequence_tokens=0,
        code_locatable=location is not None,
        code_location_method=location.method if location else None,
        code_start_char=location.start_char if location else None,
        code_end_char=location.end_char if location else None,
        starts_with_think=analysis.starts_with_think,
        has_matching_closed_think=analysis.has_matching_closed_think,
        fenced_code=analysis.fenced_code,
        final_code_preserved=False,
        artificial_truncation=False,
        reference_code_appended=False,
        eos_supervised=False,
        dropped=True,
        drop_reason=reason,
    )


def build_sft_sequence_v3(
    tokenizer: Any,
    *,
    prompt: str,
    r1_generation: str,
    reference_code: str,
    max_sequence_length: int,
    max_response_tokens_including_eos: int,
) -> SequenceExampleV3:
    if max_sequence_length < 2 or max_response_tokens_including_eos < 2:
        raise ValueError("Sequence and response budgets must leave room for response and EOS")
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        raise ValueError("Tokenizer must define eos_token_id")

    prompt_ids = list(tokenizer(prompt, add_special_tokens=False, truncation=False)["input_ids"])
    response_ids = list(tokenizer(r1_generation, add_special_tokens=False, truncation=False)["input_ids"])
    analysis = analyze_response_structure(r1_generation, reference_code)
    full_sequence_tokens = len(prompt_ids) + len(response_ids) + 1
    if not analysis.valid_structure:
        return _drop(
            prompt_tokens=len(prompt_ids),
            response_tokens=len(response_ids),
            full_sequence_tokens=full_sequence_tokens,
            analysis=analysis,
            reason=analysis.invalid_reason or "invalid_structure",
        )
    if len(response_ids) + 1 > max_response_tokens_including_eos:
        return _drop(
            prompt_tokens=len(prompt_ids),
            response_tokens=len(response_ids),
            full_sequence_tokens=full_sequence_tokens,
            analysis=analysis,
            reason="natural_response_over_budget",
        )
    if full_sequence_tokens > max_sequence_length:
        return _drop(
            prompt_tokens=len(prompt_ids),
            response_tokens=len(response_ids),
            full_sequence_tokens=full_sequence_tokens,
            analysis=analysis,
            reason="prompt_plus_response_over_sequence_budget",
        )

    input_ids = prompt_ids + [int(value) for value in response_ids] + [int(eos_id)]
    labels = response_only_labels(
        input_ids,
        prompt_tokens=len(prompt_ids),
        response_tokens_including_eos=len(response_ids) + 1,
        pad_token_id=None,
    )
    if input_ids[-1] != int(eos_id) or labels[-1] != int(eos_id):
        raise AssertionError("v3 must supervise EOS as the final causal-LM target")
    location = analysis.code_location
    return SequenceExampleV3(
        input_ids=input_ids,
        labels=labels,
        final_response_text=r1_generation,
        prompt_tokens=len(prompt_ids),
        response_tokens=len(response_ids),
        response_tokens_including_eos=len(response_ids) + 1,
        original_response_tokens=len(response_ids),
        original_reasoning_tokens=len(response_ids),
        full_sequence_tokens=full_sequence_tokens,
        final_sequence_tokens=len(input_ids),
        code_locatable=location is not None,
        code_location_method=location.method if location else None,
        code_start_char=location.start_char if location else None,
        code_end_char=location.end_char if location else None,
        starts_with_think=analysis.starts_with_think,
        has_matching_closed_think=analysis.has_matching_closed_think,
        fenced_code=analysis.fenced_code,
        final_code_preserved=True,
        artificial_truncation=False,
        reference_code_appended=False,
        eos_supervised=True,
        dropped=False,
        drop_reason=None,
    )
