from __future__ import annotations

"""Canonical Reasoning-SFT target, sequence construction, and loss masking.

The response is the single ``r1_generation`` string stored in ``reasoning``.
``reference_code`` is metadata used only to locate the executable code span.
"""

from dataclasses import dataclass
import re
from typing import Any, Sequence


@dataclass(frozen=True)
class CodeLocation:
    start_char: int
    end_char: int
    method: str


@dataclass(frozen=True)
class SequenceExample:
    input_ids: list[int]
    labels: list[int]
    prompt_tokens: int
    response_tokens: int
    original_response_tokens: int
    original_reasoning_tokens: int
    retained_reasoning_tokens: int
    retained_reasoning_ratio: float
    full_sequence_tokens: int
    final_sequence_tokens: int
    full_fit: bool
    reasoning_truncated: bool
    code_locatable: bool
    code_location_method: str | None
    code_start_char: int | None
    code_end_char: int | None
    dropped: bool
    drop_reason: str | None
    prompt_plus_preserved_code_overflow: bool


def canonical_sft_target(config: dict[str, Any], problem: str, r1_generation: str) -> str:
    """Serialize exactly one response; never append reference/OCR code."""

    template = str(config["prompt"]["template"])
    return template.format(problem=problem) + str(r1_generation)


def _normalise_whitespace_with_map(text: str) -> tuple[str, list[int]]:
    """Collapse whitespace and return a normalized-char -> original-char map."""

    chars: list[str] = []
    mapping: list[int] = []
    pending_space = False
    pending_index = 0
    for index, char in enumerate(text):
        if char.isspace():
            if chars:
                pending_space = True
                pending_index = index
            continue
        if pending_space:
            chars.append(" ")
            mapping.append(pending_index)
            pending_space = False
        chars.append(char)
        mapping.append(index)
    return "".join(chars), mapping


def _normalise_line_trailing(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines())


def _mapped_substring_span(response: str, code: str) -> tuple[int, int] | None:
    normalized_response, response_map = _normalise_whitespace_with_map(response)
    normalized_code, _ = _normalise_whitespace_with_map(code)
    if not normalized_code:
        return None
    start = normalized_response.find(normalized_code)
    if start < 0:
        return None
    end = start + len(normalized_code)
    original_start = response_map[start]
    # The final normalized character maps to its original position. Include it.
    original_end = response_map[end - 1] + 1
    return original_start, original_end


def locate_reference_code(response: str, reference_code: str) -> CodeLocation | None:
    """Locate code only by exact or deterministic whitespace normalization."""

    if not reference_code:
        return None
    exact = response.find(reference_code)
    if exact >= 0:
        return CodeLocation(exact, exact + len(reference_code), "exact")

    mapped = _mapped_substring_span(response, reference_code)
    if mapped is not None:
        return CodeLocation(mapped[0], mapped[1], "normalized_whitespace")
    return None


def _tokenize_with_offsets(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]] | None]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
    )
    ids = list(encoded["input_ids"])
    offsets = encoded.get("offset_mapping")
    if offsets is not None:
        offsets = [tuple(map(int, item)) for item in offsets]
    return ids, offsets


def _code_start_token(
    response_ids: Sequence[int],
    offsets: list[tuple[int, int]] | None,
    location: CodeLocation | None,
) -> int | None:
    if location is None:
        return None
    if offsets is not None:
        for index, (start, end) in enumerate(offsets):
            if end > location.start_char and start < location.end_char:
                return index
    # Slow/non-fast tokenizers do not expose offsets. Exact matching can still
    # be located by tokenizing the code prefix and taking its token count.
    return None


def build_sft_sequence(
    tokenizer: Any,
    *,
    prompt: str,
    r1_generation: str,
    reference_code: str,
    max_sequence_length: int,
) -> SequenceExample:
    """Build one code-preserving sequence with response-only labels."""

    if max_sequence_length < 2:
        raise ValueError("max_sequence_length must leave room for prompt, response, and EOS")
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        raise ValueError("Tokenizer must define eos_token_id")

    prompt_ids = list(tokenizer(prompt, add_special_tokens=False, truncation=False)["input_ids"])
    response_ids, offsets = _tokenize_with_offsets(tokenizer, r1_generation)
    location = locate_reference_code(r1_generation, reference_code)
    code_start = _code_start_token(response_ids, offsets, location)
    full_ids = prompt_ids + response_ids + [int(eos_id)]
    full_len = len(full_ids)
    full_fit = full_len <= max_sequence_length

    # If the response fits, retain it verbatim even if a code location is not
    # available. Location is only needed when truncation is required.
    if full_fit:
        input_ids = full_ids
        retained_reasoning = len(response_ids) if code_start is None else code_start
        original_reasoning = len(response_ids) if code_start is None else code_start
        return _make_example(
            input_ids=input_ids,
            prompt_tokens=len(prompt_ids),
            original_response_tokens=len(response_ids),
            original_reasoning_tokens=original_reasoning,
            retained_reasoning_tokens=retained_reasoning,
            full_sequence_tokens=full_len,
            full_fit=True,
            reasoning_truncated=False,
            location=location,
            code_start_char=location.start_char if location else None,
            code_end_char=location.end_char if location else None,
            code_locatable=code_start is not None,
            dropped=False,
            drop_reason=None,
            prompt_plus_preserved_code_overflow=False,
            eos_id=int(eos_id),
        )

    if code_start is None:
        return _dropped_example(
            prompt_tokens=len(prompt_ids),
            original_response_tokens=len(response_ids),
            full_len=full_len,
            location=location,
            reason="unlocatable_final_code",
        )

    preserved_code_ids = response_ids[code_start:]
    prefix_budget = max_sequence_length - len(prompt_ids) - len(preserved_code_ids) - 1
    if prefix_budget < 0:
        return _dropped_example(
            prompt_tokens=len(prompt_ids),
            original_response_tokens=len(response_ids),
            full_len=full_len,
            location=location,
            reason="prompt_plus_preserved_code_overflow",
        )

    retained_prefix = response_ids[: min(code_start, prefix_budget)]
    input_ids = prompt_ids + retained_prefix + preserved_code_ids + [int(eos_id)]
    return _make_example(
        input_ids=input_ids,
        prompt_tokens=len(prompt_ids),
        original_response_tokens=len(response_ids),
        original_reasoning_tokens=code_start,
        retained_reasoning_tokens=len(retained_prefix),
        full_sequence_tokens=full_len,
        full_fit=False,
        reasoning_truncated=len(retained_prefix) < code_start,
        location=location,
        code_start_char=location.start_char if location else None,
        code_end_char=location.end_char if location else None,
        code_locatable=True,
        dropped=False,
        drop_reason=None,
        prompt_plus_preserved_code_overflow=False,
        eos_id=int(eos_id),
    )


def _make_example(
    *,
    input_ids: list[int],
    prompt_tokens: int,
    original_response_tokens: int,
    original_reasoning_tokens: int,
    retained_reasoning_tokens: int,
    full_sequence_tokens: int,
    full_fit: bool,
    reasoning_truncated: bool,
    location: CodeLocation | None,
    code_start_char: int | None,
    code_end_char: int | None,
    code_locatable: bool,
    dropped: bool,
    drop_reason: str | None,
    prompt_plus_preserved_code_overflow: bool,
    eos_id: int,
) -> SequenceExample:
    labels = response_only_labels(
        input_ids,
        prompt_tokens=prompt_tokens,
        response_tokens=len(input_ids) - prompt_tokens - 1,
        pad_token_id=None,
    )
    ratio = (
        1.0
        if original_reasoning_tokens == 0
        else retained_reasoning_tokens / original_reasoning_tokens
    )
    return SequenceExample(
        input_ids=input_ids,
        labels=labels,
        prompt_tokens=prompt_tokens,
        response_tokens=len(input_ids) - prompt_tokens - 1,
        original_response_tokens=original_response_tokens,
        original_reasoning_tokens=original_reasoning_tokens,
        retained_reasoning_tokens=retained_reasoning_tokens,
        retained_reasoning_ratio=ratio,
        full_sequence_tokens=full_sequence_tokens,
        final_sequence_tokens=len(input_ids),
        full_fit=full_fit,
        reasoning_truncated=reasoning_truncated,
        code_locatable=code_locatable,
        code_location_method=location.method if location else None,
        code_start_char=code_start_char,
        code_end_char=code_end_char,
        dropped=dropped,
        drop_reason=drop_reason,
        prompt_plus_preserved_code_overflow=prompt_plus_preserved_code_overflow,
    )


def _dropped_example(
    *,
    prompt_tokens: int,
    original_response_tokens: int,
    full_len: int,
    location: CodeLocation | None,
    reason: str,
) -> SequenceExample:
    return SequenceExample(
        input_ids=[],
        labels=[],
        prompt_tokens=prompt_tokens,
        response_tokens=0,
        original_response_tokens=original_response_tokens,
        original_reasoning_tokens=original_response_tokens,
        retained_reasoning_tokens=0,
        retained_reasoning_ratio=0.0,
        full_sequence_tokens=full_len,
        final_sequence_tokens=0,
        full_fit=False,
        reasoning_truncated=False,
        code_locatable=location is not None,
        code_location_method=location.method if location else None,
        code_start_char=location.start_char if location else None,
        code_end_char=location.end_char if location else None,
        dropped=True,
        drop_reason=reason,
        prompt_plus_preserved_code_overflow=reason == "prompt_plus_preserved_code_overflow",
    )


def response_only_labels(
    input_ids: Sequence[int],
    *,
    prompt_tokens: int,
    response_tokens: int,
    pad_token_id: int | None,
    ignore_index: int = -100,
) -> list[int]:
    """Mask prompt and padding; leave only response labels trainable."""

    if prompt_tokens < 0 or response_tokens < 0 or prompt_tokens + response_tokens > len(input_ids):
        raise ValueError("Invalid prompt/response token boundaries")
    labels = [ignore_index] * len(input_ids)
    for index in range(prompt_tokens, prompt_tokens + response_tokens):
        token = int(input_ids[index])
        labels[index] = ignore_index if pad_token_id is not None and token == pad_token_id else token
    return labels


def shifted_loss_positions(labels: Sequence[int], ignore_index: int = -100) -> list[int]:
    """Return target positions after the causal LM one-token shift."""

    return [index for index, label in enumerate(labels[1:], start=1) if int(label) != ignore_index]
