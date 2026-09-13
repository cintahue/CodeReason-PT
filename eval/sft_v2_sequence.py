from __future__ import annotations

"""Phase 3-v2 canonical target construction.

This module is intentionally separate from the Phase 3-v1 constructor.  v2
keeps the single ``r1_generation`` response, preserves structural suffixes
when truncation is necessary, and supervises the appended EOS token.
"""

from dataclasses import dataclass
import re
from typing import Any, Sequence

from eval.sft_sequence import CodeLocation, locate_reference_code


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

# The code group is deliberately conservative.  It recognizes ordinary
# Markdown fences without trying to infer or reconstruct malformed syntax.
FENCE_RE = re.compile(
    r"```(?P<language>[A-Za-z0-9_+.#-]*)[ \t]*\n(?P<code>.*?)(?:\n```|```)",
    re.DOTALL,
)
FENCE_OPEN_RE = re.compile(r"```(?P<language>[A-Za-z0-9_+.#-]*)[ \t]*\n")


@dataclass(frozen=True)
class StructuralAnalysis:
    code_location: CodeLocation | None
    starts_with_think: bool
    has_matching_closed_think: bool
    think_open_start: int | None
    think_open_end: int | None
    think_close_start: int | None
    think_close_end: int | None
    fenced_code: bool
    fence_open_start: int | None
    fence_open_end: int | None
    fence_close_start: int | None
    fence_close_end: int | None
    suffix_start_char: int | None
    suffix_kind: str | None
    valid_structure: bool
    invalid_reason: str | None


@dataclass(frozen=True)
class SequenceExampleV2:
    input_ids: list[int]
    labels: list[int]
    final_response_text: str
    prompt_tokens: int
    response_tokens: int
    response_tokens_including_eos: int
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
    starts_with_think: bool
    has_matching_closed_think: bool
    fenced_code: bool
    mandatory_suffix_tokens: int
    mandatory_suffix_kind: str | None
    mandatory_suffix_start_char: int | None
    final_code_preserved: bool
    reference_code_appended: bool
    eos_supervised: bool
    dropped: bool
    drop_reason: str | None


def canonical_sft_target(config: dict[str, Any], problem: str, r1_generation: str) -> str:
    """Serialize exactly one response; reference/OCR code is never appended."""

    template = str(config["prompt"]["template"])
    return template.format(problem=problem) + str(r1_generation)


def _tokenize(tokenizer: Any, text: str, *, offsets: bool = False) -> tuple[list[int], list[tuple[int, int]] | None]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=offsets,
    )
    ids = [int(item) for item in encoded["input_ids"]]
    raw_offsets = encoded.get("offset_mapping") if offsets else None
    mapped_offsets = None
    if raw_offsets is not None:
        mapped_offsets = [tuple(map(int, item)) for item in raw_offsets]
    return ids, mapped_offsets


def _first_overlapping_token(
    offsets: list[tuple[int, int]] | None,
    location: CodeLocation | None,
) -> int | None:
    if offsets is None or location is None:
        return None
    for index, (start, end) in enumerate(offsets):
        if end > location.start_char and start < location.end_char:
            return index
    return None


def _find_fence_for_code(response: str, location: CodeLocation | None) -> tuple[int, int, int | None, int | None] | None:
    if location is None:
        return None
    for opening in FENCE_OPEN_RE.finditer(response):
        code_start = opening.end()
        if location.start_char < code_start:
            continue
        close_start = response.find("```", location.end_char)
        if close_start < 0:
            return opening.start(), code_start, None, None
        trailing = response[close_start : location.end_char] if location.end_char > close_start else ""
        if location.end_char <= close_start or not trailing.strip():
            return opening.start(), code_start, close_start, close_start + 3
    return None


def analyze_response_structure(response: str, reference_code: str) -> StructuralAnalysis:
    """Analyze only exact/normalized code location and explicit structures."""

    location = locate_reference_code(response, reference_code)
    leading = len(response) - len(response.lstrip())
    starts_with_think = response[leading:].startswith(THINK_OPEN)
    think_open_start = leading if starts_with_think else None
    think_open_end = leading + len(THINK_OPEN) if starts_with_think else None
    think_close_start = None
    think_close_end = None
    has_matching_closed_think = False
    if starts_with_think and think_open_end is not None:
        close = response.find(THINK_CLOSE, think_open_end)
        if close >= 0:
            think_close_start = close
            think_close_end = close + len(THINK_CLOSE)
            has_matching_closed_think = location is not None and close < location.start_char

    fence = _find_fence_for_code(response, location)
    fenced_code = fence is not None
    fence_open_start = fence[0] if fence else None
    fence_open_end = fence[1] if fence else None
    fence_close_start = fence[2] if fence else None
    fence_close_end = fence[3] if fence else None

    if location is None:
        return StructuralAnalysis(
            code_location=None,
            starts_with_think=starts_with_think,
            has_matching_closed_think=has_matching_closed_think,
            think_open_start=think_open_start,
            think_open_end=think_open_end,
            think_close_start=think_close_start,
            think_close_end=think_close_end,
            fenced_code=fenced_code,
            fence_open_start=fence_open_start,
            fence_open_end=fence_open_end,
            fence_close_start=fence_close_start,
            fence_close_end=fence_close_end,
            suffix_start_char=None,
            suffix_kind=None,
            valid_structure=False,
            invalid_reason="unlocatable_final_code",
        )

    if starts_with_think and not has_matching_closed_think:
        return StructuralAnalysis(
            code_location=location,
            starts_with_think=starts_with_think,
            has_matching_closed_think=False,
            think_open_start=think_open_start,
            think_open_end=think_open_end,
            think_close_start=think_close_start,
            think_close_end=think_close_end,
            fenced_code=fenced_code,
            fence_open_start=fence_open_start,
            fence_open_end=fence_open_end,
            fence_close_start=fence_close_start,
            fence_close_end=fence_close_end,
            suffix_start_char=None,
            suffix_kind=None,
            valid_structure=False,
            invalid_reason="unclosed_or_late_think",
        )

    if fenced_code and fence_close_start is None:
        return StructuralAnalysis(
            code_location=location,
            starts_with_think=starts_with_think,
            has_matching_closed_think=has_matching_closed_think,
            think_open_start=think_open_start,
            think_open_end=think_open_end,
            think_close_start=think_close_start,
            think_close_end=think_close_end,
            fenced_code=True,
            fence_open_start=fence_open_start,
            fence_open_end=fence_open_end,
            fence_close_start=None,
            fence_close_end=None,
            suffix_start_char=None,
            suffix_kind=None,
            valid_structure=False,
            invalid_reason="unclosed_code_fence",
        )

    if starts_with_think and think_close_start is not None:
        suffix_start = think_close_start
        suffix_kind = "think_close_and_code_suffix"
    elif fenced_code and fence_open_start is not None:
        suffix_start = fence_open_start
        suffix_kind = "fenced_code_suffix"
    else:
        suffix_start = location.start_char
        suffix_kind = "code_body_suffix"

    return StructuralAnalysis(
        code_location=location,
        starts_with_think=starts_with_think,
        has_matching_closed_think=has_matching_closed_think,
        think_open_start=think_open_start,
        think_open_end=think_open_end,
        think_close_start=think_close_start,
        think_close_end=think_close_end,
        fenced_code=fenced_code,
        fence_open_start=fence_open_start,
        fence_open_end=fence_open_end,
        fence_close_start=fence_close_start,
        fence_close_end=fence_close_end,
        suffix_start_char=suffix_start,
        suffix_kind=suffix_kind,
        valid_structure=True,
        invalid_reason=None,
    )


def format_response_diagnostics(response: str, extraction_success: bool) -> dict[str, Any]:
    """Return evaluator-only format flags without changing extraction/verifier behavior."""

    leading = len(response) - len(response.lstrip())
    starts_with_think = response[leading:].startswith(THINK_OPEN)
    think_open_end = leading + len(THINK_OPEN) if starts_with_think else None
    think_close = response.find(THINK_CLOSE, think_open_end or 0) if starts_with_think else -1
    has_closed_think = bool(starts_with_think and think_close >= 0)
    unclosed_think = bool(starts_with_think and not has_closed_think)
    fences = list(FENCE_RE.finditer(response))
    fence_markers = response.count("```")
    fenced_code = fence_markers > 0
    balanced_fence = fence_markers % 2 == 0
    first_fence_start = fences[0].start() if fences else None
    valid_transition = not starts_with_think or (
        has_closed_think and (first_fence_start is None or first_fence_start > int(think_close))
    )
    format_valid = bool(extraction_success and not unclosed_think and balanced_fence and valid_transition)
    if unclosed_think:
        strategy_class = "unclosed_think"
    elif starts_with_think and fenced_code:
        strategy_class = "think_fenced"
    elif starts_with_think:
        strategy_class = "think_unfenced"
    elif fenced_code:
        strategy_class = "fenced"
    else:
        strategy_class = "plain"
    return {
        "starts_with_think": starts_with_think,
        "closed_think": has_closed_think,
        "unclosed_think": unclosed_think,
        "fenced_code": fenced_code,
        "valid_reasoning_to_code_transition": valid_transition,
        "format_valid": format_valid,
        "format_class": strategy_class,
    }


def response_only_labels(
    input_ids: Sequence[int],
    *,
    prompt_tokens: int,
    response_tokens_including_eos: int,
    pad_token_id: int | None,
    ignore_index: int = -100,
) -> list[int]:
    """Mask prompt/padding while supervising response tokens including EOS."""

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
    """Return supervised target positions after the causal-LM one-token shift."""

    return [index for index, label in enumerate(labels[1:], start=1) if int(label) != ignore_index]


def _prefix_text_for_token_budget(
    tokenizer: Any,
    source: str,
    budget: int,
    required_prefix: str,
    tokenized: tuple[list[int], list[tuple[int, int]] | None] | None = None,
) -> str | None:
    if budget <= 0:
        return "" if not required_prefix else None
    ids, offsets = tokenized or _tokenize(tokenizer, source, offsets=True)
    if len(ids) <= budget:
        candidate = source
    elif offsets is not None and budget <= len(offsets):
        end_char = offsets[budget - 1][1]
        candidate = source[:end_char]
    else:
        candidate = str(tokenizer.decode(ids[:budget], clean_up_tokenization_spaces=False))
    if required_prefix and not candidate.startswith(required_prefix):
        return None
    return candidate


def _drop(
    *,
    prompt_tokens: int,
    original_response_tokens: int,
    full_sequence_tokens: int,
    analysis: StructuralAnalysis,
    reason: str,
    mandatory_suffix_tokens: int = 0,
) -> SequenceExampleV2:
    location = analysis.code_location
    return SequenceExampleV2(
        input_ids=[],
        labels=[],
        final_response_text="",
        prompt_tokens=prompt_tokens,
        response_tokens=0,
        response_tokens_including_eos=0,
        original_response_tokens=original_response_tokens,
        original_reasoning_tokens=original_response_tokens,
        retained_reasoning_tokens=0,
        retained_reasoning_ratio=0.0,
        full_sequence_tokens=full_sequence_tokens,
        final_sequence_tokens=0,
        full_fit=False,
        reasoning_truncated=False,
        code_locatable=location is not None,
        code_location_method=location.method if location else None,
        code_start_char=location.start_char if location else None,
        code_end_char=location.end_char if location else None,
        starts_with_think=analysis.starts_with_think,
        has_matching_closed_think=analysis.has_matching_closed_think,
        fenced_code=analysis.fenced_code,
        mandatory_suffix_tokens=mandatory_suffix_tokens,
        mandatory_suffix_kind=analysis.suffix_kind,
        mandatory_suffix_start_char=analysis.suffix_start_char,
        final_code_preserved=False,
        reference_code_appended=False,
        eos_supervised=False,
        dropped=True,
        drop_reason=reason,
    )


def _make(
    *,
    input_ids: list[int],
    prompt_tokens: int,
    original_response_tokens: int,
    original_reasoning_tokens: int,
    retained_reasoning_tokens: int,
    full_sequence_tokens: int,
    full_fit: bool,
    reasoning_truncated: bool,
    analysis: StructuralAnalysis,
    mandatory_suffix_tokens: int,
    final_response_text: str,
    eos_id: int,
) -> SequenceExampleV2:
    response_tokens_including_eos = len(input_ids) - prompt_tokens
    labels = response_only_labels(
        input_ids,
        prompt_tokens=prompt_tokens,
        response_tokens_including_eos=response_tokens_including_eos,
        pad_token_id=None,
    )
    if input_ids[-1] != eos_id or labels[-1] != eos_id:
        raise AssertionError("v2 construction must supervise the final EOS token")
    ratio = 1.0 if original_reasoning_tokens == 0 else retained_reasoning_tokens / original_reasoning_tokens
    location = analysis.code_location
    return SequenceExampleV2(
        input_ids=input_ids,
        labels=labels,
        final_response_text=final_response_text,
        prompt_tokens=prompt_tokens,
        response_tokens=response_tokens_including_eos - 1,
        response_tokens_including_eos=response_tokens_including_eos,
        original_response_tokens=original_response_tokens,
        original_reasoning_tokens=original_reasoning_tokens,
        retained_reasoning_tokens=retained_reasoning_tokens,
        retained_reasoning_ratio=ratio,
        full_sequence_tokens=full_sequence_tokens,
        final_sequence_tokens=len(input_ids),
        full_fit=full_fit,
        reasoning_truncated=reasoning_truncated,
        code_locatable=location is not None,
        code_location_method=location.method if location else None,
        code_start_char=location.start_char if location else None,
        code_end_char=location.end_char if location else None,
        starts_with_think=analysis.starts_with_think,
        has_matching_closed_think=analysis.has_matching_closed_think,
        fenced_code=analysis.fenced_code,
        mandatory_suffix_tokens=mandatory_suffix_tokens,
        mandatory_suffix_kind=analysis.suffix_kind,
        mandatory_suffix_start_char=analysis.suffix_start_char,
        final_code_preserved=True,
        reference_code_appended=False,
        eos_supervised=True,
        dropped=False,
        drop_reason=None,
    )


def build_sft_sequence_v2(
    tokenizer: Any,
    *,
    prompt: str,
    r1_generation: str,
    reference_code: str,
    max_sequence_length: int,
    max_response_tokens_including_eos: int,
) -> SequenceExampleV2:
    """Construct one v2 sequence with structure-preserving middle truncation."""

    if max_sequence_length < 2 or max_response_tokens_including_eos < 2:
        raise ValueError("Sequence and response budgets must leave room for response and EOS")
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        raise ValueError("Tokenizer must define eos_token_id")

    prompt_ids, _ = _tokenize(tokenizer, prompt)
    response_ids, response_offsets = _tokenize(tokenizer, r1_generation, offsets=True)
    analysis = analyze_response_structure(r1_generation, reference_code)
    full_ids = prompt_ids + response_ids + [int(eos_id)]
    full_len = len(full_ids)
    if not analysis.valid_structure:
        return _drop(
            prompt_tokens=len(prompt_ids),
            original_response_tokens=len(response_ids),
            full_sequence_tokens=full_len,
            analysis=analysis,
            reason=analysis.invalid_reason or "invalid_structure",
        )

    if len(response_ids) + 1 <= max_response_tokens_including_eos:
        if full_len > max_sequence_length:
            return _drop(
                prompt_tokens=len(prompt_ids),
                original_response_tokens=len(response_ids),
                full_sequence_tokens=full_len,
                analysis=analysis,
                reason="prompt_plus_response_over_sequence_budget",
            )
        code_start_token = (
            _first_overlapping_token(response_offsets, analysis.code_location) or len(response_ids)
            if analysis.code_location
            else len(response_ids)
        )
        return _make(
            input_ids=full_ids,
            prompt_tokens=len(prompt_ids),
            original_response_tokens=len(response_ids),
            original_reasoning_tokens=code_start_token,
            retained_reasoning_tokens=code_start_token,
            full_sequence_tokens=full_len,
            full_fit=True,
            reasoning_truncated=False,
            analysis=analysis,
            mandatory_suffix_tokens=len(_tokenize(tokenizer, r1_generation[analysis.suffix_start_char or 0])[0]),
            final_response_text=r1_generation,
            eos_id=int(eos_id),
        )

    if analysis.suffix_start_char is None:
        return _drop(
            prompt_tokens=len(prompt_ids),
            original_response_tokens=len(response_ids),
            full_sequence_tokens=full_len,
            analysis=analysis,
            reason="unlocatable_final_code",
        )

    suffix_text = r1_generation[analysis.suffix_start_char :]
    suffix_ids, _ = _tokenize(tokenizer, suffix_text)
    mandatory_suffix_tokens = len(suffix_ids)
    if mandatory_suffix_tokens + 1 > max_response_tokens_including_eos:
        return _drop(
            prompt_tokens=len(prompt_ids),
            original_response_tokens=len(response_ids),
            full_sequence_tokens=full_len,
            analysis=analysis,
            reason="mandatory_structural_suffix_over_response_budget",
            mandatory_suffix_tokens=mandatory_suffix_tokens,
        )

    prefix_source = r1_generation[: analysis.suffix_start_char]
    prefix_tokenized = _tokenize(tokenizer, prefix_source, offsets=True)
    required_prefix = r1_generation[: analysis.think_open_end] if analysis.starts_with_think and analysis.think_open_end is not None else ""
    available_prefix_tokens = max_response_tokens_including_eos - mandatory_suffix_tokens - 1
    code_start_token = (
        _first_overlapping_token(response_offsets, analysis.code_location) or len(response_ids)
        if analysis.code_location
        else len(response_ids)
    )
    chosen_text: str | None = None
    chosen_response_ids: list[int] | None = None
    candidate_budget = available_prefix_tokens
    while candidate_budget >= 0:
        prefix_text = _prefix_text_for_token_budget(
            tokenizer,
            prefix_source,
            candidate_budget,
            required_prefix,
            tokenized=prefix_tokenized,
        )
        if prefix_text is None:
            candidate_budget += 1
            break
        candidate_text = prefix_text + suffix_text
        candidate_ids, _ = _tokenize(tokenizer, candidate_text)
        overflow = len(candidate_ids) + 1 - max_response_tokens_including_eos
        if overflow <= 0:
            chosen_text = candidate_text
            chosen_response_ids = candidate_ids
            break
        candidate_budget -= max(1, overflow)

    if chosen_text is None or chosen_response_ids is None:
        return _drop(
            prompt_tokens=len(prompt_ids),
            original_response_tokens=len(response_ids),
            full_sequence_tokens=full_len,
            analysis=analysis,
            reason="reasoning_prefix_cannot_fit",
            mandatory_suffix_tokens=mandatory_suffix_tokens,
        )

    final_ids = prompt_ids + chosen_response_ids + [int(eos_id)]
    if len(final_ids) > max_sequence_length:
        return _drop(
            prompt_tokens=len(prompt_ids),
            original_response_tokens=len(response_ids),
            full_sequence_tokens=full_len,
            analysis=analysis,
            reason="prompt_plus_preserved_code_overflow",
            mandatory_suffix_tokens=mandatory_suffix_tokens,
        )
    retained_reasoning_tokens = max(0, len(chosen_response_ids) - mandatory_suffix_tokens)
    return _make(
        input_ids=final_ids,
        prompt_tokens=len(prompt_ids),
        original_response_tokens=len(response_ids),
        original_reasoning_tokens=code_start_token,
        retained_reasoning_tokens=retained_reasoning_tokens,
        full_sequence_tokens=full_len,
        full_fit=False,
        reasoning_truncated=True,
        analysis=analysis,
        mandatory_suffix_tokens=mandatory_suffix_tokens,
        final_response_text=chosen_text,
        eos_id=int(eos_id),
    )
