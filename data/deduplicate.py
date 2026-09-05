from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Any

from data.schemas import problem_text_for_dedup


def normalize_text(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9_]+", " ", lowered)
    return " ".join(lowered.split())


def normalized_text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def token_ngrams(text: str, *, ngram_size: int) -> set[str]:
    tokens = normalize_text(text).split()
    if not tokens:
        return set()
    if len(tokens) < ngram_size:
        return {" ".join(tokens)}
    return {" ".join(tokens[index : index + ngram_size]) for index in range(len(tokens) - ngram_size + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def detect_cross_split_overlaps(
    left_records: list[dict[str, Any]],
    right_records: list[dict[str, Any]],
    *,
    left_name: str,
    right_name: str,
    ngram_size: int,
    near_duplicate_threshold: float,
) -> dict[str, Any]:
    left_ids = {record["problem_id"] for record in left_records}
    right_ids = {record["problem_id"] for record in right_records}
    exact_id_overlap = sorted(left_ids & right_ids)

    left_hashes: dict[str, list[str]] = defaultdict(list)
    right_hashes: dict[str, list[str]] = defaultdict(list)
    for record in left_records:
        left_hashes[normalized_text_hash(problem_text_for_dedup(record))].append(record["problem_id"])
    for record in right_records:
        right_hashes[normalized_text_hash(problem_text_for_dedup(record))].append(record["problem_id"])

    exact_text_overlap: list[dict[str, Any]] = []
    for text_hash in sorted(set(left_hashes) & set(right_hashes)):
        exact_text_overlap.append(
            {
                "normalized_text_hash": text_hash,
                f"{left_name}_problem_ids": sorted(left_hashes[text_hash]),
                f"{right_name}_problem_ids": sorted(right_hashes[text_hash]),
            }
        )

    right_shingles = {
        record["problem_id"]: token_ngrams(problem_text_for_dedup(record), ngram_size=ngram_size)
        for record in right_records
    }
    right_inverted: dict[str, set[str]] = defaultdict(set)
    for right_id, shingles in right_shingles.items():
        for shingle in shingles:
            right_inverted[shingle].add(right_id)

    near_duplicates: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for record in left_records:
        left_id = record["problem_id"]
        left_set = token_ngrams(problem_text_for_dedup(record), ngram_size=ngram_size)
        candidate_right_ids: set[str] = set()
        for shingle in left_set:
            candidate_right_ids.update(right_inverted.get(shingle, set()))
        for right_id in candidate_right_ids:
            pair = (left_id, right_id)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            right_set = right_shingles[right_id]
            score = jaccard(left_set, right_set)
            if score >= near_duplicate_threshold:
                near_duplicates.append(
                    {
                        f"{left_name}_problem_id": left_id,
                        f"{right_name}_problem_id": right_id,
                        "jaccard": round(score, 6),
                    }
                )

    blocked_right_ids = set(exact_id_overlap)
    for item in exact_text_overlap:
        blocked_right_ids.update(item[f"{right_name}_problem_ids"])
    blocked_right_ids.update(item[f"{right_name}_problem_id"] for item in near_duplicates)

    return {
        f"{left_name}_count": len(left_records),
        f"{right_name}_count": len(right_records),
        "exact_id_overlap_count": len(exact_id_overlap),
        "exact_id_overlap": exact_id_overlap,
        "exact_text_overlap_count": len(exact_text_overlap),
        "exact_text_overlap": exact_text_overlap,
        "near_duplicate_count": len(near_duplicates),
        "near_duplicates": near_duplicates,
        "ngram_size": ngram_size,
        "near_duplicate_threshold": near_duplicate_threshold,
        "blocked_right_problem_ids": sorted(blocked_right_ids),
        f"blocked_{right_name}_problem_ids": sorted(blocked_right_ids),
    }


def detect_sft_pt_overlaps(
    sft_records: list[dict[str, Any]],
    pt_records: list[dict[str, Any]],
    *,
    ngram_size: int,
    near_duplicate_threshold: float,
) -> dict[str, Any]:
    return detect_cross_split_overlaps(
        sft_records,
        pt_records,
        left_name="sft",
        right_name="pt",
        ngram_size=ngram_size,
        near_duplicate_threshold=near_duplicate_threshold,
    )
