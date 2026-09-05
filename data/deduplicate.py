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


def detect_sft_pt_overlaps(
    sft_records: list[dict[str, Any]],
    pt_records: list[dict[str, Any]],
    *,
    ngram_size: int,
    near_duplicate_threshold: float,
) -> dict[str, Any]:
    sft_ids = {record["problem_id"] for record in sft_records}
    pt_ids = {record["problem_id"] for record in pt_records}
    exact_id_overlap = sorted(sft_ids & pt_ids)

    sft_hashes: dict[str, list[str]] = defaultdict(list)
    pt_hashes: dict[str, list[str]] = defaultdict(list)
    for record in sft_records:
        sft_hashes[normalized_text_hash(problem_text_for_dedup(record))].append(record["problem_id"])
    for record in pt_records:
        pt_hashes[normalized_text_hash(problem_text_for_dedup(record))].append(record["problem_id"])

    exact_text_overlap: list[dict[str, Any]] = []
    for text_hash in sorted(set(sft_hashes) & set(pt_hashes)):
        exact_text_overlap.append(
            {
                "normalized_text_hash": text_hash,
                "sft_problem_ids": sorted(sft_hashes[text_hash]),
                "pt_problem_ids": sorted(pt_hashes[text_hash]),
            }
        )

    pt_shingles = {
        record["problem_id"]: token_ngrams(problem_text_for_dedup(record), ngram_size=ngram_size)
        for record in pt_records
    }
    pt_inverted: dict[str, set[str]] = defaultdict(set)
    for pt_id, shingles in pt_shingles.items():
        for shingle in shingles:
            pt_inverted[shingle].add(pt_id)

    near_duplicates: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for record in sft_records:
        sft_id = record["problem_id"]
        sft_set = token_ngrams(problem_text_for_dedup(record), ngram_size=ngram_size)
        candidate_pt_ids: set[str] = set()
        for shingle in sft_set:
            candidate_pt_ids.update(pt_inverted.get(shingle, set()))
        for pt_id in candidate_pt_ids:
            pair = (sft_id, pt_id)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            pt_set = pt_shingles[pt_id]
            score = jaccard(sft_set, pt_set)
            if score >= near_duplicate_threshold:
                near_duplicates.append(
                    {
                        "sft_problem_id": sft_id,
                        "pt_problem_id": pt_id,
                        "jaccard": round(score, 6),
                    }
                )

    blocked_pt_ids = set(exact_id_overlap)
    for item in exact_text_overlap:
        blocked_pt_ids.update(item["pt_problem_ids"])
    blocked_pt_ids.update(item["pt_problem_id"] for item in near_duplicates)

    return {
        "sft_count": len(sft_records),
        "pt_count": len(pt_records),
        "exact_id_overlap_count": len(exact_id_overlap),
        "exact_id_overlap": exact_id_overlap,
        "exact_text_overlap_count": len(exact_text_overlap),
        "exact_text_overlap": exact_text_overlap,
        "near_duplicate_count": len(near_duplicates),
        "near_duplicates": near_duplicates,
        "ngram_size": ngram_size,
        "near_duplicate_threshold": near_duplicate_threshold,
        "blocked_pt_problem_ids": sorted(blocked_pt_ids),
    }
