from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any

from eval.phase2_common import (
    config_hash,
    file_sha256,
    git_command,
    git_status_short,
    load_jsonl,
    numeric_stats,
    package_versions,
    path_from_config,
    pretrained_load_reference,
    read_config,
    reports_dir,
    runtime_info,
    serialize_prompt,
    view_path,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile Phase 2 tokenizer lengths without truncation.")
    parser.add_argument("--config", default="configs/base_eval.yaml")
    return parser.parse_args()


def _load_tokenizer(config: dict[str, Any]):
    from transformers import AutoTokenizer

    source, kwargs = pretrained_load_reference(config, "tokenizer_revision")
    try:
        return AutoTokenizer.from_pretrained(source, **kwargs)
    except OSError as exc:
        raise RuntimeError(
            "Failed to load the fixed Qwen2.5-Coder-3B tokenizer from cache or Hugging Face. "
            "Run bash scripts/03a_phase2_prefetch_base_model.sh first; if direct Hugging Face SSL fails, "
            "retry that prefetch with HF_ENDPOINT set to a reachable endpoint."
        ) from exc


def _count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"])


def _profile_split(config: dict[str, Any], tokenizer, split: str) -> dict[str, Any]:
    records = load_jsonl(view_path(config, split))
    prompt_tokens: list[int] = []
    reasoning_tokens: list[int] = []
    ocr_code_tokens: list[int] = []
    reasoning_code_tokens: list[int] = []
    max_new_tokens = int(config["generation"]["max_new_tokens"])
    context_window = int(config["model"]["max_position_embeddings"])
    overflow_problem_ids: list[str] = []

    for record in records:
        prompt = serialize_prompt(config, str(record["prompt"]))
        reasoning = str(record["reasoning"])
        ocr_code = str(record["reference_code"])
        prompt_len = _count_tokens(tokenizer, prompt)
        reasoning_len = _count_tokens(tokenizer, reasoning)
        code_len = _count_tokens(tokenizer, ocr_code)
        total_len = _count_tokens(tokenizer, reasoning + "\n" + ocr_code)
        prompt_tokens.append(prompt_len)
        reasoning_tokens.append(reasoning_len)
        ocr_code_tokens.append(code_len)
        reasoning_code_tokens.append(total_len)
        if prompt_len + max_new_tokens > context_window:
            overflow_problem_ids.append(record["problem_id"])

    return {
        "count": len(records),
        "prompt_tokens": numeric_stats(prompt_tokens, (50, 90, 95, 99)),
        "reasoning_tokens": numeric_stats(reasoning_tokens, (50, 95, 99)),
        "ocr_code_tokens": numeric_stats(ocr_code_tokens, (50, 95, 99)),
        "reasoning_plus_code_tokens": numeric_stats(reasoning_code_tokens, (50, 95, 99)),
        "prompt_plus_max_new_tokens_over_context_count": len(overflow_problem_ids),
        "prompt_plus_max_new_tokens_over_context_problem_id_sample": overflow_problem_ids[:20],
    }


def profile_tokens(config_path: str) -> dict[str, Any]:
    config = read_config(config_path)
    tokenizer = _load_tokenizer(config)
    splits = list(config["token_profile"]["splits"])
    split_profiles = {split: _profile_split(config, tokenizer, split) for split in splits}
    overflow_count = sum(
        int(profile["prompt_plus_max_new_tokens_over_context_count"]) for profile in split_profiles.values()
    )
    if overflow_count and config["token_profile"].get("fail_if_prompt_plus_generation_exceeds_context", True):
        status = "failed_context_window_check"
    else:
        status = "completed"

    report = {
        "phase": "phase2_base_baseline",
        "step": "token_length_profile",
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": git_command(["rev-parse", "HEAD"]),
            "dirty": bool(git_status_short()),
            "status_short": git_status_short(),
        },
        "config_file": config_path,
        "config_hash": config_hash(config_path),
        "model_repo": config["model"]["repo_id"],
        "tokenizer_revision": config["model"]["tokenizer_revision"],
        "tokenizer_class": tokenizer.__class__.__name__,
        "model_max_length": getattr(tokenizer, "model_max_length", None),
        "configured_context_window": int(config["model"]["max_position_embeddings"]),
        "max_new_tokens": int(config["generation"]["max_new_tokens"]),
        "views": {split: {"path": str(view_path(config, split)), "hash": file_sha256(view_path(config, split))} for split in splits},
        "splits": split_profiles,
        "runtime": runtime_info(),
        "package_versions": package_versions(),
    }
    report_path = reports_dir(config) / "phase2_token_profile.json"
    write_json(report_path, report)
    if status != "completed":
        raise SystemExit(f"Token profile failed context-window check; see {report_path}")
    return report


def main() -> None:
    args = parse_args()
    report = profile_tokens(args.config)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
