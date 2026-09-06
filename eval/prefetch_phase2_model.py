from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from eval.phase2_common import (
    MODEL_SNAPSHOT_METADATA,
    config_hash,
    file_sha256,
    git_command,
    git_status_short,
    local_model_snapshot_status,
    model_local_dir,
    package_versions,
    phase2_model_files,
    read_config,
    reports_dir,
    runtime_info,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prefetch the fixed Phase 2 base model snapshot.")
    parser.add_argument("--config", default="configs/base_eval.yaml")
    return parser.parse_args()


def _download_url(endpoint: str, repo_id: str, revision: str, filename: str) -> str:
    quoted_file = quote(filename, safe="/._-")
    return f"{endpoint.rstrip('/')}/{repo_id}/resolve/{revision}/{quoted_file}"


def _file_complete(path: Path, spec: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    if path.stat().st_size != int(spec["size"]):
        return False
    expected_sha = spec.get("sha256")
    if expected_sha:
        return file_sha256(path) == expected_sha
    return True


def _download_one(endpoint: str, repo_id: str, revision: str, local_dir: Path, spec: dict[str, Any]) -> dict[str, Any]:
    filename = str(spec["path"])
    output_path = local_dir / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if _file_complete(output_path, spec):
        return {
            "path": filename,
            "status": "already_complete",
            "size": output_path.stat().st_size,
            "expected_size": int(spec["size"]),
        }
    if output_path.exists():
        size = output_path.stat().st_size
        expected_size = int(spec["size"])
        if size > expected_size:
            output_path.unlink()
        elif size == expected_size and spec.get("sha256"):
            output_path.unlink()

    url = _download_url(endpoint, repo_id, revision, filename)
    retries = os.environ.get("CODEREASON_CURL_RETRIES", "30")
    retry_delay = os.environ.get("CODEREASON_CURL_RETRY_DELAY", "5")
    connect_timeout = os.environ.get("CODEREASON_CURL_CONNECT_TIMEOUT", "30")
    low_speed_limit = os.environ.get("CODEREASON_CURL_LOW_SPEED_LIMIT", "1024")
    low_speed_time = os.environ.get("CODEREASON_CURL_LOW_SPEED_TIME", "120")
    command = [
        "curl",
        "--fail",
        "--location",
        "--continue-at",
        "-",
        "--retry",
        retries,
        "--retry-delay",
        retry_delay,
        "--retry-all-errors",
        "--connect-timeout",
        connect_timeout,
        "--output",
        str(output_path),
        url,
    ]
    if low_speed_limit and low_speed_time:
        command[1:1] = ["--speed-limit", low_speed_limit, "--speed-time", low_speed_time]
    subprocess.run(command, check=True)

    if not _file_complete(output_path, spec):
        actual_size = output_path.stat().st_size if output_path.exists() else 0
        raise RuntimeError(f"Downloaded file failed size/hash check: {filename}, size={actual_size}")
    return {
        "path": filename,
        "status": "downloaded",
        "size": output_path.stat().st_size,
        "expected_size": int(spec["size"]),
    }


def prefetch_model(config_path: str) -> dict[str, Any]:
    config = read_config(config_path)
    model_config = config["model"]
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    local_dir = model_local_dir(config)
    if local_dir is None:
        raise ValueError("configs/base_eval.yaml must set model.local_dir for direct-resume prefetch")
    local_dir.mkdir(parents=True, exist_ok=True)

    before = local_model_snapshot_status(config)
    downloads = [
        _download_one(endpoint, model_config["repo_id"], model_config["model_revision"], local_dir, spec)
        for spec in phase2_model_files()
    ]
    metadata_path = local_dir / MODEL_SNAPSHOT_METADATA
    write_json(
        metadata_path,
        {
            "repo_id": model_config["repo_id"],
            "model_revision": model_config["model_revision"],
            "tokenizer_revision": model_config["tokenizer_revision"],
            "download_method": "direct_curl",
            "hf_endpoint": endpoint,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "files": downloads,
        },
    )
    after = local_model_snapshot_status(config, verify_hashes=True)
    if not after["complete"]:
        raise RuntimeError(f"Local model snapshot is incomplete after prefetch: {local_dir}")

    report = {
        "phase": "phase2_base_baseline",
        "step": "prefetch_fixed_base_model",
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": git_command(["rev-parse", "HEAD"]),
            "dirty": bool(git_status_short()),
            "status_short": git_status_short(),
        },
        "config_file": config_path,
        "config_hash": config_hash(config_path),
        "repo_id": model_config["repo_id"],
        "model_revision": model_config["model_revision"],
        "tokenizer_revision": model_config["tokenizer_revision"],
        "hf_endpoint": endpoint,
        "cache_dir": model_config.get("cache_dir"),
        "local_dir": str(local_dir),
        "snapshot_metadata_path": str(metadata_path),
        "download_method": "direct_curl",
        "before": before,
        "downloads": downloads,
        "after": after,
        "runtime": runtime_info(),
        "package_versions": package_versions(),
    }
    report_path = reports_dir(config) / "phase2_model_prefetch_report.json"
    write_json(report_path, report)
    return report


def main() -> None:
    args = parse_args()
    try:
        report = prefetch_model(args.config)
    except Exception as exc:
        endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
        raise SystemExit(
            "Failed to prefetch the fixed Phase 2 base model snapshot.\n"
            f"Endpoint: {endpoint}\n"
            "This is a network/download failure before Phase 2 evaluation starts.\n"
            "Retry with a reachable endpoint, for example:\n"
            "  env HF_ENDPOINT=https://hf-mirror.com bash scripts/03a_phase2_prefetch_base_model.sh\n"
            "Partial files in model.local_dir are kept and reused with curl -C - on the next run.\n"
            "The model repo and commit SHA remain fixed by configs/base_eval.yaml."
        ) from exc
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
