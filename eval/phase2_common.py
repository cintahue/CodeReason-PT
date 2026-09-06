from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

from data.config import load_config
from data.schemas import iter_jsonl, stable_hash


ROOT = Path(__file__).resolve().parents[1]
MODEL_SNAPSHOT_METADATA = ".codereason_phase2_snapshot.json"
PHASE2_MODEL_FILES: tuple[dict[str, Any], ...] = (
    {"path": ".gitattributes", "size": 1519},
    {"path": "LICENSE", "size": 7388},
    {"path": "README.md", "size": 4068},
    {"path": "config.json", "size": 661},
    {"path": "generation_config.json", "size": 139},
    {"path": "merges.txt", "size": 1671839},
    {
        "path": "model-00001-of-00002.safetensors",
        "size": 4957560304,
        "sha256": "f4528d2c6caa86a8f922f26675abff773a9b7dd1ccf255eb97255ae76373ada9",
    },
    {
        "path": "model-00002-of-00002.safetensors",
        "size": 1214366696,
        "sha256": "007c3310312712a7762fd0d1db92de9be7cedb2e6eb6dc0afff74f143a9dbb23",
    },
    {"path": "model.safetensors.index.json", "size": 35581},
    {"path": "tokenizer.json", "size": 7031645},
    {"path": "tokenizer_config.json", "size": 7228},
    {"path": "vocab.json", "size": 2776833},
)


def read_config(path: str | Path) -> dict[str, Any]:
    config = load_config(path)
    if config.get("phase") != "phase2_base_baseline":
        raise ValueError(f"Unexpected config phase: {config.get('phase')}")
    return config


def path_from_config(config: dict[str, Any], *keys: str) -> Path:
    value: Any = config
    for key in keys:
        value = value[key]
    return Path(str(value)).expanduser()


def artifact_root(config: dict[str, Any]) -> Path:
    return path_from_config(config, "paths", "artifact_root")


def views_dir(config: dict[str, Any]) -> Path:
    return artifact_root(config) / "views"


def reports_dir(config: dict[str, Any]) -> Path:
    return artifact_root(config) / "reports"


def rollouts_dir(config: dict[str, Any]) -> Path:
    return artifact_root(config) / "rollouts"


def smoke_dir(config: dict[str, Any]) -> Path:
    return artifact_root(config) / "smoke"


def view_path(config: dict[str, Any], split: str) -> Path:
    return views_dir(config) / str(config["views"][split])


def write_json(path: str | Path, value: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def config_hash(path: str | Path) -> str:
    return file_sha256(path)


def phase2_model_files() -> list[dict[str, Any]]:
    return [dict(item) for item in PHASE2_MODEL_FILES]


def model_local_dir(config: dict[str, Any]) -> Path | None:
    local_dir = config["model"].get("local_dir")
    if not local_dir:
        return None
    return Path(str(local_dir)).expanduser()


def local_model_snapshot_status(config: dict[str, Any], *, verify_hashes: bool = False) -> dict[str, Any]:
    local_dir = model_local_dir(config)
    model_config = config["model"]
    if local_dir is None:
        return {"configured": False, "complete": False, "files": {}}

    metadata_path = local_dir / MODEL_SNAPSHOT_METADATA
    metadata: dict[str, Any] | None = None
    metadata_error = None
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            metadata_error = str(exc)

    metadata_ok = bool(
        metadata
        and metadata.get("repo_id") == model_config["repo_id"]
        and metadata.get("model_revision") == model_config["model_revision"]
        and metadata.get("tokenizer_revision") == model_config["tokenizer_revision"]
    )

    file_statuses: dict[str, dict[str, Any]] = {}
    for spec in PHASE2_MODEL_FILES:
        path = local_dir / str(spec["path"])
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        expected_size = int(spec["size"])
        status = {
            "exists": exists,
            "size": size,
            "expected_size": expected_size,
            "size_ok": exists and size == expected_size,
        }
        expected_sha = spec.get("sha256")
        if expected_sha:
            status["expected_sha256"] = expected_sha
            if verify_hashes and status["size_ok"]:
                actual_sha = file_sha256(path)
                status["sha256"] = actual_sha
                status["sha256_ok"] = actual_sha == expected_sha
            else:
                status["sha256_ok"] = bool(metadata_ok and status["size_ok"])
        file_statuses[str(spec["path"])] = status

    complete = metadata_ok and all(item["size_ok"] for item in file_statuses.values())
    if verify_hashes:
        complete = complete and all(item.get("sha256_ok", True) for item in file_statuses.values())

    return {
        "configured": True,
        "complete": complete,
        "local_dir": str(local_dir),
        "metadata_path": str(metadata_path),
        "metadata_ok": metadata_ok,
        "metadata_error": metadata_error,
        "files": file_statuses,
    }


def git_command(args: list[str]) -> str:
    try:
        completed = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ""
    return completed.stdout.strip()


def git_status_short() -> list[str]:
    try:
        completed = subprocess.run(["git", "status", "--short"], cwd=ROOT, text=True, capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    return completed.stdout.splitlines()


def stable_problem_sample(records: list[dict[str, Any]], *, count: int, seed: int) -> list[dict[str, Any]]:
    keyed = sorted(
        records,
        key=lambda record: stable_hash({"seed": seed, "problem_id": record["problem_id"]}),
    )
    return keyed[:count]


def serialize_prompt(config: dict[str, Any], problem: str) -> str:
    prompt_config = config["prompt"]
    if prompt_config.get("use_chat_template") is not False:
        raise ValueError("Phase 2 base prompt must not use chat templates")
    return str(prompt_config["template"]).format(problem=problem)


def generation_config_for_hash(config: dict[str, Any]) -> dict[str, Any]:
    generation = dict(config["generation"])
    return {
        "do_sample": bool(generation["do_sample"]),
        "num_return_sequences": int(generation["num_return_sequences"]),
        "max_new_tokens": int(generation["max_new_tokens"]),
        "use_cache": bool(generation.get("use_cache", True)),
    }


def hf_local_files_only() -> bool:
    return os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"


def model_cache_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    model_config = config["model"]
    kwargs: dict[str, Any] = {}
    cache_dir = model_config.get("cache_dir")
    if cache_dir:
        kwargs["cache_dir"] = str(Path(str(cache_dir)).expanduser())
    if hf_local_files_only():
        kwargs["local_files_only"] = True
    return kwargs


def pretrained_load_reference(config: dict[str, Any], revision_key: str) -> tuple[str, dict[str, Any]]:
    model_config = config["model"]
    local_status = local_model_snapshot_status(config)
    trust_remote_code = bool(model_config.get("trust_remote_code", False))
    if local_status.get("complete"):
        return str(model_local_dir(config)), {"trust_remote_code": trust_remote_code}
    if hf_local_files_only():
        raise FileNotFoundError(
            "Fixed Phase 2 model snapshot is not complete locally. "
            "Run bash scripts/03a_phase2_prefetch_base_model.sh before offline profile/smoke/dev."
        )
    return (
        str(model_config["repo_id"]),
        {
            "revision": model_config[revision_key],
            "trust_remote_code": trust_remote_code,
            **model_cache_kwargs(config),
        },
    )


def percentile(values: list[int | float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, int(round((q / 100.0) * (len(ordered) - 1)))))
    return ordered[index]


def numeric_stats(values: list[int | float], percentiles: tuple[int, ...]) -> dict[str, float]:
    if not values:
        return {"count": 0, "max": 0.0}
    output: dict[str, float] = {"count": float(len(values))}
    for q in percentiles:
        output[f"p{q}"] = percentile(values, q)
    output["max"] = max(float(value) for value in values)
    return output


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for module_name in (
        "torch",
        "transformers",
        "tokenizers",
        "accelerate",
        "huggingface_hub",
        "datasets",
        "pandas",
        "pyarrow",
    ):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", None)
        except ModuleNotFoundError:
            versions[module_name] = None
    return versions


def gpu_info() -> list[dict[str, Any]]:
    try:
        import torch
    except ModuleNotFoundError:
        return []
    if not torch.cuda.is_available():
        return []
    output = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        output.append(
            {
                "index": index,
                "name": props.name,
                "total_vram_mb": int(props.total_memory // (1024 * 1024)),
                "capability": [int(props.major), int(props.minor)],
            }
        )
    return output


def runtime_info() -> dict[str, Any]:
    return {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": os.sys.version,
        "package_versions": package_versions(),
        "gpu": gpu_info(),
    }


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ModuleNotFoundError:
        return
