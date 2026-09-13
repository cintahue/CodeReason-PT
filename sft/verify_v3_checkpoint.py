from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from data.config import load_config
from eval.phase2_common import file_sha256, write_json
from sft.train_v3 import _load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load-check a Phase 3-v3 LoRA checkpoint.")
    parser.add_argument("--config", default="configs/sft_v3.yaml")
    parser.add_argument("--checkpoint", default=None)
    return parser.parse_args()


def verify(config_path: str, checkpoint_override: str | None) -> dict[str, object]:
    config = load_config(config_path)
    checkpoint = Path(checkpoint_override or config["training"]["output_dir"]).expanduser()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        if not (checkpoint / name).exists():
            raise FileNotFoundError(checkpoint / name)

    from peft import PeftConfig, PeftModel

    adapter_config = PeftConfig.from_pretrained(str(checkpoint))
    model, tokenizer = _load_model_and_tokenizer(config)
    loaded = PeftModel.from_pretrained(model, str(checkpoint), is_trainable=False)
    loaded.eval()
    report = {
        "phase": "phase3_reasoning_sft_v3",
        "step": "checkpoint_save_load_check",
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(Path(config_path).resolve()),
        "config_hash": file_sha256(config_path),
        "checkpoint_path": str(checkpoint),
        "adapter_hash": file_sha256(checkpoint / "adapter_model.safetensors"),
        "base_model_name_or_path": str(getattr(adapter_config, "base_model_name_or_path", "")),
        "tokenizer_eos_token_id": int(tokenizer.eos_token_id),
        "adapter_load_check": "passed",
        "v1_v2_artifacts_touched": False,
    }
    mode = "smoke" if checkpoint == Path(str(config["smoke"]["output_dir"])).expanduser() else "train"
    report_path = Path(str(config["paths"]["artifact_root"])).expanduser() / mode / "checkpoint_load_report.json"
    write_json(report_path, report)
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    args = parse_args()
    print(json.dumps(verify(args.config, args.checkpoint), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
