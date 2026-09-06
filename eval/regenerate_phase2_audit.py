from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.phase2_common import read_config, reports_dir, write_json
from eval.run_base_baseline import _audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate Phase 2 audit from existing Dev baseline artifacts.")
    parser.add_argument("--config", default="configs/base_eval.yaml")
    return parser.parse_args()


def regenerate_audit(config_path: str) -> dict:
    config = read_config(config_path)
    report_path = reports_dir(config) / "base_dev_metrics.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Missing Phase 2 Dev metrics report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected = int(config["eval"]["expected_count"])
    if report.get("mode") != "dev" or int(report.get("evaluated", -1)) != expected:
        raise ValueError(f"Dev report does not match expected final Dev count {expected}: {report_path}")
    rollouts_path = Path(str(report["rollouts_path"]))
    if not rollouts_path.exists():
        raise FileNotFoundError(f"Missing Phase 2 Dev rollouts: {rollouts_path}")
    audit = _audit(config_path, config, report, rollouts_path)
    write_json("phase2_base_audit.json", audit)
    return audit


def main() -> None:
    args = parse_args()
    audit = regenerate_audit(args.config)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
