# CodeReason-PT

Verifier-driven post-training pipeline for code reasoning models.

This repository is being implemented phase by phase from `PLAN.md`. The current code is limited to Phase 0: data configuration, schema validation, exact-key reasoning joins, deterministic SFT/PT/Dev splitting, cross-split de-duplication, fixed reward/held-out testcase splitting, and lightweight provenance auditing.

Large artifacts should live under:

```text
/mnt/data/liangjunwei/CodeReason-PT
```

Expected raw Phase 0 inputs:

```text
raw/problems.jsonl
raw/reasoning.jsonl
```

The join is intentionally high-confidence only. Records are joined by exact `problem_id` or exact `(source, source_id)`. For OpenCodeReasoning-2 to APPS/TACO raw construction, `source_id` includes the source split, for example `train:123`, and `problem_id` becomes `taco:train:123`. No embedding, semantic, or fuzzy join is used.

To build raw files from the configured public sources:

```text
python -m data.build_raw --config configs/phase0.yaml
python -m data.prepare --config configs/phase0.yaml
python -m data.validate --config configs/phase0.yaml
python -m data.audit --config configs/phase0.yaml --output phase0_audit.json
```

The public-source builder joins OpenCodeReasoning-2 to TACO/APPS only by exact `(dataset, source split, index)`. It scans OpenCodeReasoning-2 until `max_opencode_records` or stream exhaustion, not until a fixed candidate multiplier is reached.

Phase 0 currently uses split-specific testcase thresholds:

```text
SFT >= 2 unique tests
PT  >= 10 unique tests
Dev >= 10 unique tests
```

`phase0_audit.json` is safe to commit: it contains only counts, hashes, distributions, drop summaries, overlap counts, gate status, and package versions. It does not contain prompts, reasoning, code, or testcase bodies.
