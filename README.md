# CodeReason-PT

Verifier-driven post-training pipeline for code reasoning models.

This repository is being implemented phase by phase from `PLAN.md`. The current code is limited to Phase 0: data configuration, schema validation, exact-key reasoning joins, deterministic SFT/PT/Dev splitting, SFT/PT de-duplication, and fixed reward/held-out testcase splitting.

Large artifacts should live under:

```text
/mnt/data/liangjunwei/CodeReason-PT
```

Expected raw Phase 0 inputs:

```text
raw/problems.jsonl
raw/reasoning.jsonl
```

The join is intentionally high-confidence only. Records are joined by exact `problem_id` or exact `(source, source_id)`. No embedding, semantic, or fuzzy join is used.

To build raw files from the configured public sources:

```text
python -m data.build_raw --config configs/phase0.yaml
python -m data.prepare --config configs/phase0.yaml
python -m data.validate --config configs/phase0.yaml
```

The public-source builder joins OpenCodeReasoning-2 to TACO/APPS only by exact `(dataset, split, index)`.
