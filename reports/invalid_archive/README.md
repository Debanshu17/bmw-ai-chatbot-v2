# Invalid Reports Archive

This directory contains evaluation reports and checkpoint files produced prior to the eval harness hardening.

## Reasons for Invalidation
1. **v1 Baseline runs (`eval_v1_baseline_*.json`)**: In earlier runs without API retry and error decoupling, Gemini API calls encountered HTTP 429 quota exhaustion (Free Tier limit exceeded), causing all 40 baseline queries to return `SYSTEM ERROR` and score 0.0% accuracy artificially.
2. **v2 Production runs (`eval_v2_production_*.json`)**: When API calls failed with 429 errors, the pipeline silently fell back to extractive text slicing directly from retrieved vector chunks rather than generating answers via the LLM. Older reports also lacked structured provenance auditing (`answer_source`, `judge_method`, token usage, and harness version stamping).
3. **Checkpoints (`.checkpoint_*.json`)**: Contained unvalidated or un-stamped cache entries that could cause resume contamination.

These files are retained strictly for archival and forensic reference. Current evaluation runs must use fresh, version-stamped execution.
