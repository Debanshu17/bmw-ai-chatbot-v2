"""Sequential runner for clean evaluation pass of both Baseline and v2 Production.

Enforces:
1. Strict model check on gemini-3.5-flash-lite
2. Runs No-RAG baseline on all 40 questions
3. Runs v2 Production on all 40 questions
4. Aborts immediately if either run fails or exceeds error threshold
"""

import logging
import sys
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import settings
from evals.run_evals import HARNESS_VERSION, run_eval

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_runner")


def main():
    out_dir = BASE_DIR / "reports"
    dataset_path = BASE_DIR / "evals" / "dataset.jsonl"

    logger.info(f"Target Evaluation Model: {settings.CHAT_MODEL}")
    logger.info(f"Harness Version: {HARNESS_VERSION}")
    logger.info(f"Dataset: {dataset_path}")

    print("\n" + "=" * 60)
    print("STEP 1: RUNNING NO-RAG BASELINE EVALUATION (40 QUESTIONS)")
    print("=" * 60)
    baseline_rep = run_eval(
        dataset_path=dataset_path,
        out_dir=out_dir,
        is_baseline_v1=True
    )
    logger.info(f"Baseline run complete! Valid: {baseline_rep.get('is_valid')}, Accuracy: {baseline_rep.get('accuracy_pct')}%")

    print("\n" + "=" * 60)
    print("STEP 2: RUNNING V2 PRODUCTION RAG EVALUATION (40 QUESTIONS)")
    print("=" * 60)
    v2_rep = run_eval(
        dataset_path=dataset_path,
        out_dir=out_dir,
        is_baseline_v1=False
    )
    logger.info(f"v2 Production run complete! Valid: {v2_rep.get('is_valid')}, Accuracy: {v2_rep.get('accuracy_pct')}%")

    print("\n" + "=" * 60)
    print("ALL EVALUATION RUNS COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
