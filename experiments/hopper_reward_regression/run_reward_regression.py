from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.hopper_reward_regression.reward_regression_suite import run_suite, write_report


def main():
    report = run_suite()
    out_path = write_report(report)
    print(json.dumps(report["summary"], ensure_ascii=False))
    print(f"report={out_path}")


if __name__ == "__main__":
    main()
