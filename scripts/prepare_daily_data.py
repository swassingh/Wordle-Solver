"""Convert logged daily sessions into ML training datasets.

This script reads `data/daily_sessions.jsonl` (written by `/api/log/session`)
and produces NumPy `.npz` files that can be consumed by the training pipeline.

Outputs (in `data/` by default):
  - `daily_classification.npz`  → X, y for classic model
  - `daily_regression.npz`      → X, y for min-guess model
"""

from pathlib import Path
import sys

import numpy as np

# Ensure `src` package is importable when running as a script.
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.server import SESSION_LOG_PATH  # type: ignore[attr-defined]
from src.ml.daily_data import load_daily_session_data
from src.domain.word_lists import WordLists


def main() -> None:
    """Entry point for converting daily sessions to training data."""
    data_dir = PROJECT_ROOT / "data"
    session_path = SESSION_LOG_PATH

    print(f"Reading daily sessions from: {session_path}")

    word_lists = WordLists()
    X_cls, y_cls, X_reg, y_reg = load_daily_session_data(session_path, word_lists)

    if X_cls.size == 0:
        print("No valid daily sessions found. Nothing to export.")
        return

    cls_out = data_dir / "daily_classification.npz"
    reg_out = data_dir / "daily_regression.npz"

    cls_out.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Prepared {X_cls.shape[0]} classification samples "
        f"and {X_reg.shape[0]} regression samples from daily sessions."
    )

    np.savez_compressed(cls_out, X=X_cls, y=y_cls)
    np.savez_compressed(reg_out, X=X_reg, y=y_reg)

    print(f"Classification data saved to: {cls_out}")
    print(f"Regression data saved to: {reg_out}")


if __name__ == "__main__":
    main()



