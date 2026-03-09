"""Script to train Wordle ML models (classic + min-guess)."""

import sys
from pathlib import Path
from datetime import datetime

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml.trainer import ModelTrainer
from src.domain.word_lists import WordLists


def main():
    """Train one or both ML models."""
    import argparse

    parser = argparse.ArgumentParser(description="Train Wordle ML models.")
    parser.add_argument(
        "--mode",
        choices=["classic", "min-guess", "both"],
        default="classic",
        help="Which model to train: classic (classification), min-guess (regression), or both.",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=500,
        help="Number of games to simulate for training.",
    )
    parser.add_argument(
        "--model-type",
        choices=["random_forest", "gradient_boosting", "xgb"],
        default="random_forest",
        help="Type of model to train. Use 'xgb' for GPU support.",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU for training (only for XGBoost models).",
    )
    parser.add_argument(
        "--use-daily",
        action="store_true",
        help=(
            "If set, mix in real daily-session data from data/daily_sessions.jsonl "
            "when training (if the file exists)."
        ),
    )
    parser.add_argument(
        "--daily-path",
        type=str,
        default="data/daily_sessions.jsonl",
        help="Path to JSONL file with logged daily sessions.",
    )
    parser.add_argument(
        "--daily-weight",
        type=float,
        default=0.2,
        help=(
            "Approximate fraction of daily-session samples relative to simulated "
            "samples when mixing training data (e.g. 0.2 ≈ 20% daily)."
        ),
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help=(
            "Optional tag to append to trained model filenames for versioning, "
            "e.g. --tag 20260309 will also save wordle_model_classic_20260309.pkl."
        ),
    )
    args = parser.parse_args()

    word_lists = WordLists()
    answers = word_lists.get_answer_list()
    num_games = min(args.games, len(answers))

    model_dir = project_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    daily_path = Path(args.daily_path)
    daily_session_path = daily_path if args.use_daily else None
    daily_weight = args.daily_weight

    tag_suffix = f"_{args.tag}" if args.tag else ""

    # Display GPU status if requested
    if args.use_gpu:
        if args.model_type != "xgb":
            print("Warning: --use-gpu only works with --model-type xgb. Ignoring GPU flag.")
            args.use_gpu = False
        else:
            print("GPU training enabled (will fall back to CPU if GPU unavailable)")

    if args.mode in ("classic", "both"):
        print("\n=== Training CLASSIC (classification) model ===")
        trainer_classic = ModelTrainer(
            word_lists,
            model_type=args.model_type,
            model_mode="classification",
            use_gpu=args.use_gpu,
        )
        metrics_classic = trainer_classic.train(
            num_games=num_games,
            validation_split=0.2,
            training_mode="classification",
            daily_session_path=daily_session_path,
            daily_weight=daily_weight,
        )
        classic_path = model_dir / "wordle_model_classic.pkl"
        trainer_classic.save_model(classic_path)

        if tag_suffix:
            classic_tagged = model_dir / f"wordle_model_classic{tag_suffix}.pkl"
            trainer_classic.save_model(classic_tagged)
            print(f"Tagged classic model saved to: {classic_tagged}")

        print("\nClassic model training complete!")
        print(f"Model saved to: {classic_path}")
        print(f"Metrics: {metrics_classic}")

    if args.mode in ("min-guess", "both"):
        print("\n=== Training MIN-GUESS (regression) model ===")
        trainer_reg = ModelTrainer(
            word_lists,
            model_type=args.model_type,
            model_mode="regression",
            use_gpu=args.use_gpu,
        )
        metrics_reg = trainer_reg.train(
            num_games=num_games,
            validation_split=0.2,
            training_mode="regression",
            daily_session_path=daily_session_path,
            daily_weight=daily_weight,
        )
        reg_path = model_dir / "wordle_model_min_guess.pkl"
        trainer_reg.save_model(reg_path)

        if tag_suffix:
            reg_tagged = model_dir / f"wordle_model_min_guess{tag_suffix}.pkl"
            trainer_reg.save_model(reg_tagged)
            print(f"Tagged min-guess model saved to: {reg_tagged}")

        print("\nMin-guess model training complete!")
        print(f"Model saved to: {reg_path}")
        print(f"Metrics: {metrics_reg}")


if __name__ == "__main__":
    main()

