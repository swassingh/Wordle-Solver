"""Benchmark different solver strategies on held-out Wordle answers.

This script does not run automatically; invoke it manually when you want to
compare strategies:

    python scripts/benchmark_strategies.py --strategy info
    python scripts/benchmark_strategies.py --strategy ml_classic --classic-model models/wordle_model_classic.pkl
    python scripts/benchmark_strategies.py --strategy ml_min_guess --min-guess-model models/wordle_model_min_guess.pkl
"""

import sys
from pathlib import Path
from typing import Literal

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.domain.word_lists import WordLists
from src.game.simulator import WordleSimulator
from src.metrics.tracker import MetricsTracker, Metrics
from src.solvers.strategy import SolverStrategy, StrategyName


def run_benchmark(
    strategy: StrategyName,
    games: int,
    classic_model: Path | None,
    min_guess_model: Path | None,
    metrics_json: Path | None = None,
) -> Metrics:
    word_lists = WordLists()
    answers = word_lists.get_answer_list()

    test_words = answers[:games]

    simulator = WordleSimulator(word_lists)
    tracker = MetricsTracker()

    if strategy == "info":
        strat = None
    else:
        strat = SolverStrategy(
            word_lists=word_lists,
            classic_model_path=classic_model,
            min_guess_model_path=min_guess_model,
        )

    from src.solvers.info_theory import InformationTheorySolver
    from src.solvers.ml_solver import MLSolver

    info_solver = InformationTheorySolver(word_lists)

    print(f"\nBenchmarking strategy='{strategy}' on {len(test_words)} words...\n")

    for idx, target in enumerate(test_words):
        simulator.start_game(target)
        guesses_made = 0

        while not simulator.is_game_over() and guesses_made < 6:
            game_state = simulator.get_current_state()

            if strategy == "info":
                guess = info_solver.make_guess(game_state)
            else:
                guess = strat.make_guess(strategy, game_state)  # type: ignore[union-attr]

            simulator.make_guess(guess)
            guesses_made += 1

            if simulator.is_solved():
                break

        final_state = simulator.get_current_state()
        tracker.record_game(final_state, target)

        if (idx + 1) % 50 == 0:
            print(f"  Progress: {idx + 1}/{len(test_words)} games...")

    # Print and optionally export metrics
    tracker.print_summary()

    if metrics_json is not None:
        tracker.export_json(metrics_json)

    return tracker.get_metrics()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark WordleAI strategies.")
    parser.add_argument(
        "--strategy",
        choices=["info", "ml_classic", "ml_min_guess"],
        default="info",
        help="Strategy to benchmark.",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=200,
        help="Number of answer words to test.",
    )
    parser.add_argument(
        "--classic-model",
        type=str,
        default="models/wordle_model_classic.pkl",
        help="Path to classic ML model for ml_classic strategy.",
    )
    parser.add_argument(
        "--min-guess-model",
        type=str,
        default="models/wordle_model_min_guess.pkl",
        help="Path to min-guess ML model for ml_min_guess strategy.",
    )
    parser.add_argument(
        "--metrics-json",
        type=str,
        default=None,
        help=(
            "Optional path to write benchmark metrics as JSON. "
            "Useful for comparing new models to previous runs."
        ),
    )
    parser.add_argument(
        "--min-success-rate",
        type=float,
        default=None,
        help=(
            "If set, require success rate >= this value (e.g. 0.97 for 97%). "
            "If the benchmark underperforms, the script exits with code 1."
        ),
    )
    parser.add_argument(
        "--max-average-guesses",
        type=float,
        default=None,
        help=(
            "If set, require average guesses (on solved games) <= this value. "
            "If the benchmark underperforms, the script exits with code 1."
        ),
    )

    args = parser.parse_args()

    classic_path = Path(args.classic_model) if args.classic_model else None
    min_guess_path = Path(args.min_guess_model) if args.min_guess_model else None
    metrics_json = Path(args.metrics_json) if args.metrics_json else None

    metrics = run_benchmark(
        strategy=args.strategy,
        games=args.games,
        classic_model=classic_path,
        min_guess_model=min_guess_path,
        metrics_json=metrics_json,
    )

    # Apply simple acceptance checks if thresholds were provided.
    failed_checks: list[str] = []

    if args.min_success_rate is not None:
        if metrics.success_rate < args.min_success_rate:
            failed_checks.append(
                f"success_rate {metrics.success_rate:.4f} "
                f"< required {args.min_success_rate:.4f}"
            )

    if args.max_average_guesses is not None:
        if metrics.average_guesses > args.max_average_guesses:
            failed_checks.append(
                f"average_guesses {metrics.average_guesses:.4f} "
                f"> allowed {args.max_average_guesses:.4f}"
            )

    if failed_checks:
        print("\nBenchmark acceptance checks FAILED:")
        for msg in failed_checks:
            print(f"  - {msg}")
        print("Exiting with non-zero status to signal regression.")
        sys.exit(1)
    elif args.min_success_rate is not None or args.max_average_guesses is not None:
        print("\nBenchmark acceptance checks PASSED.")


if __name__ == "__main__":
  main()


