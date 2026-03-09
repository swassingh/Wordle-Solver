"""Solver strategy selector for Wordle.

This module provides a thin layer that chooses between:
- Information theory solver
- ML classic solver (classification model)
- ML min-guess solver (regression model)

Strategies:
- \"info\"        -> pure information-theory solver
- \"ml_classic\" -> MLSolver with classification model
- \"ml_min_guess\" -> MLSolver with regression model (min guesses)
"""

from pathlib import Path
from typing import Literal, Optional

from src.domain.word_lists import WordLists
from src.domain.game_state import GameState
from src.solvers.info_theory import InformationTheorySolver
from src.solvers.ml_solver import MLSolver


StrategyName = Literal["info", "ml_classic", "ml_min_guess"]


class SolverStrategy:
    """Factory/selector for different solver strategies."""

    def __init__(
        self,
        word_lists: Optional[WordLists] = None,
        classic_model_path: Optional[Path] = None,
        min_guess_model_path: Optional[Path] = None,
    ) -> None:
        self.word_lists = word_lists or WordLists()

        self.info_solver = InformationTheorySolver(self.word_lists)

        self.ml_classic_solver: Optional[MLSolver] = None
        if classic_model_path is not None:
            self.ml_classic_solver = MLSolver(
                model_path=classic_model_path,
                word_lists=self.word_lists,
                mode="classification",
            )

        self.ml_min_guess_solver: Optional[MLSolver] = None
        if min_guess_model_path is not None:
            self.ml_min_guess_solver = MLSolver(
                model_path=min_guess_model_path,
                word_lists=self.word_lists,
                mode="regression",
            )

    def _ensure_ml_classic(self) -> MLSolver:
        if self.ml_classic_solver is None:
            raise ValueError(
                "ML classic solver not initialized. "
                "Provide classic_model_path when creating SolverStrategy."
            )
        return self.ml_classic_solver

    def _ensure_ml_min_guess(self) -> MLSolver:
        if self.ml_min_guess_solver is None:
            raise ValueError(
                "ML min-guess solver not initialized. "
                "Provide min_guess_model_path when creating SolverStrategy."
            )
        return self.ml_min_guess_solver

    def make_guess(self, strategy: StrategyName, game_state: GameState) -> str:
        """Make a guess using the requested strategy."""

        if strategy == "info":
            return self.info_solver.make_guess(game_state)

        if strategy == "ml_classic":
            solver = self._ensure_ml_classic()
            return solver.make_guess(game_state)

        if strategy == "ml_min_guess":
            solver = self._ensure_ml_min_guess()
            return solver.make_guess(game_state)

        raise ValueError(f"Unknown strategy: {strategy}")



