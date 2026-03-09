"""Utilities for turning logged daily sessions into training data.

This module reads JSONL logs produced by `/api/log/session` and converts
them into feature matrices and labels compatible with the existing
ML training pipeline (classification + regression).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Tuple, Optional

import numpy as np

from src.domain.game_state import GameState, Feedback as GameFeedback
from src.domain.word_lists import WordLists
from src.game.feedback import parse_feedback_string
from src.ml.features import FeatureExtractor


def _replay_session_to_samples(
    guesses: List[str],
    feedback: List[str],
    total_guesses_used: int,
    word_lists: WordLists,
    feature_extractor: FeatureExtractor,
) -> Tuple[List[np.ndarray], List[str], List[np.ndarray], List[float]]:
    """Turn a single session into per-step training samples.

    Returns:
        X_cls, y_cls, X_reg, y_reg lists (may be empty if session invalid).
    """
    if len(guesses) != len(feedback) or not guesses:
        return [], [], [], []

    # Build game state step-by-step, mirroring API logic.
    game_state = GameState(remaining_words=word_lists.all_valid_words.copy())

    X_cls: List[np.ndarray] = []
    y_cls: List[str] = []
    X_reg: List[np.ndarray] = []
    y_reg: List[float] = []

    # We treat the *played* word at each step as the chosen action.
    for idx, (guess_raw, fb_str) in enumerate(zip(guesses, feedback)):
        guess = guess_raw.upper().strip()

        # Basic validation – skip the whole session if anything looks wrong.
        if len(guess) != 5 or not guess.isalpha():
            return [], [], [], []
        if len(fb_str) != 5:
            return [], [], [], []

        try:
            fb_types = parse_feedback_string(fb_str)
        except ValueError:
            return [], [], [], []

        feedback_obj = GameFeedback(guess=guess, feedback=fb_types)

        # Extract features for (state, chosen_word) *before* applying this step.
        features = feature_extractor.extract_features(game_state, guess)
        X_cls.append(features)
        y_cls.append(guess)

        # For regression, we attach a noisy but useful label:
        # total guesses used in this game (proxy for difficulty / cost).
        X_reg.append(features)
        y_reg.append(float(total_guesses_used))

        # Apply step to advance state for the next iteration.
        game_state.add_guess(guess, feedback_obj)
        game_state.update_remaining_words(word_lists.all_valid_words)

    return X_cls, y_cls, X_reg, y_reg


def load_daily_session_data(
    session_path: Path,
    word_lists: Optional[WordLists] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load daily session logs and convert to ML-ready arrays.

    Args:
        session_path: Path to JSONL file written by `/api/log/session`.
        word_lists: Optional `WordLists` instance (created if omitted).

    Returns:
        X_class, y_class, X_reg, y_reg numpy arrays.
        Arrays may be empty (shape (0, n_features)) if no valid data.
    """
    wl = word_lists or WordLists()
    fx = FeatureExtractor(wl)

    X_cls_all: List[np.ndarray] = []
    y_cls_all: List[str] = []
    X_reg_all: List[np.ndarray] = []
    y_reg_all: List[float] = []

    if not session_path.exists():
        # No daily data yet – return empty arrays.
        dummy_dim = fx.get_feature_dimension()
        return (
            np.empty((0, dummy_dim), dtype=np.float32),
            np.empty((0,), dtype="<U5"),
            np.empty((0, dummy_dim), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    with session_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            guesses = obj.get("guesses") or []
            feedback = obj.get("feedback") or []
            guesses_used = obj.get("guesses_used")

            if not isinstance(guesses_used, int) or guesses_used <= 0:
                # Fallback: use length of guesses if missing.
                guesses_used = len(guesses)

            Xc, yc, Xr, yr = _replay_session_to_samples(
                guesses=list(guesses),
                feedback=list(feedback),
                total_guesses_used=guesses_used,
                word_lists=wl,
                feature_extractor=fx,
            )

            if not Xc:
                continue

            X_cls_all.extend(Xc)
            y_cls_all.extend(yc)
            X_reg_all.extend(Xr)
            y_reg_all.extend(yr)

    if not X_cls_all:
        dummy_dim = fx.get_feature_dimension()
        return (
            np.empty((0, dummy_dim), dtype=np.float32),
            np.empty((0,), dtype="<U5"),
            np.empty((0, dummy_dim), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    X_class = np.stack(X_cls_all).astype(np.float32)
    y_class = np.array(y_cls_all, dtype="<U5")
    X_reg = np.stack(X_reg_all).astype(np.float32)
    y_reg = np.array(y_reg_all, dtype=np.float32)

    return X_class, y_class, X_reg, y_reg


def mix_simulated_and_daily(
    X_sim: np.ndarray,
    y_sim: np.ndarray,
    X_daily: np.ndarray,
    y_daily: np.ndarray,
    daily_weight: float = 0.2,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Combine simulated and daily samples using a simple weighting scheme.

    The goal is to approximate a given share of daily data in the final
    training set without over-representing small logs.
    """
    if X_daily.size == 0 or y_daily.size == 0 or daily_weight <= 0.0:
        return X_sim, y_sim

    if rng is None:
        rng = np.random.default_rng()

    n_sim = X_sim.shape[0]
    # Target ratio daily : simulated ≈ daily_weight : (1 - daily_weight)
    target_daily = int((daily_weight / max(1e-6, 1.0 - daily_weight)) * n_sim)
    target_daily = max(1, min(target_daily, X_daily.shape[0]))

    indices = rng.choice(X_daily.shape[0], size=target_daily, replace=False)
    X_d = X_daily[indices]
    y_d = y_daily[indices]

    X_comb = np.concatenate([X_sim, X_d], axis=0)
    y_comb = np.concatenate([y_sim, y_d], axis=0)
    return X_comb, y_comb



