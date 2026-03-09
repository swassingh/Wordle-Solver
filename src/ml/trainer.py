"""Training pipeline for Wordle ML model."""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

from src.ml.model import WordleModel
from src.ml.features import FeatureExtractor
from src.solvers.info_theory import InformationTheorySolver
from src.game.simulator import WordleSimulator
from src.domain.word_lists import WordLists
from src.domain.game_state import GameState
from src.ml.daily_data import (
    load_daily_session_data,
    mix_simulated_and_daily,
)


class TrainingDataGenerator:
    """Generates training data by simulating games."""
    
    def __init__(self, word_lists: WordLists):
        """Initialize data generator.
        
        Args:
            word_lists: WordLists instance.
        """
        self.word_lists = word_lists
        self.solver = InformationTheorySolver(word_lists)
        self.simulator = WordleSimulator(word_lists)
        self.feature_extractor = FeatureExtractor(word_lists)
    
    def generate_training_data(
        self,
        num_games: int,
        answers: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate training data by simulating games.
        
        Args:
            num_games: Number of games to simulate.
            answers: List of answer words to use. If None, uses all answers.
        
        Returns:
            Tuple of (feature_matrix, target_words).
        """
        if answers is None:
            answers = self.word_lists.get_answer_list()
        
        # Limit to requested number
        answers = answers[:num_games]
        
        X_list = []
        y_list = []
        
        print(f"Generating training data from {len(answers)} games...")
        
        for i, target in enumerate(answers):
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(answers)} games...")
            
            # Simulate game with info theory solver
            self.simulator.start_game(target)
            
            while not self.simulator.is_game_over():
                game_state = self.simulator.get_current_state()
                
                # Get optimal guess from info theory solver
                try:
                    optimal_guess = self.solver.make_guess(game_state)
                except (ValueError, IndexError):
                    break
                
                # Extract features for this game state and optimal guess
                features = self.feature_extractor.extract_features(game_state, optimal_guess)
                X_list.append(features)
                y_list.append(optimal_guess)
                
                # Make the guess
                self.simulator.make_guess(optimal_guess)
                
                if self.simulator.is_solved():
                    break
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        print(f"Generated {len(X)} training samples")
        return X, y

    def generate_regression_training_data(
        self,
        num_games: int,
        answers: Optional[List[str]] = None,
        candidates_per_state: int = 25,
        failure_penalty: float = 7.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate regression training data for min-guess optimization.

        For each game state, we sample candidate words, force that word as the
        next guess, then complete the game with the information-theory solver.
        The label is the total number of guesses used (or a penalty if unsolved).
        """
        if answers is None:
            answers = self.word_lists.get_answer_list()

        answers = answers[:num_games]

        X_list: list[np.ndarray] = []
        y_list: list[float] = []

        print(f"Generating regression training data from {len(answers)} games...")

        all_words = list(self.word_lists.all_valid_words)

        for idx, target in enumerate(answers):
            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{len(answers)} games...")

            # Simulate a baseline game once to get history and baseline states.
            history: List[str] = []
            self.simulator.start_game(target)

            while not self.simulator.is_game_over():
                base_state = self.simulator.get_current_state()

                try:
                    baseline_guess = self.solver.make_guess(base_state)
                except (ValueError, IndexError):
                    break

                # Build candidate pool: include baseline guess + random others.
                candidates = [baseline_guess]
                # Sample additional candidates from full word list.
                rng = np.random.default_rng()
                extra_candidates = rng.choice(
                    all_words,
                    size=min(candidates_per_state - 1, len(all_words)),
                    replace=False,
                ).tolist()
                for word in extra_candidates:
                    if word not in candidates:
                        candidates.append(word)

                # Evaluate each candidate by re-simulating from history.
                for cand in candidates:
                    sim = WordleSimulator(self.word_lists)
                    sim.start_game(target)

                    # Replay history
                    for g in history:
                        try:
                            sim.make_guess(g)
                        except ValueError:
                            break

                    if sim.is_game_over():
                        continue

                    guesses_used = len(history)

                    # Forced candidate guess
                    try:
                        sim.make_guess(cand)
                    except ValueError:
                        continue

                    guesses_used += 1

                    # Finish game with info-theory solver
                    while not sim.is_game_over():
                        state = sim.get_current_state()
                        try:
                            g2 = self.solver.make_guess(state)
                        except (ValueError, IndexError):
                            break
                        sim.make_guess(g2)
                        guesses_used += 1
                        if sim.is_game_over():
                            break

                    if not sim.is_solved():
                        label = float(failure_penalty)
                    else:
                        label = float(guesses_used)

                    features = self.feature_extractor.extract_features(base_state, cand)
                    X_list.append(features)
                    y_list.append(label)

                # Advance baseline game
                self.simulator.make_guess(baseline_guess)
                history.append(baseline_guess)

                if self.simulator.is_solved():
                    break

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)

        print(f"Generated {len(X)} regression training samples")
        return X, y


class ModelTrainer:
    """Trains the Wordle ML model."""
    
    def __init__(
        self,
        word_lists: WordLists,
        model_type: str = "random_forest",
        model_mode: str = "classification",
        use_gpu: bool = False,
    ):
        """Initialize trainer.
        
        Args:
            word_lists: WordLists instance.
            model_type: Type of model to train.
            model_mode: "classification" or "regression".
            use_gpu: Whether to use GPU for training (only for XGBoost models).
        """
        self.word_lists = word_lists
        self.model = WordleModel(
            model_type=model_type,
            model_mode=model_mode,  # type: ignore[arg-type]
            use_gpu=use_gpu,
        )
        self.feature_extractor = FeatureExtractor(word_lists)
        self.data_generator = TrainingDataGenerator(word_lists)
        
        self.model.set_feature_extractor(self.feature_extractor)
        self.model.set_word_lists(word_lists)
    
    def train(
        self,
        num_games: int = 1000,
        validation_split: float = 0.2,
        answers: Optional[List[str]] = None,
        training_mode: str = "classification",
        daily_session_path: Optional[Path] = None,
        daily_weight: float = 0.2,
    ) -> dict:
        """Train the model.
        
        Args:
            num_games: Number of games to use for training.
            validation_split: Fraction of data to use for validation.
            answers: List of answer words. If None, uses all answers.
            training_mode: "classification" or "regression".
            daily_session_path: Optional path to JSONL daily-session log file.
                If provided and non-empty, the simulated data will be mixed with
                samples reconstructed from real daily games.
            daily_weight: Approximate fraction of daily data to include
                relative to simulated data (e.g. 0.2 → ~20% daily, 80% simulated).
        
        Returns:
            Dictionary with training metrics.
        """
        print("="*50)
        print("TRAINING WORDLE ML MODEL")
        print("="*50)
        
        # Generate core (simulated) training data
        if training_mode == "regression":
            X, y = self.data_generator.generate_regression_training_data(
                num_games=num_games,
                answers=answers,
            )
        else:
            X, y = self.data_generator.generate_training_data(num_games, answers)

        # Optionally augment with daily-session data
        if daily_session_path is not None and daily_weight > 0.0:
            print(
                f"\nLoading daily-session data from {daily_session_path} "
                f"with target weight {daily_weight:.2f}..."
            )
            (
                X_class_daily,
                y_class_daily,
                X_reg_daily,
                y_reg_daily,
            ) = load_daily_session_data(daily_session_path, word_lists=self.word_lists)

            if training_mode == "regression":
                X, y = mix_simulated_and_daily(
                    X_sim=X,
                    y_sim=y,
                    X_daily=X_reg_daily,
                    y_daily=y_reg_daily,
                    daily_weight=daily_weight,
                )
            else:
                X, y = mix_simulated_and_daily(
                    X_sim=X,
                    y_sim=y,
                    X_daily=X_class_daily,
                    y_daily=y_class_daily,
                    daily_weight=daily_weight,
                )
        
        # Split into train/validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"\nTraining set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        
        # Train model
        print("\nTraining model...")
        self.model.train(X_train, y_train)
        
        # Evaluate
        print("\nEvaluating model...")
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)

        if training_mode == "regression":
            # Mean Absolute Error / Mean Squared Error
            train_mae = float(np.mean(np.abs(train_pred - y_train)))
            val_mae = float(np.mean(np.abs(val_pred - y_val)))
            train_mse = float(np.mean((train_pred - y_train) ** 2))
            val_mse = float(np.mean((val_pred - y_val) ** 2))

            metrics = {
                "mode": "regression",
                "train_mae": train_mae,
                "val_mae": val_mae,
                "train_mse": train_mse,
                "val_mse": val_mse,
                "train_samples": len(X_train),
                "val_samples": len(X_val),
            }

            print(f"\nTrain MAE: {train_mae:.3f}, MSE: {train_mse:.3f}")
            print(f"Val   MAE: {val_mae:.3f}, MSE: {val_mse:.3f}")
        else:
            train_accuracy = float(np.mean(train_pred == y_train))
            val_accuracy = float(np.mean(val_pred == y_val))

            metrics = {
                "mode": "classification",
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy,
                "train_samples": len(X_train),
                "val_samples": len(X_val),
            }

            print(f"\nTraining Accuracy: {train_accuracy:.2%}")
            print(f"Validation Accuracy: {val_accuracy:.2%}")
        print("="*50)
        
        return metrics
    
    def save_model(self, filepath: Path) -> None:
        """Save trained model.
        
        Args:
            filepath: Path to save model.
        """
        self.model.save(filepath)
    
    def load_model(self, filepath: Path) -> None:
        """Load trained model.
        
        Args:
            filepath: Path to model file.
        """
        self.model.load(filepath)

