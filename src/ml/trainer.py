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


class ModelTrainer:
    """Trains the Wordle ML model."""
    
    def __init__(
        self,
        word_lists: WordLists,
        model_type: str = "random_forest"
    ):
        """Initialize trainer.
        
        Args:
            word_lists: WordLists instance.
            model_type: Type of model to train.
        """
        self.word_lists = word_lists
        self.model = WordleModel(model_type=model_type)
        self.feature_extractor = FeatureExtractor(word_lists)
        self.data_generator = TrainingDataGenerator(word_lists)
        
        self.model.set_feature_extractor(self.feature_extractor)
        self.model.set_word_lists(word_lists)
    
    def train(
        self,
        num_games: int = 1000,
        validation_split: float = 0.2,
        answers: Optional[List[str]] = None
    ) -> dict:
        """Train the model.
        
        Args:
            num_games: Number of games to use for training.
            validation_split: Fraction of data to use for validation.
            answers: List of answer words. If None, uses all answers.
        
        Returns:
            Dictionary with training metrics.
        """
        print("="*50)
        print("TRAINING WORDLE ML MODEL")
        print("="*50)
        
        # Generate training data
        X, y = self.data_generator.generate_training_data(num_games, answers)
        
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
        
        train_accuracy = np.mean(train_pred == y_train)
        val_accuracy = np.mean(val_pred == y_val)
        
        metrics = {
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'train_samples': len(X_train),
            'val_samples': len(X_val)
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

