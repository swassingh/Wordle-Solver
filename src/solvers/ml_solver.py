"""ML-based Wordle solver."""

from pathlib import Path
from typing import Optional
import numpy as np
from src.solvers.base import BaseSolver
from src.solvers.info_theory import InformationTheorySolver
from src.domain.game_state import GameState
from src.domain.word_lists import WordLists
from src.ml.model import WordleModel
from src.ml.features import FeatureExtractor


class MLSolver(BaseSolver):
    """Solver that uses a trained ML model to predict optimal guesses."""
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        word_lists: Optional[WordLists] = None,
        fallback_to_info_theory: bool = True,
        confidence_threshold: float = 0.3
    ):
        """Initialize ML solver.
        
        Args:
            model_path: Path to trained model file. If None, model must be loaded separately.
            word_lists: WordLists instance. If None, creates a new one.
            fallback_to_info_theory: If True, fall back to info theory solver when needed.
            confidence_threshold: Minimum confidence to use ML prediction.
        """
        super().__init__(word_lists)
        self.model: Optional[WordleModel] = None
        self.feature_extractor = FeatureExtractor(self.word_lists)
        self.fallback_solver: Optional[InformationTheorySolver] = None
        self.fallback_to_info_theory = fallback_to_info_theory
        self.confidence_threshold = confidence_threshold
        
        if model_path is not None:
            self.load_model(model_path)
        
        if fallback_to_info_theory:
            self.fallback_solver = InformationTheorySolver(self.word_lists)
    
    def load_model(self, model_path: Path) -> None:
        """Load trained model from file.
        
        Args:
            model_path: Path to model file.
        
        Raises:
            FileNotFoundError: If model file doesn't exist.
            ValueError: If model loading fails.
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = WordleModel()
        self.model.load(model_path)
        self.model.set_feature_extractor(self.feature_extractor)
        self.model.set_word_lists(self.word_lists)
    
    def make_guess(self, game_state: GameState) -> str:
        """Make a guess using ML model (with fallback if needed).
        
        Args:
            game_state: Current game state.
        
        Returns:
            Best guess word.
        
        Raises:
            ValueError: If no valid candidates remain or model not loaded.
        """
        candidates = self.get_valid_candidates(game_state)
        
        if not candidates:
            raise ValueError("No valid candidates remaining")
        
        # If only one candidate, guess it
        if len(candidates) == 1:
            return list(candidates)[0]
        
        # If model not loaded, use fallback
        if self.model is None or not self.model.is_trained:
            if self.fallback_solver is not None:
                return self.fallback_solver.make_guess(game_state)
            else:
                raise ValueError("Model not loaded and no fallback solver available")
        
        # Extract features for all candidates
        try:
            candidate_list = list(candidates)
            
            # Limit candidates for performance (evaluate top 200)
            if len(candidate_list) > 200:
                # Prefer answer words
                answer_candidates = [w for w in candidate_list if self.word_lists.is_valid_answer(w)]
                guess_candidates = [w for w in candidate_list if w not in answer_candidates]
                
                # Take top answers + diverse guesses
                candidate_list = answer_candidates[:100] + guess_candidates[:100]
            
            # Extract features
            X = np.array([
                self.feature_extractor.extract_features(game_state, word)
                for word in candidate_list
            ])
            
            # Get predictions with probabilities
            words_pred, probabilities = self.model.predict_proba(X)
            
            # Get top prediction for each candidate
            best_word = None
            best_score = -1.0
            
            for i, word in enumerate(candidate_list):
                # Find this word in predictions
                word_idx = np.where(words_pred[i] == word)[0]
                if len(word_idx) > 0:
                    prob = probabilities[i][word_idx[0]]
                    if prob > best_score:
                        best_score = prob
                        best_word = word
            
            # Use ML prediction if confidence is high enough
            if best_word and best_score >= self.confidence_threshold:
                return best_word
            
            # Fallback to info theory if confidence too low
            if self.fallback_solver is not None:
                return self.fallback_solver.make_guess(game_state)
            
            # Last resort: return highest probability word
            if best_word:
                return best_word
            
            # Final fallback: return first candidate
            return candidate_list[0]
        
        except Exception as e:
            # If ML prediction fails, use fallback
            if self.fallback_solver is not None:
                return self.fallback_solver.make_guess(game_state)
            else:
                raise ValueError(f"ML prediction failed: {e}")
    
    def has_model(self) -> bool:
        """Check if model is loaded and trained.
        
        Returns:
            True if model is available.
        """
        return self.model is not None and self.model.is_trained

