"""ML model architecture for Wordle solver."""

import pickle
from pathlib import Path
from typing import Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from src.ml.features import FeatureExtractor
from src.domain.word_lists import WordLists


class WordleModel:
    """ML model for predicting optimal Wordle guesses."""
    
    def __init__(self, model_type: str = "random_forest"):
        """Initialize the model.
        
        Args:
            model_type: Type of model ("random_forest" or "gradient_boosting").
        """
        self.model_type = model_type
        self.model: Optional[object] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.feature_extractor: Optional[FeatureExtractor] = None
        self.word_lists: Optional[WordLists] = None
        self.is_trained = False
    
    def _create_model(self):
        """Create the underlying sklearn model."""
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=10,
                learning_rate=0.1,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model.
        
        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target labels (word strings).
        """
        if self.model is None:
            self._create_model()
        
        # Encode labels (words) to integers
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Train model
        self.model.fit(X, y_encoded)
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict word labels.
        
        Args:
            X: Feature matrix (n_samples, n_features).
        
        Returns:
            Predicted word labels.
        
        Raises:
            ValueError: If model is not trained.
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        y_encoded = self.model.predict(X)
        y_pred = self.label_encoder.inverse_transform(y_encoded)
        return y_pred
    
    def predict_proba(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict word probabilities.
        
        Args:
            X: Feature matrix (n_samples, n_features).
        
        Returns:
            Tuple of (predicted words, probabilities).
        
        Raises:
            ValueError: If model is not trained.
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        proba = self.model.predict_proba(X)
        
        # Get top predictions
        top_indices = np.argsort(proba, axis=1)[:, ::-1]  # Sort descending
        
        # Convert indices back to words
        words = []
        probabilities = []
        for idx_row in top_indices:
            word_row = self.label_encoder.inverse_transform(idx_row)
            prob_row = proba[np.arange(len(idx_row)), idx_row]
            words.append(word_row)
            probabilities.append(prob_row)
        
        return np.array(words), np.array(probabilities)
    
    def save(self, filepath: Path) -> None:
        """Save model to file.
        
        Args:
            filepath: Path to save model.
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model.")
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: Path) -> None:
        """Load model from file.
        
        Args:
            filepath: Path to model file.
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.model_type = model_data['model_type']
        self.is_trained = model_data['is_trained']
        
        print(f"Model loaded from {filepath}")
    
    def set_feature_extractor(self, feature_extractor: FeatureExtractor) -> None:
        """Set the feature extractor (for convenience).
        
        Args:
            feature_extractor: FeatureExtractor instance.
        """
        self.feature_extractor = feature_extractor
    
    def set_word_lists(self, word_lists: WordLists) -> None:
        """Set word lists (for convenience).
        
        Args:
            word_lists: WordLists instance.
        """
        self.word_lists = word_lists

