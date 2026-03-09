"""ML model architecture for Wordle solver."""

import pickle
from pathlib import Path
from typing import Optional, Literal

import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.preprocessing import LabelEncoder

# XGBoost for GPU support (optional)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

from src.ml.features import FeatureExtractor
from src.domain.word_lists import WordLists


ModelMode = Literal["classification", "regression"]


class WordleModel:
    """ML model for Wordle solver.

    Supports two modes:
    - classification: predict best word label (current behavior)
    - regression: predict expected number of guesses remaining
    """

    def __init__(
        self,
        model_type: str = "random_forest",
        model_mode: ModelMode = "classification",
        use_gpu: bool = False,
    ):
        """Initialize the model.

        Args:
            model_type: Type of model (\"random_forest\", \"gradient_boosting\", or \"xgb\").
            model_mode: \"classification\" or \"regression\".
            use_gpu: Whether to use GPU for training (only for XGBoost models).
        """
        self.model_type = model_type
        self.model_mode: ModelMode = model_mode
        self.use_gpu = use_gpu
        self.model: Optional[object] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.feature_extractor: Optional[FeatureExtractor] = None
        self.word_lists: Optional[WordLists] = None
        self.is_trained = False
        
        # Validate GPU usage
        if self.use_gpu and self.model_type != "xgb":
            raise ValueError("GPU support is only available for XGBoost (model_type='xgb')")
        
        if self.use_gpu and not XGBOOST_AVAILABLE:
            raise ValueError("XGBoost is not installed. Install with: pip install xgboost")

    def _create_model(self) -> None:
        """Create the underlying model based on mode/type."""
        if self.model_mode == "classification":
            if self.model_type == "random_forest":
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=20,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                )
            elif self.model_type == "gradient_boosting":
                self.model = GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=10,
                    learning_rate=0.1,
                    random_state=42,
                )
            elif self.model_type == "xgb":
                if not XGBOOST_AVAILABLE:
                    raise ValueError("XGBoost is not installed. Install with: pip install xgboost")
                tree_method = "gpu_hist" if self.use_gpu else "hist"
                try:
                    self.model = xgb.XGBClassifier(
                        n_estimators=200,
                        max_depth=10,
                        learning_rate=0.1,
                        random_state=42,
                        tree_method=tree_method,
                        gpu_id=0 if self.use_gpu else None,
                        predictor="gpu_predictor" if self.use_gpu else "cpu_predictor",
                    )
                except Exception as e:
                    if self.use_gpu:
                        print(f"Warning: GPU training failed ({e}). Falling back to CPU.")
                        self.use_gpu = False
                        self.model = xgb.XGBClassifier(
                            n_estimators=200,
                            max_depth=10,
                            learning_rate=0.1,
                            random_state=42,
                            tree_method="hist",
                        )
                    else:
                        raise
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
        else:  # regression
            if self.model_type == "random_forest":
                self.model = RandomForestRegressor(
                    n_estimators=150,
                    max_depth=25,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                )
            elif self.model_type == "gradient_boosting":
                self.model = GradientBoostingRegressor(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    random_state=42,
                )
            elif self.model_type == "xgb":
                if not XGBOOST_AVAILABLE:
                    raise ValueError("XGBoost is not installed. Install with: pip install xgboost")
                tree_method = "gpu_hist" if self.use_gpu else "hist"
                try:
                    self.model = xgb.XGBRegressor(
                        n_estimators=300,
                        max_depth=6,
                        learning_rate=0.05,
                        random_state=42,
                        tree_method=tree_method,
                        gpu_id=0 if self.use_gpu else None,
                        predictor="gpu_predictor" if self.use_gpu else "cpu_predictor",
                    )
                except Exception as e:
                    if self.use_gpu:
                        print(f"Warning: GPU training failed ({e}). Falling back to CPU.")
                        self.use_gpu = False
                        self.model = xgb.XGBRegressor(
                            n_estimators=300,
                            max_depth=6,
                            learning_rate=0.05,
                            random_state=42,
                            tree_method="hist",
                        )
                    else:
                        raise
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target labels.
               - classification: word strings
               - regression: float values (expected guesses)
        """
        if self.model is None:
            self._create_model()

        if self.model_mode == "classification":
            # Encode labels (words) to integers
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
            self.model.fit(X, y_encoded)
        else:
            # Regression on numeric targets
            y_float = y.astype(float)
            self.model.fit(X, y_float)

        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict outputs based on mode.

        - classification: returns word labels (np.ndarray[str])
        - regression: returns cost estimates (np.ndarray[float])
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        if self.model_mode == "classification":
            y_encoded = self.model.predict(X)
            if self.label_encoder is None:
                raise ValueError("Label encoder missing for classification model.")
            y_pred = self.label_encoder.inverse_transform(y_encoded)
            return y_pred

        # regression
        preds = self.model.predict(X)
        return np.asarray(preds, dtype=float)

    def predict_proba(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict word probabilities (classification mode only)."""
        if self.model_mode != "classification":
            raise ValueError("predict_proba is only available in classification mode.")

        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        proba = self.model.predict_proba(X)

        # Get top predictions
        top_indices = np.argsort(proba, axis=1)[:, ::-1]  # Sort descending

        # Convert indices back to words
        if self.label_encoder is None:
            raise ValueError("Label encoder missing for classification model.")

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
            "model": self.model,
            "label_encoder": self.label_encoder,
            "model_type": self.model_type,
            "model_mode": self.model_mode,
            "use_gpu": self.use_gpu,
            "is_trained": self.is_trained,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {filepath}")

    def load(self, filepath: Path) -> None:
        """Load model from file.

        Args:
            filepath: Path to model file.
        """
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.label_encoder = model_data.get("label_encoder")
        self.model_type = model_data.get("model_type", "random_forest")
        self.model_mode = model_data.get("model_mode", "classification")
        self.use_gpu = model_data.get("use_gpu", False)
        self.is_trained = model_data.get("is_trained", True)

        print(f"Model loaded from {filepath} (mode={self.model_mode})")

    def set_feature_extractor(self, feature_extractor: FeatureExtractor) -> None:
        """Set the feature extractor (for convenience)."""
        self.feature_extractor = feature_extractor

    def set_word_lists(self, word_lists: WordLists) -> None:
        """Set word lists (for convenience)."""
        self.word_lists = word_lists

