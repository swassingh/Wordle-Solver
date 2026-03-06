"""Script to train the Wordle ML model."""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml.trainer import ModelTrainer
from src.domain.word_lists import WordLists


def main():
    """Train the model."""
    # Initialize
    word_lists = WordLists()
    trainer = ModelTrainer(word_lists, model_type="random_forest")
    
    # Train on subset of answers (adjust as needed)
    answers = word_lists.get_answer_list()
    num_games = min(1000, len(answers))  # Use up to 1000 games
    
    # Train
    metrics = trainer.train(num_games=num_games, validation_split=0.2)
    
    # Save model
    model_dir = project_root / "models"
    model_path = model_dir / "wordle_model.pkl"
    trainer.save_model(model_path)
    
    print(f"\nModel training complete!")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()

