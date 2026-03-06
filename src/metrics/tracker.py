"""Performance metrics tracking for Wordle solver."""

import json
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional
from src.domain.game_state import GameState


@dataclass
class GameResult:
    """Result of a single game."""
    target_word: str
    solved: bool
    guesses_used: int
    guesses: List[str] = field(default_factory=list)
    feedback_history: List[str] = field(default_factory=list)  # String representations
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class Metrics:
    """Aggregated metrics."""
    total_games: int = 0
    solved_games: int = 0
    failed_games: int = 0
    success_rate: float = 0.0
    average_guesses: float = 0.0
    guess_distribution: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    games: List[GameResult] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        result = asdict(self)
        result['guess_distribution'] = dict(result['guess_distribution'])
        return result


class MetricsTracker:
    """Tracks and aggregates solver performance metrics."""
    
    def __init__(self):
        """Initialize the metrics tracker."""
        self.results: List[GameResult] = []
    
    def record_game(self, game_state: GameState, target_word: str) -> GameResult:
        """Record the result of a game.
        
        Args:
            game_state: Final game state.
            target_word: The target word that was being solved.
        
        Returns:
            GameResult object.
        """
        solved = game_state.is_solved()
        guesses_used = len(game_state.guesses)
        
        # Convert feedback to string representations
        feedback_strings = [str(fb) for fb in game_state.feedback_history]
        
        result = GameResult(
            target_word=target_word,
            solved=solved,
            guesses_used=guesses_used,
            guesses=game_state.guesses.copy(),
            feedback_history=feedback_strings
        )
        
        self.results.append(result)
        return result
    
    def get_metrics(self) -> Metrics:
        """Calculate aggregated metrics.
        
        Returns:
            Metrics object with aggregated statistics.
        """
        if not self.results:
            return Metrics()
        
        total = len(self.results)
        solved = sum(1 for r in self.results if r.solved)
        failed = total - solved
        
        success_rate = solved / total if total > 0 else 0.0
        
        # Average guesses (only for solved games)
        solved_results = [r for r in self.results if r.solved]
        avg_guesses = (
            sum(r.guesses_used for r in solved_results) / len(solved_results)
            if solved_results else 0.0
        )
        
        # Guess distribution
        guess_dist = defaultdict(int)
        for result in self.results:
            if result.solved:
                guess_dist[result.guesses_used] += 1
            else:
                guess_dist[7] += 1  # Failed games (7 = didn't solve in 6)
        
        return Metrics(
            total_games=total,
            solved_games=solved,
            failed_games=failed,
            success_rate=success_rate,
            average_guesses=avg_guesses,
            guess_distribution=dict(guess_dist),
            games=self.results.copy()
        )
    
    def print_summary(self) -> None:
        """Print a summary of metrics to console."""
        metrics = self.get_metrics()
        
        print("\n" + "="*50)
        print("SOLVER PERFORMANCE METRICS")
        print("="*50)
        print(f"Total Games: {metrics.total_games}")
        print(f"Solved: {metrics.solved_games}")
        print(f"Failed: {metrics.failed_games}")
        print(f"Success Rate: {metrics.success_rate:.2%}")
        print(f"Average Guesses (solved): {metrics.average_guesses:.2f}")
        
        print("\nGuess Distribution:")
        for guesses in sorted(metrics.guess_distribution.keys()):
            count = metrics.guess_distribution[guesses]
            if guesses == 7:
                print(f"  Failed (>6): {count}")
            else:
                print(f"  {guesses} guesses: {count}")
        
        print("="*50)
    
    def export_json(self, filepath: Path) -> None:
        """Export metrics to JSON file.
        
        Args:
            filepath: Path to output JSON file.
        """
        metrics = self.get_metrics()
        metrics_dict = metrics.to_dict()
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metrics_dict, f, indent=2)
        
        print(f"Metrics exported to {filepath}")
    
    def export_csv(self, filepath: Path) -> None:
        """Export game results to CSV file.
        
        Args:
            filepath: Path to output CSV file.
        """
        import csv
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['target_word', 'solved', 'guesses_used', 'guesses'])
            
            for result in self.results:
                writer.writerow([
                    result.target_word,
                    result.solved,
                    result.guesses_used,
                    '|'.join(result.guesses)
                ])
        
        print(f"Results exported to {filepath}")
    
    def reset(self) -> None:
        """Reset all tracked metrics."""
        self.results.clear()
    
    def __len__(self) -> int:
        """Return number of games tracked."""
        return len(self.results)

