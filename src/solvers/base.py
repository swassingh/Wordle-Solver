"""Base solver interface and utilities."""

from abc import ABC, abstractmethod
from typing import List, Set, Optional
from src.domain.game_state import GameState
from src.domain.word_lists import WordLists


class BaseSolver(ABC):
    """Base class for all Wordle solvers."""
    
    def __init__(self, word_lists: Optional[WordLists] = None):
        """Initialize the solver.
        
        Args:
            word_lists: WordLists instance. If None, creates a new one.
        """
        self.word_lists = word_lists or WordLists()
    
    @abstractmethod
    def make_guess(self, game_state: GameState) -> str:
        """Make a guess based on the current game state.
        
        Args:
            game_state: Current state of the game.
        
        Returns:
            A valid 5-letter word guess.
        
        Raises:
            ValueError: If no valid guesses remain.
        """
        pass
    
    def get_valid_candidates(self, game_state: GameState) -> Set[str]:
        """Get set of valid candidate words based on game state.
        
        Args:
            game_state: Current game state.
        
        Returns:
            Set of valid candidate words.
        """
        if game_state.remaining_words:
            # Use pre-filtered remaining words if available
            candidates = game_state.remaining_words.copy()
        else:
            # Fallback: filter all words based on constraints
            candidates = self.word_lists.all_valid_words.copy()
            game_state.update_remaining_words(candidates)
            candidates = game_state.remaining_words.copy()
        
        # Ensure all candidates are valid guesses
        return {w for w in candidates if self.word_lists.is_valid_guess(w)}
    
    def get_candidate_list(self, game_state: GameState) -> List[str]:
        """Get sorted list of valid candidate words.
        
        Args:
            game_state: Current game state.
        
        Returns:
            Sorted list of valid candidate words.
        """
        return sorted(list(self.get_valid_candidates(game_state)))
    
    def score_word(self, word: str, candidates: Set[str]) -> float:
        """Score a word based on how well it narrows down candidates.
        
        This is a utility method that can be overridden by subclasses.
        Default implementation returns 0.0 (no scoring).
        
        Args:
            word: Word to score.
            candidates: Set of remaining candidate words.
        
        Returns:
            Score for the word (higher is better).
        """
        return 0.0
    
    def solve(self, game_state: GameState, max_guesses: int = 6) -> tuple[GameState, bool]:
        """Solve a game by making guesses until solved or out of guesses.
        
        Args:
            game_state: Initial game state (should have target_word set).
            max_guesses: Maximum number of guesses allowed.
        
        Returns:
            Tuple of (final game state, is_solved).
        """
        while not game_state.is_game_over():
            try:
                guess = self.make_guess(game_state)
            except (ValueError, IndexError) as e:
                # No valid guesses remaining
                break
            
            # In a real game, we'd get feedback from the game
            # For this method, we assume the game_state is updated externally
            # This is mainly for testing/benchmarking
            
            if game_state.is_solved():
                break
        
        return game_state, game_state.is_solved()
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}()"

