"""Wordle game simulator."""

import random
from typing import Optional
from src.domain.game_state import GameState
from src.domain.word_lists import WordLists
from src.game.feedback import calculate_feedback


class WordleSimulator:
    """Simulates a Wordle game."""
    
    def __init__(self, word_lists: Optional[WordLists] = None):
        """Initialize the simulator.
        
        Args:
            word_lists: WordLists instance. If None, creates a new one.
        """
        self.word_lists = word_lists or WordLists()
        self.game_state: Optional[GameState] = None
        self.target_word: Optional[str] = None
    
    def start_game(self, target_word: Optional[str] = None) -> GameState:
        """Start a new game.
        
        Args:
            target_word: The target word to guess. If None, selects a random answer.
        
        Returns:
            Initial game state.
        
        Raises:
            ValueError: If target_word is not a valid answer.
        """
        if target_word is None:
            # Select random answer
            answers = self.word_lists.get_answer_list()
            target_word = random.choice(answers)
        else:
            target_word = target_word.upper()
            if not self.word_lists.is_valid_answer(target_word):
                raise ValueError(f"'{target_word}' is not a valid answer word")
        
        self.target_word = target_word
        self.game_state = GameState(
            target_word=target_word,
            remaining_words=self.word_lists.all_valid_words.copy()
        )
        
        return self.game_state
    
    def make_guess(self, guess: str) -> tuple[GameState, bool]:
        """Make a guess and get feedback.
        
        Args:
            guess: The word to guess (5 letters, case-insensitive).
        
        Returns:
            Tuple of (updated game state, is_solved).
        
        Raises:
            ValueError: If game not started, guess invalid, or game already over.
        """
        if self.game_state is None:
            raise ValueError("Game not started. Call start_game() first.")
        
        if self.game_state.is_game_over():
            raise ValueError("Game is already over.")
        
        guess = guess.upper()
        
        # Validate guess
        if len(guess) != 5:
            raise ValueError(f"Guess must be 5 letters, got {len(guess)}")
        
        if not self.word_lists.is_valid_guess(guess):
            raise ValueError(f"'{guess}' is not a valid guess word")
        
        # Calculate feedback
        feedback = calculate_feedback(guess, self.target_word)
        
        # Update game state
        self.game_state.add_guess(guess, feedback)
        
        # Update remaining words based on constraints
        self.game_state.update_remaining_words(self.word_lists.all_valid_words)
        
        is_solved = self.game_state.is_solved()
        
        return self.game_state, is_solved
    
    def get_feedback(self, guess: str, target: Optional[str] = None) -> tuple[list, bool]:
        """Get feedback for a guess without updating game state.
        
        This is useful for testing or external use.
        
        Args:
            guess: The word to guess.
            target: The target word. If None, uses current game target.
        
        Returns:
            Tuple of (feedback list, is_solved).
        """
        if target is None:
            if self.target_word is None:
                raise ValueError("No target word set. Provide target or start a game.")
            target = self.target_word
        
        feedback = calculate_feedback(guess, target)
        is_solved = feedback.is_solved()
        
        return feedback, is_solved
    
    def reset(self) -> None:
        """Reset the simulator to initial state."""
        self.game_state = None
        self.target_word = None
    
    def get_current_state(self) -> Optional[GameState]:
        """Get the current game state.
        
        Returns:
            Current game state, or None if game not started.
        """
        return self.game_state
    
    def is_game_over(self) -> bool:
        """Check if the game is over.
        
        Returns:
            True if game is over (solved or out of guesses).
        """
        if self.game_state is None:
            return False
        return self.game_state.is_game_over()
    
    def is_solved(self) -> bool:
        """Check if the puzzle has been solved.
        
        Returns:
            True if solved.
        """
        if self.game_state is None:
            return False
        return self.game_state.is_solved()

