"""Game state and feedback types."""

from dataclasses import dataclass, field
from typing import List, Optional, Set
from enum import Enum


class FeedbackType(Enum):
    """Feedback types for each letter position."""
    CORRECT = "green"  # Correct letter, correct position
    PRESENT = "yellow"  # Correct letter, wrong position
    ABSENT = "gray"  # Letter not in word


@dataclass
class Feedback:
    """Feedback for a single guess."""
    guess: str
    feedback: List[FeedbackType]  # Length 5, one for each position
    
    def __post_init__(self):
        """Validate feedback."""
        if len(self.guess) != 5:
            raise ValueError(f"Guess must be 5 letters, got {len(self.guess)}")
        if len(self.feedback) != 5:
            raise ValueError(f"Feedback must have 5 elements, got {len(self.feedback)}")
    
    def is_solved(self) -> bool:
        """Check if this feedback indicates the puzzle is solved."""
        return all(f == FeedbackType.CORRECT for f in self.feedback)
    
    def __str__(self) -> str:
        """String representation using emoji or letters."""
        mapping = {
            FeedbackType.CORRECT: "🟩",
            FeedbackType.PRESENT: "🟨",
            FeedbackType.ABSENT: "⬛",
        }
        return "".join(mapping[f] for f in self.feedback)
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"Feedback(guess='{self.guess}', feedback={[f.value for f in self.feedback]})"


@dataclass
class GameState:
    """Current state of a Wordle game."""
    target_word: Optional[str] = None  # None if playing against real game
    guesses: List[str] = field(default_factory=list)
    feedback_history: List[Feedback] = field(default_factory=list)
    remaining_words: Optional[Set[str]] = None  # Valid words that match constraints
    
    def __post_init__(self):
        """Initialize remaining words if not set."""
        if self.remaining_words is None:
            self.remaining_words = set()
    
    def add_guess(self, guess: str, feedback: Feedback) -> None:
        """Add a guess and its feedback to the game state.
        
        Args:
            guess: The word that was guessed.
            feedback: The feedback for this guess.
        """
        if len(guess) != 5:
            raise ValueError(f"Guess must be 5 letters, got {len(guess)}")
        
        self.guesses.append(guess.upper())
        self.feedback_history.append(feedback)
    
    def is_solved(self) -> bool:
        """Check if the puzzle has been solved."""
        if not self.feedback_history:
            return False
        return self.feedback_history[-1].is_solved()
    
    def is_game_over(self) -> bool:
        """Check if the game is over (solved or out of guesses)."""
        return self.is_solved() or len(self.guesses) >= 6
    
    def get_guess_count(self) -> int:
        """Get the number of guesses made."""
        return len(self.guesses)
    
    def get_remaining_guesses(self) -> int:
        """Get the number of remaining guesses."""
        return max(0, 6 - len(self.guesses))
    
    def update_remaining_words(self, valid_words: Set[str]) -> None:
        """Update the set of remaining valid words based on constraints.
        
        Args:
            valid_words: Set of all valid words to filter from.
        """
        # Start with all valid words
        remaining = set(valid_words)
        
        # Apply constraints from each feedback
        for feedback in self.feedback_history:
            remaining = self._filter_words(remaining, feedback)
        
        self.remaining_words = remaining
    
    def _filter_words(self, words: Set[str], feedback: Feedback) -> Set[str]:
        """Filter words based on feedback constraints.
        
        Args:
            words: Set of words to filter.
            feedback: Feedback to apply as constraints.
        
        Returns:
            Filtered set of words.
        """
        filtered = set()
        guess = feedback.guess.upper()
        
        for word in words:
            if self._word_matches_feedback(word, guess, feedback):
                filtered.add(word)
        
        return filtered
    
    def _word_matches_feedback(self, word: str, guess: str, feedback: Feedback) -> bool:
        """Check if a word matches the constraints from feedback.
        
        Args:
            word: Word to check.
            guess: The guess that generated the feedback.
            feedback: The feedback to match against.
        
        Returns:
            True if word matches all constraints.
        """
        word = word.upper()
        guess = guess.upper()
        
        # Track letter counts for handling duplicates
        word_letter_counts = {}
        for letter in word:
            word_letter_counts[letter] = word_letter_counts.get(letter, 0) + 1
        
        # First pass: handle correct positions (green)
        used_positions = set()
        letter_counts = word_letter_counts.copy()
        
        for i, fb_type in enumerate(feedback.feedback):
            if fb_type == FeedbackType.CORRECT:
                if word[i] != guess[i]:
                    return False
                used_positions.add(i)
                letter_counts[guess[i]] = letter_counts.get(guess[i], 0) - 1
        
        # Second pass: handle present letters (yellow) and absent letters (gray)
        for i, fb_type in enumerate(feedback.feedback):
            if i in used_positions:
                continue
            
            letter = guess[i]
            
            if fb_type == FeedbackType.PRESENT:
                # Letter must be in word, but not at this position
                if word[i] == letter:
                    return False  # Can't be at this position
                if letter not in word or letter_counts.get(letter, 0) <= 0:
                    return False  # Must be present elsewhere
                letter_counts[letter] = letter_counts.get(letter, 0) - 1
            
            elif fb_type == FeedbackType.ABSENT:
                # Letter must not be in word (but check if it's used elsewhere)
                # If the letter appears in the guess multiple times, we need to be careful
                guess_count = sum(1 for j, g in enumerate(guess) if g == letter and j != i)
                feedback_count = sum(
                    1 for j, f in enumerate(feedback.feedback)
                    if f in (FeedbackType.CORRECT, FeedbackType.PRESENT) and guess[j] == letter
                )
                
                # If all occurrences of this letter in the guess are marked absent,
                # then the letter is not in the word
                if guess_count == 0:  # Only one occurrence
                    if letter in word:
                        return False
                else:
                    # Multiple occurrences - check if we've accounted for all
                    if feedback_count == 0 and letter in word:
                        return False
        
        return True
    
    def __repr__(self) -> str:
        """String representation."""
        status = "SOLVED" if self.is_solved() else f"{self.get_guess_count()}/6 guesses"
        return f"GameState(status={status}, guesses={self.guesses})"

