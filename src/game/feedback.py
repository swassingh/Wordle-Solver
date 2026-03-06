"""Feedback calculation for Wordle guesses."""

from typing import List
from src.domain.game_state import Feedback, FeedbackType


def calculate_feedback(guess: str, target: str) -> Feedback:
    """Calculate feedback for a guess against a target word.
    
    Wordle feedback rules:
    - Green (CORRECT): Letter is in the word at this exact position
    - Yellow (PRESENT): Letter is in the word but at a different position
    - Gray (ABSENT): Letter is not in the word
    
    For duplicate letters:
    - Each letter in the target can only match one letter in the guess
    - Green matches take priority
    - Yellow matches are assigned to remaining positions
    - If a letter appears more times in the guess than in the target,
      excess occurrences are marked as absent
    
    Args:
        guess: The guessed word (5 letters).
        target: The target word (5 letters).
    
    Returns:
        Feedback object with feedback for each position.
    
    Raises:
        ValueError: If guess or target are not 5 letters.
    """
    guess = guess.upper()
    target = target.upper()
    
    if len(guess) != 5:
        raise ValueError(f"Guess must be 5 letters, got {len(guess)}")
    if len(target) != 5:
        raise ValueError(f"Target must be 5 letters, got {len(target)}")
    
    feedback = [FeedbackType.ABSENT] * 5
    
    # Count letters in target (available for matching)
    target_letter_counts = {}
    for letter in target:
        target_letter_counts[letter] = target_letter_counts.get(letter, 0) + 1
    
    # First pass: mark correct positions (green)
    used_target_positions = set()
    for i in range(5):
        if guess[i] == target[i]:
            feedback[i] = FeedbackType.CORRECT
            used_target_positions.add(i)
            target_letter_counts[guess[i]] -= 1
    
    # Second pass: mark present letters (yellow)
    for i in range(5):
        if feedback[i] == FeedbackType.CORRECT:
            continue  # Already marked as correct
        
        letter = guess[i]
        
        # Check if this letter exists in target at other positions
        # and we haven't used up all occurrences
        if target_letter_counts.get(letter, 0) > 0:
            # Find a position in target where this letter exists and isn't used
            for j in range(5):
                if j not in used_target_positions and target[j] == letter:
                    feedback[i] = FeedbackType.PRESENT
                    used_target_positions.add(j)
                    target_letter_counts[letter] -= 1
                    break
    
    # Remaining positions are already marked as ABSENT
    
    return Feedback(guess=guess, feedback=feedback)


def parse_feedback_string(feedback_str: str) -> List[FeedbackType]:
    """Parse a feedback string into FeedbackType list.
    
    Accepts formats:
    - "GGYXX" (G=green, Y=yellow, X=gray)
    - "green yellow gray" (space-separated)
    - "🟩🟨⬛" (emoji format)
    
    Args:
        feedback_str: String representation of feedback.
    
    Returns:
        List of FeedbackType values.
    
    Raises:
        ValueError: If feedback string is invalid.
    """
    feedback_str = feedback_str.strip().upper()
    
    # Handle emoji format
    if "🟩" in feedback_str or "🟨" in feedback_str or "⬛" in feedback_str:
        result = []
        for char in feedback_str:
            if char == "🟩":
                result.append(FeedbackType.CORRECT)
            elif char == "🟨":
                result.append(FeedbackType.PRESENT)
            elif char == "⬛":
                result.append(FeedbackType.ABSENT)
        if len(result) == 5:
            return result
    
    # Handle letter format (G/Y/X)
    if len(feedback_str) == 5 and all(c in "GYX" for c in feedback_str):
        mapping = {
            "G": FeedbackType.CORRECT,
            "Y": FeedbackType.PRESENT,
            "X": FeedbackType.ABSENT,
        }
        return [mapping[c] for c in feedback_str]
    
    # Handle space-separated words
    words = feedback_str.split()
    if len(words) == 5:
        mapping = {
            "GREEN": FeedbackType.CORRECT,
            "YELLOW": FeedbackType.PRESENT,
            "GRAY": FeedbackType.ABSENT,
            "GREY": FeedbackType.ABSENT,  # British spelling
        }
        result = []
        for word in words:
            word_upper = word.upper()
            if word_upper in mapping:
                result.append(mapping[word_upper])
            else:
                raise ValueError(f"Invalid feedback word: {word}")
        return result
    
    raise ValueError(
        f"Invalid feedback format: {feedback_str}\n"
        "Expected: 'GGYXX' (G=green, Y=yellow, X=gray) or "
        "'green yellow gray green gray'"
    )


def feedback_to_string(feedback: Feedback) -> str:
    """Convert feedback to a human-readable string.
    
    Args:
        feedback: Feedback object.
    
    Returns:
        String representation (e.g., "🟩🟨⬛🟩⬛").
    """
    return str(feedback)


def feedback_to_letters(feedback: Feedback) -> str:
    """Convert feedback to letter codes (G=green, Y=yellow, X=gray).
    
    Args:
        feedback: Feedback object.
    
    Returns:
        String of letter codes (e.g., "GYXGX").
    """
    mapping = {
        FeedbackType.CORRECT: "G",
        FeedbackType.PRESENT: "Y",
        FeedbackType.ABSENT: "X",
    }
    return "".join(mapping[f] for f in feedback.feedback)
