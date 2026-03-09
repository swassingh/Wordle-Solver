"""Information theory-based Wordle solver using entropy maximization."""

import math
from collections import Counter
from typing import Set, List, Optional

from src.solvers.base import BaseSolver
from src.domain.game_state import GameState
from src.domain.word_lists import WordLists
from src.game.feedback import calculate_feedback, FeedbackType


class InformationTheorySolver(BaseSolver):
    """Solver that maximizes expected information gain using entropy."""
    
    # Precomputed best first guesses (common optimal starting words)
    BEST_FIRST_GUESSES = [
        "SALET", "REAST", "CRATE", "TRACE", "SLATE",
        "CARTE", "CARET", "ROATE", "RAISE", "ARISE"
    ]
    
    def __init__(self, word_lists: Optional[WordLists] = None, use_precomputed: bool = True):
        """Initialize the information theory solver.
        
        Args:
            word_lists: WordLists instance. If None, creates a new one.
            use_precomputed: If True, use precomputed best first guesses.
        """
        super().__init__(word_lists)
        self.use_precomputed = use_precomputed

        # Precompute letter / positional statistics over the answer list
        self._answer_list: List[str] = self.word_lists.get_answer_list()
        self._global_letter_freq = self._compute_global_letter_frequencies()
        self._positional_letter_freq = self._compute_positional_letter_frequencies()
    
    def make_guess(self, game_state: GameState) -> str:
        """Make a guess that maximizes expected information gain.
        
        Args:
            game_state: Current game state.
        
        Returns:
            Best guess word.
        
        Raises:
            ValueError: If no valid candidates remain.
        """
        candidates = self.get_valid_candidates(game_state)
        
        if not candidates:
            raise ValueError("No valid candidates remaining")
        
        # If this is the first guess and we have precomputed options, use them
        if len(game_state.guesses) == 0 and self.use_precomputed:
            # Use best precomputed guess if it's still valid
            for guess in self.BEST_FIRST_GUESSES:
                if guess in candidates:
                    return guess
        
        # If only one candidate remains, guess it
        if len(candidates) == 1:
            return list(candidates)[0]
        
        # If few candidates remain, prefer answers over guesses
        if len(candidates) <= 3:
            answer_candidates = {w for w in candidates if self.word_lists.is_valid_answer(w)}
            if answer_candidates:
                return list(answer_candidates)[0]
        
        # Calculate expected information gain for each candidate word.
        # To better optimize for *minimum guesses*, we combine entropy with:
        # - dynamic answer bias (stronger when few candidates remain)
        # - simple positional / letter-coverage heuristics
        best_word: Optional[str] = None
        best_score = -float("inf")
        
        # Limit search space for performance
        # For large candidate sets, only evaluate a subset
        words_to_evaluate = self._get_words_to_evaluate(candidates, game_state)
        
        for word in words_to_evaluate:
            score = self._score_candidate(
                word=word,
                candidates=candidates,
                guess_index=len(game_state.guesses),
            )

            if score > best_score:
                best_score = score
                best_word = word
        
        if best_word is None:
            # Fallback: return first candidate
            best_word = list(candidates)[0]
        
        return best_word
    
    def _get_words_to_evaluate(self, candidates: Set[str], game_state: GameState) -> List[str]:
        """Get list of words to evaluate for information gain.
        
        For performance, we may limit the search space.
        
        Args:
            candidates: Set of candidate words.
            game_state: Current game state.
        
        Returns:
            List of words to evaluate.
        """
        candidates_list = list(candidates)
        
        # If few candidates, evaluate all
        if len(candidates_list) <= 50:
            return candidates_list
        
        # For many candidates, prefer:
        # 1. Answer words (if any remain)
        # 2. Words with diverse letter patterns
        # 3. Limit to top 200 for performance
        
        # Always ensure all words from the official answer list are considered as answers.
        answer_words = [w for w in candidates_list if self.word_lists.is_valid_answer(w)]
        
        if answer_words:
            # Mix of answer words and diverse guess words
            diverse_guesses = self._get_diverse_words(
                [w for w in candidates_list if w not in answer_words],
                max_count=150
            )
            return answer_words[:50] + diverse_guesses
        
        # No answer words, use diverse guess words
        return self._get_diverse_words(candidates_list, max_count=200)

    # --- Heuristic scoring helpers -------------------------------------------------

    def _compute_global_letter_frequencies(self) -> Counter:
        """Compute overall letter frequency across all answer words."""
        freq = Counter()
        for word in self._answer_list:
            for ch in set(word):  # presence-based frequency
                freq[ch] += 1
        return freq

    def _compute_positional_letter_frequencies(self) -> List[Counter]:
        """Compute position-wise letter frequency across answer words."""
        pos_freq: List[Counter] = [Counter() for _ in range(5)]
        for word in self._answer_list:
            if len(word) != 5:
                continue
            for i, ch in enumerate(word):
                pos_freq[i][ch] += 1
        return pos_freq

    def _letter_coverage_score(self, word: str) -> float:
        """Score word by how well it covers common letters/positions."""
        seen = set()
        score = 0.0
        for i, ch in enumerate(word):
            if ch in seen:
                continue
            seen.add(ch)
            score += self._global_letter_freq.get(ch, 0)
            score += self._positional_letter_freq[i].get(ch, 0)
        # Normalize roughly by number of answers to keep this term small
        normalizer = max(len(self._answer_list), 1)
        return score / normalizer

    def _score_candidate(
        self,
        word: str,
        candidates: Set[str],
        guess_index: int,
    ) -> float:
        """Score a candidate word using entropy + heuristics.

        Args:
            word: candidate word
            candidates: remaining candidate set
            guess_index: 0-based index of the current guess
        """
        entropy_score = self._calculate_expected_information_gain(word, candidates)

        # Dynamic answer bias: small early, larger late-game.
        remaining = len(candidates)
        is_answer = 1.0 if self.word_lists.is_valid_answer(word) else 0.0

        if remaining > 50:
            answer_weight = 0.05
        elif remaining > 15:
            answer_weight = 0.15
        else:
            answer_weight = 0.4

        answer_bonus = answer_weight * is_answer

        # Letter coverage heuristic: prefer words that cover common letters,
        # especially in early guesses when the space is large.
        coverage_weight = 0.05 if remaining > 50 else 0.02
        coverage_score = coverage_weight * self._letter_coverage_score(word)

        # Penalize repeated letters early when many candidates remain.
        repeated_penalty = 0.0
        if remaining > 80:
            unique_letters = len(set(word))
            if unique_letters < 5:
                repeated_penalty = 0.08 * (5 - unique_letters)

        return entropy_score + answer_bonus + coverage_score - repeated_penalty
    
    def _get_diverse_words(self, words: List[str], max_count: int) -> List[str]:
        """Select diverse words (different letter patterns).
        
        Args:
            words: List of words to choose from.
            max_count: Maximum number of words to return.
        
        Returns:
            List of diverse words.
        """
        if len(words) <= max_count:
            return words
        
        # Simple diversity: prefer words with unique letter combinations
        # Sort by number of unique letters (more unique = better)
        scored = [(len(set(w)), w) for w in words]
        scored.sort(reverse=True)
        
        return [w for _, w in scored[:max_count]]
    
    def _calculate_expected_information_gain(self, word: str, candidates: Set[str]) -> float:
        """Calculate expected information gain for a word.
        
        Information gain = sum over all possible feedback patterns of:
            P(feedback) * log2(1 / P(feedback))
        
        Higher information gain = better word choice.
        
        Args:
            word: Word to evaluate.
            candidates: Set of remaining candidate words.
        
        Returns:
            Expected information gain (bits).
        """
        # Count how many candidates would match each possible feedback pattern
        feedback_counts = Counter()
        
        for candidate in candidates:
            feedback = calculate_feedback(word, candidate)
            # Convert feedback to a hashable pattern
            pattern = self._feedback_to_pattern(feedback)
            feedback_counts[pattern] += 1
        
        total = len(candidates)
        if total == 0:
            return 0.0
        
        # Calculate expected information gain (entropy)
        expected_gain = 0.0
        for count in feedback_counts.values():
            probability = count / total
            if probability > 0:
                # Information gain = -log2(P(feedback))
                # Expected value = sum P(feedback) * information_gain
                expected_gain += probability * (-math.log2(probability))
        
        return expected_gain
    
    def _feedback_to_pattern(self, feedback) -> tuple:
        """Convert feedback to a hashable pattern.
        
        Args:
            feedback: Feedback object.
        
        Returns:
            Tuple representation of feedback pattern.
        """
        return tuple(feedback.feedback)
    
    def score_word(self, word: str, candidates: Set[str]) -> float:
        """Score a word based on expected information gain.
        
        Args:
            word: Word to score.
            candidates: Set of remaining candidate words.
        
        Returns:
            Expected information gain score.
        """
        return self._calculate_expected_information_gain(word, candidates)

