"""Feature engineering for ML models."""

import numpy as np
from collections import Counter
from typing import List, Dict, Set
from src.domain.game_state import GameState, FeedbackType
from src.domain.word_lists import WordLists


class FeatureExtractor:
    """Extracts features from game state for ML models."""
    
    def __init__(self, word_lists: WordLists):
        """Initialize feature extractor.
        
        Args:
            word_lists: WordLists instance for word validation.
        """
        self.word_lists = word_lists
        self.letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    def extract_features(self, game_state: GameState, candidate_word: str) -> np.ndarray:
        """Extract features for a candidate word given game state.
        
        Args:
            game_state: Current game state.
            candidate_word: Word to extract features for.
        
        Returns:
            Feature vector as numpy array.
        """
        features = []
        
        # Basic word features
        features.extend(self._word_features(candidate_word))
        
        # Game state features
        features.extend(self._game_state_features(game_state))
        
        # Candidate-specific features relative to game state
        features.extend(self._candidate_game_features(candidate_word, game_state))
        
        # Remaining words features
        remaining = game_state.remaining_words or set()
        features.extend(self._remaining_words_features(candidate_word, remaining))
        
        return np.array(features, dtype=np.float32)
    
    def _word_features(self, word: str) -> List[float]:
        """Extract basic word features.
        
        Args:
            word: Word to extract features from.
        
        Returns:
            List of feature values.
        """
        features = []
        word = word.upper()
        
        # Letter frequency features (position-based)
        for i in range(5):
            letter = word[i]
            # One-hot encoding for each position (26 letters)
            letter_idx = ord(letter) - ord('A')
            features.append(letter_idx / 25.0)  # Normalize to [0, 1]
        
        # Unique letters count
        unique_letters = len(set(word))
        features.append(unique_letters / 5.0)
        
        # Vowel count
        vowels = sum(1 for c in word if c in 'AEIOU')
        features.append(vowels / 5.0)
        
        # Common letter features (E, A, R, I, O, T, N, S)
        common_letters = 'EAROTINS'
        common_count = sum(1 for c in word if c in common_letters)
        features.append(common_count / 5.0)
        
        return features
    
    def _game_state_features(self, game_state: GameState) -> List[float]:
        """Extract features from game state.
        
        Args:
            game_state: Current game state.
        
        Returns:
            List of feature values.
        """
        features = []
        
        # Number of guesses made
        guess_count = len(game_state.guesses)
        features.append(guess_count / 6.0)  # Normalize to [0, 1]
        
        # Feedback history features
        if game_state.feedback_history:
            last_feedback = game_state.feedback_history[-1]
            
            # Count of each feedback type in last guess
            feedback_counts = Counter(last_feedback.feedback)
            features.append(feedback_counts.get(FeedbackType.CORRECT, 0) / 5.0)
            features.append(feedback_counts.get(FeedbackType.PRESENT, 0) / 5.0)
            features.append(feedback_counts.get(FeedbackType.ABSENT, 0) / 5.0)
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Known correct positions (from all feedback)
        known_positions = [None] * 5
        known_letters = set()
        absent_letters = set()
        
        for feedback in game_state.feedback_history:
            for i, fb_type in enumerate(feedback.feedback):
                letter = feedback.guess[i]
                if fb_type == FeedbackType.CORRECT:
                    known_positions[i] = letter
                    known_letters.add(letter)
                elif fb_type == FeedbackType.PRESENT:
                    known_letters.add(letter)
                elif fb_type == FeedbackType.ABSENT:
                    # Only mark absent if letter doesn't appear elsewhere
                    if letter not in known_letters:
                        absent_letters.add(letter)
        
        # Count known positions
        known_count = sum(1 for p in known_positions if p is not None)
        features.append(known_count / 5.0)
        
        # Count known letters (not in position)
        features.append(len(known_letters) / 26.0)
        
        # Count absent letters
        features.append(len(absent_letters) / 26.0)
        
        return features
    
    def _candidate_game_features(self, candidate: str, game_state: GameState) -> List[float]:
        """Extract features relating candidate to game state.
        
        Args:
            candidate: Candidate word.
            game_state: Current game state.
        
        Returns:
            List of feature values.
        """
        features = []
        candidate = candidate.upper()
        
        # Check if candidate matches known positions
        known_positions = [None] * 5
        known_letters = set()
        absent_letters = set()
        
        for feedback in game_state.feedback_history:
            for i, fb_type in enumerate(feedback.feedback):
                letter = feedback.guess[i]
                if fb_type == FeedbackType.CORRECT:
                    known_positions[i] = letter
                    known_letters.add(letter)
                elif fb_type == FeedbackType.PRESENT:
                    known_letters.add(letter)
                elif fb_type == FeedbackType.ABSENT:
                    if letter not in known_letters:
                        absent_letters.add(letter)
        
        # Match known positions
        matches = sum(1 for i, letter in enumerate(candidate) 
                     if known_positions[i] is not None and known_positions[i] == letter)
        features.append(matches / 5.0)
        
        # Contains known letters (not in position)
        contains_known = sum(1 for c in candidate if c in known_letters)
        features.append(contains_known / 5.0)
        
        # Contains absent letters (bad)
        contains_absent = sum(1 for c in candidate if c in absent_letters)
        features.append(contains_absent / 5.0)
        
        # Overlap with previous guesses
        if game_state.guesses:
            overlap = sum(len(set(candidate) & set(guess)) for guess in game_state.guesses)
            features.append(overlap / (len(game_state.guesses) * 5.0))
        else:
            features.append(0.0)
        
        return features
    
    def _remaining_words_features(self, candidate: str, remaining: Set[str]) -> List[float]:
        """Extract features based on remaining candidate words.
        
        Args:
            candidate: Candidate word.
            remaining: Set of remaining valid words.
        
        Returns:
            List of feature values.
        """
        features = []
        candidate = candidate.upper()
        
        if not remaining:
            # 5 positional freqs + 1 overall freq + 1 in-remaining + 1 similarity +
            # 1 log(count) + 1 is_answer + 1 normalized remaining size = 10
            return [0.0] * 10
        
        remaining_list = list(remaining)
        
        # Position-based letter frequencies in remaining words
        for pos in range(5):
            letter_counts = Counter(word[pos] for word in remaining_list)
            candidate_letter = candidate[pos]
            freq = letter_counts.get(candidate_letter, 0) / len(remaining_list)
            features.append(freq)
        
        # Overall letter frequency in remaining words
        all_letters = ''.join(remaining_list)
        letter_counts = Counter(all_letters)
        candidate_letters = set(candidate)
        
        avg_freq = sum(letter_counts.get(c, 0) for c in candidate_letters) / max(len(candidate_letters), 1)
        features.append(avg_freq / (len(remaining_list) * 5.0))
        
        # Is candidate in remaining words (good if we're narrowing down)
        is_in_remaining = 1.0 if candidate in remaining else 0.0
        features.append(is_in_remaining)
        
        # Similarity to remaining words (average character overlap)
        similarities = []
        for word in remaining_list[:100]:  # Limit for performance
            overlap = len(set(candidate) & set(word))
            similarities.append(overlap / 5.0)
        
        if similarities:
            features.append(np.mean(similarities))
        else:
            features.append(0.0)
        
        # Remaining words count (log scale)
        features.append(np.log1p(len(remaining)) / np.log1p(10000))

        # Is this candidate in the official answer list?
        is_answer = 1.0 if self.word_lists.is_valid_answer(candidate) else 0.0
        features.append(is_answer)

        # Normalized remaining candidate set size
        remaining_size = len(remaining_list)
        total_words = max(len(self.word_lists.all_valid_words), 1)
        features.append(remaining_size / total_words)
        
        return features
    
    def get_feature_dimension(self) -> int:
        """Get the dimension of feature vectors.
        
        Returns:
            Feature vector size.
        """
        # Create a dummy game state to calculate feature size
        from src.domain.game_state import GameState
        dummy_state = GameState()
        dummy_word = "HELLO"
        features = self.extract_features(dummy_state, dummy_word)
        return len(features)

