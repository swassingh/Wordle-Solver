"""Word list loading and validation."""

from pathlib import Path
from typing import Set, List, Optional


class WordLists:
    """Manages valid guess and answer word lists for Wordle."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize word lists from data directory.
        
        Args:
            data_dir: Path to directory containing word list files.
                     Defaults to project data/ directory.
        """
        if data_dir is None:
            # Default to project data directory
            project_root = Path(__file__).parent.parent.parent
            data_dir = project_root / "data"
        
        self.data_dir = Path(data_dir)
        self._valid_guesses: Set[str] = set()
        self._valid_answers: Set[str] = set()
        self._all_valid_words: Set[str] = set()
        
        self._load_word_lists()
    
    def _load_word_lists(self) -> None:
        """Load word lists from files."""
        guesses_path = self.data_dir / "valid_guesses.txt"
        answers_path = self.data_dir / "valid_answers.txt"
        
        # Load valid guesses
        if guesses_path.exists():
            with open(guesses_path, 'r', encoding='utf-8') as f:
                self._valid_guesses = {
                    word.strip().upper()
                    for word in f
                    if word.strip() and len(word.strip()) == 5
                }
        else:
            raise FileNotFoundError(
                f"Word list file not found: {guesses_path}\n"
                "Run 'python scripts/download_wordlists.py' to download word lists."
            )
        
        # Load valid answers
        if answers_path.exists():
            with open(answers_path, 'r', encoding='utf-8') as f:
                self._valid_answers = {
                    word.strip().upper()
                    for word in f
                    if word.strip() and len(word.strip()) == 5
                }
        else:
            raise FileNotFoundError(
                f"Word list file not found: {answers_path}\n"
                "Run 'python scripts/download_wordlists.py' to download word lists."
            )
        
        # Combined set of all valid words
        self._all_valid_words = self._valid_guesses | self._valid_answers
        
        if not self._valid_guesses:
            raise ValueError("No valid guess words loaded. Check word list files.")
        if not self._valid_answers:
            raise ValueError("No valid answer words loaded. Check word list files.")
    
    @property
    def valid_guesses(self) -> Set[str]:
        """Get set of all valid guess words."""
        return self._valid_guesses.copy()
    
    @property
    def valid_answers(self) -> Set[str]:
        """Get set of all valid answer words."""
        return self._valid_answers.copy()
    
    @property
    def all_valid_words(self) -> Set[str]:
        """Get set of all valid words (guesses + answers)."""
        return self._all_valid_words.copy()
    
    def is_valid_guess(self, word: str) -> bool:
        """Check if a word is a valid guess.
        
        Args:
            word: Word to check (case-insensitive).
        
        Returns:
            True if word is a valid guess.
        """
        return word.upper() in self._valid_guesses or word.upper() in self._valid_answers
    
    def is_valid_answer(self, word: str) -> bool:
        """Check if a word is a valid answer.
        
        Args:
            word: Word to check (case-insensitive).
        
        Returns:
            True if word is a valid answer.
        """
        return word.upper() in self._valid_answers
    
    def filter_valid_words(self, words: List[str]) -> List[str]:
        """Filter a list of words to only include valid guesses.
        
        Args:
            words: List of words to filter.
        
        Returns:
            List of valid words.
        """
        return [w for w in words if self.is_valid_guess(w.upper())]
    
    def get_word_list(self, include_answers: bool = True) -> List[str]:
        """Get a list of all valid words.
        
        Args:
            include_answers: If True, include answer words. If False, only guesses.
        
        Returns:
            Sorted list of valid words.
        """
        if include_answers:
            return sorted(list(self._all_valid_words))
        else:
            return sorted(list(self._valid_guesses))
    
    def get_answer_list(self) -> List[str]:
        """Get a list of all valid answer words.
        
        Returns:
            Sorted list of answer words.
        """
        return sorted(list(self._valid_answers))
    
    def __len__(self) -> int:
        """Return total number of valid words."""
        return len(self._all_valid_words)
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"WordLists(guesses={len(self._valid_guesses)}, "
            f"answers={len(self._valid_answers)}, "
            f"total={len(self._all_valid_words)})"
        )

