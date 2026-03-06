"""Download NYT Wordle word lists."""

import os
import requests
from pathlib import Path

# NYT Wordle word lists from GitHub
VALID_GUESSES_URL = "https://raw.githubusercontent.com/charlesreid1/five-letter-words/master/sgb-words.txt"
VALID_ANSWERS_URL = "https://raw.githubusercontent.com/charlesreid1/five-letter-words/master/sgb-words.txt"

# Alternative: Use a more comprehensive source
# These URLs point to common Wordle word list sources
# Note: You may need to adjust these URLs based on actual availability

def download_word_list(url: str, output_path: Path) -> list[str]:
    """Download a word list from URL and save to file."""
    print(f"Downloading word list from {url}...")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Filter to 5-letter words and convert to uppercase
        words = [
            word.strip().upper()
            for word in response.text.split('\n')
            if word.strip() and len(word.strip()) == 5 and word.strip().isalpha()
        ]
        
        # Remove duplicates and sort
        words = sorted(list(set(words)))
        
        # Write to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(words))
        
        print(f"Downloaded {len(words)} words to {output_path}")
        return words
    
    except Exception as e:
        print(f"Error downloading from {url}: {e}")
        print("Creating placeholder files. You may need to manually add word lists.")
        
        # Create empty placeholder files
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("")
        
        return []


def create_nyt_word_lists():
    """Create NYT-style word lists.
    
    Since the exact NYT word lists aren't publicly available via API,
    we'll use a comprehensive 5-letter word list and note that in production
    you'd want to use the actual NYT lists.
    """
    data_dir = Path(__file__).parent.parent / "data"
    
    guesses_path = data_dir / "valid_guesses.txt"
    answers_path = data_dir / "valid_answers.txt"
    
    # Download comprehensive word list
    all_words = download_word_list(VALID_GUESSES_URL, guesses_path)
    
    # For answers, use a subset (common words)
    # In a real implementation, you'd use the actual NYT answer list
    # For now, we'll use the same list but note this limitation
    if all_words:
        # Use first 2315 words as answers (approximate NYT answer count)
        answer_words = all_words[:2315] if len(all_words) >= 2315 else all_words
        
        with open(answers_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(answer_words))
        
        print(f"Created answer list with {len(answer_words)} words at {answers_path}")
    else:
        # Create empty file
        answers_path.parent.mkdir(parents=True, exist_ok=True)
        with open(answers_path, 'w', encoding='utf-8') as f:
            f.write("")
        print(f"Created placeholder answer list at {answers_path}")
        print("NOTE: You may need to manually populate the word lists with NYT Wordle words")


if __name__ == "__main__":
    create_nyt_word_lists()
    print("\nWord list download complete!")
    print("NOTE: For production use, replace these with actual NYT Wordle word lists.")
    print("The NYT word lists are not publicly available via API, so you may need")
    print("to source them from other repositories or extract them manually.")

