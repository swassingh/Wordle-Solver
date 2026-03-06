"""Quick start script to download word lists and test the solver."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Quick start setup."""
    print("="*50)
    print("Wordle Solver - Quick Start")
    print("="*50)
    
    # Step 1: Download word lists
    print("\n1. Downloading word lists...")
    try:
        from scripts.download_wordlists import create_nyt_word_lists
        create_nyt_word_lists()
        print("   ✓ Word lists downloaded")
    except Exception as e:
        print(f"   ✗ Error downloading word lists: {e}")
        print("   You may need to manually add word lists to the data/ directory")
        return
    
    # Step 2: Test solver
    print("\n2. Testing information theory solver...")
    try:
        from src.domain.word_lists import WordLists
        from src.game.simulator import WordleSimulator
        from src.solvers.info_theory import InformationTheorySolver
        
        word_lists = WordLists()
        solver = InformationTheorySolver(word_lists)
        simulator = WordleSimulator(word_lists)
        
        # Test on a simple word
        test_word = "HELLO"
        simulator.start_game(test_word)
        
        guesses = 0
        while not simulator.is_game_over() and guesses < 6:
            game_state = simulator.get_current_state()
            guess = solver.make_guess(game_state)
            simulator.make_guess(guess)
            guesses += 1
            
            if simulator.is_solved():
                print(f"   ✓ Solved '{test_word}' in {guesses} guesses")
                break
        
        if not simulator.is_solved():
            print(f"   ✗ Failed to solve '{test_word}'")
        
    except Exception as e:
        print(f"   ✗ Error testing solver: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*50)
    print("Quick start complete!")
    print("="*50)
    print("\nNext steps:")
    print("  1. Run: python -m src.cli.main solve HELLO")
    print("  2. Run: python -m src.cli.main benchmark 100")
    print("  3. Train ML model: python -m src.cli.main train")
    print("\nFor help: python -m src.cli.main --help")


if __name__ == "__main__":
    main()

