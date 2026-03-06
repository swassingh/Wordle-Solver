"""Tests for solvers."""

import pytest
from src.solvers.info_theory import InformationTheorySolver
from src.game.simulator import WordleSimulator
from src.domain.word_lists import WordLists


def test_info_theory_solver_basic():
    """Test basic information theory solver functionality."""
    solver = InformationTheorySolver()
    simulator = WordleSimulator(solver.word_lists)
    
    simulator.start_game("HELLO")
    game_state = simulator.get_current_state()
    
    guess = solver.make_guess(game_state)
    assert len(guess) == 5
    assert solver.word_lists.is_valid_guess(guess)


def test_info_theory_solver_solves_word():
    """Test that solver can solve a word."""
    solver = InformationTheorySolver()
    simulator = WordleSimulator(solver.word_lists)
    
    target = "HELLO"
    simulator.start_game(target)
    
    max_guesses = 6
    guesses_made = 0
    
    while not simulator.is_game_over() and guesses_made < max_guesses:
        game_state = simulator.get_current_state()
        guess = solver.make_guess(game_state)
        simulator.make_guess(guess)
        guesses_made += 1
        
        if simulator.is_solved():
            break
    
    assert simulator.is_solved()
    assert guesses_made <= 6


def test_info_theory_solver_success_rate():
    """Test that solver achieves >80% success rate."""
    solver = InformationTheorySolver()
    word_lists = WordLists()
    
    # Test on a sample of answer words
    answers = word_lists.get_answer_list()
    test_words = answers[:100]  # Test on first 100 words
    
    successes = 0
    total_guesses = 0
    
    for target in test_words:
        simulator = WordleSimulator(word_lists)
        simulator.start_game(target)
        
        guesses_made = 0
        solved = False
        
        while not simulator.is_game_over() and guesses_made < 6:
            game_state = simulator.get_current_state()
            try:
                guess = solver.make_guess(game_state)
                simulator.make_guess(guess)
                guesses_made += 1
                
                if simulator.is_solved():
                    solved = True
                    break
            except (ValueError, IndexError):
                break
        
        if solved:
            successes += 1
            total_guesses += guesses_made
    
    success_rate = successes / len(test_words)
    avg_guesses = total_guesses / successes if successes > 0 else 0
    
    print(f"\nSuccess rate: {success_rate:.2%} ({successes}/{len(test_words)})")
    print(f"Average guesses: {avg_guesses:.2f}")
    
    # Should achieve at least 80% success rate
    assert success_rate >= 0.80, f"Success rate {success_rate:.2%} is below 80%"


def test_info_theory_solver_first_guess():
    """Test that solver uses good first guesses."""
    solver = InformationTheorySolver()
    word_lists = WordLists()
    
    simulator = WordleSimulator(word_lists)
    simulator.start_game("HELLO")
    
    game_state = simulator.get_current_state()
    first_guess = solver.make_guess(game_state)
    
    assert len(first_guess) == 5
    assert word_lists.is_valid_guess(first_guess)

