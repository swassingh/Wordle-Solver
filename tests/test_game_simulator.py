"""Tests for game simulator."""

import pytest
from src.game.simulator import WordleSimulator
from src.domain.word_lists import WordLists


def test_simulator_start_game():
    """Test starting a game."""
    simulator = WordleSimulator()
    game_state = simulator.start_game("HELLO")
    
    assert game_state.target_word == "HELLO"
    assert len(game_state.guesses) == 0
    assert not game_state.is_solved()


def test_simulator_make_guess():
    """Test making a guess."""
    simulator = WordleSimulator()
    simulator.start_game("HELLO")
    
    game_state, is_solved = simulator.make_guess("WORLD")
    assert len(game_state.guesses) == 1
    assert game_state.guesses[0] == "WORLD"
    assert not is_solved
    
    # Correct guess
    game_state, is_solved = simulator.make_guess("HELLO")
    assert is_solved
    assert game_state.is_solved()


def test_simulator_feedback_calculation():
    """Test feedback calculation."""
    simulator = WordleSimulator()
    simulator.start_game("HELLO")
    
    game_state, _ = simulator.make_guess("HELLO")
    feedback = game_state.feedback_history[0]
    assert feedback.is_solved()
    assert all(f == feedback.feedback[0] for f in feedback.feedback)


def test_simulator_invalid_guess():
    """Test invalid guess handling."""
    simulator = WordleSimulator()
    simulator.start_game("HELLO")
    
    with pytest.raises(ValueError):
        simulator.make_guess("INVALID")


def test_simulator_game_over():
    """Test game over conditions."""
    simulator = WordleSimulator()
    simulator.start_game("HELLO")
    
    # Make 6 wrong guesses
    for i in range(6):
        try:
            simulator.make_guess("WORLD")
        except ValueError:
            # Game is over
            break
    
    assert simulator.is_game_over()

