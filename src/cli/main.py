"""Command-line interface for Wordle Solver."""

import sys
import random
from pathlib import Path
import click
from src.domain.word_lists import WordLists
from src.domain.game_state import GameState, Feedback, FeedbackType
from src.game.simulator import WordleSimulator
from src.game.feedback import parse_feedback_string
from src.solvers.info_theory import InformationTheorySolver
from src.solvers.ml_solver import MLSolver
from src.metrics.tracker import MetricsTracker


# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@click.group()
def cli():
    """Wordle Solver - ML-powered Wordle puzzle solver."""
    pass


@cli.command()
@click.argument('target_word')
@click.option('--solver', type=click.Choice(['info', 'ml']), default='info',
              help='Solver to use (info=information theory, ml=machine learning)')
@click.option('--model-path', type=click.Path(exists=True),
              help='Path to ML model file (required for ML solver)')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed solving process')
def solve(target_word: str, solver: str, model_path: str, verbose: bool):
    """Solve a specific Wordle puzzle.
    
    TARGET_WORD: The 5-letter word to solve.
    """
    try:
        word_lists = WordLists()
        simulator = WordleSimulator(word_lists)
        
        # Validate target word
        if len(target_word) != 5 or not target_word.isalpha():
            click.echo(f"Error: Target word must be 5 letters", err=True)
            sys.exit(1)
        
        target_word = target_word.upper()
        if not word_lists.is_valid_answer(target_word):
            click.echo(f"Warning: '{target_word}' is not in the answer list", err=True)
        
        # Initialize solver
        if solver == 'ml':
            if not model_path:
                click.echo("Error: --model-path required for ML solver", err=True)
                sys.exit(1)
            solver_obj = MLSolver(model_path=Path(model_path), word_lists=word_lists)
        else:
            solver_obj = InformationTheorySolver(word_lists)
        
        # Start game
        simulator.start_game(target_word)
        
        click.echo(f"\nSolving Wordle puzzle: {target_word}")
        click.echo("=" * 50)
        
        # Solve
        guesses_made = 0
        while not simulator.is_game_over() and guesses_made < 6:
            game_state = simulator.get_current_state()
            
            try:
                guess = solver_obj.make_guess(game_state)
            except (ValueError, IndexError) as e:
                click.echo(f"\nError: {e}", err=True)
                break
            
            game_state, is_solved = simulator.make_guess(guess)
            guesses_made += 1
            
            # Display guess and feedback
            feedback = game_state.feedback_history[-1]
            click.echo(f"Guess {guesses_made}: {guess} {feedback}")
            
            if is_solved:
                click.echo(f"\n✓ Solved in {guesses_made} guesses!")
                break
        
        if not simulator.is_solved():
            click.echo(f"\n✗ Failed to solve in 6 guesses")
            click.echo(f"Target word was: {target_word}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--solver', type=click.Choice(['info', 'ml']), default='info',
              help='Solver to use')
@click.option('--model-path', type=click.Path(exists=True),
              help='Path to ML model file')
@click.option('--count', '-n', default=1, help='Number of random words to solve')
def solve_random(solver: str, model_path: str, count: int):
    """Solve random Wordle puzzles."""
    try:
        word_lists = WordLists()
        answers = word_lists.get_answer_list()
        
        # Initialize solver
        if solver == 'ml':
            if not model_path:
                click.echo("Error: --model-path required for ML solver", err=True)
                sys.exit(1)
            solver_obj = MLSolver(model_path=Path(model_path), word_lists=word_lists)
        else:
            solver_obj = InformationTheorySolver(word_lists)
        
        simulator = WordleSimulator(word_lists)
        tracker = MetricsTracker()
        
        for i in range(count):
            target = random.choice(answers)
            simulator.start_game(target)
            
            click.echo(f"\n[{i+1}/{count}] Solving: {target}")
            
            guesses_made = 0
            while not simulator.is_game_over() and guesses_made < 6:
                game_state = simulator.get_current_state()
                try:
                    guess = solver_obj.make_guess(game_state)
                except (ValueError, IndexError):
                    break
                
                simulator.make_guess(guess)
                guesses_made += 1
                
                feedback = game_state.feedback_history[-1] if game_state.feedback_history else None
                if feedback:
                    click.echo(f"  {guess} {feedback}")
                
                if simulator.is_solved():
                    break
            
            # Record result
            final_state = simulator.get_current_state()
            tracker.record_game(final_state, target)
            
            if simulator.is_solved():
                click.echo(f"  ✓ Solved in {guesses_made} guesses")
            else:
                click.echo(f"  ✗ Failed")
        
        # Show summary
        tracker.print_summary()
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('n', type=int)
@click.option('--solver', type=click.Choice(['info', 'ml']), default='info',
              help='Solver to use')
@click.option('--model-path', type=click.Path(exists=True),
              help='Path to ML model file')
@click.option('--export', type=click.Path(), help='Export metrics to JSON file')
def benchmark(n: int, solver: str, model_path: str, export: str):
    """Benchmark solver on N random words.
    
    N: Number of words to solve.
    """
    try:
        word_lists = WordLists()
        answers = word_lists.get_answer_list()
        
        # Sample N words
        test_words = random.sample(answers, min(n, len(answers)))
        
        # Initialize solver
        if solver == 'ml':
            if not model_path:
                click.echo("Error: --model-path required for ML solver", err=True)
                sys.exit(1)
            solver_obj = MLSolver(model_path=Path(model_path), word_lists=word_lists)
        else:
            solver_obj = InformationTheorySolver(word_lists)
        
        simulator = WordleSimulator(word_lists)
        tracker = MetricsTracker()
        
        click.echo(f"\nBenchmarking on {len(test_words)} words...")
        click.echo("=" * 50)
        
        for i, target in enumerate(test_words):
            if (i + 1) % 50 == 0:
                click.echo(f"  Progress: {i+1}/{len(test_words)}...")
            
            simulator.start_game(target)
            
            guesses_made = 0
            while not simulator.is_game_over() and guesses_made < 6:
                game_state = simulator.get_current_state()
                try:
                    guess = solver_obj.make_guess(game_state)
                except (ValueError, IndexError):
                    break
                
                simulator.make_guess(guess)
                guesses_made += 1
                
                if simulator.is_solved():
                    break
            
            # Record result
            final_state = simulator.get_current_state()
            tracker.record_game(final_state, target)
        
        # Show results
        tracker.print_summary()
        
        # Export if requested
        if export:
            export_path = Path(export)
            tracker.export_json(export_path)
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--num-games', '-n', default=1000, help='Number of games for training')
@click.option('--model-type', type=click.Choice(['random_forest', 'gradient_boosting']),
              default='random_forest', help='Type of ML model')
@click.option('--output', '-o', type=click.Path(), default='models/wordle_model.pkl',
              help='Output path for trained model')
def train(num_games: int, model_type: str, output: str):
    """Train the ML model."""
    try:
        from src.ml.trainer import ModelTrainer
        
        word_lists = WordLists()
        trainer = ModelTrainer(word_lists, model_type=model_type)
        
        # Train
        metrics = trainer.train(num_games=num_games, validation_split=0.2)
        
        # Save
        output_path = Path(output)
        trainer.save_model(output_path)
        
        click.echo(f"\n✓ Training complete!")
        click.echo(f"  Model saved to: {output_path}")
        click.echo(f"  Training accuracy: {metrics['train_accuracy']:.2%}")
        click.echo(f"  Validation accuracy: {metrics['val_accuracy']:.2%}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.option('--solver', type=click.Choice(['info', 'ml']), default='info',
              help='Solver to use (info=information theory, ml=machine learning)')
@click.option('--model-path', type=click.Path(exists=True),
              help='Path to ML model file (required for ML solver)')
def daily(solver: str, model_path: str):
    """Interactive mode to solve today's NYT Wordle.
    
    The solver will suggest guesses, and you enter the feedback
    you received from the NYT Wordle game.
    
    Feedback format: Enter as 'GGYXX' where:
    - G = Green (correct letter, correct position)
    - Y = Yellow (correct letter, wrong position)
    - X = Gray (letter not in word)
    
    Example: If you guess "HELLO" and get green-green-yellow-gray-gray,
    enter: GGYXX
    """
    try:
        word_lists = WordLists()
        
        # Initialize solver
        if solver == 'ml':
            if not model_path:
                click.echo("Error: --model-path required for ML solver", err=True)
                sys.exit(1)
            solver_obj = MLSolver(model_path=Path(model_path), word_lists=word_lists)
        else:
            solver_obj = InformationTheorySolver(word_lists)
        
        # Create game state (no target word since we're playing the real game)
        game_state = GameState(remaining_words=word_lists.all_valid_words.copy())
        
        click.echo("\n" + "="*60)
        click.echo("NYT Wordle Daily Solver - Interactive Mode")
        click.echo("="*60)
        click.echo("\nInstructions:")
        click.echo("  1. The solver will suggest a word")
        click.echo("  2. Enter that word in the NYT Wordle game")
        click.echo("  3. Enter the feedback you received (format: GGYXX)")
        click.echo("     G = Green, Y = Yellow, X = Gray")
        click.echo("  4. Repeat until solved!")
        click.echo("\n" + "-"*60)
        
        guess_number = 0
        while guess_number < 6:
            guess_number += 1
            
            # Get suggestion from solver
            try:
                suggested_guess = solver_obj.make_guess(game_state)
            except (ValueError, IndexError) as e:
                click.echo(f"\nError: {e}")
                click.echo("No valid guesses remaining. Check your feedback entries.")
                break
            
            click.echo(f"\n[Guess {guess_number}/6]")
            click.echo(f"Suggested word: {click.style(suggested_guess, fg='bright_cyan', bold=True)}")
            
            # Get user's actual guess (they might want to use a different word)
            user_guess = click.prompt(
                "Enter the word you guessed (or press Enter to use suggestion)",
                default=suggested_guess,
                type=str
            ).upper().strip()
            
            if len(user_guess) != 5 or not user_guess.isalpha():
                click.echo("Invalid word. Please enter a 5-letter word.")
                guess_number -= 1
                continue
            
            if not word_lists.is_valid_guess(user_guess):
                click.echo(f"Warning: '{user_guess}' may not be a valid Wordle guess.")
                if not click.confirm("Continue anyway?"):
                    guess_number -= 1
                    continue
            
            # Get feedback from user
            while True:
                feedback_input = click.prompt(
                    "Enter feedback (GGYXX format, or 'green yellow gray'):",
                    type=str
                ).strip()
                
                try:
                    feedback_types = parse_feedback_string(feedback_input)
                    break
                except ValueError as e:
                    click.echo(f"Invalid feedback format: {e}")
                    click.echo("Format: GGYXX (G=green, Y=yellow, X=gray)")
            
            # Create feedback object
            feedback = Feedback(guess=user_guess, feedback=feedback_types)
            
            # Display feedback visually
            click.echo(f"Feedback: {feedback}")
            
            # Update game state
            game_state.add_guess(user_guess, feedback)
            game_state.update_remaining_words(word_lists.all_valid_words)
            
            # Check if solved
            if feedback.is_solved():
                click.echo("\n" + "="*60)
                click.echo(click.style("🎉 SOLVED!", fg='green', bold=True))
                click.echo(f"Solved in {guess_number} guesses!")
                click.echo("="*60)
                break
            
            # Show remaining possibilities
            remaining_count = len(game_state.remaining_words) if game_state.remaining_words else 0
            if remaining_count > 0:
                click.echo(f"Remaining possibilities: {remaining_count}")
                if remaining_count <= 10:
                    click.echo(f"Possible words: {', '.join(sorted(list(game_state.remaining_words))[:10])}")
        
        if not game_state.is_solved():
            click.echo("\n" + "="*60)
            click.echo("Could not solve in 6 guesses.")
            if game_state.remaining_words and len(game_state.remaining_words) <= 20:
                click.echo(f"Remaining possibilities: {', '.join(sorted(list(game_state.remaining_words)))}")
            click.echo("="*60)
        
    except KeyboardInterrupt:
        click.echo("\n\nSolver interrupted by user.")
        sys.exit(0)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    cli()

