# SPEC: Wordle Solver (ML-Powered)

## Problem
People are curious about how machine learning can solve word games like Wordle. There's educational and entertainment value in seeing an AI model systematically solve the New York Times Wordle puzzle, demonstrating:
- How ML models can make strategic decisions with limited information
- Optimal word selection strategies based on feedback patterns
- The effectiveness of different ML approaches (e.g., information theory, pattern recognition, reinforcement learning)

## Users
- People interested in seeing a machine solving Wordle
- ML enthusiasts and students learning about applied machine learning
- Developers curious about game-solving algorithms
- Anyone fascinated by AI solving word puzzles

## Scope (MVP)
- Train an ML model capable of solving NYT Wordle puzzles
- Interface to interact with Wordle game (API integration or game simulation)
- Visual/logged demonstration of the solving process (showing guesses and feedback)
- Model that can solve puzzles within the 6-guess limit
- Support for the standard Wordle word list and rules
- Success rate tracking and basic performance metrics

## Non-goals
- Building a full Wordle game interface for human players
- Creating a web application or GUI (MVP can be CLI/script-based)
- Supporting multiple languages or word variants
- Real-time solving with sub-second response times (can be slower for MVP)
- Handling edge cases like invalid words or API failures (basic error handling only)
- Model explainability/interpretability features (focus on solving, not explanation)

## Acceptance Criteria
- Given a valid Wordle puzzle, when the model attempts to solve it, then it should find the correct word within 6 guesses at least 80% of the time
- Given a Wordle puzzle, when the model makes a guess, then it should receive and process feedback (green/yellow/gray) correctly
- Given multiple puzzles, when the model solves them, then we can track and display success rate and average number of guesses
- Given the model, when it's trained, then it should be able to solve new puzzles it hasn't seen before
- Given a puzzle-solving session, when the model completes it, then the solving process should be logged/displayed showing each guess and feedback

## API Contract
**NYT Wordle Integration:**
- Option 1: Use NYT Wordle API (if available) to fetch daily puzzles and submit guesses
- Option 2: Simulate Wordle game locally using the official word list
- Option 3: Web scraping/interaction with NYT Wordle website (if API unavailable)

**Model Interface:**
- Input: Current game state (previous guesses, feedback patterns, remaining possibilities)
- Output: Next word guess (5-letter word from valid word list)
- Feedback format: Array of 5 states (green=correct position, yellow=wrong position, gray=not in word)

**Word Lists:**
- Valid guess words (12,972 words - NYT Wordle guess list)
- Valid answer words (2,315 words - NYT Wordle answer list)

## Telemetry / Metrics
- **Success Rate**: Percentage of puzzles solved within 6 guesses
- **Average Guesses**: Mean number of guesses required to solve puzzles
- **Distribution**: Histogram of guesses needed (1-6 guesses)
- **Failure Analysis**: Cases where model fails to solve in 6 guesses
- **Model Performance**: Training accuracy, validation metrics, convergence time
- **Solving Time**: Average time per puzzle (if applicable)
- **Word Selection Patterns**: Most common first/second guesses, strategy insights