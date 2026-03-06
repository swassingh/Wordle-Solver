# Wordle Solver (ML-Powered)

A machine learning-powered solver for the New York Times Wordle game. This project implements both information theory-based and ML-based approaches to solve Wordle puzzles.

## Features

- **Information Theory Solver**: Baseline entropy-based solver that maximizes information gain
- **ML Model Solver**: Machine learning model trained on optimal solving strategies
- **Game Simulation**: Full Wordle game simulation with feedback calculation
- **Metrics Tracking**: Success rate, average guesses, and detailed performance metrics
- **CLI Interface**: Command-line tool to solve puzzles and benchmark performance

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download word lists:
   ```bash
   python scripts/download_wordlists.py
   ```

## Usage

### Solve Today's NYT Wordle (Interactive Mode)
This is the recommended way to solve the actual daily NYT Wordle puzzle:

```bash
python -m src.cli.main daily
```

The solver will:
1. Suggest a word to guess
2. You enter that word in the NYT Wordle game
3. Enter the feedback you received (format: `GGYXX` where G=green, Y=yellow, X=gray)
4. Repeat until solved!

**Example:**
```
[Guess 1/6]
Suggested word: SALET
Enter the word you guessed: SALET
Enter feedback (GGYXX format): XXYXX
Feedback: ⬛⬛🟨⬛⬛

[Guess 2/6]
Suggested word: CRANE
...
```

### Solve a specific word
```bash
python -m src.cli.main solve <target_word>
```

### Solve a random word
```bash
python -m src.cli.main solve-random
```

### Benchmark on multiple words
```bash
python -m src.cli.main benchmark 100
```

### Train the ML model
```bash
python -m src.cli.main train
```

## Project Structure

- `src/domain/`: Core data structures (game state, word lists)
- `src/game/`: Game simulation and feedback logic
- `src/solvers/`: Solver implementations (info theory, ML)
- `src/ml/`: ML model training and inference
- `src/metrics/`: Performance metrics tracking
- `src/cli/`: Command-line interface
- `data/`: Word lists (valid guesses and answers)
- `scripts/`: Utility scripts for data download and training

## Testing

Run tests with:
```bash
pytest tests/
```

## License

MIT

