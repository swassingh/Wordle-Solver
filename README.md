# WordleAI (ML-Powered)

A machine learning-powered solver for the New York Times Wordle game. This project implements both information theory-based and ML-based approaches to solve Wordle puzzles.

## Features

- **Information Theory Solver**: Baseline entropy-based solver that maximizes information gain
- **ML Classic Solver**: Classification model that imitates the information-theory solver
- **ML Min-Guess Solver**: Regression model trained to minimize the expected number of guesses
- **NYT-Style Web UI**: 6×5 board + on-screen keyboard + suggestions panel, powered by the solver
- **HTTP API (FastAPI)**: `/api/solve/step` endpoint consumed by the web UI and other clients
- **Game Simulation & Metrics**: Full Wordle simulation, success rate, average guesses, and guess distribution tracking

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

## Hosted Version

You can try WordleAI in your browser here (no local setup required):  
`https://wordle-solver-production-169e.up.railway.app/web/`

## Usage

### 1. Start API & Web UI (recommended)

```bash
python -m src.api.server
```

Then open `http://localhost:8000/web/` in your browser. You’ll see:

- A 6×5 board that mirrors the NYT Wordle grid
- An on-screen keyboard (you can also type on your physical keyboard)
- Clickable tiles that cycle feedback `X → Y → G`
- A **Suggestions** panel with:
  - Strategy selector: `ML (min guesses)`, `ML (classic)`, `Information theory`
  - Next suggested word from the chosen strategy

Workflow:
1. Use the keyboard to enter the same word you played on NYT.
2. Click each tile to match the feedback colors (X = gray, Y = yellow, G = green).
3. Click **Get suggestion** (or press Enter) to get the next recommended guess.
4. Repeat until solved.

### 2. CLI usage

#### Solve a specific word

```bash
# Information theory only
python -m src.cli.main solve HELLO --strategy info

# ML (classic) - classification model
python -m src.cli.main solve HELLO \
  --strategy ml_classic \
  --classic-model models/wordle_model_classic.pkl

# ML (min guesses) - regression model optimized for fewest guesses
python -m src.cli.main solve HELLO \
  --strategy ml_min_guess \
  --min-guess-model models/wordle_model_min_guess.pkl
```

#### Solve random words / benchmark via CLI

```bash
python -m src.cli.main solve-random -n 50 --strategy info

python -m src.cli.main solve-random -n 50 \
  --strategy ml_classic \
  --classic-model models/wordle_model_classic.pkl

python -m src.cli.main solve-random -n 50 \
  --strategy ml_min_guess \
  --min-guess-model models/wordle_model_min_guess.pkl
```

### 3. Train ML models

Train classic (classification), min-guess (regression), or both:

```bash
# Classic only (simulated data only)
python scripts/train_model.py --mode classic --games 800

# Min-guess only (simulated data only)
python scripts/train_model.py --mode min-guess --games 500

# Train both models (simulated data only)
python scripts/train_model.py --mode both --games 800

# Mix in real daily-session data (if data/daily_sessions.jsonl exists)
python scripts/train_model.py --mode both --games 800 \
  --use-daily \
  --daily-path data/daily_sessions.jsonl \
  --daily-weight 0.2 \
  --tag 20260309
```

This produces:

- `models/wordle_model_classic.pkl`
- `models/wordle_model_min_guess.pkl`

If you pass `--tag SOME_TAG`, the script will also save timestamped / tagged
versions:

- `models/wordle_model_classic_SOME_TAG.pkl`
- `models/wordle_model_min_guess_SOME_TAG.pkl`

## Project Structure

- `src/domain/`: Core data structures (game state, word lists)
- `src/game/`: Game simulation and feedback logic
- `src/solvers/`: Solver implementations (info theory, ML, strategy selector)
- `src/ml/`: ML model training and inference
- `src/metrics/`: Performance metrics tracking
- `src/api/`: FastAPI server exposing HTTP API and serving the web UI
- `src/cli/`: Command-line interface
- `web/public/`: Static web UI (HTML/CSS/JS)
- `data/`: Word lists (valid guesses and answers)
- `scripts/`: Utility scripts for data download, training, and benchmarking

## Benchmarking & Model Rollback

- **Run benchmarks** for any strategy using held-out answers:

```bash
python scripts/benchmark_strategies.py --strategy info --games 500

python scripts/benchmark_strategies.py --strategy ml_classic --games 500 \
  --classic-model models/wordle_model_classic.pkl

python scripts/benchmark_strategies.py --strategy ml_min_guess --games 500 \
  --min-guess-model models/wordle_model_min_guess.pkl
```

- **Export metrics and enforce simple acceptance thresholds** (useful in CI):

```bash
python scripts/benchmark_strategies.py --strategy ml_classic --games 500 \
  --classic-model models/wordle_model_classic_20260309.pkl \
  --metrics-json benchmarks/ml_classic_20260309.json \
  --min-success-rate 0.97 \
  --max-average-guesses 3.7
```

If the measured success rate or average guesses fall outside the thresholds,
the script exits with a non-zero status to signal a regression.

- **Recommended deployment / rollback workflow**:
  - Train new models with a unique tag (e.g. date): `--tag 20260309`.
  - Benchmark the tagged models vs the current production models using the
    commands above and compare JSON metrics if desired.
  - If the new models pass your thresholds, copy/symlink them to the
    untagged filenames used by the API:
    - `models/wordle_model_classic.pkl`
    - `models/wordle_model_min_guess.pkl`
  - To **roll back**, simply restore the previous `.pkl` files (or point the
    API back to the earlier tagged versions) and rerun quick benchmarks to
    confirm behavior.

## Testing

Run tests with:
```bash
pytest tests/
```

## License

MIT

