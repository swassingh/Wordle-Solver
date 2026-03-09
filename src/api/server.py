"""FastAPI server exposing WordleAI via HTTP API.

Endpoints:
- POST /api/solve/step: given guesses + X/Y/G feedback, return next suggested guess.
- POST /api/solve/reset: stateless API; provided for completeness.

The API is designed for use by the web UI, but can also be called directly.
"""

from pathlib import Path
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.domain.word_lists import WordLists
from src.domain.game_state import GameState, Feedback as GameFeedback
from src.game.feedback import parse_feedback_string
from src.solvers.info_theory import InformationTheorySolver
from src.solvers.ml_solver import MLSolver
from src.solvers.strategy import SolverStrategy, StrategyName
from src.metrics.session_schema import GameSession


# Default model paths (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_CLASSIC_MODEL = PROJECT_ROOT / "models" / "wordle_model_classic.pkl"
DEFAULT_MIN_GUESS_MODEL = PROJECT_ROOT / "models" / "wordle_model_min_guess.pkl"


app = FastAPI(
    title="WordleAI API",
    description="HTTP API for WordleAI (information theory + ML).",
    version="0.1.0",
)


# CORS configuration (relaxed for development; tighten for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SolveStepRequest(BaseModel):
    """Request body for /api/solve/step."""

    guesses: List[str] = Field(
        default_factory=list,
        description="List of 5-letter guesses made so far.",
    )
    feedback: List[str] = Field(
        default_factory=list,
        description=(
            "List of feedback strings (one per guess), using X/Y/G format "
            "(X=absent, Y=present, G=correct). Example: 'GGYXX'."
        ),
    )
    # Legacy field (deprecated): kept for backward compatibility.
    solver: Optional[Literal["info", "ml"]] = Field(
        default=None,
        description="(Deprecated) Solver to use: 'info' or 'ml'. Prefer 'strategy'.",
    )
    # New strategy field
    strategy: StrategyName = Field(
        default="info",
        description="Solver strategy: 'info', 'ml_classic', or 'ml_min_guess'.",
    )
    # Optional model paths for ML strategies
    classic_model_path: Optional[str] = Field(
        default=None,
        description="Path to classic ML model (.pkl) for 'ml_classic' strategy.",
    )
    min_guess_model_path: Optional[str] = Field(
        default=None,
        description="Path to min-guess ML model (.pkl) for 'ml_min_guess' strategy.",
    )


class SolveStepResponse(BaseModel):
    """Response body for /api/solve/step."""

    strategy: StrategyName
    next_guess: str
    remaining_candidates: int
    remaining_words: List[str] = Field(
        default_factory=list,
        description="Subset of remaining candidate words (capped for performance).",
    )


@app.post("/api/solve/step", response_model=SolveStepResponse)
def solve_step(payload: SolveStepRequest) -> SolveStepResponse:
    """Given guesses and feedback, return the next suggested guess."""

    if len(payload.guesses) != len(payload.feedback):
        raise HTTPException(
            status_code=400,
            detail="Length of 'guesses' and 'feedback' lists must match.",
        )

    word_lists = WordLists()

    # Build game state from incoming guesses + feedback
    game_state = GameState(remaining_words=word_lists.all_valid_words.copy())

    for guess, fb_str in zip(payload.guesses, payload.feedback):
        guess = guess.upper().strip()
        if len(guess) != 5 or not guess.isalpha():
            raise HTTPException(
                status_code=400,
                detail=f"Invalid guess '{guess}'. Guesses must be 5-letter alphabetic strings.",
            )

        try:
            fb_types = parse_feedback_string(fb_str)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        feedback_obj = GameFeedback(guess=guess, feedback=fb_types)
        game_state.add_guess(guess, feedback_obj)

    # Update remaining candidates based on all feedback
    game_state.update_remaining_words(word_lists.all_valid_words)

    # Determine strategy (map legacy solver field if needed)
    strategy: StrategyName
    if payload.solver is not None and payload.solver == "ml" and payload.strategy == "info":
        # Legacy \"solver=ml\" usage; default to ml_classic.
        strategy = "ml_classic"
    else:
        strategy = payload.strategy

    # Build solver / strategy wrapper
    if strategy == "info":
        solver_obj = InformationTheorySolver(word_lists)
    else:
        # Use provided paths or fall back to defaults
        classic_model_path = (
            Path(payload.classic_model_path)
            if payload.classic_model_path
            else DEFAULT_CLASSIC_MODEL if DEFAULT_CLASSIC_MODEL.exists() else None
        )
        min_guess_model_path = (
            Path(payload.min_guess_model_path)
            if payload.min_guess_model_path
            else DEFAULT_MIN_GUESS_MODEL if DEFAULT_MIN_GUESS_MODEL.exists() else None
        )
        try:
            solver_obj = SolverStrategy(
                word_lists=word_lists,
                classic_model_path=classic_model_path,
                min_guess_model_path=min_guess_model_path,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Ask solver for next guess
    try:
        if isinstance(solver_obj, SolverStrategy):
            next_guess = solver_obj.make_guess(strategy, game_state)
        else:
            next_guess = solver_obj.make_guess(game_state)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Solver could not find a valid guess: {exc}",
        ) from exc

    # Use current remaining words from game_state; cap list for performance
    remaining_set = game_state.remaining_words or set(word_lists.all_valid_words)
    remaining_list = sorted(list(remaining_set))
    remaining_preview = remaining_list[:50]

    return SolveStepResponse(
        strategy=strategy,
        next_guess=next_guess,
        remaining_candidates=len(remaining_set),
        remaining_words=remaining_preview,
    )


SESSION_LOG_PATH = PROJECT_ROOT / "data" / "daily_sessions.jsonl"


@app.post("/api/log/session")
def log_session(session: GameSession) -> dict:
    """Log an anonymized game session for later training.

    The session is appended as a JSON line to data/daily_sessions.jsonl.
    """

    # Basic validation: guesses/feedback lengths must match
    if len(session.guesses) != len(session.feedback):
        raise HTTPException(
            status_code=400,
            detail="Length of 'guesses' and 'feedback' lists must match.",
        )

    # Ensure all guesses look like 5-letter words and feedback strings are valid
    for guess in session.guesses:
        if len(guess) != 5 or not guess.isalpha():
            raise HTTPException(
                status_code=400,
                detail=f"Invalid guess '{guess}'. Guesses must be 5-letter alphabetic strings.",
            )
    for fb_str in session.feedback:
        if len(fb_str) != 5 or any(ch not in ("X", "Y", "G") for ch in fb_str):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid feedback '{fb_str}'. Feedback must be 5 characters of X/Y/G.",
            )

    # Append to JSONL file
    try:
        SESSION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        import json

        with SESSION_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(session.model_dump_json() + "\n")
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to log session: {exc}",
        ) from exc

    return {"status": "ok"}


@app.post("/api/solve/reset")
def reset() -> dict:
    """Stateless reset endpoint for clients (no server-side session)."""

    return {"status": "ok", "message": "Solver is stateless; start a new game client-side."}


# Serve the web UI (if present) under /web
WEB_PUBLIC_DIR = (
    Path(__file__)
    .resolve()
    .parent.parent.parent  # project root
    / "web"
    / "public"
)

if WEB_PUBLIC_DIR.exists():
    app.mount(
        "/web",
        StaticFiles(directory=str(WEB_PUBLIC_DIR), html=True),
        name="web",
    )


@app.get("/")
def root() -> dict:
    """Simple health/info endpoint."""

    return {
        "status": "ok",
        "message": "WordleAI API",
        "endpoints": [
            "/api/solve/step",
            "/api/solve/reset",
            "/docs",
            "/redoc",
            "/web (if built)",
        ],
    }


def get_app() -> FastAPI:
    """Convenience accessor for ASGI servers."""

    return app


if __name__ == "__main__":
    # Local development entrypoint:
    # python -m src.api.server
    import uvicorn

    uvicorn.run(
        "src.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


