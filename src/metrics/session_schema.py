"""Game session schema and utilities for logging daily Wordle games.

This module defines a simple Pydantic model used by the API server to
validate and persist anonymized game sessions played via the web UI.
"""

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


StrategyName = Literal["info", "ml_classic", "ml_min_guess"]


class GameSession(BaseModel):
    """An anonymized game session played by a user.

    This is intentionally minimal: no user identity, just the sequence
    of guesses and feedback plus a strategy label.
    """

    session_id: str = Field(
        ...,
        description="Client-generated UUID or random string identifying the session.",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when the session was logged.",
    )
    strategy: StrategyName = Field(
        ...,
        description="Solver strategy used during the game (info, ml_classic, ml_min_guess).",
    )
    guesses: List[str] = Field(
        default_factory=list,
        description="Ordered list of 5-letter guesses played in this session.",
    )
    feedback: List[str] = Field(
        default_factory=list,
        description=(
            "Ordered list of feedback strings (X/Y/G) matching guesses. "
            "Each string must be length 5, with X=absent, Y=present, G=correct."
        ),
    )
    solved: bool = Field(
        ...,
        description="True if the puzzle was solved within the attempt limit.",
    )
    guesses_used: int = Field(
        ...,
        ge=0,
        le=10,
        description="Number of guesses used in this session (1-6 typical, may be >6 for experiments).",
    )
    target_word: Optional[str] = Field(
        default=None,
        description=(
            "Optional actual answer word (if known). "
            "For daily Wordle, this is typically not stored to avoid spoilers."
        ),
    )



