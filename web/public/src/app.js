// WordleAI frontend
// NYT-style board + on-screen keyboard + physical keyboard input.

const API_BASE = "/api";

// Iframe handling
const wordleIframe = document.getElementById("wordle-iframe");
const iframeFallback = document.getElementById("iframe-fallback");

if (wordleIframe && iframeFallback) {
  let iframeLoadTimeout;
  let iframeLoaded = false;

  // Set timeout to detect if iframe doesn't load
  iframeLoadTimeout = setTimeout(() => {
    if (!iframeLoaded) {
      // Iframe likely blocked by X-Frame-Options
      wordleIframe.style.display = "none";
      iframeFallback.style.display = "block";
    }
  }, 5000); // 5 second timeout

  // Handle successful iframe load
  wordleIframe.onload = () => {
    iframeLoaded = true;
    clearTimeout(iframeLoadTimeout);
    iframeFallback.style.display = "none";
  };

  // Handle iframe errors (though onerror may not fire for X-Frame-Options)
  wordleIframe.onerror = () => {
    iframeLoaded = false;
    clearTimeout(iframeLoadTimeout);
    wordleIframe.style.display = "none";
    iframeFallback.style.display = "block";
  };

  // Additional check: try to access iframe content (will fail if blocked)
  try {
    // This will throw if cross-origin restrictions apply
    const testAccess = wordleIframe.contentWindow;
    // If we get here, we can access it (unlikely for NYT)
  } catch (e) {
    // Cross-origin error is expected - iframe may still load visually
    // The timeout will handle actual blocking
  }
}

const ROWS = 6;
const COLS = 5;

const KEYBOARD_ROWS = [
  ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
  ["A", "S", "D", "F", "G", "H", "J", "K", "L"],
  ["ENTER", "Z", "X", "C", "V", "B", "N", "M", "DEL"],
];

const boardEl = document.getElementById("board");
const keyboardEl = document.getElementById("keyboard");
const suggestionWordEl = document.getElementById("suggested-word");
const suggestionBtn = document.getElementById("suggestion-btn");
const resetBtn = document.getElementById("reset-btn");
const strategySelect = document.getElementById("strategy-select");
const statusEl = document.getElementById("status");
const remainingEl = document.getElementById("remaining");

// Internal board state
let letters = Array.from({ length: ROWS }, () => Array(COLS).fill(""));
let states = Array.from({ length: ROWS }, () => Array(COLS).fill("X")); // X/Y/G

// Auto-suggestion debounce timer
let autoSuggestionTimer = null;
const AUTO_SUGGESTION_DELAY = 500; // 500ms delay after tile changes

function setStatus(msg, isError = false) {
  statusEl.textContent = msg || "";
  statusEl.style.color = isError ? "#f87171" : "#aaaaaa";
}

function setRemainingInfo(count, preview) {
  if (!count) {
    remainingEl.textContent = "";
    return;
  }
  let text = `Remaining candidates: ${count}`;
  if (preview && preview.length) {
    text += `\n${preview.slice(0, 10).join(", ")}`;
  }
  remainingEl.textContent = text;
}

function isSolved() {
  // Puzzle is considered solved if there is at least one fully filled row
  // where all feedback states are green.
  for (let r = 0; r < ROWS; r++) {
    if (
      letters[r].every((ch) => ch) &&
      states[r].every((s) => s === "G")
    ) {
      return true;
    }
  }
  return false;
}

function updateSuggestionVisibility() {
  if (isSolved()) {
    suggestionBtn.style.display = "none";
    setStatus("Solved! Click “New puzzle” to start another.");
  } else {
    suggestionBtn.style.display = "";
  }
}

function renderTile(row, col) {
  const tile = boardEl.querySelector(
    `.board-tile[data-row="${row}"][data-col="${col}"]`
  );
  if (!tile) return;

  const ch = letters[row][col] || "";
  const state = states[row][col] || "X";

  tile.textContent = ch;
  tile.classList.toggle("filled", !!ch);

  // Reset styles
  tile.style.backgroundColor = "transparent";
  tile.style.borderColor = getComputedStyle(
    document.documentElement
  ).getPropertyValue("--tile-border");

  if (!ch) {
    return;
  }

  if (state === "G") {
    tile.style.backgroundColor = "#538d4e";
    tile.style.borderColor = "#538d4e";
  } else if (state === "Y") {
    tile.style.backgroundColor = "#b59f3b";
    tile.style.borderColor = "#b59f3b";
  } else if (state === "X") {
    tile.style.backgroundColor = "#3a3a3c";
    tile.style.borderColor = "#3a3a3c";
  }
}

function cycleTileState(row, col) {
  // Only cycle if a letter is present; matches NYT behavior.
  if (!letters[row][col]) return;
  const current = states[row][col] || "X";
  const next = current === "X" ? "Y" : current === "Y" ? "G" : "X";
  states[row][col] = next;
  renderTile(row, col);
  updateSuggestionVisibility();
  // Trigger auto-suggestion after feedback change (debounced)
  triggerAutoSuggestion();
}

function initBoard() {
  boardEl.innerHTML = "";
  for (let row = 0; row < ROWS; row++) {
    for (let col = 0; col < COLS; col++) {
      const tile = document.createElement("div");
      tile.className = "board-tile";
      tile.dataset.row = String(row);
      tile.dataset.col = String(col);
      tile.addEventListener("click", () => cycleTileState(row, col));
      boardEl.appendChild(tile);
    }
  }
}

function getActivePosition() {
  // First row that is not full; cursor is at first empty cell there.
  for (let r = 0; r < ROWS; r++) {
    const rowLetters = letters[r];
    const hasAny = rowLetters.some((ch) => ch);
    const isFull = rowLetters.every((ch) => ch);
    if (!hasAny || !isFull) {
      let c = 0;
      while (c < COLS && rowLetters[c]) c += 1;
      return { row: r, col: c };
    }
  }
  return { row: ROWS - 1, col: COLS };
}

function handleLetterInput(letter) {
  letter = letter.toUpperCase();
  if (!/^[A-Z]$/.test(letter)) return;

  const { row, col } = getActivePosition();
  if (row >= ROWS || col >= COLS) return;

  letters[row][col] = letter;
  states[row][col] = "X";
  renderTile(row, col);
  updateSuggestionVisibility();
  
  // Note: We don't auto-suggest when row is complete because user needs to set feedback first
  // Auto-suggestion will trigger when feedback is changed via cycleTileState
}

function handleBackspace() {
  const { row } = getActivePosition();
  if (row < 0) return;

  // Walk backward from current row to clear the last filled tile.
  for (let r = row; r >= 0; r--) {
    for (let c = COLS - 1; c >= 0; c--) {
      if (letters[r][c]) {
        letters[r][c] = "";
        states[r][c] = "X";
        renderTile(r, c);
        updateSuggestionVisibility();
        return;
      }
    }
  }
}

function handleKey(key) {
  if (key === "ENTER") {
    requestSuggestion();
  } else if (key === "DEL") {
    handleBackspace();
  } else {
    handleLetterInput(key);
  }
}

function initKeyboard() {
  keyboardEl.innerHTML = "";
  KEYBOARD_ROWS.forEach((row) => {
    const rowEl = document.createElement("div");
    rowEl.className = "keyboard-row";
    row.forEach((key) => {
      const btn = document.createElement("button");
      btn.className = "key";
      if (key === "ENTER" || key === "DEL") {
        btn.classList.add("wide");
      }
      btn.textContent = key === "DEL" ? "⌫" : key;
      btn.addEventListener("click", () => handleKey(key));
      rowEl.appendChild(btn);
    });
    keyboardEl.appendChild(rowEl);
  });
}

function getHistoryFromBoard() {
  const guesses = [];
  const feedbackStrings = [];

  for (let r = 0; r < ROWS; r++) {
    const rowLetters = letters[r];
    if (rowLetters.every((ch) => ch)) {
      guesses.push(rowLetters.join(""));
      feedbackStrings.push(states[r].join(""));
    } else {
      // Assume contiguous rows from the top; stop at first incomplete
      break;
    }
  }

  return { guesses, feedbackStrings };
}

function triggerAutoSuggestion() {
  // Clear any existing timer
  if (autoSuggestionTimer) {
    clearTimeout(autoSuggestionTimer);
  }
  
  // Set new timer to trigger suggestion after delay
  autoSuggestionTimer = setTimeout(() => {
    const { guesses, feedbackStrings } = getHistoryFromBoard();
    
    // Only auto-suggest if we have at least one complete guess with feedback
    if (guesses.length > 0 && feedbackStrings.length > 0) {
      // Check if the last row has all feedback set (not all X)
      const lastRowIndex = guesses.length - 1;
      const lastFeedback = feedbackStrings[lastRowIndex];
      // Only suggest if feedback has been set (at least one non-X)
      if (lastFeedback && !lastFeedback.split("").every((s) => s === "X")) {
        requestSuggestion();
      }
    } else if (guesses.length > 0) {
      // If we have guesses but no feedback yet, don't suggest
      // User needs to set feedback colors first
      return;
    }
  }, AUTO_SUGGESTION_DELAY);
}

async function requestSuggestion() {
  // Clear any pending auto-suggestion timer
  if (autoSuggestionTimer) {
    clearTimeout(autoSuggestionTimer);
    autoSuggestionTimer = null;
  }
  
  // Determine strategy to send to the API
  const strategy = strategySelect ? strategySelect.value : "ml_min_guess";
  // If puzzle already solved, avoid extra calls.
  if (isSolved()) {
    setStatus("Puzzle already solved. Click “New puzzle” to start again.");
    updateSuggestionVisibility();
    return;
  }

  const { guesses, feedbackStrings } = getHistoryFromBoard();

  setStatus("Contacting solver…");

  try {
    const resp = await fetch(`${API_BASE}/solve/step`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        guesses,
        feedback: feedbackStrings,
        strategy,
        // The API expects model paths to be configured server-side; we rely on defaults.
      }),
    });

    if (!resp.ok) {
      const errJson = await resp.json().catch(() => ({}));
      const detail = errJson.detail || resp.statusText;
      throw new Error(
        typeof detail === "string" ? detail : JSON.stringify(detail)
      );
    }

    const data = await resp.json();
    const nextGuess = data.next_guess;

    if (!nextGuess) {
      setStatus(
        "Solver did not return a guess. Check your board letters and colors.",
        true
      );
      return;
    }

    suggestionWordEl.textContent = nextGuess;
    setStatus(`Suggested next guess: ${nextGuess}`);
    setRemainingInfo(data.remaining_candidates, data.remaining_words || []);
    updateSuggestionVisibility();

    // If this suggestion leads to a solved row (all G), we can log the session.
    // The actual logging request is triggered from the UI when the user completes
    // the puzzle; see logSessionIfSolved().
  } catch (err) {
    console.error(err);
    setStatus(`Error contacting solver: ${err.message || err}`, true);
  }
}

function resetAll() {
  letters = Array.from({ length: ROWS }, () => Array(COLS).fill(""));
  states = Array.from({ length: ROWS }, () => Array(COLS).fill("X"));
  initBoard();
  suggestionWordEl.textContent = "Press ENTER or “Get suggestion”";
  setStatus(
    "New puzzle started. Recreate your NYT board, then request a suggestion."
  );
  setRemainingInfo(0, []);
  updateSuggestionVisibility();
}

async function logSessionIfSolved() {
  if (!isSolved()) {
    return;
  }

  const { guesses, feedbackStrings } = getHistoryFromBoard();
  if (!guesses.length || guesses.length !== feedbackStrings.length) {
    return;
  }

  try {
    const sessionId =
      window.crypto && window.crypto.randomUUID
        ? window.crypto.randomUUID()
        : `${Date.now()}-${Math.random().toString(16).slice(2)}`;

    const payload = {
      session_id: sessionId,
      strategy: strategySelect ? strategySelect.value : "ml_min_guess",
      guesses,
      feedback: feedbackStrings,
      solved: true,
      guesses_used: guesses.length,
    };

    await fetch(`${API_BASE}/log/session`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch (err) {
    // Logging failures should not affect the user experience.
    console.warn("Failed to log session", err);
  }
}

// --- Event wiring ---

suggestionBtn.addEventListener("click", () => {
  requestSuggestion();
});

resetBtn.addEventListener("click", () => {
  resetAll();
});

// Physical keyboard support
window.addEventListener("keydown", (event) => {
  const key = event.key;

  // Let browser/system shortcuts work (Ctrl/Meta/Alt combinations).
  if (event.ctrlKey || event.metaKey || event.altKey) {
    return;
  }

  if (/^[a-zA-Z]$/.test(key)) {
    event.preventDefault();
    handleLetterInput(key.toUpperCase());
    return;
  }

  if (key === "Enter") {
    event.preventDefault();
    requestSuggestion();
    // If the puzzle becomes solved after applying feedback, log the session.
    setTimeout(() => {
      logSessionIfSolved();
    }, 0);
    return;
  }

  if (key === "Backspace" || key === "Delete") {
    event.preventDefault();
    handleBackspace();
  }
});

// Initialize on load
initBoard();
initKeyboard();
setStatus(
  "Use your keyboard or click the on-screen keys. Click tiles to set colors X/Y/G."
);

 