# Cursor Settings & Workflow

## Objective
Maximize shipping speed while keeping changes reviewable, testable, and easy to debug.

## Recommended Default Workflow
1) Keep `SPEC.md` current (source of truth).
2) Ask Cursor for a file tree before generating code.
3) Build a single vertical slice end-to-end.
4) Add tests and structured logging early.
5) Iterate with small diffs.

## Context Management (Most Important)
- Scope Cursor requests to a small set of files:
  - `SPEC.md`
  - the route file
  - one service file
  - one integration file
- Prefer: “Only modify files X, Y, Z” when requesting changes.
- If Cursor starts drifting:
  - restate constraints
  - provide the expected schema
  - require a minimal patch

## Project Instructions / Rules
- Enable project rules and point Cursor at `/.cursor/rules.md`.
- Keep rules short and strict:
  - no guessing
  - minimal diffs
  - tests required

## Chat / Edit Mode Usage
- Use Chat for:
  - architecture options
  - explaining unfamiliar code
  - debugging strategy
- Use Edit/Composer for:
  - implementing a specific task
  - generating a scaffold from a file tree
  - applying a precise refactor
Rule: do not use Composer for multi-feature “build it all” prompts.

## Prompt Patterns That Work Best
### 1) “Plan + Patch + Test”
Ask Cursor:
- Provide a brief plan (bullets)
- Then give a patch (diff)
- Then add tests
- Then list run commands

### 2) “Vertical Slice”
- UI → API → service → stub data → response
- Add a contract test validating schema

### 3) “Debugging”
- root cause hypothesis
- minimal fix
- regression test

### 4) “Hardening”
- P0/P1/P2 issues
- patch P0 only

## Guardrails Against Hallucination
- Require Cursor to cite code locations:
  - “Point to the file + function where this behavior is implemented.”
- Require TODOs for unknown integrations:
  - “Do not implement IRIS client; create an interface and a fake client.”

## Code Review Hygiene
- Ask for diffs, not full files:
  - “Output only unified diffs.”
- Enforce minimal edits:
  - “No changes outside these files.”
- Require tests:
  - “Add/adjust tests for any new logic.”

## Repo Hygiene That Makes Cursor Better
Create these files (even for small projects):
- `SPEC.md` — feature scope + non-goals + acceptance criteria
- `API_CONTRACT.md` — endpoint schemas + examples
- `.env.example` — required env vars
- `/examples/` — sample payloads and responses
- `/scripts/` — one-command run/build/test scripts

## Suggested Settings (Conceptual)
(Names differ by Cursor version; match to your UI.)

- Autocomplete: ON
- Inline suggestions: ON
- Multi-file edits: ON (but use with constraints)
- Apply edits as diffs (if available): ON
- “Use project context”: ON
- “Search codebase for relevant files”: ON
- “Allow running tests” (if available): ON, but prefer you execute commands

## Personal Operating Discipline
- Never accept a big change without running:
  - tests
  - type checks (if any)
  - lint (if any)
- When stuck, do not thrash:
  - paste the exact error
  - ask Cursor for root cause + minimal patch

## Daily Checklist
- [ ] SPEC updated
- [ ] small diff
- [ ] tests updated
- [ ] logs added at boundaries
- [ ] run commands documented in PR description