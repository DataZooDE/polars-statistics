# Phase 15: Accuracy Verification Pass - Context

**Gathered:** 2026-09-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Make the five authored SKILL.md cheat-sheets trustworthy for an agent: prove every
code example runs against the shipped API, confirm each skill's `description` routes
Claude to the right cheat-sheet, and correct any signature/kwarg/return-shape drift
found — before the phase closes. Scope is verification + drift correction of the
EXISTING skill content (authored in Phase 14); no new API coverage is added here.
Out of scope: wheel bundling (Phase 16), release (Phase 17).

</domain>

<decisions>
## Implementation Decisions

### Example Execution (criterion #1)
- Build a REUSABLE, COMMITTED harness — e.g. `scripts/verify_skill_examples.py` — that
  extracts every fenced ```python code block from all five SKILL.md files and executes
  each against the live installed package (`uv run`). It reports pass/fail per block
  with the failing traceback. Committed so future releases can re-run it as a regression
  guard. Criterion #1 requires EVERY example to run — the harness must be exhaustive, not
  sampled.
- Examples that are intentionally illustrative-only (e.g. pseudo-snippets, output
  displays) must be either made runnable or explicitly marked/skipped by the harness with
  a visible reason — no silent skips.

### Description Routing (criterion #2)
- Verify routing with a REPRESENTATIVE TASK-PROMPT MAPPING: define a set of representative
  task prompts (e.g. "run an independent t-test", "fit ridge regression", "unnest a result
  struct", "two-way ANOVA", "get started / which model do I use") and confirm each routes
  to the correct skill's `description` as the best match. If any mis-routes, refine that
  skill's `description`. Document the prompt→skill mapping in the phase artifacts.

### Drift Handling (criterion #3)
- Fix drift IN-PLACE in the SKILL.md files as found (wrong signature, kwarg, or
  return-shape), then re-run the harness until all examples pass green. The phase does not
  close until the harness is green and routing is confirmed.
- **LIVE RUNTIME IS AUTHORITATIVE, NOT DOCSTRINGS.** Source docstrings in this repo are
  known-stale for some output structs (e.g. correlation.py `distance_cor`/`partial_cor`/
  `semi_partial_cor` omit `ci_lower`/`ci_upper` in the docstring but return them at
  runtime). Always confirm field lists / kwargs against the live imported package, never
  the docstring. (Phase 14 already corrected the correlation family; the harness must
  catch any remaining cases like it.)

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- The five authored SKILL.md files under `plugins/polars-statistics/skills/*/SKILL.md`.
- The package is importable via `uv run python` (reports `__version__` 0.7.0; skills pin
  0.8.0 as a version string — expected, NOT drift).
- `scripts/` directory (create if absent) for the committed harness.
- 14-RESEARCH.md has the API inventory; 14-REVIEW.md / 14-REVIEW-FIX.md document the drift
  already found and fixed in Phase 14 (useful priors for what to double-check).

### Established Patterns
- Spot-checks in Phase 14 used `uv run python -c "..."`; the harness generalizes this to
  block extraction + execution.
- Output is struct-typed; verifying return shape = comparing the documented struct field
  list against `df.select(expr).schema[...].fields` at runtime.

### Integration Points
- Corrections land in the same SKILL.md files that Phase 16 bundles into the wheel.

</code_context>

<specifics>
## Specific Ideas

- Harness path suggestion: `scripts/verify_skill_examples.py`, runnable via `uv run python
  scripts/verify_skill_examples.py`, exit non-zero if any example fails.
- Representative routing prompts should cover at least one task per skill family.
- Known prior drift class: docstring-vs-runtime output-struct field mismatches — prioritize
  re-checking every documented `struct{...}` field list against live output.

</specifics>

<deferred>
## Deferred Ideas

- Wiring the harness into CI as a permanent gate — nice-to-have; can be a follow-up. This
  phase requires it to exist and pass, not necessarily to be CI-wired.
- Any new API surface coverage — out of scope (only verify/correct existing content).

</deferred>
