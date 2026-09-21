---
phase: 15-accuracy-verification-pass
reviewed: 2026-09-21T00:00:00Z
depth: standard
files_reviewed: 7
files_reviewed_list:
  - scripts/verify_skill_examples.py
  - plugins/polars-statistics/skills/ROUTING.md
  - plugins/polars-statistics/skills/polars-statistics-anova/SKILL.md
  - plugins/polars-statistics/skills/polars-statistics-getting-started/SKILL.md
  - plugins/polars-statistics/skills/polars-statistics-hypothesis-tests/SKILL.md
  - plugins/polars-statistics/skills/polars-statistics-regression/SKILL.md
  - plugins/polars-statistics/skills/polars-statistics-results/SKILL.md
findings:
  critical: 1
  warning: 2
  info: 2
  total: 5
status: resolved
dispositions:
  - id: CR-01
    disposition: fixed
    rationale: classify_block now catches any malformed skip directive (`^#\s*verify:\s*skip\b` without a proper `-- <reason>`) and reports empty_reason instead of silently promoting to runnable. Harness re-run green (49/0/43).
  - id: WR-01
    disposition: fixed
    rationale: ROUTING.md ICC note corrected — only the anova description names ICC explicitly; hypothesis-tests covers it under "correlation tests".
  - id: WR-02
    disposition: fixed
    rationale: getting-started version-check comment now notes the current build may report 0.7.0 (0.8.0 once released), consistent with the Gotchas section.
  - id: IN-01
    disposition: fixed
    rationale: SUMMARY per-skill block counts corrected (anova 13/9/4, getting-started 11/9/2); totals were already correct.
  - id: IN-02
    disposition: accepted
    rationale: TOST skip is legitimate (block is non-runnable); the reason names type-annotation syntax. Left as-is — the skip correctly excludes a non-executable block; reason completeness is cosmetic.
---

# Phase 15: Code Review Report

**Reviewed:** 2026-09-21
**Depth:** standard
**Files Reviewed:** 7
**Status:** issues_found

## Summary

The harness runs correctly and exits green (49 PASS / 0 FAIL / 43 SKIP, exit 0). Every
runnable block genuinely executes against the live package. All 43 SKIP blocks are
legitimately non-runnable: API signature references with type-annotation syntax
(`x: list`, `successes: int`), illustrative pseudo-syntax with indented method chaining
(`.fit(x, y) -> Model`), and two blocks that require `sklearn` which is not installed.
The three documented drift fixes are present and correct. No real worked example has been
dodged behind a SKIP marker.

One correctness gap in `classify_block` can silently promote a mistyped skip marker to
runnable status. Two documentation inaccuracies were found — one in ROUTING.md, one as
an inline code comment — neither affecting execution. Two INFO items cover a stale
per-skill block-count table in SUMMARY.md and a partially inaccurate skip reason.

---

## Critical Issues

### CR-01: `classify_block` silently treats `# verify: skip --` (no trailing space) as runnable

**File:** `scripts/verify_skill_examples.py:137`

**Issue:** `SKIP_MARKER` is defined as `"# verify: skip -- "` (with a trailing space after
`--`). A block whose first line is `# verify: skip --` (two hyphens, no trailing space and
no reason text) does **not** start with `SKIP_MARKER`, and the bare-marker regex on line 145
(`^# verify:\s*skip\s*$`) also fails to match because of the ` --` suffix. The block
falls through to `return "runnable", ""` and executes silently — with no FAIL reported.

This means a developer who writes `# verify: skip --` thinking they have an empty-reason
marker will instead get the block executed against the live package. If that block fails,
it surfaces as a genuine FAIL, which is recoverable. If it happens to pass (e.g., the block
imports something and does nothing visible), it will silently be counted as PASS — defeating
the audit trail.

The gap is not triggered in any current file (confirmed by grep), but it is one character
away from a class of future mistakes (omitting the trailing space or forgetting the reason
text after `--`).

**Fix:** Add a third branch before the bare-marker check that catches the no-trailing-space
form and returns `empty_reason`:

```python
SKIP_MARKER        = "# verify: skip -- "   # with trailing space
SKIP_MARKER_NOTAIL = "# verify: skip --"    # without trailing space (two-hyphen, no space)

def classify_block(block: dict) -> tuple[str, str]:
    first = block["first_line"]

    if first.startswith(SKIP_MARKER):
        reason = first[len(SKIP_MARKER):].strip()
        if reason:
            return "skip", reason
        else:
            return "empty_reason", "(empty reason — required non-empty)"

    # Catch '# verify: skip --' without trailing space / reason
    if first == SKIP_MARKER_NOTAIL or first.startswith(SKIP_MARKER_NOTAIL + " "):
        # The startswith branch above already handles '# verify: skip -- reason',
        # so this only fires on bare '# verify: skip --' with nothing after.
        return "empty_reason", "(no reason text after '--' — use '# verify: skip -- <reason>')"

    if re.match(r'^# verify:\s*skip\s*$', first):
        return "empty_reason", "(no reason text — use '# verify: skip -- <reason>')"

    return "runnable", ""
```

---

## Warnings

### WR-01: ROUTING.md falsely claims both skill descriptions mention ICC

**File:** `plugins/polars-statistics/skills/ROUTING.md:88`

**Issue:** The routing note states "Both descriptions mention ICC" to explain why ICC prompts
can route to either `polars-statistics-anova` or `polars-statistics-hypothesis-tests`.
The anova description does mention "intraclass correlation coefficient (ICC)" explicitly.
However, the hypothesis-tests description reads:

> "…normality tests, correlation tests, chi-square, Fisher's exact, TOST equivalence tests,
> MMD, or energy distance…"

It says "correlation tests" — not "ICC". An LLM routing purely on the description field
would not match "intraclass correlation coefficient" to this skill. The ROUTING.md note
is therefore factually wrong: only one description, not both, mentions ICC.

The consequence is that a user asking about ICC in a hypothesis-testing framing (rather
than an ANOVA framing) may not be routed to the hypothesis-tests skill even though that
skill fully covers `ps.icc()` and `ps.icc_type` in its Group D section.

**Fix:** Update the routing note to accurately reflect which description mentions ICC:

```markdown
**Known ambiguity — ICC:** A user asking about inter-rater reliability / ICC may phrase
the question as a hypothesis-test ("is the agreement significant?") or as an ANOVA
variance-decomposition question. Only `polars-statistics-anova` mentions ICC explicitly
in its description ("intraclass correlation coefficient (ICC)").
`polars-statistics-hypothesis-tests` covers `ps.icc()` in Group D but does not name
it in the description field, so routing from that description alone will miss it.
If routing fails to reach the hypothesis-tests skill for ICC questions, consider adding
"ICC" or "intraclass correlation" to the hypothesis-tests `description` field.
```

---

### WR-02: Version comment in getting-started example contradicts installed package

**File:** `plugins/polars-statistics/skills/polars-statistics-getting-started/SKILL.md:63`

**Issue:** The version-check code block reads:

```python
print(ps.__version__)  # "0.8.0"
```

The installed package reports `0.7.0`. A user following this example will see `0.7.0`
printed and believe their installation is wrong or the docs are stale. The Gotchas section
at the bottom of the same file correctly acknowledges this discrepancy, but the inline
comment in the example contradicts it and is the first thing a user reads.

This also means an assertion on the version string (if ever added) would fail against the
live package — making it a latent harness-correctness risk if the check is ever tightened.

**Fix:** Change the inline comment to the accurate value, or use a range comment:

```python
print(ps.__version__)  # "0.7.0" (installed); skill targets 0.8.0 — see Gotchas
```

---

## Info

### IN-01: SUMMARY.md per-skill block-count table is stale

**File:** `.planning/phases/15-accuracy-verification-pass/15-01-SUMMARY.md` (metrics table)

**Issue:** The SUMMARY.md table reports:

| Skill | Blocks |
|-------|--------|
| anova | 16 |
| getting-started | 12 |

But the harness measures:

| Skill | Blocks |
|-------|--------|
| anova | 13 |
| getting-started | 11 |

Total claimed in table: implies ~96 blocks; actual: 92. The final-line summary (49 PASS /
0 FAIL / 43 SKIP = 92) is correct. Only the per-skill breakdown rows are stale. This
suggests the table was written during an intermediate edit pass before final block counts
settled. No harness correctness is affected.

**Fix:** Update the per-skill table to match actual counts before archiving.

---

### IN-02: TOST skip reason partially misattributes non-runnability

**File:** `plugins/polars-statistics/skills/polars-statistics-hypothesis-tests/SKILL.md:300`

**Issue:** The skip marker reads:

```
# verify: skip -- function signature reference (not runnable Python; successes: int, n: int use type annotation syntax)
```

Of the ten TOST functions in that block, only `tost_prop_one` and `tost_prop_two` use
`successes: int` / `n: int` type-annotation syntax. The other eight (`tost_t_test_*`,
`tost_wilcoxon_*`, `tost_bootstrap`, `tost_yuen`) use plain unbound names `x` and `y`
— which would cause a `NameError` because no DataFrame context is set up, not a syntax
error. The skip is still legitimate (the block cannot run), but the stated reason describes
only part of why. A future reader auditing skip legitimacy might be confused by the
mismatch between reason and what they observe in the code.

**Fix:** Broaden the reason to cover both failure modes:

```
# verify: skip -- function signature reference (unbound column-name placeholders x/y; two prop variants use type annotation syntax successes: int, n: int)
```

---

_Reviewed: 2026-09-21_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
