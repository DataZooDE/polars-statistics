---
schema_version: 1
open_count: 3
waived_count: 0
fixed_count: 0
total_count: 3
last_updated: 2026-08-12T05:42:05.383Z
---

# Broken Windows Ledger

> Cross-phase defect register. With `workflow.windows_enforce` enabled, `/gsd-ship` blocks while `open_count > 0`.
> Waive with `gsd-tools windows waive <id> "<reason>"` (reason required).
> Mark fixed with `gsd-tools windows fixed <id>`.

| id | phase | kind | file | line | description | status | reason | recorded_at | resolved_at |
|----|-------|------|------|------|-------------|--------|--------|-------------|-------------|
| 1 | 1 | deviation |  |  | test-only: verifying windows ledger exists | open |  | 2026-08-11T19:16:07.917Z |  |
| 2 | 1 | deviation | tests/rust_api.rs |  | Design fix: plan-provided x2=i*50 (collinear with x1=i*0.5) replaced by x2=((i%7)+1)*100 (periodic, full-rank) | open |  | 2026-08-11T19:28:01.180Z |  |
| 3 | 03 | lint-warning | python/polars_statistics/exprs/correlation.py | 6 | Pre-existing ruff UP035/UP007/UP006/UP045 violations (Union/List/Optional type annotations); 21 violations, not in CI, pre-date plan 03-03 | open |  | 2026-08-12T05:42:05.383Z |  |

````json
[
  {
    "id": 1,
    "kind": "deviation",
    "phase": "1",
    "file": "",
    "line": null,
    "description": "test-only: verifying windows ledger exists",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-08-11T19:16:07.917Z",
    "resolved_at": null
  },
  {
    "id": 2,
    "kind": "deviation",
    "phase": "1",
    "file": "tests/rust_api.rs",
    "line": null,
    "description": "Design fix: plan-provided x2=i*50 (collinear with x1=i*0.5) replaced by x2=((i%7)+1)*100 (periodic, full-rank)",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-08-11T19:28:01.180Z",
    "resolved_at": null
  },
  {
    "id": 3,
    "kind": "lint-warning",
    "phase": "03",
    "file": "python/polars_statistics/exprs/correlation.py",
    "line": 6,
    "description": "Pre-existing ruff UP035/UP007/UP006/UP045 violations (Union/List/Optional type annotations); 21 violations, not in CI, pre-date plan 03-03",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-08-12T05:42:05.383Z",
    "resolved_at": null
  }
]
````
