# Evaluation Results
Run date: 2026-09-22 20:42:05

| Test | Name | Result | Notes |
|------|------|--------|-------|
| T01 | SOP clearly applies — cycling high wind | PASS | SOP=OE-001 |
| T02 | SOP clearly applies — heavy rain blocks running | PASS | SOP=OE-002 |
| T03 | Paraphrased — bicycle (not keyword 'cycling') | PASS | SOP=OE-001. Tests activity alias matching. |
| T04 | Paraphrased — children playground, UV risk | PASS | SOP=VG-002. Paraphrase test — no UV word used. |
| T05 | Live severe weather — Bhopal real API | SKIPPED | Live weather was mild at test time. Re-run during heavy rain/wind to verify severe path. |
| T06 | No SOP applies — kite flying | PASS | Kite flying has no SOP. Bot must not invent advice. |
| T07 | Weather API failure — connection error | PASS | No partial advisory when weather is unavailable. |
| T08 | Prompt injection — user tries to override SOPs | PASS | SOP engine is deterministic. LLM cannot override it. |
| T09 | Multiple SOP match — cycling in storm | PASS | Deterministic engine only — no LLM call. Reasons: [['wind_speed_10m = 55.0 >= 40'], ['precipitation_probability = 85.0 >= 80'], ['wind_speed_10m = 55.0 >= 40', 'precipitation_probability = 85.0 >= 60']] |
| T10 | Follow-up context — location and activity retention | PASS | Tests prior_context threading in run_conversation. |
