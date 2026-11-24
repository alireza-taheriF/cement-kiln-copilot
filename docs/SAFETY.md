# Safety case

Cement Kiln Copilot is **decision support**. It reads plant data and shows
recommendations; a qualified operator decides and acts. This document states
the safeguards, what they assume, and what must be done at a plant before the
first recommendation is shown.

## 1. Boundaries

| The copilot does | The copilot does not |
| ---------------- | -------------------- |
| Read historian exports, lab sheets and OPC UA history | Write to the DCS, PLC or historian (no code path exists; `tests/test_connectors.py` fails if one is added) |
| Recommend lever values inside a signed-off envelope | Override interlocks, alarms or operator decisions |
| Explain each recommendation and its uncertainty | Guarantee outcomes; recommendations are forecasts |
| Log every recommendation and the operator's answer | Close the loop (see §5) |

## 2. Safeguards in the advisor (`src/advisory/advisor.py`)

1. **Signed-off envelope** (`config/safety_limits.yaml`): an absolute range and a
   maximum step per lever; process limits for every guarded variable. One file,
   one owner (the plant's process engineer).
2. **Pessimistic constraint test**: a candidate is feasible only if the
   unfavourable end of the 90 % prediction interval is inside the limit.
3. **Direction check**: before a lever is used, each constraint model's local
   response is compared with process knowledge (`expected_effects`). A model that
   contradicts physics, or shows no effect, cannot move that lever.
4. **Bounded gains**: the effect of a lever is computed with the model's gain
   clipped into step-test bounds; the interval uses the worst gain in the bounds.
5. **Rate limit**: no recommendation moves a lever more than its `max_step`.
6. **Keep-current bias**: changes below `min_improvement` are not proposed.
7. **Final gate**: the chosen recommendation is re-checked against the envelope;
   a failure is an internal error and nothing is shown (`UnsafeRecommendationError`).
8. **No safe candidate**: if nothing satisfies every limit, the copilot says so and
   asks to escalate to the shift lead; it never shows a "least bad" change.

## 3. Validation

- `tests/test_advisor_and_safety.py` applies recommendations in the simulator with
  identical random draws and checks the *true* burning-zone temperature and CO over
  the following three hours.
- `tests/test_ml_robust.py` checks out-of-sample interval coverage of the
  burning-zone model.
- Shadow mode (`src/control/shadow.py`) measures interval coverage and
  post-recommendation limit violations on the plant's own data.

## 4. Before the first recommendation at a plant

| Step | Owner | Evidence |
| ---- | ----- | -------- |
| Replace every limit in `safety_limits.yaml` with plant limits | Process engineer | Signed limit sheet |
| Step tests for each lever (fuel, ID fan) to set `gain_bounds` | Process engineer + copilot team | Step-test report |
| Read-only data account (OPC UA) or file export agreed with IT/OT | IT/OT lead | Access approval |
| Two weeks of shadow mode: recommendations logged, not shown | Copilot team | Shadow report (coverage, would-be violations) |
| Operator briefing: advisory only, how to reject with a reason | Shift leads | Attendance |
| Escalation path for "no safe candidate" and model errors | Plant manager | Written procedure |

## 5. Closed loop

Writing setpoints automatically is a different product with its own safety case:
HAZOP, management of change, DCS-side interlocks and limits, operator override,
and the APC/DCS vendor's integration. `GET /v1/control/readiness` reports the
evidence for proposing a *supervised* closed-loop trial (days covered, acceptance,
agreement, zero violations after followed recommendations, calibrated intervals).
Passing it is necessary, not sufficient. The only `SetpointWriter` in this code
base is `DryRunWriter`, which records and never connects.
