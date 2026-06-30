# Cement Kiln Copilot — Factory One-Pager

**AI decision support for kiln and mill operations**  
Advisory-only · Open-loop · Operator-in-the-loop

---

## The problem

Cement pyroprocessing is energy-intensive, non-linear, and sensitive to small control changes. Operators must continuously balance clinker quality, specific energy consumption, throughput, and emissions—often with limited visibility into how a setpoint change will affect the process minutes or hours later.

Traditional APC and expert systems are costly to deploy and maintain. Spreadsheet analysis and operator experience remain the default, which leaves efficiency gains on the table and increases the risk of kiln upsets when conditions shift.

## Why cement plants care

| Pressure | Operational impact |
| -------- | ------------------ |
| Energy cost | Fuel and electrical consumption directly affect margin |
| Quality variability | Free lime, C3S, fineness deviations drive rework and customer risk |
| Process stability | Upsets, rings, and temperature excursions reduce availability |
| Operator load | Experienced staff are scarce; shift-to-shift consistency is hard |
| Decarbonization targets | Efficiency improvements are among the fastest levers before capex |

Even modest improvements in specific heat consumption or reduced quality variability can justify a focused pilot—without requiring a full plant automation overhaul.

## What Cement Kiln Copilot does

Cement Kiln Copilot is a **decision-support system** that:

1. **Ingests** recent process signals and optional lab/QC data from the plant historian or exported time-series.
2. **Predicts** how a chosen KPI (e.g. zone temperature, specific energy proxy, quality indicator) is likely to behave over a short horizon.
3. **Recommends** ranked candidate setpoint adjustments that move the predicted KPI toward a desired target—each scored by the trained model, not by heuristics.

The system is delivered as a trained model, REST API, and operator-facing dashboard. It fits alongside existing DCS/SCADA workflows; it does not replace them.

## Inputs required

| Category | Examples |
| -------- | -------- |
| Time-series signals | Kiln zone temperatures, fuel flow, feed rate, fan speeds, O₂/CO, mill power |
| Setpoints | Current operator or DCS setpoints for adjustable tags |
| Lab/QC (optional) | Free lime, C3S, Blaine—when available and aligned to process time |
| Configuration | Target KPI, adjustable tags, safe operating bounds, resampling interval |

For a pilot, a **4–12 week historian export** at regular intervals (e.g. 1–5 minutes) is typically sufficient. Tag naming is mapped to the plant's conventions during onboarding.

## Outputs delivered

- **KPI forecast** — predicted mean, min, max, and trend over the configured horizon
- **Ranked setpoint candidates** — proposed adjustments to selected tags, with predicted KPI impact
- **Recommendation score** — model-predicted distance to the desired target (lower = closer; used for ranking only)
- **Audit trail** — API responses and dashboard views suitable for post-shift review

Operators and process engineers review recommendations and decide whether to act. The system never writes to the DCS.

## Safety model: advisory only

| Property | Cement Kiln Copilot |
| -------- | ------------------- |
| Control mode | **Open-loop / advisory** |
| DCS integration | Read-only data in; no setpoint writes |
| Actuation | None — recommendations only |
| Operator role | Final authority on every adjustment |
| Bounds | Configurable absolute limits on candidate setpoints |

This design reduces IT/OT security concerns, avoids regulatory and insurance questions associated with autonomous control, and respects the plant's existing safety interlocks and operator culture.

## Expected pilot scope

A typical **4–8 week proof-of-concept** on one line or kiln zone:

- One primary KPI (e.g. burning zone temperature, specific heat consumption proxy, or a quality tag)
- 3–8 adjustable setpoints agreed with process engineering
- Historical data for model training; live or near-real-time data for advisory runs during the final 1–2 weeks
- Weekly review with production and process engineering

Deployment options: on-premise VM, plant DMZ, or isolated pilot environment—aligned with the plant's OT policy.

## Expected value from a pilot

Outcomes are measured, not assumed. A successful pilot should demonstrate:

- **Actionable recommendations** that process engineers recognize as directionally sound
- **Improved visibility** into KPI sensitivity to setpoint changes
- **Documented baseline** for a specific operating envelope (before/after advisory use)
- **Clear go/no-go criteria** for a broader rollout

We do not promise fixed percentage energy savings in a one-pager. Value depends on feed quality, equipment condition, and current operating practice. The pilot is designed to quantify what is realistic for {{PLANT_NAME}}.

## Why now

- **Mature historian data** — most plants already log the signals needed; the gap is interpretation, not instrumentation.
- **Proven ML stack** — gradient-boosted models handle non-linear kiln dynamics without a multi-year APC project.
- **Low integration risk** — advisory-only deployment avoids closed-loop approval cycles.
- **Energy and carbon pressure** — efficiency gains are increasingly tied to commercial and reporting requirements.
- **Demo-ready platform** — training, API, recommendation engine, and operator dashboard are implemented end-to-end; pilots focus on plant-specific data and KPIs, not greenfield software.

---

**Next step:** 30-minute technical intro and live demo, followed by a scoped PoC proposal.  
Contact: {{CONTACT_NAME}} · {{CONTACT_EMAIL}} · {{CONTACT_PHONE}}
