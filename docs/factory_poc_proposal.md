# Proof-of-Concept Proposal — Cement Kiln Copilot

**Plant:** {{PLANT_NAME}}  
**Line / asset:** {{KILN_LINE}}  
**Proposal date:** {{DATE}}  
**Version:** 1.0

---

## 1. Background

{{PLANT_NAME}} operates a cement pyroprocessing line where kiln setpoints materially affect energy consumption, clinker quality, and process stability. {{VENDOR_OR_TEAM_NAME}} proposes a time-limited proof-of-concept (PoC) using **Cement Kiln Copilot** — an advisory decision-support system that predicts KPI behavior from process signals and ranks candidate setpoint adjustments for operator review.

The platform is demo-ready: data ingestion, model training, REST API, model-based recommendation engine, and operator dashboard. This PoC adapts the system to {{PLANT_NAME}}'s tags, KPIs, and operating envelope.

---

## 2. Current problem

| Area | Description |
| ---- | ----------- |
| Operational challenge | {{e.g. Burning zone temperature variability drives fuel over-use and quality excursions}} |
| Visibility gap | Operators lack forward-looking guidance on how setpoint changes will affect the target KPI |
| Data available | Historian time-series exists; interpretation into actionable recommendations is manual |
| Constraints | Plant requires **advisory-only** deployment—no closed-loop control or DCS writes |

---

## 3. Proposed pilot

A **{{4|6|8}}-week** advisory pilot on **{{KILN_LINE}}** to:

1. Train a plant-specific model on historian exports for agreed tags and one primary KPI.
2. Validate predictions and recommendations offline with the process engineering team.
3. Run advisory recommendations during selected shifts in the final phase (operator review only).
4. Document outcomes against pre-agreed success metrics.

---

## 4. Scope

### In scope

| Item | Detail |
| ---- | ------ |
| Asset | {{KILN_LINE}} — kiln pyroprocessing zone |
| Primary KPI | {{TARGET_KPI — e.g. KILN_ZONE1_TEMP, specific heat proxy, free lime model}} |
| Adjustable setpoints | {{LIST — e.g. fuel flow, ID fan, feed rate}} (max {{N}} tags) |
| Users | Process engineer(s), kiln operator(s), production supervisor |
| Deployment | {{On-premise VM / plant DMZ / isolated pilot server}} |
| Support | Weekly 45-min review; email support during business hours |

### Out of scope (see Section 11)

Closed-loop control, DCS setpoint writes, mill optimization (unless added by change order), multi-plant rollout.

---

## 5. Required data

| Data type | Requirement |
| --------- | ----------- |
| Historian export | {{WEEKS}} weeks minimum, resampled to {{INTERVAL — e.g. 1 min}} |
| Tags | {{TAG_LIST}} — final list confirmed in Week 1 |
| Setpoints | Current setpoint values or proxies for adjustable tags |
| Lab/QC (optional) | {{LAB_TAGS}} if KPI is quality-related |
| Metadata | Tag descriptions, units, valid ranges |

**Transfer method:** {{SFTP / secure share / on-premise copy}} per plant OT policy.  
**Format:** CSV or plant-standard export; long-format time-series preferred.

---

## 6. Deployment approach

| Phase | Activity |
| ----- | -------- |
| Week 0 | NDA, security questionnaire, tag list sign-off |
| Week 1–2 | Data ingest, cleaning, feature mapping, initial model training |
| Week 3–{{MID}} | Offline validation; tune bounds, lags, and adjustable tags with process team |
| Week {{MID+1}}–{{END}} | Advisory dashboard access; shift-level review of recommendations |
| Final week | PoC report, go/no-go recommendation |

**Architecture:** Read-only data in → trained model + API + Streamlit dashboard → recommendations displayed to operators. **No connection to DCS outputs.**

---

## 7. Deliverables

1. Plant-configured KPI prediction model (artifact + training config)
2. REST API endpoints: health, KPI prediction, setpoint recommendation
3. Operator-facing advisory dashboard
4. Weekly summary notes ({{N}} sessions)
5. **Final PoC report** including:
   - Model performance on hold-out data
   - Sample recommendation sessions vs. actual outcomes (where comparable)
   - Process engineering qualitative assessment
   - Recommendation for next phase (expand, refine, or stop)

---

## 8. KPIs for success

Success metrics are agreed before Week 1. Examples (select and customize):

| Metric | Target (illustrative) |
| ------ | --------------------- |
| Prediction usefulness | Process engineer rates forecast direction as correct in ≥ {{X}}% of reviewed windows |
| Recommendation relevance | ≥ {{N}} recommendation sessions where top candidate is judged "actionable" by process team |
| Safety compliance | Zero DCS writes; all bounds respected in candidate generation |
| Adoption | Operators/engineers use dashboard ≥ {{N}} times per week in final phase |
| Data pipeline | Stable ingest from historian export or read-only feed |

*No guaranteed energy savings percentage is part of this PoC. Measured operational improvement, if any, is reported honestly in the final document.*

---

## 9. Pilot timeline

| Week | Milestone |
| ---- | --------- |
| 0 | Kickoff, data access, security clearance |
| 1 | Tag mapping, data quality review, baseline statistics |
| 2 | Initial model training; offline prediction review |
| 3 | Recommendation engine configured; bounds validated |
| 4 | Process team offline review of ranked candidates |
| 5–6 | Advisory runs (optional live/near-real-time feed) |
| 7–8 | Consolidate results, final report, go/no-go meeting |

**Total duration:** {{4|6|8}} weeks from data handover.  
**Pilot start (target):** {{START_DATE}}

---

## 10. Assumptions

- {{PLANT_NAME}} provides timely historian exports and tag documentation.
- One dedicated process engineering contact attends weekly reviews.
- Adjustable setpoints and absolute bounds are agreed in writing before advisory runs.
- Pilot environment meets minimum compute: {{SPECS — e.g. 4 vCPU, 16 GB RAM, Linux VM}}.
- Plant accepts advisory-only operation; operators retain full control authority.
- Delays in data access or OT approval extend the calendar timeline accordingly.

---

## 11. Exclusions

- Autonomous or closed-loop control of any plant equipment
- Writing setpoints or commands to DCS, PLC, or SCADA
- Replacement of existing APC, expert systems, or safety interlocks
- Instrumentation upgrades, new sensors, or DCS modifications
- Guaranteed ROI, production increase, or emission reduction figures
- 24/7 on-site support (unless added by separate agreement)
- Multi-line or corporate rollout (post-PoC scope)

---

## 12. Advisory-only disclaimer

**Cement Kiln Copilot is an open-loop decision-support tool.** It generates KPI forecasts and ranked setpoint **recommendations** for human review. It does not actuate equipment, override operator judgment, or interface with control outputs. {{PLANT_NAME}} operators and process engineers are solely responsible for all control actions. This PoC does not constitute a safety instrumented system (SIS) or advanced process control (APC) deployment.

---

## 13. Commercial terms (optional — edit as needed)

| Item | Value |
| ---- | ----- |
| PoC fee | {{FEE_OR_NO_COST_PILOT}} |
| Expenses | {{TRAVEL_POLICY}} |
| IP / data | Plant data remains {{PLANT_NAME}} property; model artifacts {{USAGE_TERMS}} |
| Confidentiality | Mutual NDA {{REFERENCE}} |

---

## 14. Next step

1. Review and comment on this proposal by **{{RESPONSE_DATE}}**.
2. Schedule kickoff with process engineering and OT/security contact.
3. Execute NDA and transfer Week 0 data package.

**Prepared by:** {{SENDER_NAME}}, {{TITLE}}  
**Contact:** {{CONTACT_EMAIL}} · {{CONTACT_PHONE}}

**Plant sign-off:**

| Role | Name | Signature | Date |
| ---- | ---- | --------- | ---- |
| Process engineering | | | |
| Production / plant management | | | |
| IT/OT (data & security) | | | |
