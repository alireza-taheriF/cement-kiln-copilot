# Factory Discovery Questions — Cement Kiln Copilot

Structured checklist for first meetings with production, process, energy, and IT/OT stakeholders. Use as a conversation guide—not a questionnaire to read verbatim.

**Meeting:** {{DATE}} · **Plant:** {{PLANT_NAME}} · **Attendees:** {{NAMES/ROLES}}

---

## 1. Process and kiln operations

| # | Question | Notes |
| - | -------- | ----- |
| 1.1 | Which line(s) or kiln(s) are in scope? Capacity, type (dry/wet), preheater/calciner configuration? | |
| 1.2 | What is the typical operating mode—steady-state, frequent setpoint changes, or recovery from upsets? | |
| 1.3 | Which zones or subsystems cause the most operator attention (burning zone, calciner, cooler, preheater)? | |
| 1.4 | What are the main disturbance sources (feed quality, alternative fuels, seasonal ambient, equipment degradation)? | |
| 1.5 | Are there known constraints we must respect (minimum O₂, max CO, refractory limits, emission caps)? | |

---

## 2. Current KPIs and targets

| # | Question | Notes |
| - | -------- | ----- |
| 2.1 | What KPIs does the shift team track daily—clinker quality (free lime, C3S, LSF), specific heat consumption, throughput? | |
| 2.2 | Which KPI would deliver the most value if predicted 10–30 minutes ahead? | |
| 2.3 | What are typical target bands and alarm limits for that KPI? | |
| 2.4 | How often is lab/QC data available, and how is it aligned to process time? | |
| 2.5 | Are there existing dashboards or reports the team already trusts? | |

---

## 3. Energy pain points

| # | Question | Notes |
| - | -------- | ----- |
| 3.1 | What is the primary energy concern—fuel cost, electrical (mill/fan), or both? | |
| 3.2 | Do you measure specific heat consumption (kcal/kg clinker or GJ/t) at line level? How is it calculated? | |
| 3.3 | Have recent efficiency projects plateaued? What was tried (APC, burner tuning, alternative fuels)? | |
| 3.4 | Is there a corporate or national decarbonization target affecting this plant? | |
| 3.5 | Would a 1–2% improvement in a specific metric be meaningful at current production volumes? (Qualitative—no ROI claim.) | |

---

## 4. Variability and instability

| # | Question | Notes |
| - | -------- | ----- |
| 4.1 | How often do kiln upsets, rings, or temperature excursions occur? | |
| 4.2 | Is shift-to-shift performance consistent, or does it depend heavily on individual operators? | |
| 4.3 | Which setpoint changes are most debated between shifts? | |
| 4.4 | After an upset, what signals do operators watch first to recover? | |
| 4.5 | Would ranked "what-if" setpoint guidance help reduce trial-and-error during recovery? | |

---

## 5. Operator workflow

| # | Question | Notes |
| - | -------- | ----- |
| 5.1 | Who adjusts setpoints today—kiln operator, process engineer, both? | |
| 5.2 | How are setpoint changes documented (logbook, DCS event journal, informal)? | |
| 5.3 | Would operators accept recommendations on a separate screen, or must it integrate into an existing HMI? | |
| 5.4 | What would make a recommendation **trustworthy** vs. **ignored**? (Explainability, bounds, engineer sign-off?) | |
| 5.5 | Is there union or site policy on decision-support tools vs. automated control? | |

---

## 6. Historian, PLC, SCADA, and data access

| # | Question | Notes |
| - | -------- | ----- |
| 6.1 | What historian or data platform is in use (PI, Wonderware, OPC-UA aggregator, other)? | |
| 6.2 | Typical sampling interval for kiln tags (1 s, 1 min, 5 min)? | |
| 6.3 | Can you export 4–12 weeks of time-series for selected tags (CSV, API, ODBC)? | |
| 6.4 | Who owns tag naming and metadata—automation, process engineering, IT? | |
| 6.5 | Are lab results in the same system or a separate LIMS/spreadsheet? | |
| 6.6 | For a live pilot: read-only OPC/historian feed vs. batch export only? | |
| 6.7 | Data residency, air-gap, or export approval requirements? | |

**Candidate tags to confirm (map to plant names):**

- Kiln zone temperatures (e.g. burning zone, secondary air)
- Fuel flow / burner setting
- Feed rate, kiln speed, ID fan
- O₂, CO, NOx (if available)
- Mill power / fineness (if mill scope added later)

---

## 7. Pilot constraints

| # | Question | Notes |
| - | -------- | ----- |
| 7.1 | Preferred pilot duration—4, 6, or 8 weeks? Blackout periods (shutdown, audit, peak season)? | |
| 7.2 | One kiln line only, or multi-line comparison? | |
| 7.3 | On-premise VM, plant DMZ, or vendor-hosted isolated environment? | |
| 7.4 | Cybersecurity review required before any data leaves OT? Typical lead time? | |
| 7.5 | NDA and data anonymization requirements? | |
| 7.6 | Who must sign off before operators see recommendations during a shift? | |
| 7.7 | Success criteria the plant would accept as "worth continuing" vs. "stop here"? | |

---

## 8. Approval, budget, and decision-makers

| # | Question | Notes |
| - | -------- | ----- |
| 8.1 | Who sponsors operational pilots—plant manager, production director, corporate digital/energy team? | |
| 8.2 | Who is the process engineering owner for kiln optimization? | |
| 8.3 | Does IT/OT security need to be in the first meeting or only before data transfer? | |
| 8.4 | Is there a budget band for pilot-scale industrial software (internal charge code, innovation fund)? | |
| 8.5 | What is the typical procurement path after a successful PoC—PO, framework agreement, corporate vendor onboarding? | |
| 8.6 | Are there incumbent APC vendors or internal projects that overlap? How do we position as complementary? | |
| 8.7 | Target decision date for go/no-go on a pilot? | |

---

## Post-meeting summary (fill in before sending follow-up)

**Primary KPI for pilot:**  
**Adjustable setpoints (draft):**  
**Data path:**  
**OT/security contact:**  
**Decision-maker:**  
**Proposed pilot window:**  
**Open risks / blockers:**  
**Agreed next step:**
