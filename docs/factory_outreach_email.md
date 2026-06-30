# Factory Outreach Emails — Cement Kiln Copilot

Three templates for different outreach contexts. Replace `{{...}}` tokens before sending.

---

## 1. Short cold outreach

**Subject:** Decision support for kiln setpoints — advisory only, no DCS writes

**Body:**

Dear {{RECIPIENT_NAME}},

I am reaching out because {{PLANT_NAME}} operates a dry-process kiln where small setpoint decisions materially affect energy use and clinker quality.

We have built **Cement Kiln Copilot** — an advisory system that uses recent process signals to predict KPI behavior and suggest ranked setpoint candidates. It is **open-loop only**: operators review recommendations; nothing is sent to the DCS automatically.

We are looking for one or two plants to run a focused 4–8 week pilot. The goal is to validate whether model-scored recommendations are useful in your operating context, not to sell a black-box controller.

Would a 20-minute call next week make sense to see if this is worth a deeper look?

Best regards,  
{{SENDER_NAME}}  
{{SENDER_TITLE}}  
{{CONTACT_EMAIL}}

**CTA:** Reply with two times for a 20-minute intro call.

---

## 2. Warmer outreach (after intro or referral)

**Subject:** Following up — Cement Kiln Copilot pilot for {{PLANT_NAME}}

**Body:**

Dear {{RECIPIENT_NAME}},

Thank you for the introduction via {{REFERRER_NAME}}. As mentioned, we work on **Cement Kiln Copilot** — decision support for kiln and mill operations.

In short: the system reads historian time-series, forecasts a chosen KPI (e.g. zone temperature or a specific-energy proxy), and ranks candidate setpoint adjustments. Each recommendation is scored by a trained model. The operator always decides whether to act; the system does not close the loop.

{{REFERRER_NAME}} thought this might be relevant given {{SPECIFIC_CONTEXT — e.g. recent fuel-cost pressure / kiln variability / quality excursions}}.

I can walk your process team through a live demo (~10 minutes) and share a draft PoC scope tailored to your line. No commitment beyond understanding fit.

Are you available for a brief call on {{PROPOSED_DATE_RANGE}}?

Best regards,  
{{SENDER_NAME}}  
{{CONTACT_EMAIL}}

**CTA:** Confirm a 30-minute slot for demo + scoping discussion.

---

## 3. Accelerator-friendly version

**Subject:** Cement Kiln Copilot — advisory AI for industrial decarbonization (pilot-ready)

**Body:**

Dear {{RECIPIENT_NAME}},

**Cement Kiln Copilot** is an open-loop AI advisory platform for cement pyroprocessing. It ingests plant time-series, predicts KPI trajectories, and recommends setpoint candidates—ranked by model score—for operator review.

**Problem:** Kiln operations are energy-intensive and non-linear. Operators balance quality, throughput, and fuel consumption with limited forward visibility. Full APC deployments are slow and costly; advisory tools that respect OT boundaries are under-served.

**Approach:** End-to-end MVP — data ingestion, ML training, REST API, recommendation engine, Streamlit operator dashboard. **Advisory only**; no autonomous control or DCS writes.

**Traction / readiness:** Demo-ready on synthetic and seeded plant-style data. Seeking a cement plant or industrial partner for a 4–8 week pilot to validate on real historian exports and agreed KPIs.

**Ask:** Introduction to a plant production manager, process engineer, or energy manager open to a low-risk pilot; or feedback from your technical review panel on pilot design.

Happy to share a one-pager, PoC template, and a live demo at your convenience.

Best regards,  
{{SENDER_NAME}}  
{{ORGANIZATION}}  
{{CONTACT_EMAIL}} · {{WEBSITE_OR_REPO_LINK}}

**CTA:** 20-minute review call or warm intro to a pilot partner.
