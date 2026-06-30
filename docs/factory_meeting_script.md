# Factory Meeting Script — Cement Kiln Copilot

**Duration:** ~10 minutes (demo + pilot ask)  
**Audience:** Production manager, process engineer, plant manager, energy manager  
**Prerequisites:** Demo environment running (`./scripts/run_demo.sh` or backend + dashboard separately)

---

## English version

### 1. Opening (1 min)

> "Thank you for your time. I'm {{SENDER_NAME}}. Today I'll show **Cement Kiln Copilot** — a decision-support tool for kiln operations.  
>  
> Before we start: this is **advisory only**. It suggests setpoint changes for your team to review. It does **not** connect to your DCS and does **not** control the kiln automatically.  
>  
> The goal of this meeting is to see whether a short pilot on your data would be useful—not to sell closed-loop automation."

*Pause for acknowledgment.*

---

### 2. Problem framing (1.5 min)

> "On a dry-process kiln, operators constantly trade off clinker quality, fuel use, and stability. Experienced judgment matters, but it's hard to see **in advance** how a small change—for example in fuel flow or zone temperature setpoint—will affect the burning zone or specific energy ten or twenty minutes later.  
>  
> Most plants already log the signals in a historian. The gap is turning that data into **forward-looking guidance** without a multi-year APC project.  
>  
> Cement Kiln Copilot fills that gap: predict a KPI, then rank setpoint candidates that move you toward a target you choose."

---

### 3. Demo walkthrough (4 min)

**Step A — System health (30 sec)**

- Open the Streamlit dashboard.
- Click *Check backend status* in the sidebar.
- Point out: `Health: OK` confirms the API and model are loaded.

> "This is the operator-facing view. In production it would sit beside your existing tools—read-only from the historian, no writes to control."

**Step B — Load process context (30 sec)**

- Click *Load demo scenario (all tags)*.
- Mention tags: `KILN_ZONE1_TEMP`, `KILN_ZONE2_TEMP`, `KILN_FUEL_FLOW` (mapped to your plant's tag names in a real deployment).

> "Here we use a recent signal window—the same structure we'd pull from your historian export or live feed."

**Step C — KPI prediction (1 min)**

- Click *Run KPI Prediction*.
- Highlight the forecast chart and mean / min / max summary.

> "The model forecasts where the chosen KPI is heading over the configured horizon. This is the same model used for recommendations—not a separate heuristic."

**Step D — Setpoint recommendations (2 min)**

- Set `desired_target` slightly above or below the current operating point.
- Click *Run Recommendation*.
- Show the ranked table and score bar chart.

> "The engine perturbs adjustable setpoints within safe bounds, re-runs the model for each candidate, and ranks them. The top row is the best match to your desired target."

---

### 4. What the recommendation score means (1 min)

> "The **score** is the model's predicted distance to your desired target—mean absolute error across the forecast window. **Lower is better.** It is a ranking metric, not a guarantee of plant outcome.  
>  
> You also see **predicted mean, min, and max** for each candidate so you can judge stability, not just the average.  
>  
> Operators and process engineers decide whether a recommendation is sensible given current feed, equipment, and safety limits. The system never acts on its own."

---

### 5. Pilot proposal (1.5 min)

> "A typical pilot runs **4–8 weeks** on one kiln line or zone:  
>  
> - Week 1–2: agree KPIs, adjustable setpoints, and historian export; train a plant-specific model  
> - Week 3–5: offline validation with your process team; tune bounds and tags  
> - Week 6–8: advisory runs during shifts; weekly review of recommendations vs. actual outcomes  
>  
> Deliverables: trained model, API access, dashboard, and a short report with success metrics we define together—no fixed ROI promise upfront.  
>  
> We can deploy on-premise or in an isolated VM per your OT policy."

*Offer to share the PoC proposal template (`docs/factory_poc_proposal.md`).*

---

### 6. Handling: "Is this controlling the kiln automatically?" (30 sec)

> "**No.** Cement Kiln Copilot is open-loop advisory only. It does not write setpoints to the DCS, does not bypass interlocks, and does not run without an operator in the loop.  
>  
> Think of it as a ranked second opinion based on your recent process data—similar to an advanced what-if tool, but model-scored at scale.  
>  
> If you ever moved toward tighter integration, that would be a separate phase with your automation team and explicit approval. This pilot does not require that."

---

### 7. Closing ask (30 sec)

> "If this direction is interesting, I'd suggest three next steps:  
>  
> 1. A 45-minute discovery session with your process engineer on KPIs and data access  
> 2. A short historian sample—anonymized is fine—to confirm tag coverage  
> 3. A signed one-page PoC scope with timeline and success criteria  
>  
> What would be the right person on your side for step one—and is a pilot something you'd consider this quarter or next?"

*Capture: decision-maker, data owner, OT/security contact, preferred timeline.*

---

## نسخه فارسی (جلسه محلی — خلاصه)

**مدت:** حدود ۱۰ دقیقه

### ۱. شروع (۱ دقیقه)

> «سلام و وقت بخیر. من {{SENDER_NAME}} هستم. امروز **Cement Kiln Copilot** را نشان می‌دهم — یک ابزار **پشتیبان تصمیم** برای عملیات کوره سیمان.  
>  
> نکته مهم: این سیستم **فقط مشاوره‌ای** است. پیشنهاد تغییر ست‌پوینت می‌دهد؛ اپراتور تصمیم نهایی را می‌گیرد. **هیچ فرمانی به DCS ارسال نمی‌شود** و کوره به‌صورت خودکار کنترل نمی‌شود.  
>  
> هدف جلسه این است ببینیم یک پایلوت کوتاه روی داده‌های کارخانه شما مفید است یا نه.»

### ۲. بیان مسئله (۱ دقیقه)

> «در کوره خشک، اپراتور همیشه بین کیفیت کلینکر، مصرف انرژی و پایداری فرآیند تعادل برقرار می‌کند. پیش‌بینی اینکه یک تغییر کوچک در سوخت یا دما، ده دقیقه بعد چه اثری دارد، سخت است.  
>  
> داده‌ها معمولاً در historian موجود است؛ کمبود، **راهنمای پیش‌رو** است — بدون پروژه چندساله APC.»

### ۳. دمو (۴ دقیقه)

1. وضعیت سلامت بک‌اند را نشان دهید (*Check backend status*).
2. سناریوی دمو را بارگذاری کنید (*Load demo scenario*).
3. *Run KPI Prediction* — نمودار و میانگین/حداقل/حداکثر پیش‌بینی.
4. *desired_target* را تغییر دهید و *Run Recommendation* — جدول رتبه‌بندی و نمودار امتیاز.

> «موتور توصیه، چند کاندیدای ست‌پوینت را با مدل واقعی امتیازدهی می‌کند. ردیف اول بهترین تطابق با هدف شماست.»

### ۴. معنی امتیاز (Score)

> «**امتیاز** فاصله پیش‌بینی‌شده تا هدف است — **کمتر بهتر است**. تضمین نتیجه در کارخانه نیست؛ فقط برای رتبه‌بندی کاندیداهاست. تصمیم نهایی با تیم فرآیند و اپراتور است.»

### ۵. پیشنهاد پایلوت

> «پایلوت معمولاً **۴ تا ۸ هفته**: تعریف KPI و داده، آموزش مدل، اعتبارسنجی با مهندس فرآیند، و در هفته‌های پایانی اجرای مشاوره‌ای در شیفت. گزارش نهایی با معیارهای موفقیت توافق‌شده.»

### ۶. پاسخ: «آیا خودکار کوره را کنترل می‌کند؟»

> «**خیر.** کاملاً بازحلقه و مشاوره‌ای. هیچ نوشتنی روی DCS ندارد. مثل یک نظر دوم رتبه‌بندی‌شده بر اساس داده‌های اخیر — نه کنترل خودکار.»

### ۷. درخواست پایانی

> «اگر مایل باشید: جلسه کشف ۴۵ دقیقه‌ای با مهندس فرآیند، نمونه خروجی historian، و پیش‌نویس PoC با زمان‌بندی.  
>  
> چه کسی در کارخانه برای گام بعد مناسب است؟»

---

## Meeting checklist

- [ ] Demo environment tested before the call
- [ ] One-pager or PoC template ready to send after the meeting
- [ ] Discovery questions doc open for note-taking
- [ ] Confirm advisory-only positioning in the first 60 seconds
- [ ] Capture: KPI priorities, data access path, OT constraints, decision-maker
