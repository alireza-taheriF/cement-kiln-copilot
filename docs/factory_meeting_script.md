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

**Step B — Kiln advisor (2 min)**

- Open the *Kiln advisor* tab, click *Load simulated plant data*, then *Get advice*.
- Point out: the proposed fuel per kg clinker, the ID-fan setting, and the daily saving in Gcal and litres of mazut.
- Open *Why?*: the forecast burning-zone temperature and CO with their 90 % range, the limiting constraint, and any lever held because the model disagreed with process physics.

> "The copilot only proposes a change if the pessimistic end of every forecast stays inside your own limits, and only in small steps. If nothing is safe, it says so."

- Record a rejection with a reason code to show the feedback loop.

**Step C — Mill schedule (1 min)**

- Open *Mill schedule*, keep the default quota (no power 19–23) and tariffs, click *Optimize schedule*.

> "Same cement in the silo at the end, lower power cost: grinding moves to allowed, cheap hours and free-market purchases drop. This is the fastest saving and needs no change in the kiln."

**Step D — Free lime, equipment, shift report (1 min)**

- *Free lime*: a lab-quality estimate every few minutes, with its range and when the lab result for that clinker is due.
- *Equipment*: the simulated mill-2 bearing and shell hot spot flagged days ahead.
- *Shift report*: the Persian handover report built from data.

---

### 4. How savings are proven (1 min)

> "Before the copilot runs we fit a baseline of your fuel and power use against production for several weeks. Afterwards we report every month what energy *would have been* at your actual production, minus what it was, with the statistical uncertainty (ASHRAE Guideline 14). Only significant savings count. That report is what the savings share is invoiced on."
>
> "Everything you saw is a simulated plant. The first real numbers come from your data in the audit."

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

1. وضعیت سلامت بک‌اند را نشان دهید.
2. زبانهٔ **دستیار کوره**: «بارگذاری دادهٔ کارخانهٔ شبیه‌سازی‌شده» و «دریافت توصیه»؛ مصرف حرارت ویژهٔ پیشنهادی، دور فن ID، صرفه‌جویی روزانه (گیگاکالری و لیتر مازوت) و بخش «چرا؟» با بازهٔ ۹۰٪ دمای زون پخت و CO.
3. یک بازخورد «رد کردم» با دلیل ثبت کنید تا چرخهٔ بازخورد دیده شود.
4. زبانهٔ **زمان‌بندی آسیاب**: «محاسبهٔ برنامهٔ بهینه»؛ صرفه‌جویی برق با همان موجودی سیلو در پایان.
5. **آهک آزاد**، **تجهیزات** (یاتاقان آسیاب ۲ و نقطهٔ داغ پوسته، چند روز زودتر) و **گزارش شیفت** فارسی.

> «سامانه فقط وقتی تغییر پیشنهاد می‌دهد که بدبینانه‌ترین سر پیش‌بینی‌ها هم داخل حدود خود شما باشد، و فقط با گام کوچک. اگر گزینهٔ ایمنی نباشد، صریح می‌گوید.»

### ۴. صرفه‌جویی چطور ثابت می‌شود

> «قبل از شروع، چند هفته مصرف سوخت و برق را در برابر تولید مدل می‌کنیم (خط مبنا). بعد هر ماه گزارش می‌دهیم مصرف با تولید واقعی *چقدر می‌شد* منهای *چقدر شد*، با عدم‌قطعیت آماری (ASHRAE Guideline 14). فقط صرفه‌جویی معنادار حساب می‌شود و فاکتور سهم از صرفه‌جویی روی همین گزارش است. هرچه دیدید دادهٔ شبیه‌سازی‌شده است؛ اولین عدد واقعی از دادهٔ شما در ممیزی می‌آید.»

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
