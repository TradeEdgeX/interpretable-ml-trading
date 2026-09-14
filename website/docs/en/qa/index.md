# Q & A

**One-liner:** The Lab Q&A, said again in plain language. Local `mlbot lab` keeps the same questions with more jargon; this page is for people first.

Three sentences are enough:

1. **Rules can still cheat**, just differently: they pick a pretty threshold and a pretty window.
2. **Report bear, bull, and recent separately.** One blended curve washes out the bad years.
3. **A correlation scan cannot close the case.** Close on the three-window five-KPI table. You write the conclusion.

## Can a rule still cheat? {#can-a-rule-cheat}

### Do rule experiments need cross-validation?

**No.** That is not laziness. The gate is different.

Cross-validation (cut time into folds, train on the left, score the right) is for **models that can memorize a path**: trees, linear models, scorers with thousands of fitted knobs. This public extract has no training command.

A rule experiment does not fit-then-score. The hypothesis is frozen. Bear, bull, and recent each run once. The question is **whether the sentence still works when market structure changes**, not whether coefficients memorized the training set.

### Few knobs, cannot memorize a path — so rules cannot overfit?

**They still can. The shape is different.**

A rule cannot remember how one day in 2022 traded. It only locks one or two thresholds. The right threshold looks like a coarse structure; a coincidental one dies when the regime changes.

Rule cheating is: sweep thresholds, swap conditions, keep the window that looks good. The boring gates: a correlation scan cannot close; report three windows separately (a fail in one is a fail); turn the kill switch off when comparing, so drawdown cannot hide.

### What does a model fight vs a rule? When is cross-validation required?

| | Model | Rule |
|---|---|---|
| Capacity | Large; can memorize residuals | Small; cannot memorize a path |
| How it cheats | Looks great on the train fold, dies next door | Lucky threshold, dies when the regime changes |
| Defense | Cross-validation: do not memorize that fold | Three windows + a story a human can tell |
| Correlation scan | A clue for picking features | Same: a clue, not evidence |

Use cross-validation in the **train inner loop** only when you actually fit a model. The court is still the three-window backtest. Do not stack two rulers.

## Why three windows? {#three-windows}

### Is one blended bull-and-bear curve more stable?

**Do not blend.** A blended curve hides the bad regime and looks like “fine in the long run.” When markets are non-stationary, a longer average just washes a dead rule clean.

The three windows are economic states (down, up, recent chop), not exam folds on a calendar. Even a time split is mostly bear then bull. That is weaker than **explicitly** requiring the same sign on all three and no worse drawdown.

What *is* better: variant × window on one card, so a flip is visible.

### What are the three windows? Why does the Lab scorecard spread out?

Default crypto windows:

- **Bear**: the 2022 dump
- **Bull**: the 2023–2024 rise
- **Recent**: the latest wide-range / turn-to-bear stretch — not “average again”

Headline only five numbers: CAGR, Calmar, win rate, max drawdown, Sharpe. Do not lead with “total R.”

A single row is usually one artifact (often baseline / bear). Variants and the recent window must be listed, or a chop failure hides under a bear number. That is why Lab `/rd` is a table, not a headline.

## Why a pretty correlation is not a score {#correlation-is-not-a-score}

### If a factor was found by correlation, can it still predict?

**Correlation only says it moved together in the past. It does not promise the next stretch.**

It is not “the model remembered a simple law.” It only says: on **this table and this answer key**, the number moved with later returns.

If it later works, that coarse relation **still holds next door** — the story did not change, and fees do not eat it. It may also be: a hit after hundreds of features; an answer key that is not the production objective; a relation that only exists in one regime.

The three-window court asks whether it still holds next door. Pass, and it looks like structure. Fail, and it was sample correlation.

### So did it “remember” a simple law?

**Sometimes you are seeing the shadow of that law. It did not remember it.**

The “simple law” is the **hypothesis** you write down (fade after crowding, revert after a washout, do not open trend in chop). The scan is only in-sample association. They look the same.

- Few thresholds, a tellable story, same sign on three windows, drawdown not worse → more like structure
- Correlation only / one window / a blended curve → treat it as neither memory nor discovery

That is why step one can never close the case. The process exists so you do not take the shadow as proof.

### Why can the first step (the correlation scan) never close?

**It measures a different thing.**

Step one’s answer key is often “what the next few bars do.” The court uses realized CAGR, Calmar, win rate, max drawdown, and Sharpe, plus slots, slippage, and regime filters.

Features that look good on correlation often fail the full backtest: too many scans, invisible fees, a regime that already moved. A longer sample sometimes just averages in more dead rules.

Only the bear / bull / recent run with the kill switch off can go to close. You still write the conclusion after you read the report.

### Do the first two steps tune and grade on the same history? Is that too optimistic?

**Yes, if you treat those steps as the score.** They are not train-then-test.

| Step | What it does | Same history? |
|---|---|---|
| 1 | Scan correlation on the whole table | Yes — discovery |
| 2 | A human writes the threshold | Yes — lock the knobs |
| 3 | Bear / bull / recent backtest, kill switch off | Mostly the same calendar, different objective |

There is no classic “hold out an unseen exam.” Rules also skip cross-validation on purpose. The cheat channel is sweeping thresholds and keeping the pretty window.

Gates: step one cannot close; three windows separately; kill switch off. Step three changes the ruler (five KPIs) and the market structure. It is not future data. A true extra sample looks like a second stock universe, not “the last calendar stretch is the only exam.”

## Should we save an “unseen future”? {#unseen-future}

### Can the last half-year be the unseen exam?

**Do not replace the court with “last stretch = exam.”**

A gap at a cut only blocks the answer key leaking across that cut. It does not block picking a lucky threshold.

- The last stretch is usually one structure (today: chop / turn-to-bear). Using it as the only exam is “pretty recent window, pass.”
- If step one never sees that structure, it will pick trend thresholds.
- Re-searching thresholds every window is the forbidden rolling optimize. Replaying a frozen rule is monitoring, not retuning.

If you want a stricter tail: freeze the knobs, then replay once on a later stretch you do not touch. Fail the tail → no. Pass the tail → still not enough alone. The three-window court stays.

### If a gap blocks leakage, why also cut time into small blocks?

**Two tools. The gap blocks leaked answers. The small blocks ask “does it still work next door?” more than once.**

**The gap:** a model learns on the left using “return over the next few bars” as the answer. Cut on one day with no gap, and the last training answers already contain the start of the test path. The score is inflated. Leave a gap as long as the answer.

**The small blocks:** a model can look great on a long fit. One last-half-year test is one number, often one regime. The train loop asks several times: fit on earlier data, score the next stretch. Blocks exist to repeat the question, not to break correlation. Correlation is still the gap at each cut.

Rules do not fit-then-score the same answer, so the first three steps need neither. Re-searching thresholds on monthly blocks is the forbidden rolling optimize. Rules want large frozen blocks: the same gates on bear / bull / recent.

### If the label is “return over the next few bars,” do we need cross-validation?

**The trigger is fitting, not the name of the label.**

You need the trio only when three things happen together: you fit on the left, you score the same answer key on the right, and you cut time. Not because someone said “tree” or “forward return.”

This extract has no training command. Rules go: scan → freeze the threshold → event backtest. No such cut.

If you later fit a tree / linear / scorer on a look-ahead answer, then:

- **Gap:** answers overlap across the cut; leave at least as many bars as the label looks ahead
- **Drop crossing rows:** if a row’s answer already used the exam path, do not learn on it
- **Cross-validation:** because you are fitting, ask “next stretch still works?” more than once — not because the horizon is 10 bars vs 3. The horizon only widens the gap

A rule’s first-step correlation on the same “next few bars” does not make this trio the court. Correlation cannot close. The court is the realized three-window path.

### What does “the answer already stepped into the exam” mean?

It means that row’s **answer key** already used prices from the exam stretch.

Say the answer is “return over the next 10 bars.” The row on day 1 uses days 1 to 11. Practice is days 1–100; the exam starts on day 101. Day 95 is still on the practice calendar, but its answer uses days 95–105 — and 101–105 are already the exam. The practice target contains the exam path.

**Stepped across:** that row’s answer window overlaps the exam.  
**Drop it:** do not learn on those rows. Keep only rows whose whole answer ends before the cut.

A gap leaves unused days between the two sides. Dropping rows deletes specific overlapping lines inside an already-drawn fold. One empties a stretch of days; the other deletes specific rows.

## Short horizon, bursts, HFT {#short-horizon}

### Does this extract ship a live sleeve? Should we still hunt a short-horizon tree?

**No live sleeve.** This repo is a hypothesis validator. The public practice sentence is a moving-average cross, not a shop of production books.

Short-horizon trees and 15-minute order flow often measure “moving at the same time,” not “leading by a few bars.” After fees, little is left. Do not treat “hunt something shorter” as another tree scan.

If you still want short, change to a trade clock (follow trades, not bars) and beat fees first. Measured sentences live in the [gallery](../gallery/index.md).

### After a huge burst, does price fade or continue? Can we assume one side?

**Count both paths, and split them.** Do not assume “it must fade” or “it must continue.”

A large burst stacks two machines: temporary impact / inventory (fade) and information / liquidation cascade (continue). One average cancels them and correlation goes to about zero — it looks like “no signal.”

Ask it this way: split the two legs with a rule, see if the signs actually differ; then subtract taker round-trip fees and see if anything tradeable remains. A slight average fade smaller than fees does not make a sentence.

### I have a trading stack and rebates. Is HFT just engineering?

**No.** Regular take is about 5bp; the arithmetic is right. HFT does not forecast two minutes. The identity is spread + rebate − getting run over − inventory.

A retail limit often does not even cover the maker fee — BTC perp spread is often one tick. Negative maker is a high-volume market-making contract, not a “make it a limit” flag. An order stack, a feature store, and trade prints are not queue position, colocation, or a venue rebate contract.

Without those three, HFT is paying retail fees and canceling slowly into the adverse tick. Closer engineering: mark out fills you already have, or measure hold-style spreads (funding, basis). Do not rescan the feature store to imitate HFT.

## Fine print

- Local Lab: `mlbot lab` → `/rd/qa`
- Same questions in YAML: [docs/agent/rd_qa.yaml](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/agent/rd_qa.yaml)
- [Three windows](../quant/three-windows.md) · [Five KPIs](../quant/five-kpis.md) · [Who writes what](../tech/who-writes-what.md)
