# R&D Q&A

How rule experiments fight overfitting, why TimeSeriesSplit / Qlib CV is not the court, and why an IC factor can look predictive. Tree-train embargo/purge is not the same as cutting time into folds. Close on three segments and five KPIs, not a CV average.

Lab: `/rd/qa` · source: `docs/agent/rd_qa.yaml`

Related:
- `docs/agent/rd_playbook.md`
- `config/experiments/LAYER_PROMOTION_CRITERIA.md`
- `docs/strategy/methodology_R_and_D_flow_CN.md`

## Do rule experiments need causal TimeSeriesSplit, or Qlib-style CV?

**No.** This is not "rules cannot overfit, so skip CV." The overfitting channel is different, and this repo already gates it another way.

TimeSeriesSplit asks whether a model with thousands of fitted parameters memorized one path. Qlib and `mlbot train` need that: fit on earlier folds, score later folds.

A rule experiment does not fit. The hypothesis is frozen. Each of the three segments is a separate run. The question is **whether the rule still works when market structure changes**, not whether coefficients remembered the training set.

## Rules have few knobs and cannot memorize the training path. Does that mean they cannot overfit?

**They still overfit, just differently.** A rule almost cannot interpolate a whole path (it cannot remember how one bar in May 2022 traded). It only locks 1–3 gates. The right gate looks like a coarse structure; a coincidental gate dies when the regime changes.

Rule overfitting is: sweeping thresholds, swapping conditions, keeping the window that looks good. The gates are: Phase 1 cannot close, three-segment Pareto (a fail in one segment is a fail), kill switch OFF.

Example: BPC entry v2 looked fine on the compression-leg label and on bear/bull, then recent/chop CAGR flipped → reject.

## Is pooling bull and bear into one run better, and closer to TimeSeriesSplit?

**Do not pool.** A pooled curve hides the bad regime and looks like "fine in the long run." That is the documented trap: under non-stationarity, a longer average just washes a drifted rule clean.

The three segments are economic states (bear / bull / chop), not calendar K-folds. Even a TimeSeriesSplit on 2022–2026 would mostly be bear then bull. That is weaker than **explicitly** requiring the same sign on all three segments and no worse MaxDD.

What *is* better is showing variant × segment on one card, not adding another CV cut.

## If IC only measures a historical panel, why can a factor still look predictive in a model or a rule?

IC is **not** "the model remembered a low-dimensional law." It only says: on **this panel and this label**, the number moved with future returns.

Real prediction (if any) means that low-dimensional relation **still holds on the next stretch** — the economic story did not change, and execution costs do not eat it. Not because the factor memorized the training set.

It may also be:
- a multiple-testing hit after hundreds of features × thresholds
- a label (`forward_rr` / success flag) that is not the production objective (realized path, slots, slippage, MaxDD)
- a relation that only exists in one market structure

The three-segment court asks whether it still holds next door. Pass, and it looks like structure. Fail, and it was sample correlation.

## So did the factor "remember" a decisive low-dimensional logic?

**Sometimes you are seeing the shadow of that logic. It did not "remember" it.**

"Decisive low-dimensional logic" is the **hypothesis** you write down (fade after crowding, mean-reversion after a washout, do not open trend in chop). IC is only in-sample association. They look the same.

- Few gates, a story a human can tell, same sign on three segments, MaxDD not worse → more like structure
- IC only / one window / a pooled curve → treat it as neither memory nor discovery

Phase 1 can never promote. That sentence is the process.

## What overfitting does a model fight vs a rule? When is TimeSeriesSplit actually required?

| | Model | Rule |
|---|---|---|
| Capacity | Large; can fit residuals | Small; cannot memorize a path |
| Overfit shape | Interpolating the train fold | Wrong gate / threshold sweep |
| Defense | TimeSeriesSplit (do not memorize the fold) | Three-segment Pareto + a tellable mechanism |
| Role of IC | Feature-selection clue | Same: clue, not evidence |

Use TimeSeriesSplit in the **train inner loop** only when you actually fit a model (scorer, tree, linear). The court is still the three-segment backtest. Do not stack a Qlib CV ruler on top of the court.

## Why can Phase 1 (IC / label lift) never close an experiment?

The objective is different. Labels use `forward_rr` or a success flag. The court uses realized CAGR / Calmar / win rate / MaxDD / Sharpe, plus slots, slippage, and regime filters.

Features that look good on IC often fail the full backtest: multiple testing, invisible execution friction, non-stationary regimes. A longer sample often makes a drifted rule look cleaner.

Only Phase 3 on the canonical three segments with kill switch OFF can enter `mlbot research close`. `verdict` still needs a human `--declare`.

## What are the three court segments? Why does the Lab scorecard show a table, not one row?

Default crypto segments (`config/market_segment.yaml`):

- **Bear** `bear_2022`
- **Bull** `bull_2023_2024`
- **Chop** `recent_range_to_bear` (recent wide-range / turn-to-bear; not "average again")

Headline KPIs are only five: CAGR / Calmar / WR / MaxDD / Sharpe. Do not lead with Total R.

A single KPI row is usually the **first cited** artifact in DECISION (often baseline / bear). Variant × segment must be listed, or a chop failure hides under a bear number. Lab: the `/rd` scorecard table; this page is the "why."

## Do Phase 1 + Phase 2 tune and test on the same sample? Is that overfit / too optimistic?

**Yes, optimistic if you treat Phase 1+2 as the score.** Those steps are not train-then-test.

| Step | What it does | Same history? |
|---|---|---|
| Phase 1 | IC / plateau on the full labeled parquet | Discovery, in-sample association |
| Phase 2 | Human writes τ / lookback | Same parquet |
| Phase 3 | Bear / bull / chop event backtest, KS off | Court; calendar mostly still the same |

No classic holdout. Rule experiments also skip TimeSeriesSplit on purpose. The overfit channel is sweeping thresholds and keeping the pretty window, not memorizing a path.

Gates: Phase 1 cannot close; three-segment Pareto; kill switch off. Phase 3 changes the objective (CAGR / Calmar / WR / MaxDD / Sharpe) and the market structure. It is not future data. A true extra sample looks like A-share's second pool (core554 → full2000), not "the last calendar stretch is the only exam."

## Should Phase 1–3 use a purged/embargoed window, with the last stretch as OOS?

**Do not replace the court with "last stretch = OOS."** Purge/embargo only blocks label leak at a cut. It does not block picking the lucky gate.

- The last stretch is usually one structure (today: chop / turn-to-bear). Using it as the only OOS is the rejected `recent_6m_oos`-only promote.
- If Phase 1 never sees that structure, it will pick trend gates.
- Re-searching τ each window is the forbidden rolling optimize. Replaying a frozen rule is monitoring.

If you want a stricter tail: freeze Phase 1+2 before `T_cut` (optional gap of label horizon H); still run the three-segment court; replay once after `T_cut + H` and do not retune. Fail the tail → reject. Pass the tail → still not enough to promote alone.

## If a gap blocks correlation, why also cut time into small blocks?

**The gap blocks correlation. The small blocks are a different tool.**

**Embargo**  
A tree fits on the left and scores the same `y` on the right. `y` looks ahead. With no gap, the last training labels already contain the start of the test path, and the score is inflated. One cut, one gap as long as the label horizon.

**TimeSeriesSplit**  
A tree can look great on a long fit. One last-half-year test is one number, often one regime. The train loop asks several times: fit on earlier data, score the next stretch. The blocks exist to repeat that question, not to break correlation. Correlation is still the gap at each cut.

Rules do not fit-then-score the same `y`, so Phase 1–3 need neither. Re-searching τ on monthly blocks is the forbidden rolling optimize. Rules want large frozen blocks: same gates on bear / bull / chop.

## Are trees tied to the train loop? If the label is forward_rr over 10 bars, do we need TSS + embargo + purge?

**The trigger is fit + the same `y` + a time cut. Not the words "tree" or "forward_rr."**

In this repo, trees go through `mlbot train`: fit on the left, score the same label on the right. That is why trees are tied to the train cut. Rules go scan → frozen yaml → event backtest. No such cut.

| Tool | Why you need it | What `forward_rr` over 10 bars changes |
|---|---|---|
| **embargo** | `y` overlaps across the cut | Gap ≥ 10 bars on that timeframe (20h at 120T). Tree `holdout_embargo_minutes` is this |
| **purge** | A sample's `[t, t+10)` lands in the other fold | Drop those samples; larger H drops more |
| **TimeSeriesSplit** | You are fitting; ask "next stretch still works?" more than once | **Not because H=10.** H=3 needs it too. H only widens the gap |

So: if you fit a tree / linear / scorer on a look-ahead label, use TSS, and size embargo/purge to H.  
A rule Phase 1 IC on the same `forward_rr@10` does not make this trio the court — IC cannot close; the court is realized three-segment paths.  
If a rule later gets a freeze-at-`T` then replay-only tail, put a gap of H on that one cut. Still do not replace the three-segment court with the last stretch.

## What does "drop samples whose label already stepped into the other fold" mean?

It means that row's **answer** already used prices from the test fold.

If the label is "return over the next 10 bars," the row on day 1 uses prices from day 1 to 11. Train is days 1–100; test starts on day 101. The row on day 95 is still on the train calendar, but its answer uses days 95–105 — and 101–105 are already test. The training target contains the exam path.

**Stepped into the other fold** = that sample's label window `[t, t+H)` overlaps the test fold.  
**Purge** = do not fit on those rows. Keep only rows whose whole answer ends before the cut.

Embargo leaves an unused gap between the two sides. Purge deletes specific overlapping rows inside an already-drawn fold.

## What are SRB, fade, and A-share? Should we still hunt a 2h tree or 15-minute order flow?

Live money is the three sleeves (plus Rolling), not a mid-horizon tree:

| Sleeve | Role |
|---|---|
| **SRB** | Structure detector (follow a break). Times trend beta; harvests the right tail. Not a macro on/off switch. |
| **fade** | Weak-quality alpha: failed pierce + `box_pos_60` edge. Not classic RSI/BB. Separate SA account. |
| **A-share** | Cross-sectional oversold: fire day L1, 8 equal slots, flatten at 40 sessions. |
| **Rolling U** | Slower trend-regime harvest (U-only). |

The 2h six-coin trees (`20260831_short_term_swing_h20_court` / `20260901_…_mfe_top20`) are **reject**: columns are contemporaneous, not a 20-bar lead. 15T kline IC (`20260831_fast_scalp_15t_ic`) is fade-signed; the wide-pool tree holdout Pearson is ~0. Bar-aggregated OF on 1-minute sides also failed the gate.

**Do not** rescan `tree_full` to forecast a 2h path. **Do not** treat "hunt something shorter" as another 15T order-flow tree. A short sleeve must change to trade clock and beat fees first.

## After a P99 taker burst, does price fade or continue? Why not assume one side?

**Count both paths, and split them.** A large taker stacks two machines: temporary impact / inventory (fade) and information / liquidation cascade (continue). One average cancels them and IC goes to ~0.

Trade-clock study (`20260901_btc_taker_burst_markout`, declared reject): BTC 1s clock; 10s volume ≥ trailing 1h P99 and |imbalance|≥0.6; absorbed = no break of the prior 60s range; displaced = same-side break. 6,182 events over three months:

- First ±5bp: opposite 54.8% / same 44.0%; absorbed ≈ displaced
- Same-side markout: −0.38bp at +5s, −0.16bp at +2min; both sleeves same sign
- After 10bp taker round-trip both legs are about −10bp
- 81% labeled displaced — a 60s range is too easy for BTC P99, so the split is lopsided

The average path fades slightly, far below fees. H1 (opposite signs after the split) and H2 (tradeable +2min after fees) failed. No `event_backtest`. Do not train another 15T tree.

## 5bp is not VIP and they get paid to make. If I have a trading stack and rebates, is HFT just engineering?

**5bp is regular taker; the arithmetic is right. HFT does not forecast two minutes. The identity is spread + rebate − adverse selection − inventory.**

USD-M retail is about maker 2bp / taker 5bp (10bp taker round-trip). BTC perp spread is often one tick (~0.1–0.2bp), so a retail limit does not even cover the maker fee. Negative maker is a high-volume VIP / MM contract, not a GTC flag.

The −0.4bp we measured is the markout after a P99 hit. Market makers live by canceling before that flow and collecting the noise fills. OMS, the feature bus, and aggTrades are **not** an HFT stack: no queue position, no colocation, no stable touch, no venue rebate contract.

Cross-venue delay is millisecond HFT. Cash-and-carry is capital + basis. 8h funding is a hold spread, not HFT. Only with a rebate contract, the front of the queue, and cancels that beat toxic flow does MM become "just engineering." Without those, HFT is paying retail fees to catch the −0.4bp hit.

Closer engineering: mark out your own SRB / fade fills (save slippage), or hold-style funding / basis. Do not rescan FeatureStore to imitate HFT.
