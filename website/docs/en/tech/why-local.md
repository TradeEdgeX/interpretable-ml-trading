# Why local

**One-liner:** All data and features live on your machine. Switch the chat model and the table stays the same. Evidence is re-checkable, results are reproducible.

## Wrong vs right

| Wrong | Right |
|---|---|
| “I asked a web AI; it said the golden cross is a classic trend.” | “The feature store and trades on my machine printed this table: golden-cross recent-window CAGR −3.1%.” |

A web AI gives you “a story from its training data” that you cannot re-check. Local gives you a table printed by “trades you downloaded + features you computed + a locked court”, which you can re-check line by line.

## What “local” means

- Data: monthly parquet in `data/parquet_data/`.
- Features: monthly parquet in `feature_store/`.
- Court: locked scripts in `scripts/`.
- Results: KPI tables and logs in `results/`.

Switch the AI model — as long as these files are there, the table stays the same.

## Fine print

- [Stack and paths](stack.md)
- [docs/philosophy.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/philosophy.md)
