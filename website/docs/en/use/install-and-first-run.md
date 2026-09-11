# Install and first run

**One line:** Install once so this machine can download, build features, and run the ruler. First practice: the MA golden cross.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
mlbot --help
```

Python 3.12. Full data / feature / backtest commands: repo [usage.en.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/usage.en.md) — this site does not copy the encyclopedia.

## First practice

1. Read [A strategy is a contract](../quant/what-is-a-strategy.md) and [Closed bar](../quant/closed-bar.md).
2. Open the MA-cross table in the repo README.
3. Tell the AI the same sentence; without “measure this”, it should only help with the template and directory.
4. Public practice pack: `config/strategies/ma_cross/`.

## Deeper docs

- [Commands map](../tech/commands-map.md)
- [Usage](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/usage.en.md)
