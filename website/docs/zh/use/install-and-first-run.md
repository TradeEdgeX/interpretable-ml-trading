# 安装与第一次

**一句话：** 装一次，给本机能拉数据、算特征、跑尺子。第一条练习是均线金叉。

## 安装

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
mlbot --help
```

Python 3.12。完整数据 / 特征 / 回测命令见仓库 [usage.md](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/usage.md)——站点不复制百科。

## 第一次练什么

1. 读 [策略是合同](../quant/what-is-a-strategy.md) 和 [闭棒](../quant/closed-bar.md)。
2. 打开仓库 README 的金叉对照表。
3. 跟 AI 说同一句金叉；没说「测一下」时，它只该帮你填模板、建目录。
4. 公开练习包：`config/strategies/ma_cross/`。

## 还想看细则

- [命令地图](../tech/commands-map.md)
- [使用方法](https://github.com/TradeEdgeX/interpretable-ml-trading/blob/main/docs/usage.md)
