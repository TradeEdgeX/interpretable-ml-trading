# 十倍股是不是都得小市值、拿三四年

实验：[config/experiments/20260911_tenbagger_smallcap/](../../config/experiments/20260911_tenbagger_smallcap/)  
评测机是 **cohort_hold**，不是 2h `event_backtest`。已 `--declare reject`。

---

## 能量什么

| 人想做什么 | 本仓库 |
|---|---|
| 调研**未来**谁会十倍、叙事还没走完 | **没有用。** 没有成交、没有闭棒列。 |
| 量一句**历史规则**：「入场日市值小、死拿 3–4 年，十倍是不是比大票更密」 | **有用。** 肥尾 / 小市值 beta，不是选点 alpha。 |

---

## 模板（可 `validate`）

```bash
PYTHONPATH=src python -m cli.main research validate 20260911_tenbagger_smallcap
```

| 格 | 这一句 |
|---|---|
| 社会学 | 十倍是彩票；付费方是追故事的人。 |
| 数学 | 闭棒、入场日流通市值；固定持有年数；禁止事后贴「十倍股」标签。 |
| 统计学 | **肥尾 + 小市值 beta**。去最大几个十倍只分类，不单独否。 |
| 验证标准 | 任一段相对对照年化 ≤ 0，或十倍率无差异且回撤更深。 |
| 数据范围 | A 股日线 + 时点市值 + `market_segment_ashare.yaml`。美股未量。 |

---

## 数据怎么来（人说了测）

```bash
# 沪深在市 + 2014 后退市，约十年前复权日线（换手率写成百分数）
PYTHONPATH=src python -m cli.main data download-ashare \
  --universe listed,delisted --start-date 2016-01-01 --backend sina --workers 4

PYTHONPATH=src python scripts/research/cohort_hold.py \
  --mcap-yi 100 --hold-years 3,4
```

本次本机：在市 5213/5215 只有日线；2014-06 后退市 255 只有（更早退市新浪无带）。季度 PIT 46 个时点。市值 = 当日成交额 / (换手率/100)，不是今天的市值往回填。

---

## 数字（熔断关 · 入场日分桶）

`n` = 名字 × 入场季。`bull_924` / `chop_recent` 还拿不满 3 年 → 无样本。

**3 年。** crash_2015：小 −4.26% / 大 −4.15%，十倍都是 0。bear_2018：小 +2.75% / 大 +3.31%。covid_2020：小 +7.75% / 大 +5.96%，小票十倍 0、大票 0.13%。bear_2021：小 +9.09% / 大 −0.03%，十倍 0.09% vs 0.12%。

**4 年。** crash_2015 小仍差于大（−3.07% vs −2.78%）。bear_2018 / covid / bear_2021 小年化高于大；十倍率仍在 0–0.4%，两边同一量级。

全部完成队列：3 年十倍率都是 **0.11%**（88 vs 33 次）。小票只是池子大，个数多，密度一样。去 Top-3 几乎不动。小票 3 年路径里 1520 条退市，大票 216 条。

证伪条件已在 crash_2015（3y/4y）和 bear_2018 的 3 年窗触发（相对年化 ≤ 0）。

分类：**肥尾极薄，主形态是小市值 beta。** 网上「十倍股都是小盘」是事后数个数，不是入场日密度。

已 `--declare reject`。产物：`results/tenbagger_smallcap/experiments/20260911_tenbagger_smallcap`

---

## 美股呢？

同一句数学，换日历和市值单位。公开抽取没有美股下载，这次没量。

---

## 和别的例子怎么摆

| 例子 | 能量什么 | 不能能量什么 |
|---|---|---|
| 金叉 / 费率 / 周一 / 山寨 / P99 | 已经发生的进出合同 | 「下次一定行」 |
| 本页 | 历史上小市值长持的倍数分布 | 未来哪只票会十倍 |

框架里各层怎么选：[framework.md](../framework.md)。
