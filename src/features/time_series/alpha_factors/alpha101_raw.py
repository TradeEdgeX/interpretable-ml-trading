from .alpha_utils import *


def alpha001(c, r):
    """(rank(ts_argmax(power(((returns < 0)
    ? ts_std(returns, 20)
    : close), 2.), 5)) -0.5)"""
    c[r < 0] = ts_std(r, 20)
    return rank(ts_argmax(power(c, 2), 5)).mul(-0.5).stack().swaplevel()


def alpha002(o, c, v):
    """(-1 * ts_corr(rank(ts_delta(log(volume), 2)), rank(((close - open) / open)), 6))"""
    s1 = rank(ts_delta(log(v), 2))
    s2 = rank((c / o) - 1)
    alpha = -ts_corr(s1, s2, 6)
    return alpha.stack("ticker").swaplevel().replace([-np.inf, np.inf], np.nan)


def alpha003(o, v):
    """(-1 * ts_corr(rank(open), rank(volume), 10))"""

    return (
        -ts_corr(rank(o), rank(v), 10)
        .stack("ticker")
        .swaplevel()
        .replace([-np.inf, np.inf], np.nan)
    )


def alpha004(l):
    """(-1 * Ts_Rank(rank(low), 9))"""
    return -ts_rank(rank(l), 9).stack("ticker").swaplevel()


def alpha005(o, vwap, c):
    """(rank((open - ts_mean(vwap, 10))) * (-1 * abs(rank((close - vwap)))))"""
    return (
        rank(o.sub(ts_mean(vwap, 10)))
        .mul(rank(c.sub(vwap)).mul(-1).abs())
        .stack("ticker")
        .swaplevel()
    )


def alpha006(o, v):
    """(-ts_corr(open, volume, 10))"""
    return -ts_corr(o, v, 10).stack("ticker").swaplevel()


def alpha007(c, v, adv20):
    """(adv20 < volume)
    ? ((-ts_rank(abs(ts_delta(close, 7)), 60)) * sign(ts_delta(close, 7)))
    : -1
    """

    delta7 = ts_delta(c, 7)
    return (
        -ts_rank(abs(delta7), 60)
        .mul(sign(delta7))
        .where(adv20 < v, -1)
        .stack("ticker")
        .swaplevel()
    )


def alpha008(o, r):
    """-rank(((ts_sum(open, 5) * ts_sum(returns, 5)) -
    ts_lag((ts_sum(open, 5) * ts_sum(returns, 5)),10)))
    """
    return (
        -(
            rank(
                (
                    (ts_sum(o, 5) * ts_sum(r, 5))
                    - ts_lag((ts_sum(o, 5) * ts_sum(r, 5)), 10)
                )
            )
        )
        .stack("ticker")
        .swaplevel()
    )


def alpha009(c):
    """(0 < ts_min(ts_delta(close, 1), 5)) ? ts_delta(close, 1)
    : ((ts_max(ts_delta(close, 1), 5) < 0)
    ? ts_delta(close, 1) : (-1 * ts_delta(close, 1)))
    """
    close_diff = ts_delta(c, 1)
    alpha = close_diff.where(
        ts_min(close_diff, 5) > 0,
        close_diff.where(ts_max(close_diff, 5) < 0, -close_diff),
    )
    return alpha.stack("ticker").swaplevel()


def alpha010(c):
    """rank(((0 < ts_min(ts_delta(close, 1), 4))
    ? ts_delta(close, 1)
    : ((ts_max(ts_delta(close, 1), 4) < 0)
        ? ts_delta(close, 1)
        : (-1 * ts_delta(close, 1)))))
    """
    close_diff = ts_delta(c, 1)
    alpha = close_diff.where(
        ts_min(close_diff, 4) > 0,
        close_diff.where(ts_min(close_diff, 4) > 0, -close_diff),
    )

    return rank(alpha).stack("ticker").swaplevel()


def alpha011(c, vwap, v):
    """(rank(ts_max((vwap - close), 3)) +
    rank(ts_min(vwap - close), 3)) *
    rank(ts_delta(volume, 3))
    """
    return (
        rank(ts_max(vwap.sub(c), 3))
        .add(rank(ts_min(vwap.sub(c), 3)))
        .mul(rank(ts_delta(v, 3)))
        .stack("ticker")
        .swaplevel()
    )


def alpha012(v, c):
    """(sign(ts_delta(volume, 1)) *
    (-1 * ts_delta(close, 1)))
    """
    return sign(ts_delta(v, 1)).mul(-ts_delta(c, 1)).stack("ticker").swaplevel()


def alpha013(c, v):
    """-rank(ts_cov(rank(close), rank(volume), 5))"""
    return -rank(ts_cov(rank(c), rank(v), 5)).stack("ticker").swaplevel()


def alpha014(o, v, r):
    """
    (-rank(ts_delta(returns, 3))) * ts_corr(open, volume, 10))
    """

    alpha = -rank(ts_delta(r, 3)).mul(
        ts_corr(o, v, 10).replace([-np.inf, np.inf], np.nan)
    )
    return alpha.stack("ticker").swaplevel()


def alpha015(h, v):
    """(-1 * ts_sum(rank(ts_corr(rank(high), rank(volume), 3)), 3))"""
    alpha = -ts_sum(
        rank(ts_corr(rank(h), rank(v), 3).replace([-np.inf, np.inf], np.nan)), 3
    )
    return alpha.stack("ticker").swaplevel()


def alpha016(h, v):
    """(-1 * rank(ts_cov(rank(high), rank(volume), 5)))"""
    return -rank(ts_cov(rank(h), rank(v), 5)).stack("ticker").swaplevel()


def alpha017(c, v):
    """(((-1 * rank(ts_rank(close, 10))) * rank(ts_delta(ts_delta(close, 1), 1))) *rank(ts_rank((volume / adv20), 5)))"""
    adv20 = ts_mean(v, 20)
    return (
        -rank(ts_rank(c, 10))
        .mul(rank(ts_delta(ts_delta(c, 1), 1)))
        .mul(rank(ts_rank(v.div(adv20), 5)))
        .stack("ticker")
        .swaplevel()
    )


def alpha018(o, c):
    """-rank((ts_std(abs((close - open)), 5) + (close - open)) +
    ts_corr(close, open,10))
    """
    return (
        -rank(
            ts_std(c.sub(o).abs(), 5)
            .add(c.sub(o))
            .add(ts_corr(c, o, 10).replace([-np.inf, np.inf], np.nan))
        )
        .stack("ticker")
        .swaplevel()
    )


def alpha019(c, r):
    """((-1 * sign(((close - ts_lag(close, 7)) + ts_delta(close, 7)))) *
    (1 + rank((1 + ts_sum(returns,250)))))
    """
    return (
        -sign(ts_delta(c, 7) + ts_delta(c, 7))
        .mul(1 + rank(1 + ts_sum(r, 250)))
        .stack("ticker")
        .swaplevel()
    )


def alpha020(o, h, l, c):
    """-rank(open - ts_lag(high, 1)) *
    rank(open - ts_lag(close, 1)) *
    rank(open -ts_lag(low, 1))"""
    return (
        rank(o - ts_lag(h, 1))
        .mul(rank(o - ts_lag(c, 1)))
        .mul(rank(o - ts_lag(l, 1)))
        .mul(-1)
        .stack("ticker")
        .swaplevel()
    )


def alpha021(c, v):
    """ts_mean(close, 8) + ts_std(close, 8) < ts_mean(close, 2)
    ? -1
    : (ts_mean(close,2) < ts_mean(close, 8) - ts_std(close, 8)
        ? 1
        : (volume / adv20 < 1
            ? -1
            : 1))
    """
    sma2 = ts_mean(c, 2)
    sma8 = ts_mean(c, 8)
    std8 = ts_std(c, 8)

    cond_1 = sma8.add(std8) < sma2
    cond_2 = sma8.add(std8) > sma2
    cond_3 = v.div(ts_mean(v, 20)) < 1

    val = np.ones_like(c)
    alpha = pd.DataFrame(
        np.select(condlist=[cond_1, cond_2, cond_3], choicelist=[-1, 1, -1], default=1),
        index=c.index,
        columns=c.columns,
    )

    return alpha.stack("ticker").swaplevel()


def alpha022(h, c, v):
    """-(ts_delta(ts_corr(high, volume, 5), 5) *
    rank(ts_std(close, 20)))
    """

    return (
        ts_delta(ts_corr(h, v, 5).replace([-np.inf, np.inf], np.nan), 5)
        .mul(rank(ts_std(c, 20)))
        .mul(-1)
        .stack("ticker")
        .swaplevel()
    )


def alpha023(h, c):
    """((ts_mean(high, 20) < high)
    ? (-1 * ts_delta(high, 2))
    : 0
    """

    return (
        ts_delta(h, 2).mul(-1).where(ts_mean(h, 20) < h, 0).stack("ticker").swaplevel()
    )


def alpha024(c):
    """((((ts_delta((ts_mean(close, 100)), 100) / ts_lag(close, 100)) <= 0.05)
    ? (-1 * (close - ts_min(close, 100)))
    : (-1 * ts_delta(close, 3)))
    """
    cond = ts_delta(ts_mean(c, 100), 100) / ts_lag(c, 100) <= 0.05

    return (
        c.sub(ts_min(c, 100))
        .mul(-1)
        .where(cond, -ts_delta(c, 3))
        .stack("ticker")
        .swaplevel()
    )


def alpha025(h, c, r, vwap, adv20):
    """rank((-1 * returns) * adv20 * vwap * (high - close))"""
    return rank(-r.mul(adv20).mul(vwap).mul(h.sub(c))).stack("ticker").swaplevel()


def alpha026(h, v):
    """(-1 * ts_max(ts_corr(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))"""
    return (
        ts_max(
            ts_corr(ts_rank(v, 5), ts_rank(h, 5), 5).replace([-np.inf, np.inf], np.nan),
            3,
        )
        .mul(-1)
        .stack("ticker")
        .swaplevel()
    )


def alpha027(v, vwap):
    """((0.5 < rank(ts_mean(ts_corr(rank(volume), rank(vwap), 6), 2)))
    ? -1
    : 1)"""
    cond = rank(ts_mean(ts_corr(rank(v), rank(vwap), 6), 2))
    alpha = cond.notnull().astype(float)
    return alpha.where(cond <= 0.5, -alpha).stack("ticker").swaplevel()


def alpha028(h, l, c, v, adv20):
    """scale(((ts_corr(adv20, low, 5) + (high + low) / 2) - close))"""
    return (
        scale(
            ts_corr(adv20, l, 5)
            .replace([-np.inf, np.inf], 0)
            .add(h.add(l).div(2).sub(c))
        )
        .stack("ticker")
        .swaplevel()
    )


def alpha029(c, r):
    """(ts_min(ts_product(rank(rank(scale(log(ts_sum(ts_min(rank(rank((-1 *
        rank(ts_delta((close - 1),5))))), 2), 1))))), 1), 5)
    + ts_rank(ts_lag((-1 * returns), 6), 5))
    """
    return (
        ts_min(
            rank(rank(scale(log(ts_sum(rank(rank(-rank(ts_delta((c - 1), 5)))), 2))))),
            5,
        )
        .add(ts_rank(ts_lag((-1 * r), 6), 5))
        .stack("ticker")
        .swaplevel()
    )


def alpha030(c, v):
    """(((1.0 - rank(((sign((close - ts_lag(close, 1))) +
    sign((ts_lag(close, 1) - ts_lag(close, 2)))) +
    sign((ts_lag(close, 2) - ts_lag(close, 3)))))) *
    ts_sum(volume, 5)) / ts_sum(volume, 20))"""
    close_diff = ts_delta(c, 1)
    return (
        rank(
            sign(close_diff)
            .add(sign(ts_lag(close_diff, 1)))
            .add(sign(ts_lag(close_diff, 2)))
        )
        .mul(-1)
        .add(1)
        .mul(ts_sum(v, 5))
        .div(ts_sum(v, 20))
        .stack("ticker")
        .swaplevel()
    )


def alpha031(l, c, adv20):
    """((rank(rank(rank(ts_weighted_mean((-1 * rank(rank(ts_delta(close, 10)))), 10)))) +
    rank((-1 * ts_delta(close, 3)))) + sign(scale(ts_corr(adv20, low, 12))))
    """
    return (
        rank(rank(rank(ts_weighted_mean(rank(rank(ts_delta(c, 10))).mul(-1), 10))))
        .add(rank(ts_delta(c, 3).mul(-1)))
        .add(sign(scale(ts_corr(adv20, l, 12).replace([-np.inf, np.inf], np.nan))))
        .stack("ticker")
        .swaplevel()
    )


def alpha032(c, vwap):
    """scale(ts_mean(close, 7) - close) +
    (20 * scale(ts_corr(vwap, ts_lag(close, 5),230)))"""
    return (
        scale(ts_mean(c, 7).sub(c))
        .add(20 * scale(ts_corr(vwap, ts_lag(c, 5), 230)))
        .stack("ticker")
        .swaplevel()
    )


def alpha033(o, c):
    """rank(-(1 - (open / close)))"""
    return rank(o.div(c).mul(-1).add(1).mul(-1)).stack("ticker").swaplevel()


def alpha034(c, r):
    """rank(((1 - rank((ts_std(returns, 2) / ts_std(returns, 5)))) + (1 - rank(ts_delta(close, 1)))))"""

    return (
        rank(
            rank(ts_std(r, 2).div(ts_std(r, 5)).replace([-np.inf, np.inf], np.nan))
            .mul(-1)
            .sub(rank(ts_delta(c, 1)))
            .add(2)
        )
        .stack("ticker")
        .swaplevel()
    )


def alpha035(h, l, c, v, r):
    """((ts_Rank(volume, 32) *
    (1 - ts_Rank(((close + high) - low), 16))) *
    (1 -ts_Rank(returns, 32)))
    """
    return (
        ts_rank(v, 32)
        .mul(1 - ts_rank(c.add(h).sub(l), 16))
        .mul(1 - ts_rank(r, 32))
        .stack("ticker")
        .swaplevel()
    )


def alpha036(o, c, v, r, adv20, vwap):
    """2.21 * rank(ts_corr((close - open), ts_lag(volume, 1), 15)) +
    0.7 * rank((open- close)) +
    0.73 * rank(ts_Rank(ts_lag(-1 * returns, 6), 5)) +
    rank(abs(ts_corr(vwap,adv20, 6))) +
    0.6 * rank(((ts_mean(close, 200) - open) * (close - open)))
    """

    return (
        rank(ts_corr(c.sub(o), ts_lag(v, 1), 15))
        .mul(2.21)
        .add(rank(o.sub(c)).mul(0.7))
        .add(rank(ts_rank(ts_lag(-r, 6), 5)).mul(0.73))
        .add(rank(abs(ts_corr(vwap, adv20, 6))))
        .add(rank(ts_mean(c, 200).sub(o).mul(c.sub(o))).mul(0.6))
        .stack("ticker")
        .swaplevel()
    )


def alpha037(o, c):
    """(rank(ts_corr(ts_lag((open - close), 1), close, 200)) + rank((open - close)))"""
    return (
        rank(ts_corr(ts_lag(o.sub(c), 1), c, 200))
        .add(rank(o.sub(c)))
        .stack("ticker")
        .swaplevel()
    )


def alpha038(o, c):
    """ "-1 * rank(ts_rank(close, 10)) * rank(close / open)"""
    return (
        rank(ts_rank(o, 10))
        .mul(rank(c.div(o).replace([-np.inf, np.inf], np.nan)))
        .mul(-1)
        .stack("ticker")
        .swaplevel()
    )


def alpha039(c, v, r, adv20):
    """-rank(ts_delta(close, 7) * (1 - rank(ts_weighted_mean(volume / adv20, 9)))) *
    (1 + rank(ts_sum(returns, 250)))"""
    return (
        rank(ts_delta(c, 7).mul(rank(ts_weighted_mean(v.div(adv20), 9)).mul(-1).add(1)))
        .mul(-1)
        .mul(rank(ts_mean(r, 250).add(1)))
        .stack("ticker")
        .swaplevel()
    )


def alpha040(h, v):
    """((-1 * rank(ts_std(high, 10))) * ts_corr(high, volume, 10))"""
    return (
        rank(ts_std(h, 10)).mul(ts_corr(h, v, 10)).mul(-1).stack("ticker").swaplevel()
    )


def alpha041(h, l, vwap):
    """power(high * low, 0.5 - vwap"""
    return power(h.mul(l), 0.5).sub(vwap).stack("ticker").swaplevel()


def alpha042(c, vwap):
    """rank(vwap - close) / rank(vwap + close)"""
    return rank(vwap.sub(c)).div(rank(vwap.add(c))).stack("ticker").swaplevel()


def alpha043(adv20, c, v):
    """(ts_rank((volume / adv20), 20) * ts_rank((-1 * ts_delta(close, 7)), 8))"""

    return (
        ts_rank(v.div(adv20), 20)
        .mul(ts_rank(ts_delta(c, 7).mul(-1), 8))
        .stack("ticker")
        .swaplevel()
    )


def alpha044(h, v):
    """-ts_corr(high, rank(volume), 5)"""

    return (
        ts_corr(h, rank(v), 5)
        .replace([-np.inf, np.inf], np.nan)
        .mul(-1)
        .stack("ticker")
        .swaplevel()
    )


def alpha045(c, v):
    """-(rank((ts_mean(ts_lag(close, 5), 20)) *
    ts_corr(close, volume, 2)) *
    rank(ts_corr(ts_sum(close, 5), ts_sum(close, 20), 2)))"""

    return (
        rank(ts_mean(ts_lag(c, 5), 20))
        .mul(ts_corr(c, v, 2).replace([-np.inf, np.inf], np.nan))
        .mul(rank(ts_corr(ts_sum(c, 5), ts_sum(c, 20), 2)))
        .mul(-1)
        .stack("ticker")
        .swaplevel()
    )


def alpha046(c):
    """0.25 < ts_lag(ts_delta(close, 10), 10) / 10 - ts_delta(close, 10) / 10
    ? -1
    : ((ts_lag(ts_delta(close, 10), 10) / 10 - ts_delta(close, 10) / 10 < 0)
        ? 1
        : -ts_delta(close, 1))
    """

    cond = ts_lag(ts_delta(c, 10), 10).div(10).sub(ts_delta(c, 10).div(10))
    alpha = pd.DataFrame(-np.ones_like(cond), index=c.index, columns=c.columns)
    alpha[cond.isnull()] = np.nan
    return (
        cond.where(cond > 0.25, -alpha.where(cond < 0, -ts_delta(c, 1)))
        .stack("ticker")
        .swaplevel()
    )


def alpha047(h, c, v, vwap, adv20):
    """((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) /
    (ts_sum(high, 5) /5))) - rank((vwap - ts_lag(vwap, 5))))"""

    return (
        rank(c.pow(-1))
        .mul(v)
        .div(adv20)
        .mul(h.mul(rank(h.sub(c)).div(ts_mean(h, 5))).sub(rank(ts_delta(vwap, 5))))
        .stack("ticker")
        .swaplevel()
    )


def alpha48(c, industry):
    """(indneutralize(((ts_corr(ts_delta(close, 1), ts_delta(ts_lag(close, 1), 1), 250) *
    ts_delta(close, 1)) / close), IndClass.subindustry) /
    ts_sum(((ts_delta(close, 1) / ts_lag(close, 1))^2), 250))"""
    pass


def alpha049(c):
    """ts_delta(ts_lag(close, 10), 10).div(10).sub(ts_delta(close, 10).div(10)) < -0.1 * c
    ? 1
    : -ts_delta(close, 1)"""
    cond = ts_delta(ts_lag(c, 10), 10).div(10).sub(ts_delta(c, 10).div(10)) >= -0.1 * c
    return -ts_delta(c, 1).where(cond, 1).stack("ticker").swaplevel()


def alpha050(v, vwap):
    """-ts_max(rank(ts_corr(rank(volume), rank(vwap), 5)), 5)"""
    return (
        ts_max(rank(ts_corr(rank(v), rank(vwap), 5)), 5)
        .mul(-1)
        .stack("ticker")
        .swaplevel()
    )


def alpha051(c):
    """ts_delta(ts_lag(close, 10), 10).div(10).sub(ts_delta(close, 10).div(10)) < -0.05 * c
    ? 1
    : -ts_delta(close, 1)"""
    cond = ts_delta(ts_lag(c, 10), 10).div(10).sub(ts_delta(c, 10).div(10)) >= -0.05 * c
    return -ts_delta(c, 1).where(cond, 1).stack("ticker").swaplevel()


def alpha052(l, v, r):
    """(ts_lag(ts_min(low, 5), 5) - ts_min(low, 5)) *
    rank((ts_sum(returns, 240) - ts_sum(returns, 20)) / 220) *
    ts_rank(volume, 5)
    """
    return (
        ts_delta(ts_min(l, 5), 5)
        .mul(rank(ts_sum(r, 240).sub(ts_sum(r, 20)).div(220)))
        .mul(ts_rank(v, 5))
        .stack("ticker")
        .swaplevel()
    )


def alpha053(h, l, c):
    """-1 * ts_delta(1 - (high - close) / (close - low), 9)"""
    inner = (c.sub(l)).add(1e-6)
    return (
        ts_delta(h.sub(c).mul(-1).add(1).div(c.sub(l).add(1e-6)), 9)
        .mul(-1)
        .stack("ticker")
        .swaplevel()
    )


def alpha054(o, h, l, c):
    """-(low - close) * power(open, 5) / ((low - high) * power(close, 5))"""
    return (
        l.sub(c)
        .mul(o.pow(5))
        .mul(-1)
        .div(l.sub(h).replace(0, -0.0001).mul(c**5))
        .stack("ticker")
        .swaplevel()
    )


def alpha055(c, h, l, v):
    """(-1 * ts_corr(rank(((close - ts_min(low, 12)) /
            (ts_max(high, 12) - ts_min(low,12)))),
    rank(volume), 6))"""

    return (
        ts_corr(
            rank(
                c.sub(ts_min(l, 12)).div(
                    ts_max(h, 12).sub(ts_min(l, 12)).replace(0, 1e-6)
                )
            ),
            rank(v),
            6,
        )
        .replace([-np.inf, np.inf], np.nan)
        .mul(-1)
        .stack("ticker")
        .swaplevel()
    )


def alpha056(r, cap):
    """-rank(ts_sum(returns, 10) / ts_sum(ts_sum(returns, 2), 3)) *
    rank((returns * cap))
    """
    pass


def alpha057(c, vwap):
    """-(close - vwap) / ts_weighted_mean(rank(ts_argmax(close, 30)), 2)"""
    return (
        c.sub(vwap.add(1e-5))
        .div(ts_weighted_mean(rank(ts_argmax(c, 30))))
        .mul(-1)
        .stack("ticker")
        .swaplevel()
    )


def alpha58(v, wvap, sector):
    """(-1 * ts_rank(ts_weighted_mean(ts_corr(IndNeutralize(vwap, IndClass.sector), volume, 3), 7), 5))"""
    pass


def alpha59(v, wvap, industry):
    """-ts_rank(ts_weighted_mean(ts_corr(IndNeutralize(vwap, IndClass.industry), volume, 4), 16), 8)"""
    pass


def alpha060(l, h, c, v):
    """-((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) -scale(rank(ts_argmax(close, 10))))"""
    return (
        scale(rank(c.mul(2).sub(l).sub(h).div(h.sub(l).replace(0, 1e-5)).mul(v)))
        .mul(2)
        .sub(scale(rank(ts_argmax(c, 10))))
        .mul(-1)
        .stack("ticker")
        .swaplevel()
    )


def alpha061(v, vwap):
    """rank((vwap - ts_min(vwap, 16))) < rank(ts_corr(vwap, adv180, 17))"""

    return (
        rank(vwap.sub(ts_min(vwap, 16)))
        .lt(rank(ts_corr(vwap, ts_mean(v, 180), 18)))
        .astype(int)
        .stack("ticker")
        .swaplevel()
    )


def alpha062(o, h, l, vwap, adv20):
    """((rank(ts_corr(vwap, ts_sum(adv20, 22.4101), 9.91009)) <
    rank(((rank(open) + rank(open)) < (rank(((high + low) / 2)) + rank(high))))) * -1)
    """
    return (
        rank(ts_corr(vwap, ts_sum(adv20, 22), 9))
        .lt(rank(rank(o).mul(2)).lt(rank(h.add(l).div(2)).add(rank(h))))
        .mul(-1)
        .stack("ticker")
        .swaplevel()
    )


def alpha63(v, wvap, industry):
    """((rank(ts_weighted_mean(ts_delta(IndNeutralize(close, IndClass.industry), 2), 8)) -
    rank(ts_weighted_mean(ts_corr(((vwap * 0.318108) + (open * (1 - 0.318108))),
                                    ts_sum(adv180, 37), 13), 12))) * -1)
    """
    pass


def alpha064(o, h, l, v, vwap):
    """((rank(ts_corr(ts_sum(((open * 0.178404) + (low * (1 - 0.178404))), 12.7054),ts_sum(adv120, 12.7054), 16.6208)) <
    rank(ts_delta(((((high + low) / 2) * 0.178404) + (vwap * (1 -0.178404))), 3.69741))) * -1)
    """
    w = 0.178404
    return (
        rank(
            ts_corr(
                ts_sum(o.mul(w).add(l.mul(1 - w)), 12), ts_sum(ts_mean(v, 120), 12), 16
            )
        )
        .lt(rank(ts_delta(h.add(l).div(2).mul(w).add(vwap.mul(1 - w)), 3)))
        .mul(-1)
        .stack("ticker")
        .swaplevel()
    )


def alpha065(o, v, vwap):
    """((rank(ts_corr(((open * 0.00817205) + (vwap * (1 - 0.00817205))),
                    ts_sum(adv60,8.6911), 6.40374)) <
    rank((open - ts_min(open, 13.635)))) * -1)
    """
    w = 0.00817205
    return (
        rank(ts_corr(o.mul(w).add(vwap.mul(1 - w)), ts_mean(ts_mean(v, 60), 9), 6))
        .lt(rank(o.sub(ts_min(o, 13))))
        .mul(-1)
        .stack("ticker")
        .swaplevel()
    )


def alpha066(h, l, o, vwap):
    """((rank(ts_weighted_mean(ts_delta(vwap, 3.51013), 7.23052)) +
    ts_rank(ts_weighted_mean(((((low* 0.96633) + (low *
                                (1 - 0.96633))) - vwap) /
                                (open - ((high + low) / 2))), 11.4157), 6.72611)) * -1)
    """
    w = 0.96633
    return (
        rank(ts_weighted_mean(ts_delta(vwap, 4), 7))
        .add(
            ts_rank(
                ts_weighted_mean(
                    l.mul(w)
                    .add(l.mul(1 - w))
                    .sub(vwap)
                    .div(o.sub(h.add(l).div(2)).add(1e-3)),
                    11,
                ),
                7,
            )
        )
        .mul(-1)
        .stack("ticker")
        .swaplevel()
    )


def alpha067(h, v, sector, subindustry):
    """(power(rank((high - ts_min(high, 2.14593))),
    rank(ts_corr(IndNeutralize(vwap,IndClass.sector),
            IndNeutralize(adv20, IndClass.subindustry), 6.02936))) * -1)
    """
    pass


def alpha068(c, h, l, v):
    """((ts_rank(ts_corr(rank(high), rank(adv15), 8.91644), 13.9333) <
    rank(ts_delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157))) * -1)
    """
    w = 0.518371
    return (
        ts_rank(ts_corr(rank(h), rank(ts_mean(v, 15)), 9), 14)
        .lt(rank(ts_delta(c.mul(w).add(l.mul(1 - w)), 1)))
        .mul(-1)
        .stack("ticker")
        .swaplevel()
    )


def alpha069(c, vwap, industry):
    """((power(rank(ts_max(ts_delta(IndNeutralize(vwap, IndClass.industry), 2.72412),4.79344)),
    Ts_Rank(ts_corr(((close * 0.490655) + (vwap * (1 - 0.490655))), adv20, 4.92416),9.0615))) * -1)
    """
    pass


def alpha076(c, v, vwap, industry):
    """((power(rank(ts_delta(vwap, 1.29456)),
    ts_rank(ts_corr(IndNeutralize(close, IndClass.industry), adv50, 17.8256), 17.9171))) * -1)
    """
    pass


def alpha071(c, l, o, v, vwap):
    """max(ts_rank(ts_weighted_mean(ts_corr(ts_rank(close, 3.43976), ts_rank(adv180,12.0647), 18.0175), 4.20501), 15.6948),
    ts_rank(ts_weighted_mean((rank(((low + open) - (vwap +vwap)))^2), 16.4662), 4.4388))
    """

    s1 = ts_rank(
        ts_weighted_mean(ts_corr(ts_rank(c, 3), ts_rank(ts_mean(v, 180), 12), 18), 4),
        16,
    )
    s2 = ts_rank(ts_weighted_mean(rank(l.add(o).sub(vwap.mul(2))).pow(2), 16), 4)
    return s1.where(s1 > s2, s2).stack("ticker").swaplevel()


def alpha072(h, l, v, vwap):
    """(rank(ts_weighted_mean(ts_corr(((high + low) / 2), adv40, 8.93345), 10.1519)) /
    rank(ts_weighted_mean(ts_corr(ts_rank(vwap, 3.72469), ts_rank(volume, 18.5188), 6.86671), 2.95011)))
    """
    return (
        rank(ts_weighted_mean(ts_corr(h.add(l).div(2), ts_mean(v, 40), 9), 10))
        .div(rank(ts_weighted_mean(ts_corr(ts_rank(vwap, 3), ts_rank(v, 18), 6), 2)))
        .stack("ticker")
        .swaplevel()
    )


def alpha073(l, o, vwap):
    """(max(rank(ts_weighted_mean(ts_delta(vwap, 4.72775), 2.91864)),
    ts_rank(ts_weighted_mean(((ts_delta(((open * 0.147155) +
        (low * (1 - 0.147155))), 2.03608) /
        ((open *0.147155) + (low * (1 - 0.147155)))) * -1), 3.33829), 16.7411)) * -1)
    """
    w = 0.147155
    s1 = rank(ts_weighted_mean(ts_delta(vwap, 5), 3))
    s2 = ts_rank(
        ts_weighted_mean(
            ts_delta(o.mul(w).add(l.mul(1 - w)), 2).div(
                o.mul(w).add(l.mul(1 - w)).mul(-1)
            ),
            3,
        ),
        16,
    )

    print(s2)
    return s1.where(s1 > s2, s2).mul(-1).stack("ticker").swaplevel()


def alpha074(c, h, v, vwap):
    """((rank(ts_corr(close, ts_sum(adv30, 37.4843), 15.1365)) <
    rank(ts_corr(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11.4791)))* -1)
    """

    w = 0.0261661
    return (
        rank(ts_corr(c, ts_mean(ts_mean(v, 30), 37), 15))
        .lt(rank(ts_corr(rank(h.mul(w).add(vwap.mul(1 - w))), rank(v), 11)))
        .mul(-1)
        .stack("ticker")
        .swaplevel()
    )


def alpha075(l, v, vwap):
    """(rank(ts_corr(vwap, volume, 4.24304)) <
    rank(ts_corr(rank(low), rank(adv50),12.4413)))
    """

    return (
        rank(ts_corr(vwap, v, 4))
        .lt(rank(ts_corr(rank(l), rank(ts_mean(v, 50)), 12)))
        .astype(int)
        .stack("ticker")
        .swaplevel()
    )


def alpha076(l, vwap, sector):
    """(max(rank(ts_weighted_mean(ts_delta(vwap, 1.24383), 11.8259)),
    ts_rank(ts_weighted_mean(ts_rank(ts_corr(IndNeutralize(low, IndClass.sector), adv81,8.14941), 19.569), 17.1543), 19.383)) * -1)
    """
    pass


def alpha077(h, l, v, vwap):
    """min(rank(ts_weighted_mean(((((high + low) / 2) + high) - (vwap + high)), 20.0451)),
    rank(ts_weighted_mean(ts_corr(((high + low) / 2), adv40, 3.1614), 5.64125)))
    """

    s1 = rank(ts_weighted_mean(h.add(l).div(2).sub(vwap), 20))
    s2 = rank(ts_weighted_mean(ts_corr(h.add(l).div(2), ts_mean(v, 40), 3), 5))
    return s1.where(s1 < s2, s2).stack("ticker").swaplevel()


def alpha078(l, v, vwap):
    """(rank(ts_corr(ts_sum(((low * 0.352233) + (vwap * (1 - 0.352233))), 19.7428),
    ts_sum(adv40, 19.7428), 6.83313))^rank(ts_corr(rank(vwap), rank(volume), 5.77492)))
    """

    w = 0.352233
    return (
        rank(
            ts_corr(
                ts_sum((l.mul(w).add(vwap.mul(1 - w))), 19),
                ts_sum(ts_mean(v, 40), 19),
                6,
            )
        )
        .pow(rank(ts_corr(rank(vwap), rank(v), 5)))
        .stack("ticker")
        .swaplevel()
    )


def alpha079(o, v, sector):
    """(rank(ts_delta(IndNeutralize(((close * 0.60733) + (open * (1 - 0.60733))),IndClass.sector), 1.23438)) <
    rank(ts_corr(Ts_Rank(vwap, 3.60973), Ts_Rank(adv150,9.18637), 14.6644)))
    """
    pass


def alpha080(h, industry):
    """((power(rank(sign(ts_delta(IndNeutralize(((open * 0.868128) + (high * (1 - 0.868128))),IndClass.industry), 4.04545))),
    ts_rank(ts_corr(high, adv10, 5.11456), 5.53756)) * -1)
    """
    pass


def alpha081(v, vwap):
    """-(rank(log(ts_product(rank((rank(ts_corr(vwap, ts_sum(adv10, 49.6054),8.47743))^4)), 14.9655))) <
    rank(ts_corr(rank(vwap), rank(volume), 5.07914)))"""

    return (
        rank(
            log(
                ts_product(
                    rank(rank(ts_corr(vwap, ts_sum(ts_mean(v, 10), 50), 8)).pow(4)), 15
                )
            )
        )
        .lt(rank(ts_corr(rank(vwap), rank(v), 5)))
        .mul(-1)
        .stack("ticker")
        .swaplevel()
    )


def alpha082(o, v, sector):
    """(min(rank(ts_weighted_mean(ts_delta(open, 1.46063), 14.8717)),
    ts_rank(ts_weighted_mean(ts_corr(IndNeutralize(volume, IndClass.sector),
    ((open * 0.634196) +(open * (1 - 0.634196))), 17.4842), 6.92131), 13.4283)) * -1)

    """
    pass


def alpha083(c, h, l, v, vwap):
    """(rank(ts_lag((high - low) / ts_mean(close, 5), 2)) * rank(rank(volume)) /
    (((high - low) / ts_mean(close, 5) / (vwap - close)))
    """
    s = h.sub(l).div(ts_mean(c, 5))

    return (
        rank(rank(ts_lag(s, 2)).mul(rank(rank(v))).div(s).div(vwap.sub(c).add(1e-3)))
        .stack("ticker")
        .swaplevel()
        .replace((np.inf, -np.inf), np.nan)
    )


def alpha084(c, vwap):
    """power(ts_rank((vwap - ts_max(vwap, 15.3217)), 20.7127),
    ts_delta(close,4.96796))"""
    return (
        rank(power(ts_rank(vwap.sub(ts_max(vwap, 15)), 20), ts_delta(c, 6)))
        .stack("ticker")
        .swaplevel()
    )


def alpha085(c, h, l, v):
    """power(rank(ts_corr(((high * 0.876703) + (close * (1 - 0.876703))), adv30,9.61331)),
    rank(ts_corr(ts_rank(((high + low) / 2), 3.70596),
                 ts_rank(volume, 10.1595),7.11408)))
    """
    w = 0.876703
    return (
        rank(ts_corr(h.mul(w).add(c.mul(1 - w)), ts_mean(v, 30), 10))
        .pow(rank(ts_corr(ts_rank(h.add(l).div(2), 4), ts_rank(v, 10), 7)))
        .stack("ticker")
        .swaplevel()
    )


def alpha086(c, v, vwap):
    """((ts_rank(ts_corr(close, ts_sum(adv20, 14.7444), 6.00049), 20.4195) <
    rank(((open + close) - (vwap + open)))) * -1)
    """
    return (
        ts_rank(ts_corr(c, ts_mean(ts_mean(v, 20), 15), 6), 20)
        .lt(rank(c.sub(vwap)))
        .mul(-1)
        .stack("ticker")
        .swaplevel()
    )


def alpha087(c, vwap, industry):
    """(max(rank(ts_weighted_mean(ts_delta(((close * 0.369701) + (vwap * (1 - 0.369701))),1.91233), 2.65461)),
    ts_rank(ts_weighted_mean(abs(ts_corr(IndNeutralize(adv81,IndClass.industry), close, 13.4132)), 4.89768), 14.4535)) * -1)
    """
    pass


def alpha088(o, h, l, c, v):
    """min(rank(ts_weighted_mean(((rank(open) + rank(low)) - (rank(high) + rank(close))),8.06882)),
    ts_rank(ts_weighted_mean(ts_corr(ts_rank(close, 8.44728),
            ts_rank(adv60,20.6966), 8.01266), 6.65053), 2.61957))"""

    s1 = rank(ts_weighted_mean(rank(o).add(rank(l)).sub(rank(h)).add(rank(c)), 8))
    s2 = ts_rank(
        ts_weighted_mean(ts_corr(ts_rank(c, 8), ts_rank(ts_mean(v, 60), 20), 8), 6), 2
    )

    return s1.where(s1 < s2, s2).stack("ticker").swaplevel()


def alpha089(l, v, vwap, industry):
    """(ts_rank(ts_weighted_mean(ts_corr(((low * 0.967285) +
    (low * (1 - 0.967285))), adv10,6.94279), 5.51607), 3.79744) -
    ts_rank(ts_weighted_mean(ts_delta(IndNeutralize(vwap,IndClass.industry), 3.48158), 10.1466), 15.3012))
    """
    pass


def alpha090(c, l, subindustry):
    """((rank((close - ts_max(close, 4.66719)))
    ^ts_rank(ts_corr(IndNeutralize(adv40,IndClass.subindustry), low, 5.38375), 3.21856)) * -1)
    """
    pass


def alpha091(v, vwap, industry):
    """((ts_rank(ts_weighted_mean(ts_weighted_mean(ts_corr(IndNeutralize(close,IndClass.industry), volume, 9.74928), 16.398), 3.83219), 4.8667) -
    rank(ts_weighted_mean(ts_corr(vwap, adv30, 4.01303), 2.6809))) * -1)
    """
    pass


def alpha092(c, h, l, o, v):
    """min(ts_rank(ts_weighted_mean(((((high + low) / 2) + close) < (low + open)), 14.7221),18.8683),
    ts_rank(ts_weighted_mean(ts_corr(rank(low), rank(adv30), 7.58555), 6.94024),6.80584))
    """
    p1 = ts_rank(ts_weighted_mean(h.add(l).div(2).add(c).lt(l.add(o)), 15), 18)
    p2 = ts_rank(ts_weighted_mean(ts_corr(rank(l), rank(ts_mean(v, 30)), 7), 6), 6)

    return p1.where(p1 < p2, p2).stack("ticker").swaplevel()


def alpha093(c, v, vwap, industry):
    """(ts_rank(ts_weighted_mean(ts_corr(IndNeutralize(vwap, IndClass.industry), adv81,17.4193), 19.848), 7.54455) /
    rank(ts_weighted_mean(ts_delta(((close * 0.524434) + (vwap * (1 -0.524434))), 2.77377), 16.2664)))
    """
    pass


def alpha094(v, vwap):
    """((rank((vwap - ts_min(vwap, 11.5783)))^ts_rank(ts_corr(ts_rank(vwap,19.6462),
    ts_rank(adv60, 4.02992), 18.0926), 2.70756)) * -1)
    """

    return (
        rank(vwap.sub(ts_min(vwap, 11)))
        .pow(ts_rank(ts_corr(ts_rank(vwap, 20), ts_rank(ts_mean(v, 60), 4), 18), 2))
        .mul(-1)
        .stack("ticker")
        .swaplevel()
    )


def alpha095(h, l, o, v):
    """(rank((open - ts_min(open, 12.4105))) <
    ts_rank((rank(ts_corr(ts_sum(((high + low)/ 2), 19.1351), ts_sum(adv40, 19.1351), 12.8742))^5), 11.7584))
    """

    return (
        rank(o.sub(ts_min(o, 12)))
        .lt(
            ts_rank(
                rank(
                    ts_corr(
                        ts_mean(h.add(l).div(2), 19), ts_sum(ts_mean(v, 40), 19), 13
                    ).pow(5)
                ),
                12,
            )
        )
        .astype(int)
        .stack("ticker")
        .swaplevel()
    )


def alpha096(c, v, vwap):
    """(max(ts_rank(ts_weighted_mean(ts_corr(rank(vwap), rank(volume), 5.83878),4.16783), 8.38151),
    ts_rank(ts_weighted_mean(ts_argmax(ts_corr(ts_rank(close, 7.45404), ts_rank(adv60, 4.13242), 3.65459), 12.6556), 14.0365), 13.4143)) * -1)
    """

    s1 = ts_rank(ts_weighted_mean(ts_corr(rank(vwap), rank(v), 10), 4), 8)
    s2 = ts_rank(
        ts_weighted_mean(
            ts_argmax(ts_corr(ts_rank(c, 7), ts_rank(ts_mean(v, 60), 10), 10), 12), 14
        ),
        13,
    )
    return s1.where(s1 > s2, s2).mul(-1).stack("ticker").swaplevel()


def alpha097(l):
    """((rank(ts_weighted_mean(ts_delta(IndNeutralize(((low * 0.721001) +
    (vwap * (1 - 0.721001))),IndClass.industry), 3.3705), 20.4523)) -
    ts_rank(ts_weighted_mean(ts_rank(ts_corr(Ts_Rank(low,7.87871),
    ts_rank(adv60, 17.255), 4.97547), 18.5925), 15.7152), 6.71659)) * -1)
    """
    pass


def alpha098(o, v, vwap):
    """(rank(ts_weighted_mean(ts_corr(vwap, ts_sum(adv5, 26.4719), 4.58418), 7.18088)) -
    rank(ts_weighted_mean(ts_tank(ts_argmin(ts_corr(rank(open),
    rank(adv15), 20.8187), 8.62571),6.95668), 8.07206)))
    """
    adv5 = ts_mean(v, 5)
    adv15 = ts_mean(v, 15)
    return (
        rank(ts_weighted_mean(ts_corr(vwap, ts_mean(adv5, 26), 4), 7))
        .sub(
            rank(
                ts_weighted_mean(
                    ts_rank(ts_argmin(ts_corr(rank(o), rank(adv15), 20), 8), 6)
                )
            )
        )
        .stack("ticker")
        .swaplevel()
    )


def alpha099(h, l, v):
    """((rank(ts_corr(ts_sum(((high + low) / 2), 19.8975),
    ts_sum(adv60, 19.8975), 8.8136)) <
    rank(ts_corr(low, volume, 6.28259))) * -1)"""

    return (
        (
            rank(ts_corr(ts_sum((h.add(l).div(2)), 19), ts_sum(ts_mean(v, 60), 19), 8))
            .lt(rank(ts_corr(l, v, 6)))
            .mul(-1)
        )
        .stack("ticker")
        .swaplevel()
    )


def alpha100(r, cap):
    """(0 - (1 * (((1.5 * scale(indneutralize(
                indneutralize(rank(((((close - low) - (high -close)) / (high - low)) * volume)),
                                IndClass.subindustry), IndClass.subindustry))) -
    scale(indneutralize((ts_corr(close, rank(adv20), 5) - rank(ts_argmin(close, 30))), IndClass.subindustry))) * (volume / adv20))))
    """
    pass


def alpha101(o, h, l, c):
    """((close - open) / ((high - low) + .001))"""
    return c.sub(o).div(h.sub(l).add(1e-3)).stack("ticker").swaplevel()
