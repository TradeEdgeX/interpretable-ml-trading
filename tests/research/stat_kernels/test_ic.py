import pandas as pd

from src.research.stat_kernels.ic import ic_decay_rows, shift_target_by_horizon


def test_shift_target_by_horizon():
    df = pd.DataFrame({"forward_rr": [1.0, 2.0, 3.0, 4.0]})
    y = shift_target_by_horizon(df["forward_rr"], 2, df)
    assert y.iloc[0] == 3.0
    assert pd.isna(y.iloc[-1])


def test_shift_target_by_horizon_underscore_symbol():
    df = pd.DataFrame(
        {
            "_symbol": ["A", "A", "A", "B", "B", "B"],
            "forward_rr": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        }
    )
    y = shift_target_by_horizon(df["forward_rr"], 2, df)
    assert y.iloc[0] == 3.0
    assert y.iloc[3] == 30.0


def test_ic_decay_rows_with_shift():
    df = pd.DataFrame(
        {
            "feat": [1.0, 2.0, 3.0, 4.0, 5.0] * 25,
            "forward_rr": list(range(125)),
        }
    )
    rows = ic_decay_rows(df, ["feat"], [1, 3], "forward_rr")
    h3 = [r for r in rows if r["horizon"] == 3][0]
    assert h3["shifted"] is True
    assert "shift" in h3["target_col"]
