from __future__ import annotations

import unittest.mock as mock

import pandas as pd
import pytest
import torch

from mlquant.data import make_panel
from mlquant.data.akshare_loader import load_akshare_panel


def test_load_akshare_panel_mock():
    # 构造包含 akshare 真实中文列名的模拟数据
    def make_mock_df(*, symbol, period, start_date, end_date, adjust):
        assert symbol in {"600519", "000001"}
        assert period == "daily"
        assert start_date == "20230101"
        assert end_date == "20230104"
        assert adjust == "qfq"
        return pd.DataFrame(
            {
                "日期": ["2023-01-03", "2023-01-04"],
                "开盘": [10.0, 10.5],
                "收盘": [10.5, 11.5],
                "最高": [11.0, 12.0],
                "最低": [9.0, 10.0],
                "成交量": [1000, 2000],
                "成交额": [10000, 22000],
                "振幅": [20.0, 19.0],
                "涨跌幅": [5.0, 9.5],
                "涨跌额": [0.5, 1.0],
                "换手率": [1.0, 2.0],
            }
        )

    with mock.patch("akshare.stock_zh_a_hist") as mock_hist:
        mock_hist.side_effect = make_mock_df

        tickers = ["600519", "000001", "600519"]  # 包含重复代码
        panel = load_akshare_panel(tickers, "2023-01-01", "2023-01-04")

        # 去重后只调用两次 akshare
        assert mock_hist.call_count == 2

        # 检查 Panel 维度和股票顺序
        assert panel.n_stocks == 2
        assert panel.n_dates == 2
        assert list(panel.stocks) == ["600519", "000001"]

        expected_close = torch.tensor(
            [[10.5, 10.5], [11.5, 11.5]], dtype=torch.float32
        )
        assert torch.allclose(panel.close, expected_close)

        expected_vwap = torch.tensor(
            [[10.125, 10.125], [11.0, 11.0]], dtype=torch.float32
        )
        assert torch.allclose(panel.vwap, expected_vwap)

        assert panel.mask.all()


def test_load_akshare_panel_empty_tickers():
    with pytest.raises(ValueError, match="Tickers list cannot be empty"):
        load_akshare_panel([], "2023-01-01", "2023-01-04")


def test_make_panel_dispatches_to_akshare():
    expected_panel = mock.sentinel.panel
    with mock.patch("mlquant.data.load_akshare_panel", return_value=expected_panel) as loader:
        panel = make_panel(
            source="akshare",
            tickers=["600519", "000001"],
            start="2023-01-01",
            end="2023-01-04",
            adjust="hfq",
        )

    assert panel is expected_panel
    loader.assert_called_once_with(
        ["600519", "000001"],
        "2023-01-01",
        "2023-01-04",
        adjust="hfq",
    )
