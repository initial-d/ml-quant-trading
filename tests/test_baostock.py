from __future__ import annotations

import unittest.mock as mock
import numpy as np
import pandas as pd
import torch
import pytest
from mlquant.data.baostock_loader import load_baostock_panel

def test_load_baostock_panel_mock():
    # Mock baostock
    with mock.patch("baostock.login") as mock_login, \
         mock.patch("baostock.logout") as mock_logout, \
         mock.patch("baostock.query_history_k_data_plus") as mock_query:
        
        # Setup mock login
        mock_login.return_value.error_code = "0"
        
        # Setup mock query results
        mock_rs = mock.Mock()
        mock_rs.error_code = "0"
        # We'll return 2 rows for each ticker
        mock_rs.next.side_effect = [True, True, False, True, True, False]
        mock_rs.get_row_data.side_effect = [
            ["2023-01-01", "sh.600000", "10.0", "11.0", "9.0", "10.5", "10.0", "1000", "10000", "1"],
            ["2023-01-02", "sh.600000", "10.5", "12.0", "10.0", "11.5", "10.5", "2000", "22000", "1"],
            ["2023-01-01", "sz.000001", "20.0", "21.0", "19.0", "20.5", "20.0", "500", "10250", "1"],
            ["2023-01-02", "sz.000001", "20.5", "22.0", "20.0", "21.5", "20.5", "600", "12600", "1"],
        ]
        mock_query.return_value = mock_rs
        
        tickers = ["sh.600000", "sz.000001", "sh.600000"] # Duplicate included
        panel = load_baostock_panel(tickers, "2023-01-01", "2023-01-02")
        
        # Check login/logout called
        mock_login.assert_called_once()
        mock_logout.assert_called_once()
        
        # Check query called for unique tickers
        assert mock_query.call_count == 2
        
        # Check panel properties
        assert panel.n_stocks == 2
        assert panel.n_dates == 2
        assert list(panel.stocks) == ["sh.600000", "sz.000001"]
        
        # Check close prices (T x N)
        expected_close = torch.tensor([[10.5, 20.5], [11.5, 21.5]], dtype=torch.float32)
        assert torch.allclose(panel.close, expected_close)

        expected_vwap = torch.tensor(
            [[10.125, 20.125], [11.0, 21.0]],
            dtype=torch.float32,
        )
        assert torch.allclose(panel.vwap, expected_vwap)
        
        # Check mask
        assert panel.mask.all()

def test_load_baostock_panel_empty_tickers():
    with pytest.raises(ValueError, match="Tickers list cannot be empty"):
        load_baostock_panel([], "2023-01-01", "2023-01-02")


def test_load_baostock_panel_rejects_raw_vwap():
    with pytest.raises(ValueError, match="adjustment-consistent raw VWAP"):
        load_baostock_panel(
            ["sh.600000"],
            "2023-01-01",
            "2023-01-02",
            proxy_vwap=False,
        )


def test_load_baostock_panel_login_fail():
    with mock.patch("baostock.login") as mock_login:
        mock_login.return_value.error_code = "1"
        mock_login.return_value.error_msg = "Login failed"
        
        with pytest.raises(RuntimeError, match="Baostock login failed: Login failed"):
            load_baostock_panel(["sh.600000"], "2023-01-01", "2023-01-02")
