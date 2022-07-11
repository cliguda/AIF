"""
AIF - Artificial Intelligence for Finance
Copyright (C) 2022 Christian Liguda

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import pandas as pd

from aif.common.config import settings
from aif.strategies.strategy_trading_type import TradingType
from aif.strategies.trade_risk_control import TradeRiskControl


def test_get_tp_price():
    price_data_df = pd.DataFrame(data={'Close': [1000, 950, 900, 1050]})

    # Get tp by relative to closing price
    risk_control = TradeRiskControl(tp=0.04, sl=0.02)

    assert risk_control.get_tp_price(price_data_df, trading_type=TradingType.LONG) == 1092
    assert risk_control.get_tp_price(price_data_df, trading_type=TradingType.SHORT) == 1008

    # Get tp by function call
    risk_control = TradeRiskControl(tp=lambda df: df.iloc[-3]['Close'], sl=0.02)
    assert risk_control.get_tp_price(price_data_df, trading_type=TradingType.LONG) == 950
    assert risk_control.get_tp_price(price_data_df, trading_type=TradingType.SHORT) == 950

    # Get tp by column name
    risk_control = TradeRiskControl(tp='Close', sl='Close')
    assert risk_control.get_tp_price(price_data_df, trading_type=TradingType.LONG) == 1050
    assert risk_control.get_tp_price(price_data_df, trading_type=TradingType.SHORT) == 1050


def test_get_sl_price():
    price_data_df = pd.DataFrame(data={'Close': [1000, 950, 900, 1050]})

    # Get sl by relative to closing price
    risk_control = TradeRiskControl(tp=0.04, sl=0.02)

    assert risk_control.get_sl_price(price_data_df, trading_type=TradingType.LONG) == 1029
    assert risk_control.get_sl_price(price_data_df, trading_type=TradingType.SHORT) == 1071

    # Get sl by function call
    risk_control = TradeRiskControl(tp=0.04, sl=lambda df: df.iloc[-4]['Close'])
    assert risk_control.get_sl_price(price_data_df, trading_type=TradingType.LONG) == 1000
    assert risk_control.get_sl_price(price_data_df, trading_type=TradingType.SHORT) == 1000

    # Get sl by column name
    risk_control = TradeRiskControl(tp=0.04, sl='Close')
    assert risk_control.get_sl_price(price_data_df, trading_type=TradingType.LONG) == 1050
    assert risk_control.get_sl_price(price_data_df, trading_type=TradingType.SHORT) == 1050


def test_get_leverage():
    price_data_df = pd.DataFrame(data={'Close': [1000, 950, 900, 1050]})
    settings.trading.leverage_reduction = 0.1

    # Get leverage if set
    risk_control = TradeRiskControl(tp=0.04, sl=0.02)

    assert risk_control.get_leverage_from_data(price_data_df, trading_type=TradingType.LONG, max_leverage=60) == 45
    # If Leverage is max_leverage (and therefore the sl is hit before liquidation, no leverage adjustment is necessary)
    assert risk_control.get_leverage_from_data(price_data_df, trading_type=TradingType.LONG, max_leverage=40) == 40
