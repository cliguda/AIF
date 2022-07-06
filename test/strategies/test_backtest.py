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

from datetime import datetime

import pandas as pd

import aif.strategies.backtest as backtest
from aif import settings
from aif.data_manangement.definitions import Asset, Timeframe
from aif.data_manangement.price_data import PriceDataComplete
from aif.strategies.strategy import Strategy
from aif.strategies.strategy_trading_type import TradingType
from aif.strategies.trade_risk_control import TradeRiskControl


def test_get_profit():
    settings.trading.leverage_reduction = 0.1

    df = pd.DataFrame(
        data={
            'Open': [0, 0, 0, 0, 0, 0, 0, 0],
            'High': [0, 0, 0, 0, 0, 0, 0, 0],
            'Low': [1000, 1055, 1045, 1065, 1015, 1030, 1060, 1015],
            'Close': [1000, 1055, 1045, 1065, 1015, 1030, 1060, 1015],
            'Volume': [0, 0, 0, 0, 0, 0, 0, 0],
        },
        index=[
            datetime.fromisoformat('2021-01-01'), datetime.fromisoformat('2021-01-02'),
            datetime.fromisoformat('2021-01-03'), datetime.fromisoformat('2021-01-04'),
            datetime.fromisoformat('2021-01-05'), datetime.fromisoformat('2021-01-06'),
            datetime.fromisoformat('2021-01-07'), datetime.fromisoformat('2021-01-08'),
        ])

    price_data = PriceDataComplete(price_data_df=df, timeframe=Timeframe.DAILY, asset=Asset.BTCUSD, aggregations=[])

    risk_control = TradeRiskControl(tp=None, sl=0.1)
    strategy = Strategy(name='', trading_type=TradingType.LONG, preprocessor=[], entry_signal='Close > 1050',
                        exit_signal='Close < 1020', risk_control=risk_control, convert_data_for_classifier=False)

    strategy.initialize(price_data=price_data, max_leverage=20)
    classifier_performance = backtest.evaluate_performance(strategy=strategy, price_data=price_data, fees_per_trade=0.0)

    adjusted_leverage = 10 * (1 - 0.1)
    assert len(classifier_performance.performance_detailed) == 2
    assert abs(classifier_performance.performance_detailed[0] - ((1015 - 1055) / 1055 * adjusted_leverage)) < 0.001
    assert abs(classifier_performance.performance_detailed[1] - ((1015 - 1060) / 1060 * adjusted_leverage)) < 0.001


def test__get_profit_for_signal():
    """ Profit by exit strategy (Relevant: Closing price on exit signal) - Long """
    # SL is only needed for inferring the leverage.
    risk_control = TradeRiskControl(tp=None, sl=0.1)
    strategy = Strategy(name='', trading_type=TradingType.LONG, preprocessor=[], entry_signal='False',
                        exit_signal='Close > 1050', risk_control=risk_control, convert_data_for_classifier=False)

    df = pd.DataFrame(data=[[900, 1100, 850, 1000, 0], [1000, 1080, 960, 1060, 0]],
                      columns=['Open', 'High', 'Low', 'Close', 'Volume'],
                      index=[datetime.fromisoformat('2021-01-01'), datetime.fromisoformat('2021-01-02')]
                      )

    price_data = PriceDataComplete(price_data_df=df, timeframe=Timeframe.DAILY, asset=Asset.BTCUSD, aggregations=[])
    strategy.initialize(price_data=price_data, max_leverage=50)
    price_data_df = strategy._get_data_with_signals(price_data)

    settings.trading.leverage_reduction = 0.1
    pl = backtest._get_profit_for_signal(strategy=strategy, price_data_df=price_data_df,
                                         idx=datetime.fromisoformat('2021-01-01'), max_leverage=20, fees_per_trade=0.0)
    assert abs(pl[0] - 0.54) < 0.001

    # Short
    strategy = Strategy(name='', trading_type=TradingType.SHORT, preprocessor=[], entry_signal='False',
                        exit_signal='Close > 1050', risk_control=risk_control, convert_data_for_classifier=False)
    strategy.initialize(price_data=price_data, max_leverage=50)
    price_data_df = strategy._get_data_with_signals(price_data)

    pl = backtest._get_profit_for_signal(strategy=strategy, price_data_df=price_data_df,
                                         idx=datetime.fromisoformat('2021-01-01'), max_leverage=20, fees_per_trade=0.0)
    assert abs(pl[0] - -0.54) < 0.001

    """Profit by TP hit"""
    # Long
    risk_control = TradeRiskControl(tp=0.07, sl=0.1)
    strategy = Strategy(name='', trading_type=TradingType.LONG, preprocessor=[], entry_signal='False',
                        exit_signal='Close > 1050', risk_control=risk_control, convert_data_for_classifier=False)
    strategy.initialize(price_data=price_data, max_leverage=50)

    df = pd.DataFrame(data=[[900, 1100, 850, 1000, 0], [1000, 1080, 960, 1060, 0]],
                      columns=['Open', 'High', 'Low', 'Close', 'Volume'],
                      index=[datetime.fromisoformat('2021-01-01'), datetime.fromisoformat('2021-01-02')]
                      )

    price_data = PriceDataComplete(price_data_df=df, timeframe=Timeframe.DAILY, asset=Asset.BTCUSD, aggregations=[])
    price_data_df = strategy._get_data_with_signals(price_data)

    pl = backtest._get_profit_for_signal(strategy=strategy, price_data_df=price_data_df,
                                         idx=datetime.fromisoformat('2021-01-01'), max_leverage=20, fees_per_trade=0.0)
    assert abs(pl[0] - 0.63) < 0.001  # TP is hit, because High > 1070

    # Short
    risk_control = TradeRiskControl(tp=0.03, sl=0.1)
    strategy = Strategy(name='', trading_type=TradingType.SHORT, preprocessor=[], entry_signal='False',
                        exit_signal='Close > 1050', risk_control=risk_control, convert_data_for_classifier=False)
    strategy.initialize(price_data=price_data, max_leverage=50)

    price_data_df = strategy._get_data_with_signals(price_data)

    pl = backtest._get_profit_for_signal(strategy=strategy, price_data_df=price_data_df,
                                         idx=datetime.fromisoformat('2021-01-01'), max_leverage=20, fees_per_trade=0.0)
    assert abs(pl[0] - 0.27) < 0.001

    """Profit by SL hit"""
    # Long
    risk_control = TradeRiskControl(tp=0.07, sl=0.1)
    strategy = Strategy(name='', trading_type=TradingType.LONG, preprocessor=[], entry_signal='False',
                        exit_signal='Close > 1050', risk_control=risk_control, convert_data_for_classifier=False)
    strategy.initialize(price_data=price_data, max_leverage=50)

    df = pd.DataFrame(data=[[900, 1100, 850, 1000, 0], [1000, 1080, 890, 1060, 0]],
                      columns=['Open', 'High', 'Low', 'Close', 'Volume'],
                      index=[datetime.fromisoformat('2021-01-01'), datetime.fromisoformat('2021-01-02')]
                      )

    price_data = PriceDataComplete(price_data_df=df, timeframe=Timeframe.DAILY, asset=Asset.BTCUSD, aggregations=[])
    price_data_df = strategy._get_data_with_signals(price_data)

    pl = backtest._get_profit_for_signal(strategy=strategy, price_data_df=price_data_df,
                                         idx=datetime.fromisoformat('2021-01-01'), max_leverage=20, fees_per_trade=0.0)
    assert abs(pl[0] - -0.9) < 0.001

    # Short
    risk_control = TradeRiskControl(tp=0.03, sl=0.1)
    strategy = Strategy(name='', trading_type=TradingType.SHORT, preprocessor=[], entry_signal='False',
                        exit_signal='Close > 1050', risk_control=risk_control, convert_data_for_classifier=False)
    strategy.initialize(price_data=price_data, max_leverage=50)
    price_data_df = strategy._get_data_with_signals(price_data)

    pl = backtest._get_profit_for_signal(strategy=strategy, price_data_df=price_data_df,
                                         idx=datetime.fromisoformat('2021-01-01'), max_leverage=20, fees_per_trade=0.0)
    assert abs(pl[0] - 0.27) < 0.001


def test__get_profit_for_trade():
    # Long
    pl = backtest._get_profit_for_trade(entry_price=1000, exit_price=1050,
                                        trading_type=TradingType.LONG, leverage=10, fees_per_trade=0.0)

    assert abs(pl - 0.5) < 0.001

    # Short
    pl = backtest._get_profit_for_trade(entry_price=1000, exit_price=1050,
                                        trading_type=TradingType.SHORT, leverage=10, fees_per_trade=0.0)

    assert abs(pl - -0.5) < 0.001
