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
from functools import partial

import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from aif import settings
from aif.common.ml.weighted_distance import WeightedDistance
from aif.data_manangement.data_provider import DataProvider
from aif.data_manangement.definitions import Asset, Timeframe
from aif.data_manangement.price_data import ENTER_TRADE_COLUMN, EXIT_TRADE_COLUMN, PriceDataComplete
from aif.data_preparation import ta
from aif.strategies import backtest
from aif.strategies.library.tpsl_classifier_preperation import mark_tpsl_signals
from aif.strategies.strategy_trading_type import TradingType
from aif.strategies.strategy import Strategy, StrategyPerformance
from aif.strategies.trade_risk_control import TradeRiskControl


@pytest.fixture()
def dp():
    return DataProvider(initialize=False)


""" Tests for rule based strategies. """


def test_get_data_with_signals():
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

    strategy.initialize(price_data=price_data, max_leverage=50)
    price_data_df = strategy._get_data_with_signals(price_data)

    assert all(price_data_df[ENTER_TRADE_COLUMN] == [False, True, False, True, False, False, True, False])
    assert all(price_data_df[EXIT_TRADE_COLUMN] == [True, False, False, False, True, False, False, True])


def test_apply_strategy():
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

    with pytest.raises(RuntimeError):
        # Strategy was not initialized
        strategy.apply_entry_strategy(price_data)

    strategy.initialize(price_data=price_data, max_leverage=50)
    # Set pseudo performance, because its not relevant
    strategy.set_performance(StrategyPerformance(win_rate=1.0, pps=1.0, performance_detailed=[]))

    order_information = strategy.apply_entry_strategy(price_data)
    assert order_information is None

    df = df.iloc[:-1]
    price_data = PriceDataComplete(price_data_df=df, timeframe=Timeframe.DAILY, asset=Asset.BTCUSD, aggregations=[])

    order_information = strategy.apply_entry_strategy(price_data)
    assert order_information is not None

    assert order_information.from_strategy == strategy
    assert order_information.asset == price_data.asset
    assert order_information.trading_type == TradingType.LONG
    assert abs(order_information.pps - 1.0) < 0.001
    assert order_information.entering_price_planned == 1060
    assert order_information.leverage == 9  # Leverage is adjusted
    assert order_information.tp_price is None
    assert abs(order_information.sl_price - 954) < 0.001


""" Test for TPSL strategy. """


def test_build_and_apply_tpsl(dp):
    settings.trading.leverage_reduction = 0.1

    filename = f'{settings.common.project_path}{settings.data_provider.filename_testing}'
    price_data_tf = dp.get_historical_data_from_file(filename, Asset.BTCUSD, Timeframe.HOURLY)
    price_data_tf.price_data_df = price_data_tf.price_data_df.iloc[:10000, :]
    price_data = PriceDataComplete.create_from_timeframe(price_data_tf, aggregations=[Timeframe.DAILY, Timeframe.WEEKLY])

    indicator_conf = {
        Timeframe.HOURLY: [
            ta.IndicatorConfiguration('EMA', 20, None),
            ta.IndicatorConfiguration('EMA', 55, None),
            ta.IndicatorConfiguration('EMASlope', 55, args={'slope_window': 4}),
            ta.IndicatorConfiguration('RSI', 14, None),
            ta.IndicatorConfiguration('BollingerBands', 20, None),
        ],
        Timeframe.DAILY: [
            ta.IndicatorConfiguration('EMA', 20, None),
            ta.IndicatorConfiguration('EMA', 55, None),
            ta.IndicatorConfiguration('EMASlope', 55, args={'slope_window': 4}),
            ta.IndicatorConfiguration('RSI', 14, None),
            ta.IndicatorConfiguration('BollingerBands', 20, None),
        ],
        Timeframe.WEEKLY: [
            ta.IndicatorConfiguration('EMA', 20, None),
            ta.IndicatorConfiguration('RSI', 14, None),
        ]
    }
    ta.add_indicators(price_data, indicator_conf)

    risk_control = TradeRiskControl(tp=0.1, sl=0.02)
    classifier = Pipeline([
        ('scalar', StandardScaler()),
        ('model', KNeighborsClassifier(n_neighbors=1))
    ])

    s = Strategy(name='TPSL KNN Classifier', trading_type=TradingType.LONG, preprocessor=[],
                 entry_signal=classifier, exit_signal=None, risk_control=risk_control,
                 convert_data_for_classifier=True,
                 prepare_classifier_data=partial(mark_tpsl_signals, tp_threshold=risk_control.tp,
                                                 sl_threshold=risk_control.sl, trading_type=TradingType.LONG))

    wd = WeightedDistance(weights=np.ones(len(price_data.get_price_data(convert=True).columns)))
    s.initialize(price_data=price_data, max_leverage=100,
                 classifier_parameters={'model__metric': wd})
    classifier_performance = backtest.evaluate_performance(strategy=s, price_data=price_data, fees_per_trade=0.0)
    s.set_performance(strategy_performance=classifier_performance)

    # Create data for testing
    price_data_t1_df = price_data.get_price_data_for_timeframe(Timeframe.HOURLY).get_price_data_df()
    price_data_t1_df = price_data_t1_df.loc[:'2016-03-05 15:00', :]
    price_data_t1 = PriceDataComplete(price_data_df=price_data_t1_df, asset=price_data.asset, timeframe=price_data.timeframe,
                                      aggregations=[Timeframe.DAILY, Timeframe.WEEKLY])
    ta.add_indicators(price_data_t1, indicator_conf)

    classification_t1 = s.apply_entry_strategy(price_data_t1)

    entry_price = price_data_t1_df['Close'][-1]
    assert classification_t1.leverage == 44
    assert classification_t1.entering_price_planned == entry_price
    assert abs(classification_t1.tp_price - (1.1 * entry_price)) < 0.0001
    assert abs(classification_t1.sl_price - (0.98 * entry_price)) < 0.0001

    price_data_t2_df = price_data.get_price_data_for_timeframe(Timeframe.HOURLY).get_price_data_df()
    price_data_t2_df = price_data_t2_df.loc[:'2016-03-05 16:00', :]
    price_data_t2 = PriceDataComplete(price_data_df=price_data_t2_df, asset=price_data.asset,
                                      timeframe=price_data.timeframe,
                                      aggregations=[Timeframe.DAILY, Timeframe.WEEKLY])
    ta.add_indicators(price_data_t2, indicator_conf)
    classification_t2 = s.apply_entry_strategy(price_data_t2)
    assert classification_t2 is None
