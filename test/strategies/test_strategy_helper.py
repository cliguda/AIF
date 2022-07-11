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

import aif.strategies.strategy_helper
from aif.common.config import settings
from aif.bot.order_management.portfolio_information import ExchangeAssetInformation
from aif.data_manangement.definitions import Asset, Timeframe
from aif.data_manangement.price_data import PriceDataComplete
from aif.strategies.strategy import Strategy
from aif.strategies.strategy_trading_type import TradingType
from aif.strategies.trade_risk_control import TradeRiskControl


def test__get_exit_for_entry_signal():
    settings.trading.default_max_leverage = 50

    df = pd.DataFrame(
        data={
            'Open': [0, 0, 0, 0, 0, 0],
            'High': [1000, 1021, 1035, 1045, 1070, 1080],
            'Low': [990, 979, 915, 950, 951, 940],
            'Close': [1000, 1018, 1030, 1020, 1050, 1060],
            'Volume': [0, 0, 0, 0, 0, 0],
        },
        index=[
            datetime.fromisoformat('2021-01-01'), datetime.fromisoformat('2021-01-02'),
            datetime.fromisoformat('2021-01-03'), datetime.fromisoformat('2021-01-04'),
            datetime.fromisoformat('2021-01-05'), datetime.fromisoformat('2021-01-06'),
        ])
    price_data = PriceDataComplete(price_data_df=df, timeframe=Timeframe.DAILY, asset=Asset.BTCUSD, aggregations=[])
    # -- TP/SL Only
    # Test Long - SL hit
    risk_control = TradeRiskControl(tp=0.08, sl=0.04)
    strategy = Strategy(name='', trading_type=TradingType.LONG, preprocessor=[], entry_signal='False',
                        exit_signal='False', risk_control=risk_control, convert_data_for_classifier=False)
    strategy.initialize(price_data=price_data)

    price_data_df = strategy._get_data_with_signals(price_data)
    sl_price = strategy.risk_control.get_sl_price(price_data_df=price_data_df.loc[:'2021-01-01'],
                                                  trading_type=strategy.trading_type)
    tp_price = strategy.risk_control.get_tp_price(price_data_df=price_data_df.loc[:'2021-01-01'],
                                                  trading_type=strategy.trading_type)
    exit_signal = aif.strategies.strategy_helper.get_exit_for_entry_signal(price_data_for_signal_df=price_data_df,
                                                                           sl_price=sl_price,
                                                                           tp_price=tp_price,
                                                                           trading_type=strategy.trading_type)

    assert exit_signal.idx == datetime.fromisoformat('2021-01-03')
    assert exit_signal.exit_price == 960

    # Test Short - TP hit
    strategy = Strategy(name='', trading_type=TradingType.SHORT, preprocessor=[], entry_signal='False',
                        exit_signal='False', risk_control=risk_control, convert_data_for_classifier=False)
    strategy.initialize(price_data=price_data)

    price_data_df = strategy._get_data_with_signals(price_data)
    sl_price = strategy.risk_control.get_sl_price(price_data_df=price_data_df.loc[:'2021-01-01'],
                                                  trading_type=strategy.trading_type)
    tp_price = strategy.risk_control.get_tp_price(price_data_df=price_data_df.loc[:'2021-01-01'],
                                                  trading_type=strategy.trading_type)
    exit_signal = aif.strategies.strategy_helper.get_exit_for_entry_signal(price_data_for_signal_df=price_data_df,
                                                                           sl_price=sl_price,
                                                                           tp_price=tp_price,
                                                                           trading_type=strategy.trading_type)

    assert exit_signal.idx == datetime.fromisoformat('2021-01-03')
    assert exit_signal.exit_price == 920

    # -- TP/SL + Exit strategy
    # Test Long - Exit strategy hit
    risk_control = TradeRiskControl(tp=0.08, sl=0.04)
    strategy = Strategy(name='', trading_type=TradingType.LONG, preprocessor=[], entry_signal='False',
                        exit_signal='Close > 1015', risk_control=risk_control, convert_data_for_classifier=False)
    strategy.initialize(price_data=price_data)

    price_data_df = strategy._get_data_with_signals(price_data)
    sl_price = strategy.risk_control.get_sl_price(price_data_df=price_data_df.loc[:'2021-01-01'],
                                                  trading_type=strategy.trading_type)
    tp_price = strategy.risk_control.get_tp_price(price_data_df=price_data_df.loc[:'2021-01-01'],
                                                  trading_type=strategy.trading_type)
    exit_signal = aif.strategies.strategy_helper.get_exit_for_entry_signal(price_data_for_signal_df=price_data_df,
                                                                           sl_price=sl_price,
                                                                           tp_price=tp_price,
                                                                           trading_type=strategy.trading_type)

    assert exit_signal.idx == datetime.fromisoformat('2021-01-02')
    assert exit_signal.exit_price == 1018

    # Test Long - TP/SL hit before exit strategy hit
    risk_control = TradeRiskControl(tp=0.08, sl=0.02)
    strategy = Strategy(name='', trading_type=TradingType.LONG, preprocessor=[], entry_signal='False',
                        exit_signal='Close > 1015', risk_control=risk_control, convert_data_for_classifier=False)
    strategy.initialize(price_data=price_data)

    price_data_df = strategy._get_data_with_signals(price_data)
    sl_price = strategy.risk_control.get_sl_price(price_data_df=price_data_df.loc[:'2021-01-01'],
                                                  trading_type=strategy.trading_type)
    tp_price = strategy.risk_control.get_tp_price(price_data_df=price_data_df.loc[:'2021-01-01'],
                                                  trading_type=strategy.trading_type)
    exit_signal = aif.strategies.strategy_helper.get_exit_for_entry_signal(price_data_for_signal_df=price_data_df,
                                                                           sl_price=sl_price,
                                                                           tp_price=tp_price,
                                                                           trading_type=strategy.trading_type)

    assert exit_signal.idx == datetime.fromisoformat('2021-01-02')
    assert exit_signal.exit_price == 980

    # Test Short - Exit strategy hit
    risk_control = TradeRiskControl(tp=0.08, sl=0.04)
    strategy = Strategy(name='', trading_type=TradingType.SHORT, preprocessor=[], entry_signal='False',
                        exit_signal='Close > 1015', risk_control=risk_control, convert_data_for_classifier=False)
    strategy.initialize(price_data=price_data)

    price_data_df = strategy._get_data_with_signals(price_data)
    sl_price = strategy.risk_control.get_sl_price(price_data_df=price_data_df.loc[:'2021-01-01'],
                                                  trading_type=strategy.trading_type)
    tp_price = strategy.risk_control.get_tp_price(price_data_df=price_data_df.loc[:'2021-01-01'],
                                                  trading_type=strategy.trading_type)
    exit_signal = aif.strategies.strategy_helper.get_exit_for_entry_signal(price_data_for_signal_df=price_data_df,
                                                                           sl_price=sl_price,
                                                                           tp_price=tp_price,
                                                                           trading_type=strategy.trading_type)

    assert exit_signal.idx == datetime.fromisoformat('2021-01-02')
    assert exit_signal.exit_price == 1018

    # Test Short - TP/SL hit before exit strategy hit
    risk_control = TradeRiskControl(tp=0.08, sl=0.02)
    strategy = Strategy(name='', trading_type=TradingType.SHORT, preprocessor=[], entry_signal='False',
                        exit_signal='Close > 1015', risk_control=risk_control, convert_data_for_classifier=False)
    strategy.initialize(price_data=price_data)

    price_data_df = strategy._get_data_with_signals(price_data)
    sl_price = strategy.risk_control.get_sl_price(price_data_df=price_data_df.loc[:'2021-01-01'],
                                                  trading_type=strategy.trading_type)
    tp_price = strategy.risk_control.get_tp_price(price_data_df=price_data_df.loc[:'2021-01-01'],
                                                  trading_type=strategy.trading_type)
    exit_signal = aif.strategies.strategy_helper.get_exit_for_entry_signal(price_data_for_signal_df=price_data_df,
                                                                           sl_price=sl_price,
                                                                           tp_price=tp_price,
                                                                           trading_type=strategy.trading_type)

    assert exit_signal.idx == datetime.fromisoformat('2021-01-02')
    assert exit_signal.exit_price == 1020


def test__mark_sl_hit():
    df = pd.DataFrame(
        data={
            'Open': [0, 0, 0, 0, 0, 0],
            'High': [1000, 1020, 1060, 1040, 1070, 1080],
            'Low': [990, 980, 955, 945, 951, 940],
            'Close': [0, 0, 0, 0, 0, 0],
            'Volume': [0, 0, 0, 0, 0, 0],
        },
        index=[
            datetime.fromisoformat('2021-01-01'), datetime.fromisoformat('2021-01-02'),
            datetime.fromisoformat('2021-01-03'), datetime.fromisoformat('2021-01-04'),
            datetime.fromisoformat('2021-01-05'), datetime.fromisoformat('2021-01-06'),
        ])

    df['_sl_hit'] = aif.strategies.strategy_helper._mark_sl_hit(df, 950, TradingType.LONG)
    assert all(df['_sl_hit'] == [False, False, False, True, False, False])

    df['_sl_hit'] = aif.strategies.strategy_helper._mark_sl_hit(df, 1050, TradingType.SHORT)
    assert all(df['_sl_hit'] == [False, False, True, False, False, False])


def test__mark_tp_hit():
    df = pd.DataFrame(
        data={
            'Open': [0, 0, 0, 0, 0, 0],
            'High': [1000, 1020, 1060, 1040, 1070, 1080],
            'Low': [990, 980, 955, 945, 951, 940],
            'Close': [0, 0, 0, 0, 0, 0],
            'Volume': [0, 0, 0, 0, 0, 0],
        },
        index=[
            datetime.fromisoformat('2021-01-01'), datetime.fromisoformat('2021-01-02'),
            datetime.fromisoformat('2021-01-03'), datetime.fromisoformat('2021-01-04'),
            datetime.fromisoformat('2021-01-05'), datetime.fromisoformat('2021-01-06'),
        ])

    df['_tp_hit'] = aif.strategies.strategy_helper._mark_tp_hit(df, 1050, TradingType.LONG)
    assert all(df['_tp_hit'] == [False, False, True, False, False, False])

    df['_tp_hit'] = aif.strategies.strategy_helper._mark_tp_hit(df, 950, TradingType.SHORT)
    assert all(df['_tp_hit'] == [False, False, False, True, False, False])
