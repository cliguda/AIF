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

import pytest

from aif.common.config import settings
from aif.data_manangement.data_provider import DataProvider
from aif.data_manangement.definitions import Asset, Context, Timeframe
from aif.data_manangement.price_data import PriceData, PriceDataComplete
from aif.strategies.strategy import Strategy, StrategyPerformance
from aif.strategies.strategy_manager import StrategyManager
from aif.strategies.strategy_trading_type import TradingType
from aif.strategies.trade_risk_control import TradeRiskControl


@pytest.fixture()
def dp():
    return DataProvider(initialize=False)


def test_strategy_manager(dp):
    settings.strategies.threshold_strategy_winrate = 0.0
    settings.strategies.threshold_strategy_pps = 0.0
    settings.strategies.ignore_strategies_with_negative_folds = False

    filename = f'{settings.common.project_path}{settings.data_provider.filename_testing}'
    price_data_tf = dp.get_historical_data_from_file(filename, Asset.BTCUSD, Timeframe.HOURLY)

    price_data = PriceDataComplete.create_from_timeframe(price_data_tf)

    sm = StrategyManager()

    sm.add_entry_strategy(_get_pseudo_strategy(name='Pseudo-1', price_data=price_data,
                                               performance=StrategyPerformance(win_rate=0.1, pps=0.1,
                                                                               performance_detailed=[])),
                          asset=price_data.asset, timeframe=price_data.timeframe)

    pseudo_strategy_2 = _get_pseudo_strategy(name='Pseudo-2', price_data=price_data,
                                             performance=StrategyPerformance(win_rate=0.1, pps=0.2,
                                                                             performance_detailed=[]))
    sm.add_entry_strategy(pseudo_strategy_2, asset=price_data.asset, timeframe=price_data.timeframe)

    sm.add_entry_strategy(_get_pseudo_strategy(name='Pseudo-3', price_data=price_data,
                                               performance=StrategyPerformance(win_rate=0.1, pps=0.25,
                                                                               performance_detailed=[]),
                                               signal='False'),
                          asset=price_data.asset, timeframe=price_data.timeframe)

    sm.add_entry_strategy(_get_pseudo_strategy(name='Pseudo-4', price_data=price_data,
                                               performance=StrategyPerformance(win_rate=0.1, pps=0.15,
                                                                               performance_detailed=[])),
                          asset=price_data.asset, timeframe=price_data.timeframe)

    sm.add_entry_strategy(_get_pseudo_strategy(name='Pseudo-4b', price_data=price_data,
                                               performance=StrategyPerformance(win_rate=0.1, pps=0.15,
                                                                               performance_detailed=[])),
                          asset=Asset.ETHUSD, timeframe=price_data.timeframe)

    # Apply strategies for price data
    s = sm.apply(price_data)

    assert s is not None
    assert s.pps == 0.2
    assert s.from_strategy == pseudo_strategy_2

    # Check for correct number of strategies per Context
    assert len(sm.strategies[Context(asset=price_data.asset, timeframe=price_data.timeframe)]) == 4
    assert len(sm.strategies[Context(asset=Asset.ETHUSD, timeframe=price_data.timeframe)]) == 1

    # Testing rejection winrate
    settings.strategies.threshold_strategy_winrate = 0.5
    sm = StrategyManager()

    s = _get_pseudo_strategy(name='Pseudo-1', price_data=price_data,
                             performance=StrategyPerformance(win_rate=0.1, pps=0.1,
                                                             performance_detailed=[]))

    assert not sm.add_entry_strategy(s, asset=price_data.asset, timeframe=price_data.timeframe)

    # Testing rejection pps
    settings.strategies.threshold_strategy_winrate = 0.0
    settings.strategies.threshold_strategy_pps = 0.5

    s = _get_pseudo_strategy(name='Pseudo-1', price_data=price_data,
                             performance=StrategyPerformance(win_rate=0.1, pps=0.1,
                                                             performance_detailed=[]))
    assert not sm.add_entry_strategy(s, asset=price_data.asset, timeframe=price_data.timeframe)

    # Testing negative fold
    settings.strategies.threshold_strategy_winrate = 0.0
    settings.strategies.threshold_strategy_pps = 0.0
    settings.strategies.allowed_negative_folds = 1

    s = _get_pseudo_strategy(name='Pseudo-1', price_data=price_data,
                             performance=StrategyPerformance(win_rate=0.1, pps=0.1,
                                                             performance_detailed=[0.3, -0.1, -4.0]))
    assert not sm.add_entry_strategy(s, asset=price_data.asset, timeframe=price_data.timeframe)

    # Last test to add strategy again
    s = _get_pseudo_strategy(name='Pseudo-1', price_data=price_data,
                             performance=StrategyPerformance(win_rate=0.1, pps=0.1,
                                                             performance_detailed=[0.3, -0.1, 4.0]))
    assert sm.add_entry_strategy(s, asset=price_data.asset, timeframe=price_data.timeframe)


def _get_pseudo_strategy(name: str, price_data: PriceData, performance: StrategyPerformance,
                         signal: str = 'True'):
    s = Strategy(name=name,
                 trading_type=TradingType.LONG,
                 preprocessor=[],
                 entry_signal=signal,
                 exit_signal='False',
                 risk_control=TradeRiskControl(tp=None, sl=0.1),
                 convert_data_for_classifier=False
                 )

    s.initialize(price_data=price_data, max_leverage=1)
    s.set_performance(performance)

    return s
