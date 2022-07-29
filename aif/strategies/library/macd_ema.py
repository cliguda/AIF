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

from aif.data_manangement.definitions import Timeframe
from aif.data_preparation.indicator_config import IndicatorConfiguration, PriceDataConfiguration
from aif.strategies.prep_command import Command, CommandDescription
from aif.strategies.strategy import Strategy
from aif.strategies.strategy_definitions import StrategyConfiguration
from aif.strategies.strategy_trading_type import TradingType
from aif.strategies.trade_risk_control import TradeRiskControl


def get_long_strategy_configuration() -> StrategyConfiguration:
    """
    Name: MACD + EMA Strategy
    Description: The strategy is inspired by: https://www.youtube.com/watch?v=mC5Bmz8RMW8
    Note: Original was developed for 15m, therefore it needed some changes.
    """

    # Define all relevant indicators
    indicator_conf = {
        Timeframe.HOURLY: [
            IndicatorConfiguration('EMA', 60, None),
            IndicatorConfiguration('MACD', 26, None),
            IndicatorConfiguration('ATRBands', 14, None),
        ],
    }
    price_data_configurations = PriceDataConfiguration(indicator_conf)

    # Definition of the strategy
    # Preprocessor
    preprocessor = [
        CommandDescription(Command.SHIFT, {'COLUMN': 'MACD_Hist'}),
    ]

    # Entry/Exit signals
    entry_signal = '((MACD_Hist / Close) > 0.001) & ' \
                   '(MACD_Hist_Shift_1 < 0) & ' \
                   '(Close > EMA_60) & ' \
                   '(ATR_Lower_14 < 0.99 * Close)'

    exit_signal = None

    # Risk control
    risk_control = TradeRiskControl(tp='Close + 2 * (Close - ATR_Lower_14)', sl='ATR_Lower_14')

    s = Strategy(name='MACD_EMA_Strategy',
                 trading_type=TradingType.LONG,
                 preprocessor=preprocessor,
                 entry_signal=entry_signal,
                 exit_signal=exit_signal,
                 risk_control=risk_control,
                 convert_data_for_classifier=False
                 )

    return StrategyConfiguration(price_data_configuration=price_data_configurations, strategy=s)


def get_short_strategy_configuration() -> StrategyConfiguration:
    """
    Name: MACD + EMA Strategy
    Description: The strategy is inspired by: https://www.youtube.com/watch?v=mC5Bmz8RMW8
    Note: Original was developed for 15m, therefore it needed some changes.
    """

    # Define all relevant indicators
    indicator_conf = {
        Timeframe.HOURLY: [
            IndicatorConfiguration('EMA', 100, None),
            IndicatorConfiguration('MACD', 26, None),
            IndicatorConfiguration('ATRBands', 14, None),
            IndicatorConfiguration('VolumeRelativeToAverage', 50, None),
        ],
    }
    price_data_configurations = PriceDataConfiguration(indicator_conf)

    # Definition of the strategy
    # Preprocessor
    preprocessor = [
        CommandDescription(Command.SHIFT, {'COLUMN': 'MACD_Hist'}),
    ]

    # Entry/Exit signals
    entry_signal = '((MACD_Hist / Close) < -0.0012) & ' \
                   '(MACD_Hist_Shift_1 > 0) & ' \
                   '(Close < EMA_100) & ' \
                   '(ATR_Upper_14 > 1.01 * Close) & ' \
                   '(Volume_Relative_50 > 1.2)'

    exit_signal = None

    # Risk control
    risk_control = TradeRiskControl(tp='Close - 2 * (ATR_Upper_14 - Close)', sl='ATR_Upper_14')

    s = Strategy(name='MACD_EMA_Strategy',
                 trading_type=TradingType.SHORT,
                 preprocessor=preprocessor,
                 entry_signal=entry_signal,
                 exit_signal=exit_signal,
                 risk_control=risk_control,
                 convert_data_for_classifier=False
                 )

    return StrategyConfiguration(price_data_configuration=price_data_configurations, strategy=s)
