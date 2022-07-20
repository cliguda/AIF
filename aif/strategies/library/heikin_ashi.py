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
    Name: Heikin Ashi Candle + EMA
    Description: Strategy based on 200 EMA and the Stochastic Indicator.
    The strategy is inspired by: https://www.youtube.com/watch?v=p7ZYrxZo_38
    """

    # Define all relevant indicators
    indicator_conf = {
        Timeframe.HOURLY: [
            IndicatorConfiguration('EMA', 10, None),
            IndicatorConfiguration('EMA', 30, None),
            IndicatorConfiguration('EMA', 200, None),
            IndicatorConfiguration('LastLow', 10, None),
            IndicatorConfiguration('LastLow', 10, {'prev': 1}),
            IndicatorConfiguration('LastHigh', 10, None),
            IndicatorConfiguration('LastHigh', 10, {'prev': 1}),
            IndicatorConfiguration('HeikinAshi', 1, None),
            IndicatorConfiguration('VolumeRelativeToAverage', 50, None),
        ],
        Timeframe.FOURHOURLY: [
            IndicatorConfiguration('LastLow', 10, None),
            IndicatorConfiguration('LastLow', 10, {'prev': 1}),
            IndicatorConfiguration('LastHigh', 10, None),
            IndicatorConfiguration('LastHigh', 10, {'prev': 1}),
        ]
    }
    price_data_configurations = PriceDataConfiguration(indicator_conf)

    # Definition of the strategy
    # Preprocessor
    preprocessor = [
        CommandDescription(Command.SHIFT, {'COLUMN': 'HA_Low'}),
        CommandDescription(Command.SHIFT, {'COLUMN': 'EMA_10'}),
    ]

    # Entry/Exit signals
    entry_signal = '(Last_Low > Last_Low_Prev_1) & (Last_High > Last_High_Prev_1) & ' \
                   '(Last_Low < 0.99 * Close) & (Last_Low > 0.95 * Close) &' \
                   '(EMA_10 > EMA_30) & ' \
                   '(HA_Close > EMA_10) & ' \
                   '(HA_Low_Shift_1 < EMA_10_Shift_1) & ' \
                   '(HA_Open == HA_Low) & ' \
                   '(Volume_Relative_50 > 1.2)'

    exit_signal = '(HA_Close < HA_Open) & (HA_Open == HA_High) & (Volume_Relative_50 > 1.1)'

    # Risk control
    risk_control = TradeRiskControl(tp=None, sl='Last_Low')

    s = Strategy(name='Heikin Ashi Candle + EMA',
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
    Name: Heikin Ashi Candle + EMA
    Description: Strategy based on 200 EMA and the Stochastic Indicator.
    The strategy is inspired by: https://www.youtube.com/watch?v=p7ZYrxZo_38
    """

    # Define all relevant indicators
    indicator_conf = {
        Timeframe.HOURLY: [
            IndicatorConfiguration('EMA', 10, None),
            IndicatorConfiguration('EMA', 30, None),
            IndicatorConfiguration('EMA', 200, None),
            IndicatorConfiguration('LastLow', 10, None),
            IndicatorConfiguration('LastLow', 10, {'prev': 1}),
            IndicatorConfiguration('LastHigh', 10, None),
            IndicatorConfiguration('LastHigh', 10, {'prev': 1}),
            IndicatorConfiguration('HeikinAshi', 1, None),
            IndicatorConfiguration('VolumeRelativeToAverage', 50, None),
        ],
        Timeframe.FOURHOURLY: [
            IndicatorConfiguration('LastLow', 10, None),
            IndicatorConfiguration('LastLow', 10, {'prev': 1}),
            IndicatorConfiguration('LastHigh', 10, None),
            IndicatorConfiguration('LastHigh', 10, {'prev': 1}),
        ]
    }
    price_data_configurations = PriceDataConfiguration(indicator_conf)

    # Definition of the strategy
    # Preprocessor
    preprocessor = [
        CommandDescription(Command.SHIFT, {'COLUMN': 'HA_High'}),
        CommandDescription(Command.SHIFT, {'COLUMN': 'EMA_10'}),
    ]

    # Entry/Exit signals
    entry_signal = '(Last_Low < Last_Low_Prev_1) & (Last_High < Last_High_Prev_1) & ' \
                   '(Last_High > 1.01 * Close) & ' \
                   '(EMA_10 < EMA_30) & ' \
                   '(HA_Close < EMA_10) & ' \
                   '(HA_High_Shift_1 > EMA_10_Shift_1) & ' \
                   '(HA_Open == HA_High) & ' \
                   '(Volume_Relative_50 > 1.2)'

    exit_signal = '(HA_Close > HA_Open) & (HA_Open == HA_Low) & (Volume_Relative_50 > 1.2)'

    # Risk control
    risk_control = TradeRiskControl(tp=None, sl='Last_High')

    s = Strategy(name='Heikin Ashi Candle + EMA',
                 trading_type=TradingType.SHORT,
                 preprocessor=preprocessor,
                 entry_signal=entry_signal,
                 exit_signal=exit_signal,
                 risk_control=risk_control,
                 convert_data_for_classifier=False
                 )

    return StrategyConfiguration(price_data_configuration=price_data_configurations, strategy=s)
