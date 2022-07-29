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
    Name: STC + EMA Strategy
    Description: The strategy is inspired by https://www.youtube.com/watch?v=RHoXDctnCzE, but with modifications
    Note: Overall not working very good. On some assets, it has a good win-rate.
    """

    # Define all relevant indicators
    indicator_conf = {
        Timeframe.HOURLY: [
            IndicatorConfiguration('HeikinAshi', 1, None),
            IndicatorConfiguration('EMA', 200, None),
            IndicatorConfiguration('STC', 50, {'closing_col': 'HA_Close'}),
            IndicatorConfiguration('LastLow', 10, {'low_column': 'HA_Low'}),
            IndicatorConfiguration('VolumeRelativeToAverage', 50, None),
        ],
    }
    price_data_configurations = PriceDataConfiguration(indicator_conf)

    # Definition of the strategy
    # Preprocessor
    preprocessor = [
        CommandDescription(Command.SHIFT, {'COLUMN': 'STC_50'}),
        CommandDescription(Command.SHIFT, {'COLUMN': 'STC_50', 'INTERVALS': 2}),
        CommandDescription(Command.SHIFT, {'COLUMN': 'STC_50', 'INTERVALS': 3}),
    ]

    # Entry/Exit signals
    entry_signal = '(Close > 1.05 * EMA_200) ' \
                   '& (STC_50 > STC_50_Shift_1) ' \
                   '& ( ((STC_50_Shift_1 <= STC_50_Shift_2) & (STC_50_Shift_1 <= 5)) ' \
                   '   | ( (STC_50_Shift_1 >= STC_50_Shift_2) & (STC_50_Shift_2 <= 5) ' \
                   '      &(STC_50_Shift_2 <= STC_50_Shift_3)) ) ' \
                   '& (Last_Low < 0.99 * Close) ' \
                   '& (Volume_Relative_50 > 1.2) '

    exit_signal = None

    # Risk control
    risk_control = TradeRiskControl(tp='Close + 2 * (Close - Last_Low)', sl='Last_Low')

    s = Strategy(name='STC_EMA_Strategy',
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
    Name: STC + EMA Strategy
    Description: The strategy is inspired by https://www.youtube.com/watch?v=RHoXDctnCzE, but with modifications
    Note: Overall not working very good. On some assets, it has a good win-rate.
    """

    # Define all relevant indicators
    indicator_conf = {
        Timeframe.HOURLY: [
            IndicatorConfiguration('HeikinAshi', 1, None),
            IndicatorConfiguration('EMA', 200, None),
            IndicatorConfiguration('STC', 50, {'closing_col': 'HA_Close'}),
            IndicatorConfiguration('LastHigh', 10, {'low_column': 'HA_High'}),
            IndicatorConfiguration('VolumeRelativeToAverage', 50, None),
        ],
    }
    price_data_configurations = PriceDataConfiguration(indicator_conf)

    # Definition of the strategy
    # Preprocessor
    preprocessor = [
        CommandDescription(Command.SHIFT, {'COLUMN': 'STC_50'}),
        CommandDescription(Command.SHIFT, {'COLUMN': 'STC_50', 'INTERVALS': 2}),
        CommandDescription(Command.SHIFT, {'COLUMN': 'STC_50', 'INTERVALS': 3}),
    ]

    # Entry/Exit signals
    entry_signal = '(Close < 0.95 * EMA_200) ' \
                   '& (STC_50 < STC_50_Shift_1) ' \
                   '& ( ((STC_50_Shift_1 >= STC_50_Shift_2) & (STC_50_Shift_1 >= 95)) ' \
                   '   | ( (STC_50_Shift_1 <= STC_50_Shift_2) & (STC_50_Shift_2 >= 95) ' \
                   '      &(STC_50_Shift_2 >= STC_50_Shift_3)) ) ' \
                   '& (Last_High > 1.01 * Close) ' \
                   '& (Volume_Relative_50 > 1.2) '

    exit_signal = None

    # Risk control
    risk_control = TradeRiskControl(tp='Close - 2 * (Last_High - Close)', sl='Last_High')

    s = Strategy(name='STC_EMA_Strategy',
                 trading_type=TradingType.SHORT,
                 preprocessor=preprocessor,
                 entry_signal=entry_signal,
                 exit_signal=exit_signal,
                 risk_control=risk_control,
                 convert_data_for_classifier=False
                 )

    return StrategyConfiguration(price_data_configuration=price_data_configurations, strategy=s)
