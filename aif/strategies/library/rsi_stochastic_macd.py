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
    Name: RSI + Stochastic + MACD Strategy
    Description: The strategy is inspired by https://www.youtube.com/watch?v=hh3BKTFE1dc
    """

    # Define all relevant indicators
    indicator_conf = {
        Timeframe.HOURLY: [
            IndicatorConfiguration('MACD', 26, None),
            IndicatorConfiguration('Stochastic', 14, None),
            IndicatorConfiguration('RSI', 14, None),
            IndicatorConfiguration('LastLow', 10, None),
        ]
    }
    price_data_configurations = PriceDataConfiguration(indicator_conf)

    # Definition of the strategy
    # Preprocessor
    preprocessor = [
        CommandDescription(Command.MARK, {'EXPR': '(Stochastic_K_14 < 20) & (Stochastic_D_14 < 20)', 'VALUE': 1,
                                          'NEW_COLUMN': 'MACD_SIGNAL'}),
        CommandDescription(Command.MARK, {'EXPR': '(Stochastic_K_14 > 80) & (Stochastic_D_14 > 80)', 'VALUE': -1,
                                          'NEW_COLUMN': 'MACD_SIGNAL'}),
        CommandDescription(Command.FFILL, {'COLUMN': 'MACD_SIGNAL'}),
        CommandDescription(Command.SHIFT, {'COLUMN': 'MACD_Hist'})
    ]

    # Entry/Exit signals
    entry_signal = '(RSI_14 > 50) & (MACD_Hist > 0) & (MACD_Hist_Shift_1 < 0) & (MACD_SIGNAL == 1) & ' \
                   '(Stochastic_K_14 > 20) & (Stochastic_D_14 > 20)'
    exit_signal = None

    # Risk control
    risk_control = TradeRiskControl(tp='Close + 1.5 * (Close - Last_Low)', sl='Last_Low')

    s = Strategy(name='RSI_+_Stochastic_+_MACD_Strategy',
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
    Name: RSI + Stochastic + MACD Strategy
    Description: The strategy is inspired by https://www.youtube.com/watch?v=hh3BKTFE1dc
    """

    # Define all relevant indicators
    indicator_conf = {
        Timeframe.HOURLY: [
            IndicatorConfiguration('MACD', 26, None),
            IndicatorConfiguration('Stochastic', 14, None),
            IndicatorConfiguration('RSI', 14, None),
            IndicatorConfiguration('LastHigh', 10, None),
        ]
    }
    price_data_configurations = PriceDataConfiguration(indicator_conf)

    # Definition of the strategy
    # Preprocessor
    preprocessor = [
        CommandDescription(Command.MARK, {'EXPR': '(Stochastic_K_14 < 20) & (Stochastic_D_14 < 20)', 'VALUE': 1,
                                          'NEW_COLUMN': 'MACD_SIGNAL'}),
        CommandDescription(Command.MARK, {'EXPR': '(Stochastic_K_14 > 80) & (Stochastic_D_14 > 80)', 'VALUE': -1,
                                          'NEW_COLUMN': 'MACD_SIGNAL'}),
        CommandDescription(Command.FFILL, {'COLUMN': 'MACD_SIGNAL'}),
        CommandDescription(Command.SHIFT, {'COLUMN': 'MACD_Hist'})
    ]

    # Entry/Exit signals
    entry_signal = '(RSI_14 < 50) & (MACD_Hist < 0) & (MACD_Hist_Shift_1 > 0) & (MACD_SIGNAL == -1) & ' \
                   '(Stochastic_K_14 < 80) & (Stochastic_D_14 < 80)'
    exit_signal = None

    # Risk control
    risk_control = TradeRiskControl(tp='Close - 1.5 * (Last_High - Close)', sl='Last_High')

    s = Strategy(name='RSI_+_Stochastic_+_MACD_Strategy',
                 trading_type=TradingType.SHORT,
                 preprocessor=preprocessor,
                 entry_signal=entry_signal,
                 exit_signal=exit_signal,
                 risk_control=risk_control,
                 convert_data_for_classifier=False
                 )

    return StrategyConfiguration(price_data_configuration=price_data_configurations, strategy=s)
