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
from aif.strategies.strategy_definitions import StrategyConfiguration
from aif.strategies.strategy_trading_type import TradingType
from aif.strategies.prep_command import Command, CommandDescription
from aif.strategies.strategy import Strategy
from aif.strategies.trade_risk_control import TradeRiskControl


def get_long_strategy_configuration() -> StrategyConfiguration:
    """
    Name: EMA + Stochastic Strategy
    Description: Strategy based on 200 EMA and the Stochastic Indicator. The strategy is based on
    Credits: https://www.youtube.com/watch?v=vLbLZWi_Ypc, but with modifications (Downtrend + RSI Filter).
    Status: Profitable in some areas on 1h
    """

    # Define all relevant indicators
    indicator_conf = {
        Timeframe.HOURLY: [
            IndicatorConfiguration('EMA', 200, None),
            IndicatorConfiguration('Stochastic', 14, None),
            IndicatorConfiguration('RSI', 14, None),
            IndicatorConfiguration('LastLow', 10, None),
        ]
    }
    price_data_configurations = PriceDataConfiguration(indicator_conf)

    # Definition of the strategy
    # Preprocessor
    preprocessor = [
        CommandDescription(Command.SHIFT, {'COLUMN': 'Stochastic_K_14'}),
        CommandDescription(Command.SHIFT, {'COLUMN': 'Stochastic_D_14'}),
    ]

    # Entry/Exit signals
    entry_signal = '(EMA_200 > Close) & (Stochastic_K_14 > 25) & (Stochastic_K_14_Shift_1 < 20) & ' \
                   '(Last_Low < 0.99 * Close) & (RSI_14 >= 30) & (RSI_14 <= 50)'
    exit_signal = None

    # Risk control
    risk_control = TradeRiskControl(tp='Close + 2 * (Close - Last_Low)', sl='Last_Low')

    s = Strategy(name='EMA + Stochastic Strategy',
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
    Name: EMA + Stochastic Strategy
    Description: Strategy based on 200 EMA and the Stochastic Indicator. The strategy is based on
    Credits: https://www.youtube.com/watch?v=vLbLZWi_Ypc, but with modifications (Downtrend + RSI Filter).
    Status: Profitable in some areas on 1h
    """

    # Define all relevant indicators
    indicator_conf = {
        Timeframe.HOURLY: [
            IndicatorConfiguration('EMA', 200, None),
            IndicatorConfiguration('Stochastic', 14, None),
            IndicatorConfiguration('RSI', 14, None),
            IndicatorConfiguration('Last_High', 10, None),
        ]
    }
    price_data_configurations = PriceDataConfiguration(indicator_conf)

    # Definition of the strategy
    # Preprocessor
    preprocessor = [
        CommandDescription(Command.SHIFT, {'COLUMN': 'Stochastic_K_14'}),
        CommandDescription(Command.SHIFT, {'COLUMN': 'Stochastic_D_14'}),
    ]

    # Entry/Exit signals
    entry_signal = '(EMA_200 < Close) & (Stochastic_K_14 < 75) & (Stochastic_K_14_Shift_1 > 80) & ' \
                   '(Last_High > 1.01 * Close)'
    exit_signal = None

    # Risk control
    risk_control = TradeRiskControl(tp='Close - 2 * (Last_High - Close)', sl='Last_High')

    s = Strategy(name='EMA + Stochastic Strategy',
                 trading_type=TradingType.SHORT,
                 preprocessor=preprocessor,
                 entry_signal=entry_signal,
                 exit_signal=exit_signal,
                 risk_control=risk_control,
                 convert_data_for_classifier=False
                 )

    return StrategyConfiguration(price_data_configuration=price_data_configurations, strategy=s)
