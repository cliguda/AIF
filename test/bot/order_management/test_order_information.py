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

from aif.data_manangement.definitions import Asset
from aif.strategies.strategy import OrderInformation, Strategy
from aif.strategies.strategy_trading_type import TradingType
from aif.strategies.trade_risk_control import TradeRiskControl


def test__validate_order():
    pseudo_strategy_no_exit = Strategy(name='Mockup',
                                       trading_type=TradingType.LONG,
                                       preprocessor=[],
                                       entry_signal='Pseudo',
                                       exit_signal=None,
                                       risk_control=TradeRiskControl(tp=None, sl=0.1),
                                       convert_data_for_classifier=False
                                       )

    pseudo_strategy_with_exit = Strategy(name='Mockup',
                                         trading_type=TradingType.LONG,
                                         preprocessor=[],
                                         entry_signal='Pseudo',
                                         exit_signal='Pseudo',
                                         risk_control=TradeRiskControl(tp=None, sl=0.1),
                                         convert_data_for_classifier=False
                                         )

    # Valid order with TP/SL
    OrderInformation(from_strategy=pseudo_strategy_no_exit, asset=Asset.BTCUSD, trading_type=TradingType.LONG, pps=0.5,
                     entering_price_planned=1000, leverage=10, tp_price=1100, sl_price=900)

    # Valid order with np TP, but exit-strategy
    OrderInformation(from_strategy=pseudo_strategy_with_exit, asset=Asset.BTCUSD, trading_type=TradingType.LONG, pps=0.5,
                     entering_price_planned=1000, leverage=20, tp_price=None, sl_price=900, )

    # Invalid order without tp and exit-strategy
    with pytest.raises(ValueError):
        OrderInformation(from_strategy=pseudo_strategy_no_exit, asset=Asset.BTCUSD, trading_type=TradingType.LONG, pps=0.5,
                         entering_price_planned=1000, leverage=20, tp_price=None, sl_price=900)
