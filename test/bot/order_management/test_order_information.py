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
from aif.strategies.strategy import OrderInformation
from aif.strategies.strategy_trading_type import TradingType


def test__validate_order():
    order = OrderInformation(from_strategy='s', asset=Asset.BTCUSD, trading_type=TradingType.LONG, pps=0.5,
                             entering_price_planned=1000, leverage=10, tp_price=1100, sl_price=900)

    assert order._validate_leverage()

    with pytest.raises(ValueError):
        OrderInformation(from_strategy='s', asset=Asset.BTCUSD, trading_type=TradingType.LONG, pps=0.5,
                         entering_price_planned=1000, leverage=20, tp_price=1100, sl_price=900)
