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

from abc import ABC, abstractmethod
from typing import Optional

from aif.bot.order_management.order_status import OrderStatus
from aif.bot.order_management.portfolio_information import ExchangeAssetInformation, PositionInformation, WalletBalance
from aif.data_manangement.definitions import Asset
from aif.strategies.strategy import OrderInformation


class Exchange(ABC):
    """Abstract base class for concrete exchange implementations, e.g. to Bybit..."""

    @abstractmethod
    def get_equity(self) -> WalletBalance:
        pass

    @abstractmethod
    def get_asset_information(self, asset: Asset) -> Optional[ExchangeAssetInformation]:
        pass

    @abstractmethod
    def get_active_positions(self, asset: Asset) -> list[PositionInformation]:
        pass

    @abstractmethod
    def place_order(self, order: OrderInformation) -> OrderStatus:
        pass

    @abstractmethod
    def exit_trade(self, asset: Asset) -> OrderStatus:
        pass
