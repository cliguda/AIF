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

from collections import namedtuple
from dataclasses import dataclass

"""Equity is the total amount and used for calculating the position size. available_balance is the amount that can 
be used for opening a trade."""
WalletBalance = namedtuple('WalletBalance', ['equity', 'available_balance'])

"""Information about existing trades."""
PositionInformation = namedtuple('PositionInformation', ['position_size', 'trading_type', 'leverage'])


@dataclass
class ExchangeAssetInformation:
    """The maximum leverage as well as the fees for a market order are different, depending on the asset and exchange
    that is used."""
    max_leverage: int
    fees_market_order: float
