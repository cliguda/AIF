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

from enum import Enum


class OrderStatus(Enum):
    """Returns the status from an exchange, when an order is created."""
    ACCEPTED = 'ACCEPTED'
    EXISTING_ORDER = 'EXISTING_ORDER'
    MAX_OPEN_POSITIONS = 'MAX_OPEN_POSITIONS'
    LEVERAGE_ERROR = 'LEVERAGE_ERROR'
    TRADING_TYPE_ERROR = 'TRADING_TYPE_ERROR'
    AVAILABLE_BALANCE_ERROR = 'AVAILABLE_BALANCE_ERROR'
    REQUEST_ERROR = 'REQUEST_ERROR'
    SIMULATION_ONLY = 'SIMULATION_ONLY'
    NO_TRADE_TO_CLOSE = 'NO_TRADE_TO_CLOSE'
