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

from dataclasses import dataclass

from aif.data_preparation.indicator_config import PriceDataConfiguration
from aif.strategies.strategy import Strategy


@dataclass
class StrategyConfiguration:
    """Strategies that are defined in the library use this class, to return the necessary configuration of the price
    data as well as the strategy."""
    price_data_configuration: PriceDataConfiguration
    strategy: Strategy
