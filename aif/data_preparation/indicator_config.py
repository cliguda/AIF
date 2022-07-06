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

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

from aif.data_manangement.definitions import Timeframe


@dataclass
class IndicatorConfiguration:
    """Contains all information to add an indicator."""
    indicator: str  # e.g. EMA
    window: int     # e.g. 20 - Must be the longest window, if different are used (e.g. the Stochastic has 2 windows)
    args: Optional[dict[str, object]]   # Some indicators need more then one argument (e.g. the Stochastic)


class PriceDataConfiguration:
    """Contains the complete configuration of all indicators for all timeframes. Because different strategies can be
    applied to a PriceData object, different indicators are necessary. Therefore merging of different configurations
    is also possible, to get a complete configuration based on different strategies."""

    def __init__(self, conf: Optional[dict[Timeframe, list[IndicatorConfiguration]]] = None):
        if conf is None:
            self.configurations: dict[Timeframe, list[IndicatorConfiguration]] = {}
        else:
            self.configurations = conf

    def merge(self, merge_with: Union[PriceDataConfiguration, dict[Timeframe, list[IndicatorConfiguration]]]):
        if isinstance(merge_with, PriceDataConfiguration):
            merging_conf = merge_with.configurations
        else:
            merging_conf = merge_with

        for tf in merging_conf.keys():
            if tf not in self.configurations.keys():
                self.configurations[tf] = []

            for indicator_conf in merging_conf[tf]:
                if indicator_conf not in self.configurations[tf]:
                    self.configurations[tf].append(indicator_conf)
