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
from enum import Enum


class Asset(Enum):
    BTCUSD = 'BTCUSD'
    ETHUSD = 'ETHUSD'
    BNBUSD = 'BNBUSD'
    XRPUSD = 'XRPUSD'
    ADAUSD = 'ADAUSD'
    SOLUSD = 'SOLUSD'


class Timeframe(Enum):
    # Note: The name of a timeframe must be one word, no underscores, etc.
    HOURLY = 'HOURLY'
    FOURHOURLY = 'FOURHOURLY'
    DAILY = 'DAILY'
    WEEKLY = 'WEEKLY'

    def __eq__(self, other):
        if isinstance(other, Timeframe):
            return self.value == other.value
        else:
            return False

    def __hash__(self):
        return hash(self.value)

    @staticmethod
    def convert(from_timeframe: Timeframe, to_timeframe: Timeframe) -> int:
        """Calculates how many times a timeframe from_ timeframe is fitting into to_timeframe (e.g. HOURLY is fitting
        24 times into DAILY)"""
        time_frame_conversions = {
            Timeframe.HOURLY: (Timeframe.FOURHOURLY, 4),
            Timeframe.FOURHOURLY: (Timeframe.DAILY, 6),
            Timeframe.DAILY: (Timeframe.WEEKLY, 7),
        }

        res_factor = 1
        tf_factor = (from_timeframe, 1)
        while tf_factor[0] != to_timeframe:
            tf_factor = time_frame_conversions.get(tf_factor[0], None)
            if tf_factor is None:
                raise ValueError(f'Could not convert {from_timeframe.name} to {to_timeframe.name}')

            res_factor *= tf_factor[1]

        return res_factor


@dataclass(frozen=True, eq=True)
class Context:
    """ Whenever multiple assets and timeframes are involved, a list of Context objects makes organisation easier."""
    asset: Asset
    timeframe: Timeframe

    def __str__(self):
        return f'{self.asset.name} on timeframe {self.timeframe.name}'
