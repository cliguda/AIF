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


def test_timeframe_eq():
    assert Timeframe.HOURLY == Timeframe.HOURLY
    assert Timeframe.DAILY == Timeframe.DAILY
    assert Timeframe.WEEKLY == Timeframe.WEEKLY
    assert Timeframe.HOURLY != Timeframe.WEEKLY
    assert Timeframe.WEEKLY != Timeframe.HOURLY
    assert Timeframe.DAILY != Timeframe.HOURLY


def test_timeframe_convert():
    assert Timeframe.convert(Timeframe.HOURLY, Timeframe.FOURHOURLY) == 4
    assert Timeframe.convert(Timeframe.HOURLY, Timeframe.DAILY) == 24
    assert Timeframe.convert(Timeframe.DAILY, Timeframe.WEEKLY) == 7
    assert Timeframe.convert(Timeframe.HOURLY, Timeframe.WEEKLY) == 7 * 24
    assert Timeframe.convert(Timeframe.HOURLY, Timeframe.HOURLY) == 1

    try:
        # Trying invalid parameters
        Timeframe.convert(Timeframe.DAILY, Timeframe.HOURLY)
        assert False
    except ValueError:
        # This is what should happen.
        assert True
