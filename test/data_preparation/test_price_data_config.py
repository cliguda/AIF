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


def test_merge_price_data_configuration():
    conf = PriceDataConfiguration()

    c1 = {
        Timeframe.HOURLY: [
            IndicatorConfiguration(indicator='Ind1', window=10, args=None),
            IndicatorConfiguration(indicator='Ind2', window=10, args=None),
        ]
    }

    c2 = {
        Timeframe.HOURLY: [
            IndicatorConfiguration(indicator='Ind1', window=10, args=None),
            IndicatorConfiguration(indicator='Ind1', window=20, args=None),
            IndicatorConfiguration(indicator='Ind2', window=10, args={'arg1': 52}),
        ]

    }

    conf.merge(merge_with=c1)
    assert conf.configurations == c1

    conf.merge(merge_with=c2)
    assert len(conf.configurations) == 1
    assert len(conf.configurations.get(Timeframe.HOURLY)) == 4  # Ind 1 with window 10 is doubled
