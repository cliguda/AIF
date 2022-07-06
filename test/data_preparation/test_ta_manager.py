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

import aif.data_preparation.ta as ta
from aif.common.config import settings
from aif.data_manangement.data_provider import DataProvider
from aif.data_manangement.definitions import Asset, Timeframe
from aif.data_manangement.price_data import PriceDataComplete


@pytest.fixture()
def dp():
    return DataProvider(initialize=False)


def test_add_indicators(dp):
    filename = f'{settings.common.project_path}{settings.data_provider.filename_testing}'
    price_data_tf = dp.get_historical_data_from_file(filename, Asset.BTCUSD, Timeframe.HOURLY)

    price_data = PriceDataComplete.create_from_timeframe(price_data_tf, aggregations=[Timeframe.DAILY, Timeframe.WEEKLY])

    indicator_conf = {
        Timeframe.HOURLY: [
            ta.IndicatorConfiguration('EMA', 20, None),
            ta.IndicatorConfiguration('EMA', 55, None),
        ],
        Timeframe.DAILY: [
            ta.IndicatorConfiguration('EMA', 7, None),
        ],
    }
    ta.add_indicators(price_data, indicator_conf)

    assert 'EMA_20' in price_data.get_price_data_for_timeframe(Timeframe.HOURLY).price_data_df.columns
    assert 'EMA_55' in price_data.get_price_data_for_timeframe(Timeframe.HOURLY).price_data_df.columns
    assert 'EMA_7' in price_data.get_price_data_for_timeframe(Timeframe.DAILY).price_data_df.columns
