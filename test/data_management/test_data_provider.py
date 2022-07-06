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

from datetime import datetime

import pytest

from aif.common.config import settings
from aif.data_manangement.data_provider import DataProvider
from aif.data_manangement.definitions import Asset, Timeframe
from aif.data_manangement.price_data import PriceDataComplete


@pytest.fixture()
def dp():
    return DataProvider(initialize=False)


def test_get_historical_data(dp):
    settings.data_provider.exchange = 'binance'
    price_data_tf = dp.get_historical_data(Asset.BTCUSD, Timeframe.HOURLY)
    assert len(price_data_tf.price_data_df.columns) == 5
    assert len(price_data_tf.price_data_df) > 0
    assert price_data_tf.asset == Asset.BTCUSD
    assert price_data_tf.timeframe == Timeframe.HOURLY

    settings.data_provider.exchange = 'gemini'
    price_data_tf = dp.get_historical_data(Asset.BTCUSD, Timeframe.HOURLY)
    assert len(price_data_tf.price_data_df.columns) == 5
    assert len(price_data_tf.price_data_df) > 0
    assert price_data_tf.asset == Asset.BTCUSD
    assert price_data_tf.timeframe == Timeframe.HOURLY


def test_get_historical_data_from_file(dp):
    filename = f'{settings.common.project_path}{settings.data_provider.filename_testing}'
    price_data_tf = dp.get_historical_data_from_file(filename, Asset.BTCUSD,
                                                     Timeframe.HOURLY)
    assert len(price_data_tf.price_data_df.columns) == 5
    assert len(price_data_tf.price_data_df) == 52928
    assert price_data_tf.asset == Asset.BTCUSD
    assert price_data_tf.timeframe == Timeframe.HOURLY


@pytest.mark.skipif(settings.testing.skip_update_price_data, reason='No data updating for testing.')
def test_update_price_data():

    if len(settings.binance.api_key) > 0 and len(settings.binance.api_secret) > 0:
        settings.data_provider.exchange = 'binance'
        dp = DataProvider(initialize=True)
        __check_updated_price(dp)

    settings.data_provider.exchange = 'gemini'
    dp = DataProvider(initialize=True)
    __check_updated_price(dp)


@pytest.mark.skipif(settings.testing.skip_update_historical_data, reason='No data refresh for testing.')
def test_update_historical_data(dp):
    dp = DataProvider(initialize=True)
    settings.data_provider.exchange = 'binance'
    dp.update_historical_data(asset=Asset.BTCUSD, timeframe=Timeframe.HOURLY)

    settings.data_provider.exchange = 'gemini'
    dp.update_historical_data(asset=Asset.BTCUSD, timeframe=Timeframe.HOURLY)


def __check_updated_price(dp):
    price_data_tf = dp.get_historical_data(Asset.BTCUSD, Timeframe.HOURLY)
    columns = len(price_data_tf.price_data_df.columns)
    rows = len(price_data_tf.price_data_df)

    price_data = PriceDataComplete.create_from_timeframe(price_data_tf)
    price_data = dp.get_updated_price_data(price_data, use_lookback_window=False)
    price_data_df = price_data.get_price_data(convert=False)

    assert columns == len(price_data_df.columns)
    assert rows <= len(price_data_df)

    dt_now = datetime.utcnow()
    max_date = max(price_data_df.index)
    assert dt_now.year == max_date.year
    assert dt_now.month == max_date.month
    assert dt_now.day == max_date.day
    assert dt_now.hour - 1 == max_date.hour  # The last full hour is returned.
    assert 0 == max_date.minute
