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

import datetime as dt

import pytest

from aif.common.config import settings
from aif.common.ml.price_data_split import PriceDataSplit
from aif.data_manangement.data_provider import DataProvider
from aif.data_manangement.definitions import Asset, Timeframe


@pytest.fixture()
def dp():
    return DataProvider(initialize=False)


def test_split_testing_phase(dp):
    filename = f'{settings.common.project_path}{settings.data_provider.filename_testing}'
    price_data_tf = dp.get_historical_data_from_file(filename, Asset.BTCUSD,
                                                     Timeframe.HOURLY)

    price_data_df = price_data_tf.price_data_df

    # Just some asserts which make the checking of the splits more understandable
    assert min(price_data_df.index) == dt.datetime(2015, 10, 8, 14)
    assert max(price_data_df.index) == dt.datetime(2021, 10, 22)
    assert len(price_data_df) == 52928

    cv = PriceDataSplit(timeframe=Timeframe.HOURLY, validation_phase=False)

    # We set folds and fold size manually to be independent of configuration settings
    cv.folds = 3
    cv.fold_size = 720

    splits = [(idx_test, idx_train) for (idx_test, idx_train) in cv.split(price_data_df)]

    # 1st fold
    # - Training data
    assert price_data_df.index[splits[0][0][0]] == dt.datetime(2015, 10, 8, 14)
    assert price_data_df.index[splits[0][0][52928 - (3 * 720) - 1]] == dt.datetime(2021, 7, 24)

    # - Testing data (30 days after training data)
    assert price_data_df.index[splits[0][1][0]] == dt.datetime(2021, 7, 24, 1)
    assert price_data_df.index[splits[0][1][720 - 1]] == dt.datetime(2021, 8, 23)

    # 2nd fold
    # - Training data
    assert price_data_df.index[splits[1][0][0]] == dt.datetime(2015, 10, 8, 14)
    assert price_data_df.index[splits[1][0][52928 - (2 * 720) - 1]] == dt.datetime(2021, 8, 23)

    # - Testing data (30 days after training data)
    assert price_data_df.index[splits[1][1][0]] == dt.datetime(2021, 8, 23, 1)
    assert price_data_df.index[splits[1][1][720 - 1]] == dt.datetime(2021, 9, 22)

    # 3rd fold
    # - Training data
    assert price_data_df.index[splits[2][0][0]] == dt.datetime(2015, 10, 8, 14)
    assert price_data_df.index[splits[2][0][52928 - (1 * 720) - 1]] == dt.datetime(2021, 9, 22)

    # - Testing data (30 days after training data)
    assert price_data_df.index[splits[2][1][0]] == dt.datetime(2021, 9, 22, 1)
    assert price_data_df.index[splits[2][1][720 - 1]] == dt.datetime(2021, 10, 22)


def test_split_validation_phase(dp):
    filename = f'{settings.common.project_path}{settings.data_provider.filename_testing}'
    price_data_tf = dp.get_historical_data_from_file(filename, Asset.BTCUSD,
                                                     Timeframe.HOURLY)

    price_data_df = price_data_tf.price_data_df

    # Just some asserts which make the checking of the splits more understandable
    assert min(price_data_df.index) == dt.datetime(2015, 10, 8, 14)
    assert max(price_data_df.index) == dt.datetime(2021, 10, 22)
    assert len(price_data_df) == 52928

    cv = PriceDataSplit(timeframe=Timeframe.HOURLY, validation_phase=True)
    # For testing we override the configuration
    cv.folds = 2
    cv.fold_size = 720
    cv.offset = 3 * 720

    splits = [(idx_test, idx_train) for (idx_test, idx_train) in cv.split(price_data_df)]

    # 1st fold
    # - Training data
    assert price_data_df.index[splits[0][0][0]] == dt.datetime(2015, 10, 8, 14)
    assert len(splits[0][0]) == 52928 - (2 * 720) - (3 * 720)
    assert price_data_df.index[splits[0][0][52928 - (2 * 720) - (3 * 720) - 1]] == dt.datetime(2021, 5, 25)

    # - Testing data (30 days after training data)
    assert price_data_df.index[splits[0][1][0]] == dt.datetime(2021, 5, 25, 1)
    assert price_data_df.index[splits[0][1][720 - 1]] == dt.datetime(2021, 6, 24)

    # 2nd fold
    # - Training data
    assert price_data_df.index[splits[1][0][0]] == dt.datetime(2015, 10, 8, 14)
    assert price_data_df.index[splits[1][0][52928 - (1 * 720) - (3 * 720) - 1]] == dt.datetime(2021, 6, 24)

    # - Testing data (30 days after training data)
    assert price_data_df.index[splits[1][1][0]] == dt.datetime(2021, 6, 24, 1)
    assert price_data_df.index[splits[1][1][720 - 1]] == dt.datetime(2021, 7, 24)
