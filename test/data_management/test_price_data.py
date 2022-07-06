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

import numpy as np
import pytest

from aif.common.config import settings
from aif.data_manangement.data_provider import DataProvider
from aif.data_manangement.definitions import Asset, Timeframe
from aif.data_manangement.price_data import OHLCV_COLUMNS, PriceDataComplete, PriceDataMirror


@pytest.fixture()
def dp():
    return DataProvider(initialize=False)


""" Testing PriceDataTimeframe """


def test_get_price_data_df(dp):
    filename = f'{settings.common.project_path}{settings.data_provider.filename_testing}'
    price_data_tf = dp.get_historical_data_from_file(filename, Asset.BTCUSD,
                                                     Timeframe.HOURLY)

    # Test getting plain data
    price_data_df = price_data_tf.get_price_data_df(convert_ohl=False, convert_indicators=False,
                                                    drop_close_volume_column=False)
    price_data_df = price_data_df[price_data_df.index == '2021-10-04 00:00:00']

    assert price_data_df.columns.to_list() == OHLCV_COLUMNS
    assert price_data_df['Open'].iloc[0] == 48242.26
    assert price_data_df['Low'].iloc[0] == 47887.15
    assert price_data_df['High'].iloc[0] == 48324.9
    assert price_data_df['Close'].iloc[0] == 47916.32

    # Adding two fake indicators, one as absolut and one as relative
    price_data_tf.price_data_df.loc[:, 'Indicator1'] = 100
    price_data_tf.price_data_df.loc[:, 'Indicator2'] = 40000
    price_data_tf.add_relative_column('Indicator2')

    # Test converting olh data with removing closing price as default (default)
    price_data_df = price_data_tf.get_price_data_df(convert_ohl=True, convert_indicators=False)
    price_data_df = price_data_df[price_data_df.index == '2021-10-04 00:00:00']

    assert abs(price_data_df['Open'].iloc[0] - ((48242.26 - 47916.32) / 47916.32)) < 0.000001
    assert abs(price_data_df['Low'].iloc[0] - ((47887.15 - 47916.32) / 47916.32)) < 0.000001
    assert abs(price_data_df['High'].iloc[0] - ((48324.9 - 47916.32) / 47916.32)) < 0.000001
    assert 'Close' not in price_data_df.columns

    assert price_data_df['Indicator1'].iloc[0] == 100
    assert price_data_df['Indicator2'].iloc[0] == 40000

    # Test converting olh data without removing closing price
    price_data_df = price_data_tf.get_price_data_df(convert_ohl=True, convert_indicators=False,
                                                    drop_close_volume_column=False)
    price_data_df = price_data_df[price_data_df.index == '2021-10-04 00:00:00']
    assert price_data_df['Close'].iloc[0] == 47916.32

    # Test converting indicators
    price_data_df = price_data_tf.get_price_data_df(convert_ohl=True, convert_indicators=True,
                                                    drop_close_volume_column=False)
    price_data_df = price_data_df[price_data_df.index == '2021-10-04 00:00:00']
    assert price_data_df['Indicator1'].iloc[0] == 100
    assert abs(price_data_df['Indicator2'].iloc[0] - ((40000 - 47916.32) / 47916.32)) < 0.000001


""" Testing PriceData """


def test_create_price_data(dp):
    filename = f'{settings.common.project_path}{settings.data_provider.filename_testing}'
    price_data_tf = dp.get_historical_data_from_file(filename, Asset.BTCUSD,
                                                     Timeframe.HOURLY)
    price_data_df = price_data_tf.get_price_data_df(convert_ohl=False, convert_indicators=False)
    price_data_df = price_data_df[(price_data_df.index >= '2021-10-01 15:00:00') &
                                  (price_data_df.index < '2021-10-18 00:00:00')]

    price_data = PriceDataComplete(price_data_df=price_data_df, asset=price_data_tf.asset,
                                   timeframe=price_data_tf.timeframe,
                                   aggregations=[Timeframe.FOURHOURLY, Timeframe.DAILY, Timeframe.WEEKLY])

    assert price_data.get_price_data_for_timeframe(Timeframe.HOURLY) is not None
    assert str(min(price_data.get_price_data_for_timeframe(Timeframe.HOURLY).price_data_df.index)) == \
           '2021-10-04 00:00:00'
    pd_h = price_data.get_price_data_for_timeframe(Timeframe.HOURLY).get_price_data_df(convert_ohl=False,
                                                                                       convert_indicators=False)

    assert price_data.get_price_data_for_timeframe(Timeframe.FOURHOURLY) is not None
    assert str(min(price_data.get_price_data_for_timeframe(Timeframe.FOURHOURLY).price_data_df.index)) == \
           '2021-10-04 00:00:00'
    pd_4h = price_data.get_price_data_for_timeframe(Timeframe.FOURHOURLY).get_price_data_df(convert_ohl=False,
                                                                                            convert_indicators=False)
    assert len(pd_4h) == 84

    assert pd_4h[pd_4h.index == '2021-10-04 00:00:00']['Open'].iloc[0] == \
           pd_h[pd_h.index == '2021-10-04 00:00:00']['Open'].iloc[0]
    assert pd_4h[pd_4h.index == '2021-10-04 00:00:00']['Close'].iloc[0] == \
           pd_h[pd_h.index == '2021-10-04 03:00:00']['Close'].iloc[0]

    assert price_data.get_price_data_for_timeframe(Timeframe.DAILY) is not None
    assert str(min(price_data.get_price_data_for_timeframe(Timeframe.DAILY).price_data_df.index)) == \
           '2021-10-04 00:00:00'
    pd_d = price_data.get_price_data_for_timeframe(Timeframe.DAILY).get_price_data_df(convert_ohl=False,
                                                                                      convert_indicators=False)
    assert len(pd_d) == 14

    assert pd_d[pd_d.index == '2021-10-04 00:00:00']['Open'].iloc[0] == \
           pd_h[pd_h.index == '2021-10-04 00:00:00']['Open'].iloc[0]
    assert pd_d[pd_d.index == '2021-10-04 00:00:00']['Close'].iloc[0] == \
           pd_h[pd_h.index == '2021-10-04 23:00:00']['Close'].iloc[0]

    assert price_data.get_price_data_for_timeframe(Timeframe.WEEKLY) is not None
    assert str(min(price_data.get_price_data_for_timeframe(Timeframe.WEEKLY).price_data_df.index)) == \
           '2021-10-04 00:00:00'
    pd_w = price_data.get_price_data_for_timeframe(Timeframe.WEEKLY).get_price_data_df(convert_ohl=False,
                                                                                       convert_indicators=False)
    assert len(pd_w) == 2

    assert pd_w[pd_w.index == '2021-10-04 00:00:00']['Open'].iloc[0] == \
           pd_h[pd_h.index == '2021-10-04 00:00:00']['Open'].iloc[0]
    assert pd_w[pd_w.index == '2021-10-04 00:00:00']['Close'].iloc[0] == \
           pd_h[pd_h.index == '2021-10-10 23:00:00']['Close'].iloc[0]


def test_create_from_timeframe(dp):
    filename = f'{settings.common.project_path}{settings.data_provider.filename_testing}'
    price_data_tf = dp.get_historical_data_from_file(filename, Asset.BTCUSD,
                                                     Timeframe.HOURLY)
    price_data_df = price_data_tf.get_price_data_df(convert_ohl=False, convert_indicators=False)
    price_data_df = price_data_df[(price_data_df.index >= '2021-10-04 00:00:00') &
                                  (price_data_df.index < '2021-10-18 00:00:00')]
    price_data_tf.price_data_df = price_data_df

    price_data = PriceDataComplete.create_from_timeframe(price_data_tf, aggregations=[Timeframe.DAILY, Timeframe.WEEKLY])

    assert price_data.get_price_data_for_timeframe(Timeframe.HOURLY) is not None
    pd_h = price_data.get_price_data_for_timeframe(Timeframe.HOURLY).get_price_data_df(convert_ohl=False,
                                                                                       convert_indicators=False)

    assert price_data.get_price_data_for_timeframe(Timeframe.DAILY) is not None
    pd_d = price_data.get_price_data_for_timeframe(Timeframe.DAILY).get_price_data_df(convert_ohl=False,
                                                                                      convert_indicators=False)
    assert len(pd_d) == 14

    assert pd_d[pd_d.index == '2021-10-04 00:00:00']['Open'].iloc[0] == \
           pd_h[pd_h.index == '2021-10-04 00:00:00']['Open'].iloc[0]
    assert pd_d[pd_d.index == '2021-10-04 00:00:00']['Close'].iloc[0] == \
           pd_h[pd_h.index == '2021-10-04 23:00:00']['Close'].iloc[0]

    assert price_data.get_price_data_for_timeframe(Timeframe.WEEKLY) is not None
    pd_w = price_data.get_price_data_for_timeframe(Timeframe.WEEKLY).get_price_data_df(convert_ohl=False,
                                                                                       convert_indicators=False)
    assert len(pd_w) == 2

    assert pd_w[pd_w.index == '2021-10-04 00:00:00']['Open'].iloc[0] == \
           pd_h[pd_h.index == '2021-10-04 00:00:00']['Open'].iloc[0]
    assert pd_w[pd_w.index == '2021-10-04 00:00:00']['Close'].iloc[0] == \
           pd_h[pd_h.index == '2021-10-10 23:00:00']['Close'].iloc[0]


def test_get_price_data_without_converting(dp):
    filename = f'{settings.common.project_path}{settings.data_provider.filename_testing}'
    price_data_tf = dp.get_historical_data_from_file(filename, Asset.BTCUSD,
                                                     Timeframe.HOURLY)

    # Filter data
    price_data_df = price_data_tf.get_price_data_df(convert_ohl=False, convert_indicators=False)
    price_data_df = price_data_df[(price_data_df.index >= '2021-10-04 00:00:00') &
                                  (price_data_df.index < '2021-10-18 00:00:00')]

    price_data = PriceDataComplete(price_data_df=price_data_df, asset=price_data_tf.asset,
                                   timeframe=price_data_tf.timeframe, aggregations=[Timeframe.DAILY, Timeframe.WEEKLY])

    # Adding two fake indicators (one as absolut and one as relative) on hourly and daily data
    price_data_h = price_data.get_price_data_for_timeframe(Timeframe.HOURLY)
    price_data_h.price_data_df.loc[:, 'Indicator1'] = 100
    price_data_h.price_data_df.loc[:, 'Indicator2'] = 40000
    price_data_h.add_relative_column('Indicator2')

    price_data_d = price_data.get_price_data_for_timeframe(Timeframe.DAILY)
    price_data_d.price_data_df.loc[:, 'Indicator3'] = 200
    price_data_d.price_data_df.loc[:, 'Indicator4'] = 50000
    price_data_d.add_relative_column('Indicator4')

    # Get merged data and start testing
    pd_merged = price_data.get_price_data(convert=False)

    # Test correct merging of daily data
    assert pd_merged[pd_merged.index == '2021-10-04 00:00:00']['Open'].iloc[0] == \
           pd_merged[pd_merged.index == '2021-10-05 00:00:00']['Open_DAILY'].iloc[0]

    assert pd_merged[pd_merged.index == '2021-10-04 23:00:00']['Close'].iloc[0] == \
           pd_merged[pd_merged.index == '2021-10-05 00:00:00']['Close_DAILY'].iloc[0]

    # Test filling of merged daily data
    assert pd_merged[pd_merged.index == '2021-10-04 00:00:00']['Open'].iloc[0] == \
           pd_merged[pd_merged.index == '2021-10-05 10:00:00']['Open_DAILY'].iloc[0]

    assert pd_merged[pd_merged.index == '2021-10-04 23:00:00']['Close'].iloc[0] == \
           pd_merged[pd_merged.index == '2021-10-05 10:00:00']['Close_DAILY'].iloc[0]

    # Test correct merging of weekly data
    assert pd_merged[pd_merged.index == '2021-10-04 00:00:00']['Open'].iloc[0] == \
           pd_merged[pd_merged.index == '2021-10-11 00:00:00']['Open_WEEKLY'].iloc[0]

    assert pd_merged[pd_merged.index == '2021-10-10 23:00:00']['Close'].iloc[0] == \
           pd_merged[pd_merged.index == '2021-10-11 00:00:00']['Close_WEEKLY'].iloc[0]

    # Test filling of merged weekly data
    assert pd_merged[pd_merged.index == '2021-10-04 00:00:00']['Open'].iloc[0] == \
           pd_merged[pd_merged.index == '2021-10-13 10:00:00']['Open_WEEKLY'].iloc[0]

    assert pd_merged[pd_merged.index == '2021-10-10 23:00:00']['Close'].iloc[0] == \
           pd_merged[pd_merged.index == '2021-10-13 10:00:00']['Close_WEEKLY'].iloc[0]

    # Testing the indicators
    assert pd_merged[pd_merged.index == '2021-10-05 00:00:00']['Indicator1'].iloc[0] == 100
    assert 'Indicator1_DAILY' not in pd_merged.columns
    assert pd_merged[pd_merged.index == '2021-10-05 00:00:00']['Indicator2'].iloc[0] == 40000
    assert 'Indicator2_DAILY' not in pd_merged.columns
    assert pd_merged[pd_merged.index == '2021-10-05 00:00:00']['Indicator3_DAILY'].iloc[0] == 200
    assert 'Indicator3' not in pd_merged.columns
    assert pd_merged[pd_merged.index == '2021-10-05 00:00:00']['Indicator4_DAILY'].iloc[0] == 50000
    assert 'Indicator4' not in pd_merged.columns


def test_get_price_data_with_converting(dp):
    filename = f'{settings.common.project_path}{settings.data_provider.filename_testing}'
    price_data_tf = dp.get_historical_data_from_file(filename, Asset.BTCUSD,
                                                     Timeframe.HOURLY)

    # Filter data
    price_data_df = price_data_tf.get_price_data_df(convert_ohl=False, convert_indicators=False)
    price_data_df = price_data_df[(price_data_df.index >= '2021-10-04 00:00:00') &
                                  (price_data_df.index < '2021-10-18 00:00:00')]

    price_data = PriceDataComplete(price_data_df=price_data_df, asset=price_data_tf.asset,
                                   timeframe=price_data_tf.timeframe, aggregations=[Timeframe.DAILY, Timeframe.WEEKLY])

    # Adding two fake indicators (one as absolut and one as relative) on hourly and daily data
    price_data_h = price_data.get_price_data_for_timeframe(Timeframe.HOURLY)
    price_data_h.price_data_df.loc[:, 'Indicator1'] = 100
    price_data_h.price_data_df.loc[:, 'Indicator2'] = 40000
    price_data_h.add_relative_column('Indicator2')

    price_data_d = price_data.get_price_data_for_timeframe(Timeframe.DAILY)
    price_data_d.price_data_df.loc[:, 'Indicator3'] = 200
    price_data_d.price_data_df.loc[:, 'Indicator4'] = 50000
    price_data_d.add_relative_column('Indicator4')

    # Get merged data and start testing
    pd_merged = price_data.get_price_data(convert=True)
    assert 'Close' not in pd_merged.columns

    # Testing indicators (Closing price: 49537.52)
    pd_merged = pd_merged[pd_merged.index == '2021-10-05 00:00:00']
    assert pd_merged['Indicator1'].iloc[0] == 100
    assert abs(pd_merged['Indicator2'].iloc[0] - ((40000 - 49537.52) / 49537.52)) < 0.000001

    assert pd_merged['Indicator3_DAILY'].iloc[0] == 200
    assert abs(pd_merged['Indicator4_DAILY'].iloc[0] - ((50000 - 49537.52) / 49537.52)) < 0.000001


def test_update_max_window(dp):
    filename = f'{settings.common.project_path}{settings.data_provider.filename_testing}'
    price_data_tf = dp.get_historical_data_from_file(filename, Asset.BTCUSD,
                                                     Timeframe.HOURLY)

    price_data = PriceDataComplete.create_from_timeframe(price_data_tf, aggregations=[Timeframe.DAILY, Timeframe.WEEKLY])

    price_data.get_price_data_for_timeframe(Timeframe.HOURLY).update_max_window(30)
    assert price_data.get_lookback_window() == 31

    price_data.get_price_data_for_timeframe(Timeframe.HOURLY).update_max_window(60)
    assert price_data.get_lookback_window() == 61

    price_data.get_price_data_for_timeframe(Timeframe.DAILY).update_max_window(3)
    assert price_data.get_lookback_window() == 97

    price_data.get_price_data_for_timeframe(Timeframe.WEEKLY).update_max_window(1)
    assert price_data.get_lookback_window() == (24 * 7 * 2) + 1


def test_price_data_wrapper(dp):
    filename = f'{settings.common.project_path}{settings.data_provider.filename_testing}'
    price_data_tf = dp.get_historical_data_from_file(filename, Asset.BTCUSD, Timeframe.HOURLY)

    price_data = PriceDataComplete.create_from_timeframe(price_data_tf=price_data_tf, aggregations=[Timeframe.DAILY,
                                                                                                    Timeframe.WEEKLY])

    # Adding two fake indicators (one as absolut and one as relative) on hourly and daily data
    price_data_h = price_data.get_price_data_for_timeframe(Timeframe.HOURLY)
    price_data_h.price_data_df.loc[:, 'Indicator1'] = 100
    price_data_h.price_data_df.loc[:, 'Indicator2'] = 40000
    price_data_h.add_relative_column('Indicator2')

    price_data_d = price_data.get_price_data_for_timeframe(Timeframe.DAILY)
    price_data_d.price_data_df.loc[:, 'Indicator3'] = 200
    price_data_d.price_data_df.loc[:, 'Indicator4'] = 50000
    price_data_d.add_relative_column('Indicator4')

    price_data_wrapped = PriceDataMirror(price_data=price_data)

    # Get merged data and start testing
    df_converted = price_data.get_price_data(convert=True)
    df_unconverted = price_data.get_price_data(convert=False)

    assert df_converted.equals(price_data_wrapped.get_price_data(convert=True))
    assert df_unconverted.equals(price_data_wrapped.get_price_data(convert=False))

    index = np.array([52836 + i for i in range(0, 10)])

    price_data_wrapped = PriceDataMirror(price_data=price_data, idx_filter=index)

    assert df_converted.iloc[-10:, ].equals(price_data_wrapped.get_price_data(convert=True))
