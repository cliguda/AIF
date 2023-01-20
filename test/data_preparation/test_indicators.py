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
import pandas as pd
import pytest

import aif.data_preparation.indicators as indicators
from aif.common.config import settings
from aif.data_manangement.data_provider import DataProvider
from aif.data_manangement.definitions import Asset, Timeframe
from aif.data_manangement.price_data import PriceDataTimeframe

"""
The tests of indicators do NOT test the concrete calculations, but the correct settings of the meta-information instead.
"""


@pytest.fixture()
def dp():
    return DataProvider(initialize=False)


def test_atr(dp):
    filename = f'{settings.common.project_path}{settings.data_provider.filename_testing}'
    price_data_tf = dp.get_historical_data_from_file(filename, Asset.BTCUSD, Timeframe.HOURLY)

    columns = price_data_tf.price_data_df.columns.values
    indicators.ATR.add_indicator(price_data_tf, 14)

    # No testing of values, because its a direct implementation of python libraries
    columns = np.append(columns, ['ATR_14'])
    assert all(columns == price_data_tf.price_data_df.columns)
    assert price_data_tf.max_window == 14
    assert price_data_tf.relative_cols == []


def test_atr_bands(dp):
    filename = f'{settings.common.project_path}{settings.data_provider.filename_testing}'
    price_data_tf = dp.get_historical_data_from_file(filename, Asset.BTCUSD, Timeframe.HOURLY)

    columns = price_data_tf.price_data_df.columns.values
    indicators.ATRBands.add_indicator(price_data_tf, 14)

    columns = np.append(columns, ['ATR_Upper_14', 'ATR_Lower_14'])
    assert all(columns == price_data_tf.price_data_df.columns)
    assert price_data_tf.max_window == 14
    assert price_data_tf.relative_cols == ['ATR_Upper_14', 'ATR_Lower_14']

    # Checked on tradingview
    assert abs(price_data_tf.price_data_df.loc[price_data_tf.price_data_df.index == '2021-10-22 00:00:00',
                                           'ATR_Upper_14'][0] - 64900) < 500
    assert abs(price_data_tf.price_data_df.loc[price_data_tf.price_data_df.index == '2021-10-22 00:00:00',
                                           'ATR_Lower_14'][0] - 59800) < 100


def test_macd(dp):
    filename = f'{settings.common.project_path}{settings.data_provider.filename_testing}'
    price_data_tf = dp.get_historical_data_from_file(filename, Asset.BTCUSD, Timeframe.HOURLY)

    columns = price_data_tf.price_data_df.columns.values
    indicators.MACD.add_indicator(price_data_tf, 26)

    # No testing of values, because its a direct implementation of python libraries
    columns = np.append(columns, 'MACD_Hist')
    assert all(columns == price_data_tf.price_data_df.columns)
    assert price_data_tf.max_window == 26
    assert price_data_tf.relative_cols == []
    assert price_data_tf.price_data_df.loc[price_data_tf.price_data_df.index == '2021-10-20 21:00:00',
                                           'MACD_Hist'][0] > 0
    assert price_data_tf.price_data_df.loc[price_data_tf.price_data_df.index == '2021-10-20 22:00:00',
                                           'MACD_Hist'][0] < 0


def test_volume_relative_to_average(dp):
    filename = f'{settings.common.project_path}{settings.data_provider.filename_testing}'
    price_data_tf = dp.get_historical_data_from_file(filename, Asset.BTCUSD, Timeframe.HOURLY)

    columns = price_data_tf.price_data_df.columns.values
    indicators.VolumeRelativeToAverage.add_indicator(price_data_tf, 20)

    # No testing of values, because its a direct implementation of python libraries
    columns = np.append(columns, 'Volume_Relative_20')
    assert all(columns == price_data_tf.price_data_df.columns)
    assert price_data_tf.max_window == 20
    assert price_data_tf.relative_cols == []


def test_ema(dp):
    filename = f'{settings.common.project_path}{settings.data_provider.filename_testing}'
    price_data_tf = dp.get_historical_data_from_file(filename, Asset.BTCUSD,
                                                     Timeframe.HOURLY)
    indicators.EMA.add_indicator(price_data_tf, 20)
    indicators.EMA.add_indicator(price_data_tf, 50)

    # No testing of values, because its a direct implementation of python libaries
    assert 'EMA_20' in price_data_tf.price_data_df.columns
    assert 'EMA_50' in price_data_tf.price_data_df.columns
    assert 'EMA_20' in price_data_tf.relative_cols
    assert 'EMA_50' in price_data_tf.relative_cols
    assert price_data_tf.max_window == 50


def test_ema_slope(dp):
    filename = f'{settings.common.project_path}{settings.data_provider.filename_testing}'
    price_data_tf = dp.get_historical_data_from_file(filename, Asset.BTCUSD,
                                                     Timeframe.HOURLY)
    indicators.EMASlope.add_indicator(price_data_tf, 55, slope_window=50)

    assert 'EMA_55_Slope' in price_data_tf.price_data_df.columns
    assert price_data_tf.max_window == 55


def test_rsi(dp):
    filename = f'{settings.common.project_path}{settings.data_provider.filename_testing}'
    price_data_tf = dp.get_historical_data_from_file(filename, Asset.BTCUSD,
                                                     Timeframe.HOURLY)
    columns = price_data_tf.price_data_df.columns.values
    indicators.RSI.add_indicator(price_data_tf, 14)

    columns = np.append(columns, 'RSI_14')
    assert all(columns == price_data_tf.price_data_df.columns)
    assert price_data_tf.max_window == 14


def test_stochastic(dp):
    filename = f'{settings.common.project_path}{settings.data_provider.filename_testing}'
    price_data_tf = dp.get_historical_data_from_file(filename, Asset.BTCUSD,
                                                     Timeframe.HOURLY)
    columns = price_data_tf.price_data_df.columns.values
    indicators.Stochastic.add_indicator(price_data_tf, 14)

    columns = np.append(columns, 'Stochastic_K_14')
    columns = np.append(columns, 'Stochastic_D_14')
    assert all(columns == price_data_tf.price_data_df.columns)
    assert price_data_tf.max_window == 14


def test_bollinger_bands(dp):
    filename = f'{settings.common.project_path}{settings.data_provider.filename_testing}'
    price_data_tf = dp.get_historical_data_from_file(filename, Asset.BTCUSD,
                                                     Timeframe.HOURLY)
    columns = price_data_tf.price_data_df.columns.values
    indicators.BollingerBands.add_indicator(price_data_tf, 20)

    columns = np.append(columns, 'BB_Upper_20')
    columns = np.append(columns, 'BB_Lower_20')
    assert all(columns == price_data_tf.price_data_df.columns)
    assert price_data_tf.max_window == 20

    assert 'BB_Upper_20' in price_data_tf.relative_cols
    assert 'BB_Lower_20' in price_data_tf.relative_cols


def test_squeezing_momentum_indicators(dp):
    filename = f'{settings.common.project_path}{settings.data_provider.filename_testing}'
    price_data_tf = dp.get_historical_data_from_file(filename, Asset.BTCUSD,
                                                     Timeframe.HOURLY)
    indicators.SqueezingMomentumIndicator.add_indicator(price_data_tf, 20)

    assert 'SQZ_MNT' in price_data_tf.price_data_df.columns
    assert price_data_tf.max_window == 20


def test_keltner_channel(dp):
    filename = f'{settings.common.project_path}{settings.data_provider.filename_testing}'
    price_data_tf = dp.get_historical_data_from_file(filename, Asset.BTCUSD, Timeframe.HOURLY)

    columns = price_data_tf.price_data_df.columns.values
    indicators.KeltnerChannel.add_indicator(price_data_tf, 20)

    # No testing of values, because its a direct implementation of python libraries
    columns = np.append(columns, ['KC_Upper_20', 'KC_Lower_20'])
    assert all(columns == price_data_tf.price_data_df.columns)
    assert price_data_tf.max_window == 20
    assert price_data_tf.relative_cols == ['KC_Upper_20', 'KC_Lower_20']
    assert all(price_data_tf.price_data_df['KC_Upper_20'] >= price_data_tf.price_data_df['KC_Lower_20'])


def test_adx(dp):
    filename = f'{settings.common.project_path}{settings.data_provider.filename_testing}'
    price_data_tf = dp.get_historical_data_from_file(filename, Asset.BTCUSD, Timeframe.HOURLY)

    columns = price_data_tf.price_data_df.columns.values
    indicators.ADX.add_indicator(price_data_tf, 14)

    # No testing of values, because its a direct implementation of python libraries
    columns = np.append(columns, ['ADX_14'])
    assert all(columns == price_data_tf.price_data_df.columns)
    assert price_data_tf.max_window == 14
    assert price_data_tf.relative_cols == []

    assert min(price_data_tf.price_data_df['ADX_14']) == 0.0
    assert max(price_data_tf.price_data_df['ADX_14']) <= 100


def test_vortex(dp):
    filename = f'{settings.common.project_path}{settings.data_provider.filename_testing}'
    price_data_tf = dp.get_historical_data_from_file(filename, Asset.BTCUSD, Timeframe.HOURLY)

    columns = price_data_tf.price_data_df.columns.values
    indicators.Vortex.add_indicator(price_data_tf, 14)

    # No testing of values, because its a direct implementation of python libraries
    columns = np.append(columns, ['Vortex_14'])
    assert all(columns == price_data_tf.price_data_df.columns)
    assert price_data_tf.max_window == 14
    assert price_data_tf.relative_cols == []


def test_last_low():
    # Create data
    values = [5, 4, 3, 4.5, 5.5, 6, 5.1, 4.1, 5.2, 4.3, 3.5, 7, 8, 9, 8, 7, 6, 7, 8, 9, 10, 9]
    df = pd.DataFrame({'Open': values, 'High': values, 'Low': values, 'Close': values, 'Volume': values})
    price_data_tf = PriceDataTimeframe(price_data_df=df, timeframe=Timeframe.HOURLY, asset=Asset.BTCUSD)

    # Define command
    indicators.LastLow.add_indicator(price_data_tf, window=10)

    assert price_data_tf.max_window == 10

    assert 'Last_Low' in price_data_tf.price_data_df.columns
    assert all(np.isnan(price_data_tf.price_data_df.loc[0:14, 'Last_Low']))
    assert all(price_data_tf.price_data_df.loc[15:20, 'Last_Low'] == 3.5)
    assert all(price_data_tf.price_data_df.loc[21:, 'Last_Low'] == 6)

    indicators.LastLow.add_indicator(price_data_tf, window=10, prev=1)
    assert 'Last_Low_Prev_1' in price_data_tf.price_data_df.columns
    assert all(np.isnan(price_data_tf.price_data_df.loc[0:20, 'Last_Low_Prev_1']))
    assert all(price_data_tf.price_data_df.loc[21:, 'Last_Low_Prev_1'] == 3.5)


def test_last_high():
    # Create data
    values = [5, 4, 3, 4.5, 5.5, 6, 5.1, 4.1, 5.2, 4.3, 3.5, 7, 8, 9, 8, 7, 8, 7.5, 6.5, 6.5]
    df = pd.DataFrame({'Open': values, 'High': values, 'Low': values, 'Close': values, 'Volume': values})
    price_data_tf = PriceDataTimeframe(price_data_df=df, timeframe=Timeframe.HOURLY, asset=Asset.BTCUSD)

    # Define command
    indicators.LastHigh.add_indicator(price_data_tf, window=10)

    assert price_data_tf.max_window == 10

    assert 'Last_High' in price_data_tf.price_data_df.columns
    assert all(np.isnan(price_data_tf.price_data_df.loc[0:9, 'Last_High']))
    assert all(price_data_tf.price_data_df.loc[10:17, 'Last_High'] == 6.0)
    assert all(price_data_tf.price_data_df.loc[18:, 'Last_High'] == 9.0)

    indicators.LastHigh.add_indicator(price_data_tf, window=10, prev=1)

    assert 'Last_High_Prev_1' in price_data_tf.price_data_df.columns
    assert all(np.isnan(price_data_tf.price_data_df.loc[0:17, 'Last_High_Prev_1']))
    assert all(price_data_tf.price_data_df.loc[18:, 'Last_High_Prev_1'] == 6.0)


def test_heikin_ashi():
    df = pd.DataFrame({'Open': [1, 2], 'High': [3, 4], 'Low': [0, 1], 'Close': [2, 3], 'Volume': [0, 0]})
    price_data_tf = PriceDataTimeframe(price_data_df=df, timeframe=Timeframe.HOURLY, asset=Asset.BTCUSD)

    # Define command
    indicators.HeikinAshi.add_indicator(price_data_tf, window=1)

    assert price_data_tf.max_window == 1
    assert price_data_tf.price_data_df.loc[1, 'HA_Open'] == 1.5
    assert price_data_tf.price_data_df.loc[1, 'HA_Close'] == 2.5
    assert price_data_tf.price_data_df.loc[1, 'HA_High'] == 4
    assert price_data_tf.price_data_df.loc[1, 'HA_Low'] == 1


def test_mfi(dp):
    filename = f'{settings.common.project_path}{settings.data_provider.filename_testing}'
    price_data_tf = dp.get_historical_data_from_file(filename, Asset.BTCUSD, Timeframe.HOURLY)

    columns = price_data_tf.price_data_df.columns.values
    indicators.MFI.add_indicator(price_data_tf, window=14)

    columns = np.append(columns, 'MFI_14')
    assert all(columns == price_data_tf.price_data_df.columns)
    assert price_data_tf.max_window == 14

    assert len(price_data_tf.relative_cols) == 0
    assert np.nanmax(price_data_tf.price_data_df['MFI_14']) == 100
    assert np.nanmin(price_data_tf.price_data_df['MFI_14']) == 0
