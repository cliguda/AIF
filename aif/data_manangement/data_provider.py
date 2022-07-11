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
import math
import ssl

import numpy as np
import pandas as pd
import requests
from binance.client import Client
from dateutil import tz

import aif.common.logging as logging
from aif.common.config import settings
from aif.data_manangement.definitions import Asset, Timeframe
from aif.data_manangement.price_data import OHLCV_COLUMNS, PriceData, PriceDataComplete, PriceDataTimeframe


class DataProviderException(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __repr__(self):
        return self.msg


class DataProvider:
    """The DataProvider provides data for different assets on different timeframes.
    Note: The API for the historical data sometimes changes :-(."""

    def __init__(self, initialize: bool = True):
        # initialize can be set to false, if no updated data is needed (e.g. for plotting/experiments...)
        self.binance_client = None
        if initialize and settings.data_provider.exchange == 'binance' and len(settings.binance.api_key) > 0 and len(
                settings.binance.api_secret) > 0:
            logging.get_aif_logger(__name__).info('Setting up binance client.')
            self.binance_client = Client(settings.binance.api_key, settings.binance.api_secret)

    @staticmethod
    def get_historical_data(asset: Asset, timeframe: Timeframe) -> PriceDataTimeframe:
        """Method to load historical data. It's a good idea to run update_historical_data first."""
        filename_template = f'{settings.common.project_path}{settings.data_provider.filename_template}'
        filename = filename_template.format(asset=asset.name, timeframe=timeframe.name,
                                            exchange=settings.data_provider.exchange)

        try:
            price_data = pd.read_csv(filename, parse_dates=['Date'], index_col='Date')
            price_data = price_data[OHLCV_COLUMNS]
            return PriceDataTimeframe(price_data_df=price_data, asset=asset, timeframe=timeframe)
        except Exception:
            raise DataProviderException(f'No historical data found for {asset.name} {timeframe.name}.')

    @staticmethod
    def get_historical_data_from_file(filename: str, asset: Asset, timeframe: Timeframe) -> PriceDataTimeframe:
        """This method is mainly used for testing, where the data should not change any more."""
        try:
            price_data = pd.read_csv(filename, parse_dates=['Date'], index_col='Date')
            price_data = price_data[OHLCV_COLUMNS]
            return PriceDataTimeframe(price_data_df=price_data, asset=asset, timeframe=timeframe)
        except Exception:
            raise DataProviderException(f'Could not read historical data from {filename}.')

    @staticmethod
    def update_historical_data(asset: Asset, timeframe: Timeframe) -> None:
        if settings.data_provider.exchange == 'binance':
            price_data_df = DataProvider._get_historical_data_binance(asset=asset, timeframe=timeframe)
        elif settings.data_provider.exchange == 'gemini':
            price_data_df = DataProvider._get_historical_data_gemini(asset=asset, timeframe=timeframe)
        else:
            raise DataProviderException(f'Unknown exchange for price data: {settings.data_provider.exchange}')

        price_data_df = price_data_df.loc[price_data_df['Open'] != 0]
        price_data_df.sort_index(ascending=True, inplace=True)

        # Verify that the number of entries is
        verification_factor = {
            Timeframe.HOURLY: 60 * 60,
            Timeframe.DAILY: 60 * 60 * 24,
            Timeframe.WEEKLY: 60 * 60 * 24 * 7,
        }.get(timeframe)
        theoretical_size = (max(price_data_df.index) - min(price_data_df.index)).total_seconds() // verification_factor

        if (theoretical_size - len(price_data_df)) / len(price_data_df) > 0.005:
            raise DataProviderException(f'Historical data for {asset} on {timeframe} has invalid size.')

        filename_template = f'{settings.common.project_path}{settings.data_provider.filename_template}'
        filename = filename_template.format(asset=asset.name, timeframe=timeframe.name,
                                            exchange=settings.data_provider.exchange)
        price_data_df.to_csv(filename, index=True)

        logging.get_aif_logger(__name__).info(f'Updated historical data for {asset.name} on {timeframe.name}. '
                                              f'Data from {min(price_data_df.index)} until {max(price_data_df.index)}.')

    def get_updated_price_data(self, price_data: PriceData, use_lookback_window: bool) -> PriceDataComplete:
        """Returns a new PriceData instance with update price information. If use_lookback_window is True, the
           lookback_window from price_data will be used as filter (Useful to minimize the number of rows to calculate
           the indicators for)."""

        # Get basic PriceDataTimeframe and update data
        price_data_df = price_data.get_price_data(convert=False)
        price_data_df = self._update_price_data(price_data_df=price_data_df, asset=price_data.asset,
                                                timeframe=price_data.timeframe)

        # Filter by lookback window
        lookback_window = price_data.get_lookback_window()
        if use_lookback_window and lookback_window > 0:
            lookback_window_adjusted = math.floor(
                lookback_window * settings.strategies.lookback_window_adjustment_factor)
            price_data_df = price_data_df.iloc[-lookback_window_adjusted:]

        # Create PriceDataTimeframe with filtered data and create final PriceData afterwards.
        pd_current_tf = PriceDataTimeframe(price_data_df=price_data_df, asset=price_data.asset,
                                           timeframe=price_data.timeframe)
        price_data = PriceDataComplete.create_from_timeframe(price_data_tf=pd_current_tf,
                                                             aggregations=price_data.aggregations,
                                                             asset_information=price_data.asset_information)

        return price_data

    def _update_price_data(self, price_data_df: pd.DataFrame, asset: Asset, timeframe: Timeframe) -> pd.DataFrame:
        """Append current price information to price_data_df. Only MAIN_COLUMNS are used!"""
        price_data_df = price_data_df[OHLCV_COLUMNS]

        pd_current_df = None
        if settings.data_provider.exchange == 'binance':
            pd_current_df = self._get_last_binance_data(asset=asset, timeframe=timeframe)
        elif settings.data_provider.exchange == 'gemini':
            pd_current_df = DataProvider._get_last_gemini_data(asset=asset, timeframe=timeframe)

        # Filter out current hour. TODO: How to handle 5m, 15m... update?
        current_time = dt.datetime.utcnow()
        current_time_filter = np.datetime64(dt.datetime(current_time.year, current_time.month, current_time.day,
                                                        current_time.hour))
        pd_current_df = pd_current_df[pd_current_df.index != current_time_filter]

        max_historical_date = max(price_data_df.index)
        min_current_date = min(pd_current_df.index)

        if min_current_date > max_historical_date:
            raise DataProviderException('Could not update price data, historical data is too old.')

        pd_current_df = pd_current_df[pd_current_df.index >= max_historical_date]
        price_data_df = price_data_df[price_data_df.index < max_historical_date]

        price_data_df = pd.concat([price_data_df, pd_current_df]).sort_index(ascending=True)
        return price_data_df

    @staticmethod
    def _get_last_gemini_data(asset: Asset, timeframe: Timeframe) -> pd.DataFrame:
        """Download current price data from Gemini. The last entry is the price of the current hour and is not final."""

        url_param_timeframe = {
            Timeframe.HOURLY: '1hr'
        }.get(timeframe)

        base_url = f'https://api.gemini.com/v2/candles/{asset.name}/{url_param_timeframe}'

        try:
            response = requests.get(base_url)
            price_data_current = pd.DataFrame(response.json(), columns=['Date'] + OHLCV_COLUMNS)
        except Exception:
            raise DataProviderException(f'Could not load gemini data for {asset.name} {timeframe.name}')

        price_data_current['Date'] = price_data_current['Date']. \
            apply(lambda d: dt.datetime.fromtimestamp(d / 1000, tz=tz.UTC).replace(tzinfo=None))

        price_data_current.set_index('Date', inplace=True)
        return price_data_current

    def _get_last_binance_data(self, asset: Asset, timeframe: Timeframe) -> pd.DataFrame:
        """Download current price data from Binance.
           The last entry is the price of the current hour and is not final.
        """
        url_param_timeframe = {
            Timeframe.HOURLY: '1h'
        }.get(timeframe)

        asset_name = asset.name.replace('USD', 'USDT')
        start_date = str(dt.date.today() - dt.timedelta(days=2))
        try:
            data = self.binance_client.get_historical_klines(asset_name, url_param_timeframe, start_str=start_date)
            price_data_current = pd.DataFrame(data).iloc[:, 0:6]
        except Exception:
            raise DataProviderException(f'Could not load binance data for {asset.name} {timeframe.name}')

        price_data_current.columns = ['Date'] + OHLCV_COLUMNS

        price_data_current['Date'] = price_data_current['Date']. \
            apply(lambda d: dt.datetime.fromtimestamp(d // 1000, tz=tz.UTC).replace(tzinfo=None))

        price_data_current.set_index('Date', inplace=True)
        price_data_current[OHLCV_COLUMNS] = price_data_current[OHLCV_COLUMNS].astype(float)

        return price_data_current

    @staticmethod
    def _get_historical_data_gemini(asset: Asset, timeframe: Timeframe) -> pd.DataFrame:
        """Download historical data for Gemini"""
        url_param_timeframe = {
            Timeframe.HOURLY: '1h'
        }.get(timeframe)
        asset_name = asset.name

        url = f'https://www.cryptodatadownload.com/cdd/Gemini_{asset_name}_{url_param_timeframe}.csv'

        ssl._create_default_https_context = ssl._create_unverified_context
        try:
            price_data_df = pd.read_csv(url, skiprows=1, parse_dates=['date'], index_col='date')
        except Exception:
            raise DataProviderException(f'Could not update historical data for {asset.name} {timeframe.name}. ({url=})')

        vol_column = [x for x in price_data_df.columns if 'Volume' in x and x != 'Volume USD'][0]
        price_data_df = price_data_df[['open', 'high', 'low', 'close', vol_column]]
        price_data_df.columns = OHLCV_COLUMNS
        price_data_df.index.name = 'Date'

        return price_data_df

    @staticmethod
    def _get_historical_data_binance(asset: Asset, timeframe: Timeframe) -> pd.DataFrame:
        """Download historical data for Binance"""
        url_param_timeframe = {
            Timeframe.HOURLY: '1h'
        }.get(timeframe)
        asset_name = asset.name.replace('USD', 'USDT')

        url = f'https://www.cryptodatadownload.com/cdd/Binance_{asset_name}_{url_param_timeframe}.csv'
        try:
            price_data = pd.read_csv(url, skiprows=1)
        except Exception:
            raise DataProviderException(f'Could not update historical data for {asset.name} {timeframe.name}. ({url=})')

        # Data from Binance is a bit messy, therefore we need some cleanup here.
        p1 = price_data[np.isnan(price_data['tradecount'])].copy()
        p1.loc[:, 'Date'] = p1['unix'].apply(lambda d: dt.datetime.fromtimestamp(d, tz=tz.UTC).replace(tzinfo=None))
        p2 = price_data[~np.isnan(price_data['tradecount'])].copy()
        p2.loc[:, 'Date'] = p2['unix'].apply(
            lambda d: dt.datetime.fromtimestamp(d // 1000, tz=tz.UTC).replace(tzinfo=None))

        p = pd.concat([p1, p2])
        date_entries = p.groupby('Date')['Date'].count()
        date_entries.name = 'date_entries'  # Needed for merging

        p = p.set_index('Date')
        p = p.merge(date_entries, left_index=True, right_index=True)
        p['remove'] = False
        p.loc[(p['date_entries'] == 2) & (np.isnan(p['tradecount'])), 'remove'] = True
        p = p[~p['remove']]

        vol_columns = list(filter(lambda c: 'Volume' in c, p.columns))
        if 'Volume USDT' in vol_columns:
            vol_columns.remove('Volume USDT')

        if len(vol_columns) != 1:
            raise ValueError(f'No unique volume column found for {asset.name} on {timeframe.name}')

        vol_column = vol_columns[0]
        p = p[['open', 'high', 'low', 'close', vol_column]]
        p.columns = OHLCV_COLUMNS

        return p
