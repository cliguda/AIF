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

from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Optional, Union

import numpy as np
import pandas as pd

from aif.data_manangement.definitions import Asset, Timeframe
from aif.data_preparation.indicator_config import IndicatorConfiguration

OHLCV_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']
ENTER_TRADE_COLUMN = 'Enter_Trade_Signal'
EXIT_TRADE_COLUMN = 'Exit_Trade_Signal'


# Internal helping function


def _convert_to_relative(price_data_df: pd.DataFrame, column: str) -> None:
    price_data_df.loc[:, column] = (price_data_df[column] - price_data_df['Close']) / price_data_df['Close']


class PriceDataException(Exception):
    """A PriceDataException should be raised, if the overall status of PriceData/PriceDataTimeframe/... is invalid."""
    def __init__(self, msg):
        self.msg = msg

    def __repr__(self):
        return self.msg

# Classes to abstract from a DataFrame


class PriceDataTimeframe:
    """
    Contains the price data (as Dataframe) of one asset at one timeframe as well as some meta-information. This class
    is mostly used for constructing a PriceData object, which is the main object to use.
    """

    def __init__(self, price_data_df: pd.DataFrame, asset: Asset, timeframe: Timeframe):
        if not all([c in price_data_df.columns for c in OHLCV_COLUMNS]):
            raise PriceDataException("Invalid price data. Missing main columns.")

        self.price_data_df = price_data_df
        self.asset = asset
        self.timeframe = timeframe
        self.relative_cols: list[str] = []
        self.indicator_configurations: list[IndicatorConfiguration] = []
        self.max_window = 0

    def add_relative_column(self, column_name: Union[str, list[str]]) -> None:
        """Some indicators that are added to the price data, can also be represented as relative to the closing price
         (e.g. EMA can be converted as relative to the closing price, RSI cannot). """
        columns = [column_name] if isinstance(column_name, str) else column_name

        for c in columns:
            if c in self.price_data_df.columns:
                self.relative_cols.append(c)
            else:
                raise ValueError(f'Adding non existing column {c} as relative.')

    def add_indicator_configuration(self, indicator_configuration: IndicatorConfiguration) -> None:
        """Adds a configuration for an indicator. If the price data is updated later, all relevant information to
        calculate the indicators are available."""
        self.indicator_configurations.append(indicator_configuration)

    def update_max_window(self, window: int) -> None:
        """Number of entries that are needed to calculate the indicators
        (e.g. 200-EMA need 200 entries to be calculated)."""
        self.max_window = max(self.max_window, window)

    def get_price_data_df(self, convert_ohl: bool = False, convert_indicators: bool = False,
                          drop_close_volume_column: bool = True) -> pd.DataFrame:
        """
        Get price data as dataframe. By default, no data is converted to relative values. OLH and indicators are
        converted independently, because on merging with lower timeframes only OLH columns can be converted directly
        (e.g. The daily EMA-20 can be converted relative to the current hourly closing price, while the daily opening
        price is converted relative to the daily closing price).
        After converting, the original column is removed. Because the column "Close" is sometimes needed later for
        merging, the dropping can be prevented by drop_column_close.
        :param convert_ohl: Convert open, high and low to relative to close and volume relative to average
        :param convert_indicators: Convert indicators (that have been added by add_relative) to relative to close
        :param drop_close_volume_column: Drop column "close" and "volume" after converting olh columns. No effect,
        if convert_olh=False
        :return: DataFrame
        """
        price_data_prep_df = self.price_data_df.copy()

        if convert_ohl:
            # Convert OHL columns
            _convert_to_relative(price_data_prep_df, 'Open')
            _convert_to_relative(price_data_prep_df, 'High')
            _convert_to_relative(price_data_prep_df, 'Low')

        if convert_indicators:
            # Convert relative indicators
            for col in self.relative_cols:
                _convert_to_relative(price_data_prep_df, col)

        if drop_close_volume_column and convert_ohl:
            price_data_prep_df.drop(columns=['Close'], inplace=True)
            price_data_prep_df.drop(columns=['Volume'], inplace=True)

        return price_data_prep_df

    def reset(self):
        """Drop all columns except OHLCV and reset meta-information. Normally done before updating data."""
        self.price_data_df = self.price_data_df[OHLCV_COLUMNS]
        self.relative_cols = []
        self.max_window = 0


class PriceData(ABC):
    """The PriceData resp. their implementations combine data of different timeframes. The main functionality that
    is needed for most methods is to get a final DataFrame that combines the data from different timeframes and is
    defined by the method get_price_data. The context (asset/base timeframe) as well as some meta information need to
    be available as well."""

    def __init__(self, asset: Asset, timeframe: Timeframe, aggregations: list[Timeframe] = None):
        self.asset = asset
        self.timeframe = timeframe

        if aggregations is None:
            self.aggregations = []
        else:
            self.aggregations = aggregations

    @abstractmethod
    def get_price_data(self, convert: bool) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_indicator_configuration(self) -> dict[Timeframe, list[IndicatorConfiguration]]:
        pass

    @abstractmethod
    def get_lookback_window(self) -> int:
        pass


class PriceDataComplete(PriceData):
    """This class ist used to construct a PriceData object. It holds all information including the complete raw data
    for each timeframe."""

    def __init__(self, price_data_df: pd.DataFrame, asset: Asset, timeframe: Timeframe,
                 aggregations: list[Timeframe] = None):
        super().__init__(asset=asset, timeframe=timeframe, aggregations=aggregations)

        if not all([c in price_data_df.columns for c in OHLCV_COLUMNS]):
            raise PriceDataException("Invalid price data. Missing main columns.")

        price_data_df_copy = price_data_df[OHLCV_COLUMNS].copy()
        self._price_data = {
            timeframe: PriceDataTimeframe(price_data_df_copy, asset, timeframe)
        }
        for tf in self.aggregations:
            self._price_data[tf] = PriceDataTimeframe(PriceDataComplete._aggregate_to_timeframe(price_data_df_copy,
                                                                                                timeframe=tf), asset, tf)

        # Adjust aggregated timeframes, so that they all have the same start date
        if len(self.aggregations) > 0:
            # Bring all aggregations to the same starting date
            starting_date = min(price_data_df.index)
            # 1) Remove all entries from aggregated data before starting data
            #    (e.g. 2021-01-01 14:00 is aggregated to a daily entry for 2021-01-01 00:00 -> should be removed,
            #     because it is not complete)
            for tf in self.aggregations:
                self._price_data[tf].price_data_df = \
                    self._price_data[tf].price_data_df[self._price_data[tf].price_data_df.index >= starting_date]
            # 2) Now get the maximum starting date, over all timeframes
            #    (e.g. if the daily entry 2021-01-01 00:00:00 is removed, the next is 2021-01-02 00:00:00)
            max_starting_date = max([min(self._price_data[tf].price_data_df.index) for tf in self.aggregations])
            # 3) Deleting all data before max_starting_entry, so all dataframes start on the same date
            for tf in self._price_data.keys():
                self._price_data[tf].price_data_df = \
                    self._price_data[tf].price_data_df[self._price_data[tf].price_data_df.index >= max_starting_date]

    @classmethod
    def create_from_timeframe(cls, price_data_tf: PriceDataTimeframe,
                              aggregations: list[Timeframe] = None) -> PriceDataComplete:
        return cls(price_data_df=price_data_tf.price_data_df, asset=price_data_tf.asset,
                   timeframe=price_data_tf.timeframe, aggregations=aggregations)

    def get_indicator_configuration(self) -> dict[Timeframe, list[IndicatorConfiguration]]:
        """Return the configuration of indicators on the different timeframes."""
        config = {}
        for timeframe in self._price_data.keys():
            c = self._price_data[timeframe].indicator_configurations
            if len(c) > 0:
                config[timeframe] = c

        return config

    def get_price_data_for_timeframe(self, timeframe: Timeframe) -> PriceDataTimeframe:
        if timeframe in self._price_data:
            return self._price_data.get(timeframe)
        else:
            raise PriceDataException(f'Requested timeframe {timeframe.name} not available.')

    def get_price_data(self, convert: bool) -> pd.DataFrame:
        """
        This is the most relevant method to combine the data from different timeframes into a single Datafrome.

        :param convert: Convert OHL and relevant indicators as relative values (drop column Close and Volume, since they
        cannot be converted to relative values).

        :return: Data from different timeframes merged in one data frame

        Algorithm:
         a) Get default price data (convert as set in the parameter)
         b) Merge higher timeframes, convert only OHL prices (set by parameter "convert")
         c) Convert relative indicators of higher timeframes regarding to current closing price.
        """

        # Algorithm - Step a)
        price_data_df = self.get_price_data_for_timeframe(timeframe=self.timeframe).get_price_data_df(
            convert_ohl=convert, convert_indicators=convert, drop_close_volume_column=False)

        # Algorithm - Step b)
        # Create list of timeframes to merge
        timeframes = {
            tf: tf.name for tf in
            list(filter(lambda tf_key: tf_key != self.timeframe, self._price_data.keys()))
        }

        # Iterate over all aggregation levels (except the default one) and merge with df
        for timeframe, suffix in timeframes.items():
            price_data_tf = self.get_price_data_for_timeframe(timeframe)
            # OLH data are converted regarding to closing price of the same timeframe. Indicators are merged
            # to the current closing price.
            price_data_tf_df = price_data_tf.get_price_data_df(convert_ohl=convert, convert_indicators=False,
                                                               drop_close_volume_column=False)

            # Adjust date for correct merging.
            # (No entry should include future data from a higher timeframes after merging)
            if timeframe == Timeframe.FOURHOURLY:
                price_data_tf_df.index = price_data_tf_df.index + timedelta(hours=4)
            elif timeframe == Timeframe.DAILY:
                price_data_tf_df.index = price_data_tf_df.index + timedelta(days=1)
            elif timeframe == Timeframe.WEEKLY:
                price_data_tf_df.index = price_data_tf_df.index + timedelta(days=7)

            # Add pseudo columns to original dataframe to assure that all columns end with a suffix after merging
            missing_columns = list(filter(lambda col: col not in price_data_df.columns, price_data_tf_df.columns))
            for c in missing_columns:
                price_data_df.loc[:, c] = np.nan
            price_data_df = price_data_df.merge(price_data_tf_df, how='left', left_index=True, right_index=True,
                                                suffixes=('', f'_{suffix}'))

            price_data_df.drop(columns=missing_columns, inplace=True)

        # The merging is just on one entry per entry on the higher timeframe. Propagate values for all other entries
        for c in filter(lambda col: col.split('_')[-1] in timeframes.values(), price_data_df.columns):
            price_data_df[c] = price_data_df[c].ffill()

        # Algorithm - Step c)
        # Convert columns to relative values.
        if convert:
            for timeframe, suffix in timeframes.items():
                price_data_tf = self.get_price_data_for_timeframe(timeframe)

                for rc in price_data_tf.relative_cols + ['Close']:
                    col_name = f'{rc}_{suffix}'
                    _convert_to_relative(price_data_df, col_name)

            # Drop all volume columns and the main 'Close' column, since they cannot be converted.
            price_data_df.drop(columns=['Close'], inplace=True)
            price_data_df.drop(columns=[c for c in price_data_df.columns if 'volume' in c.lower()], inplace=True)

        return price_data_df

    def get_lookback_window(self) -> int:
        """
        Returns the lookback window that is needed to calculate all indicators on all timeframes.
        """
        lookback_window = 0
        for (tf, pd_tf) in self._price_data.items():
            # Since higher timeframes needs to be shifted, we need to adjust their lookback_window by 1
            # (e.g. the data from the day before is merged to current hourly data)
            adjustment_factor = 1 if tf != self.timeframe and pd_tf.max_window > 0 else 0

            lookback_window = max(lookback_window, Timeframe.convert(self.timeframe, tf) *
                                  (pd_tf.max_window + adjustment_factor))

        return lookback_window + 1

    def reset(self):
        for price_data_tf in self._price_data.values():
            price_data_tf.reset()

    @staticmethod
    def _aggregate_to_timeframe(price_data: pd.DataFrame, timeframe: Timeframe) -> pd.DataFrame:
        """Helping method that aggregates a DataFrome to a higher timeframe."""
        aggregation_symbol = {
            Timeframe.HOURLY: 'H',
            Timeframe.FOURHOURLY: '4H',
            Timeframe.DAILY: 'D',
            Timeframe.WEEKLY: 'W'
        }.get(timeframe)

        price_data_df = price_data.resample(aggregation_symbol).aggregate({
            'Open': 'first',
            'Low': 'min',
            'High': 'max',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()

        if timeframe == Timeframe.WEEKLY:
            # Mon-Sun data is aggregated to Sunday. Therefore we subtract six days, so that Mondays provide the
            # aggregated data of the week, like 00:00:00 contains aggregated data of the day
            price_data_df.index = price_data_df.index + timedelta(days=-6)

        return price_data_df


class PriceDataMirror(PriceData):
    """For cross validation the PriceData must be filtered. Because of the different timeframes this can be quite
    tricky. Therefore the PriceDataMirror can be used, that looks from the outside the same as PriceDataComplete, but
    contains the already converted dataframes, which can be filtered easier.."""

    def __init__(self, price_data: PriceData, idx_filter: Optional[np.ndarray] = None):
        """The PriceDataMirror is constructed from an existing PriceData object and the idx_filter can be used,
        to filter the data for cross_validation or other tasks."""
        super().__init__(asset=price_data.asset, timeframe=price_data.timeframe, aggregations=price_data.aggregations)

        self.price_data_df_unconverted = price_data.get_price_data(convert=False)
        self.price_data_df_converted = price_data.get_price_data(convert=True)
        self.price_data_configuration = price_data.get_indicator_configuration()
        self.lookback_window = price_data.get_lookback_window()

        if idx_filter is not None:
            self.price_data_df_unconverted = self.price_data_df_unconverted.iloc[idx_filter]
            self.price_data_df_converted = self.price_data_df_converted.iloc[idx_filter]

    def get_price_data(self, convert: bool) -> pd.DataFrame:
        if convert:
            df = self.price_data_df_converted.copy(deep=False)
        else:
            df = self.price_data_df_unconverted.copy(deep=False)

        return df

    def get_indicator_configuration(self) -> dict[Timeframe, list[IndicatorConfiguration]]:
        return self.price_data_configuration

    def get_lookback_window(self) -> int:
        return self.lookback_window
