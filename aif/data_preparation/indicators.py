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

import math
from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
import pandas as pd
import talib as ta

import aif.common.logging as log
from aif.data_manangement.price_data import PriceDataTimeframe
from aif.data_preparation.indicator_config import IndicatorConfiguration

"""
Implementation of all indicators.
Structure of this file:
- Definition of the abstract class Indicator
- Price indicators
- Momentum indicators
- Volume indicators
- Volatility indicators
- Others indicators
- Pattern related Features

"""


class Indicator(ABC):
    """Base class for all indicators. Indicators should always be implemented by implementing a subclass of Indicator
     to ensure, that all meta-information are correctly set in the PriceData object."""

    @classmethod
    def add_indicator(cls, price_data_tf: PriceDataTimeframe, window: int, **kwargs) -> None:
        column_name = cls.indicator_implementation(price_data_tf.price_data_df, window, **kwargs)
        price_data_tf.add_indicator_configuration(
            IndicatorConfiguration(indicator=cls.__name__, window=window, args=kwargs))

        log.get_aif_logger(__name__).debug(
            f'Added indicator {cls.__name__} on timeframe {price_data_tf.timeframe} with {window=}, {kwargs=}')
        price_data_tf.update_max_window(window)

        if column_name is not None:
            price_data_tf.add_relative_column(column_name)

    @classmethod
    @abstractmethod
    def indicator_implementation(cls, price_data_df: pd.DataFrame, window: int, **kwargs) -> \
            Optional[Union[str, list[str]]]:
        """Concrete implementation for the indicator.
        Returns:
             None, if the added indicator is not relative to the closing-price (e.g. RSI)
             str or list[str] for relative indicators (e.g. EMA, BB) the name of the new column(s) is/are returned.
        """
        raise NotImplementedError()


# Price Indicators

class EMA(Indicator):
    """Simple EMA indicator (Can be converted relative to the closing price.)"""

    @classmethod
    def indicator_implementation(cls, price_data_df: pd.DataFrame, window: int, **kwargs) -> Optional[str]:
        col_name = 'EMA_' + str(window)
        price_data_df.loc[:, col_name] = ta.EMA(price_data_df['Close'], timeperiod=window)
        return col_name


# Momentum Indicators

class EMASlope(Indicator):
    """
    Adds the slope of EMA. The window is used for the EMA, the number of candles considered for the regression
    is provided by kwargs.
    """

    @classmethod
    def indicator_implementation(cls, price_data_df: pd.DataFrame, window: int, **kwargs) -> Optional[str]:
        """
            :param kwargs: slope_window: Number of candles to consider for regression (<= window for ema)
        """
        slope_window = kwargs.get('slope_window')
        if slope_window > window:
            raise ValueError(f'Error adding indicator {cls.__name__}: slope_window > window')

        col_name = f'EMA_{str(window)}_Slope'

        x = np.array(range(0, slope_window))
        price_data_df.loc[:, '_EMA'] = ta.EMA(price_data_df['Close'], timeperiod=window)
        price_data_df[col_name] = price_data_df['_EMA'].rolling(window=slope_window).apply(
            lambda gr: np.polyfit(x, gr, 1)[0], raw=True)
        price_data_df.drop(columns=['_EMA'], inplace=True)

        return None


class RSI(Indicator):
    """ The RSI is calculated on the last 14 candles by default. The implementation is from Tradingview."""

    @classmethod
    def indicator_implementation(cls, price_data_df: pd.DataFrame, window: int, **kwargs) -> Optional[str]:
        col_name = f'RSI_{str(window)}'
        price_data_df.loc[:, col_name] = ta.RSI(price_data_df['Close'], timeperiod=window)

        return None


class Stochastic(Indicator):

    @classmethod
    def indicator_implementation(cls, price_data_df: pd.DataFrame, window: int, **kwargs) -> Optional[str]:
        col_name_k = f'Stochastic_K_{str(window)}'
        col_name_d = f'Stochastic_D_{str(window)}'

        res = ta.STOCHF(price_data_df['High'], price_data_df['Low'], price_data_df['Close'], fastk_period=window,
                        fastd_period=3, fastd_matype=0)

        price_data_df.loc[:, col_name_k] = res[0]
        price_data_df.loc[:, col_name_d] = res[1]

        return None


class SqueezingMomentumIndicator(Indicator):
    """
    Squeezing Momentum Indicator like provided by Tradingview.
    Possible problem: Only the changing entry is marked
    as long or short, all other values are 0. Maybe find a better option.
    """

    @classmethod
    def indicator_implementation(cls, price_data_df: pd.DataFrame, window: int, **kwargs) -> Optional[str]:
        """
        :param window: default is 20
        :param kwargs: length_kc=20 (<= window), mult_kc=1.5
        """
        length_kc = kwargs.get('length_kc', 20)
        mult_kc = kwargs.get('mult_kc', 1.5)

        if length_kc > window:
            raise ValueError(f'Error adding indicator {cls.__name__}: length_kc > window')

        col_name = 'SQZ_MNT'

        # calculate Bollinger Bands
        bb = ta.BBANDS(price_data_df['Close'], timeperiod=window, nbdevup=2, nbdevdn=2, matype=0)
        price_data_df.loc[:, '_upper_BB'] = bb[0]
        price_data_df.loc[:, '_lower_BB'] = bb[2]

        # Calculate Keltner Channel first we need to calculate True Range
        price_data_df['_tr0'] = abs(price_data_df["High"] - price_data_df["Low"])
        price_data_df['_tr1'] = abs(price_data_df["High"] - price_data_df["Close"].shift())
        price_data_df['_tr2'] = abs(price_data_df["Low"] - price_data_df["Close"].shift())
        price_data_df['_tr'] = price_data_df[['_tr0', '_tr1', '_tr2']].max(axis=1)  # moving average of the TR
        range_ma = price_data_df['_tr'].rolling(window=length_kc).mean()

        m_avg = price_data_df['Close'].rolling(window=window).mean()
        price_data_df['_upper_KC'] = m_avg + range_ma * mult_kc
        price_data_df['_lower_KC'] = m_avg - range_ma * mult_kc

        # check for 'squeeze'
        price_data_df['_squeeze_off'] = (price_data_df['_lower_BB'] < price_data_df['_lower_KC']) & (
                price_data_df['_upper_BB'] > price_data_df['_upper_KC'])

        # calculate momentum value
        highest = price_data_df['High'].rolling(window=length_kc).max()
        lowest = price_data_df['Low'].rolling(window=length_kc).min()
        m1 = (highest + lowest) / 2
        price_data_df['_value'] = (price_data_df['Close'] - (m1 + m_avg) / 2)
        fit_y = np.array(range(0, length_kc))
        price_data_df['_value'] = price_data_df['_value'].rolling(window=length_kc).apply(
            lambda x: np.polyfit(fit_y, x, 1)[0] * (length_kc - 1) + np.polyfit(fit_y, x, 1)[1], raw=True)

        # Shift to compare squeeze data to previous candle
        price_data_df.loc[:, '_squeeze_off_yesterday'] = price_data_df['_squeeze_off'].shift(1)
        price_data_df.loc[:, '_squeeze_off_yesterday'] = price_data_df['_squeeze_off_yesterday'].fillna(False)

        # The squeeze is released
        price_data_df.loc[:, '_sqz_released'] = price_data_df['_squeeze_off'] & ~price_data_df['_squeeze_off_yesterday']

        # entry point for long/short position:
        price_data_df.loc[:, '_sqz_mom_long'] = price_data_df['_sqz_released'] & (price_data_df['_value'] > 0)
        price_data_df.loc[:, '_sqz_mom_short'] = price_data_df['_sqz_released'] & (price_data_df['_value'] < 0)

        price_data_df.loc[price_data_df['_sqz_mom_long'], col_name] = 1
        price_data_df.loc[price_data_df['_sqz_mom_short'], col_name] = -1

        price_data_df.loc[:, col_name] = price_data_df[col_name].fillna(0).astype(int)

        # Cleanup - Add 'value_yesterday', if momentum_sign/momentum_increasing are activated
        price_data_df.drop(
            columns=['_upper_BB', '_lower_BB', '_tr0', '_tr1', '_tr2', '_tr', '_upper_KC', '_lower_KC',
                     '_squeeze_off', '_squeeze_off_yesterday', '_value', '_sqz_released', '_sqz_mom_long',
                     '_sqz_mom_short'],
            inplace=True)

        return None


# Volume Indicators

class VolumeRelativeToAverage(Indicator):
    """Volume relative to the average volume for the past "window" candles."""

    @classmethod
    def indicator_implementation(cls, price_data_df: pd.DataFrame, window: int, **kwargs) -> Optional[str]:
        col_name = f'Volume_Relative_{str(window)}'
        price_data_df.loc[:, '_Vol_Avg'] = price_data_df['Volume'].rolling(window).mean()
        price_data_df.loc[:, col_name] = \
            (price_data_df['Volume'] - price_data_df['_Vol_Avg']) / price_data_df['_Vol_Avg']
        price_data_df.drop(columns=['_Vol_Avg'], inplace=True)

        return None


# Volatility Indicators


# Others indicators

class BollingerBands(Indicator):
    """Bollinger Bands. Default window = 20."""

    @classmethod
    def indicator_implementation(cls, price_data_df: pd.DataFrame, window: int, **kwargs) -> \
            Optional[Union[str, list[str]]]:
        col_name_upper = f'BB_Upper_{str(window)}'
        col_name_lower = f'BB_Lower_{str(window)}'

        bb = ta.BBANDS(price_data_df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

        price_data_df.loc[:, col_name_upper] = bb[0]
        price_data_df.loc[:, col_name_lower] = bb[2]

        return [col_name_upper, col_name_lower]


# Pattern related features

class LastLow(Indicator):
    """Adds the last swing low as feature (that can be converted relative to closing price). To get previous lows,
    use the argument prev=n, for the n.th-previous low before the last low. A good default window is 10."""

    @classmethod
    def indicator_implementation(cls, price_data_df: pd.DataFrame, window: int, **kwargs) -> \
            Optional[Union[str, list[str]]]:

        n = kwargs.get('prev', 0)

        if n == 0:
            col_name = 'Last_Low'
        else:
            col_name = f'Last_Low_Prev_{n}'

        price_data_df['_lowest_low'] = price_data_df['Low'].rolling(window=window, center=True).min()

        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=math.floor(window / 2))
        price_data_df['_highest_low_left'] = price_data_df['Low'].rolling(window=math.floor(window / 2)).max()
        price_data_df['_highest_low_right'] = price_data_df['Low'].rolling(window=indexer).max()

        price_data_df['_is_low_'] = (price_data_df['Low'] == price_data_df['_lowest_low']) & \
                                    (price_data_df['Low'] * 1.005 < price_data_df['_highest_low_left']) & \
                                    (price_data_df['Low'] * 1.005 < price_data_df['_highest_low_right'])

        price_data_df.loc[price_data_df['_is_low_'], col_name] = price_data_df.loc[price_data_df['_is_low_'], 'Low']
        if n > 0:
            price_data_df.loc[price_data_df['_is_low_'], col_name] = \
                price_data_df.loc[price_data_df['_is_low_'], col_name].shift(n)
        price_data_df.loc[:, col_name] = price_data_df[col_name].ffill()

        price_data_df.drop(columns=['_lowest_low', '_highest_low_left', '_highest_low_right', '_is_low_'], inplace=True)

        return col_name


class LastHigh(Indicator):
    """Adds the last swing high as feature (that can be converted relative to closing price). To get previous highs,
    use the argument prev=n, for the n.th-previous high before the last high."""

    @classmethod
    def indicator_implementation(cls, price_data_df: pd.DataFrame, window: int, **kwargs) -> \
            Optional[Union[str, list[str]]]:
        n = kwargs.get('prev', 0)

        if n == 0:
            col_name = 'Last_High'
        else:
            col_name = f'Last_High_Prev_{n}'

        price_data_df['_highest_high'] = price_data_df['High'].rolling(window=window, center=True).max()

        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=math.floor(window / 2))
        price_data_df['_lowest_high_left'] = price_data_df['High'].rolling(window=math.floor(window / 2)).min()
        price_data_df['_lowest_high_right'] = price_data_df['High'].rolling(window=indexer).min()

        price_data_df['_is_high_'] = (price_data_df['High'] == price_data_df['_highest_high']) & \
                                     (price_data_df['High'] * 0.995 > price_data_df['_lowest_high_left']) & \
                                     (price_data_df['High'] * 0.995 > price_data_df['_lowest_high_right'])

        price_data_df.loc[price_data_df['_is_high_'], col_name] = price_data_df.loc[price_data_df['_is_high_'], 'High']
        if n > 0:
            price_data_df.loc[price_data_df['_is_high_'], col_name] = \
                price_data_df.loc[price_data_df['_is_high_'], col_name].shift(n)
        price_data_df.loc[:, col_name] = price_data_df[col_name].ffill()

        price_data_df.drop(columns=['_highest_high', '_lowest_high_left', '_lowest_high_right', '_is_high_'],
                           inplace=True)

        return col_name


class HeikinAshi(Indicator):
    """Adds the HeikinAshi candles to the price_data. They are considered to be converted relative to the closing
    price."""

    @classmethod
    def indicator_implementation(cls, price_data_df: pd.DataFrame, window: int, **kwargs) -> \
            Optional[Union[str, list[str]]]:
        price_data_df['_last_open'] = price_data_df['Open'].shift(1)
        price_data_df['_last_close'] = price_data_df['Close'].shift(1)

        price_data_df['HA_Open'] = (price_data_df['_last_open'] + price_data_df['_last_close']) / 2
        price_data_df['HA_Close'] = (price_data_df['Open'] + price_data_df['High'] +
                                     price_data_df['Low'] + price_data_df['Close']) / 4
        price_data_df['HA_High'] = price_data_df[['High', 'HA_Open', 'HA_Close']].max(axis=1)
        price_data_df['HA_Low'] = price_data_df[['Low', 'HA_Open', 'HA_Close']].min(axis=1)

        price_data_df.drop(columns=['_last_open', '_last_close'], inplace=True)

        return ['HA_Open', 'HA_High', 'HA_Low', 'HA_Close']
