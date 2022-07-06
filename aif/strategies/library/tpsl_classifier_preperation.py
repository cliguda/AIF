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

import numba as nb
import numpy as np
import pandas as pd

from aif.data_manangement.price_data import ENTER_TRADE_COLUMN, OHLCV_COLUMNS, PriceData
from aif.strategies.strategy_trading_type import TradingType

"""For the TPSLClassifierStrategy, the price_data must be augmented by a classification, that can be learned. 
   A method for preparation can be provided with the parameter prepare_classifier_data in Strategy 
   (see tpsl_classifier_strategy for examples)."""


def mark_tpsl_signals(price_data: PriceData, tp_threshold, sl_threshold, trading_type: TradingType) -> pd.Series:
    """The methods marks all entries, that would hit tp, before hitting sl. (E.g. if price_data contains a closing price
    of 100 and tp=0.05, sl=0.02, then the entry would be marked with 1, if the price rises to 105 without going below
    98 for a long trade)."""
    price_data_df = price_data.get_price_data(convert=False)
    price_data_prep = price_data_df[OHLCV_COLUMNS].drop(columns=['Volume'])

    tp_prep = tp_threshold
    sl_prep = sl_threshold
    ws_prep = 100

    if trading_type == TradingType.LONG:
        signals = price_data_prep.rolling(window=pd.api.indexers.FixedForwardWindowIndexer(
            window_size=ws_prep), method='table').apply(
            lambda x: _evaluate_long_group(x, sl_prep, tp_prep), raw=True, engine='numba')
    else:
        signals = price_data_prep.rolling(window=pd.api.indexers.FixedForwardWindowIndexer(
            window_size=ws_prep), method='table').apply(
            lambda x: _evaluate_short_group(x, sl_prep, tp_prep), raw=True, engine='numba')

    price_data_df[ENTER_TRADE_COLUMN] = signals['Open']  # The apply function returns a dataframe (bad format)
    price_data_df.loc[:, ENTER_TRADE_COLUMN] = price_data_df[ENTER_TRADE_COLUMN].fillna(0).astype(int)

    return price_data_df[ENTER_TRADE_COLUMN]


@nb.jit("i8(f8[:,:],f8,f8)", nopython=True)
def _evaluate_long_group(m: np.array, sl_threshold: float, tp_threshold: float) -> int:
    """Analyze prices after a concrete signal. Returns 1, -1 or 0 depending of price hitting tp, sl or none."""
    initial_price = m[0, 3]

    threshold_sl_price = (1 - sl_threshold) * initial_price
    threshold_tp_price = (1 + tp_threshold) * initial_price

    m = m[1:]

    m = np.append(m, np.reshape(m[:, 1] <= threshold_sl_price, (-1, 1)), axis=1)  # 4
    m = np.append(m, np.reshape(m[:, 2] >= threshold_tp_price, (-1, 1)), axis=1)  # 5

    sl_index = min(np.nonzero(m[:, 4] == 1)[0]) if len(np.nonzero(m[:, 4] == 1)[0]) > 0 else 10000000
    tp_index = min(np.nonzero(m[:, 5] == 1)[0]) if len(np.nonzero(m[:, 5] == 1)[0]) > 0 else 10000000

    return 1 if tp_index < sl_index else 0


@nb.jit("i8(f8[:,:],f8,f8)", nopython=True)
def _evaluate_short_group(m: np.array, sl_threshold: float, tp_threshold: float) -> int:
    """Analyze prices after a concrete signal. Returns 1, -1 or 0 depending of price hitting tp, sl or none."""
    initial_price = m[0, 3]

    threshold_sl_price = (1 + sl_threshold) * initial_price
    threshold_tp_price = (1 - tp_threshold) * initial_price

    m = m[1:]

    m = np.append(m, np.reshape(m[:, 2] <= threshold_sl_price, (-1, 1)), axis=1)  # 4
    m = np.append(m, np.reshape(m[:, 1] >= threshold_tp_price, (-1, 1)), axis=1)  # 5

    sl_index = min(np.nonzero(m[:, 4] == 1)[0]) if len(np.nonzero(m[:, 4] == 1)[0]) > 0 else 10000000
    tp_index = min(np.nonzero(m[:, 5] == 1)[0]) if len(np.nonzero(m[:, 5] == 1)[0]) > 0 else 10000000

    return 1 if tp_index < sl_index else 0