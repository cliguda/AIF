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

import itertools
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from aif.strategies.strategy_trading_type import TradingType
from aif.data_manangement.price_data import EXIT_TRADE_COLUMN


@dataclass
class ExitSignal:
    """A class to describe an exit signal. When? At what price?
    (If TP/SL was hit, the exit price is not the closing price, but needs to be calculated separately.
    Therefore we store the correct exit price here.)"""
    idx: pd.DatetimeIndex
    exit_price: Optional[float]


def get_exit_for_entry_signal(price_data_for_signal_df: pd.DataFrame, sl_price: Optional[float],
                              tp_price: Optional[float], trading_type: TradingType) -> Optional[ExitSignal]:
    """A helping function that is used for backtesting as well as for plotting. For a trading signal the exit can happen
    via an exit signal or the TP/SL and the exit as well as the exit price are returned by this method."""
    min_exit_idx = None
    min_sl_idx = None
    min_tp_idx = None

    price_data_for_signal_df = price_data_for_signal_df.iloc[1:].copy()  # The first row is the entry signal

    if EXIT_TRADE_COLUMN in price_data_for_signal_df.columns:
        exit_indices = price_data_for_signal_df[price_data_for_signal_df[EXIT_TRADE_COLUMN]].index
        if len(exit_indices) > 0:
            min_exit_idx = min(exit_indices)

    if sl_price is not None:
        price_data_for_signal_df['_sl_hit'] = _mark_sl_hit(price_data_df=price_data_for_signal_df, sl_price=sl_price,
                                                           trading_type=trading_type)

        sl_indices = price_data_for_signal_df[price_data_for_signal_df['_sl_hit']].index
        if len(sl_indices) > 0:
            min_sl_idx = min(sl_indices)

    if tp_price is not None:
        price_data_for_signal_df['_tp_hit'] = _mark_tp_hit(price_data_df=price_data_for_signal_df, tp_price=tp_price,
                                                           trading_type=trading_type)

        tp_indices = price_data_for_signal_df[price_data_for_signal_df['_tp_hit']].index
        if len(tp_indices) > 0:
            min_tp_idx = min(tp_indices)

    possible_exit_indices = [x for x in [min_exit_idx, min_sl_idx, min_tp_idx] if x is not None]
    if len(possible_exit_indices) == 0:
        return None
    else:
        exit_trade_index = min(possible_exit_indices)
        if exit_trade_index == min_sl_idx:
            return ExitSignal(idx=exit_trade_index, exit_price=sl_price)
        elif exit_trade_index == min_tp_idx:
            return ExitSignal(idx=exit_trade_index, exit_price=tp_price)
        else:
            exit_price = price_data_for_signal_df.loc[min_exit_idx, 'Close']
            return ExitSignal(idx=exit_trade_index, exit_price=exit_price)


def _mark_tp_hit(price_data_df: pd.DataFrame, tp_price: Optional[float], trading_type: TradingType) -> pd.Series:
    """Internal function to mark all TP signals."""
    if tp_price is None:
        return pd.Series({'_tp_signal': itertools.repeat(False, len(price_data_df))})

    if trading_type == TradingType.LONG:
        price_data_df['_tp_signal'] = price_data_df['High'] > tp_price
    else:
        price_data_df['_tp_signal'] = price_data_df['Low'] < tp_price

    # Only first signal is relevant
    if len(price_data_df.loc[price_data_df['_tp_signal'], :].index) > 0:
        first_signal = min(price_data_df.loc[price_data_df['_tp_signal'], :].index)
        price_data_df.loc[:, '_tp_signal'] = False
        price_data_df.loc[first_signal, '_tp_signal'] = True

    signals = price_data_df['_tp_signal']
    price_data_df.drop(columns='_tp_signal', inplace=True)

    return signals


def _mark_sl_hit(price_data_df: pd.DataFrame, sl_price: Optional[float], trading_type: TradingType) -> pd.Series:
    """Internal function to mark all SL signals."""
    if sl_price is None:
        return pd.Series({'_sl_signal': itertools.repeat(False, len(price_data_df))})

    if trading_type == TradingType.LONG:
        price_data_df['_sl_signal'] = price_data_df['Low'] < sl_price
    else:
        price_data_df['_sl_signal'] = price_data_df['High'] > sl_price

    # Only first signal is relevant
    if len(price_data_df.loc[price_data_df['_sl_signal'], :].index) > 0:
        first_signal = min(price_data_df.loc[price_data_df['_sl_signal'], :].index)
        price_data_df.loc[:, '_sl_signal'] = False
        price_data_df.loc[first_signal, '_sl_signal'] = True

    signals = price_data_df['_sl_signal']
    price_data_df.drop(columns='_sl_signal', inplace=True)

    return signals
