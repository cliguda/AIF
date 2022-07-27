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
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

import aif.common.logging as logging
from aif.common.ml.price_data_split import PriceDataSplit
from aif.data_manangement.price_data import ENTER_TRADE_COLUMN, PriceData, PriceDataMirror
from aif.strategies.strategy import Strategy, StrategyPerformance
from aif.strategies.strategy_helper import get_exit_for_entry_signal
from aif.strategies.strategy_trading_type import TradingType


def cross_validate_strategy(strategy: Strategy, price_data: PriceData,
                            classifier_parameters: Optional[dict] = None) -> StrategyPerformance:
    """This method is for classical cross validation of a strategy. Thereby the settings cv_folds_testing and
    cv_fold_size_x_testing are used, and the type of cross validation ist implemented in PriceDataSplit.
    Note: strategy has not to be initialized and will be initialized for every fold (therefore we provide the asset
    information separate. """
    price_data_df = price_data.get_price_data(convert=False)  # We just need the index for splitting.
    cv = PriceDataSplit(timeframe=price_data.timeframe, validation_phase=False)

    performance_per_fold_detailed = []

    for idx_train, idx_test in cv.split(X=price_data_df):
        logging.get_aif_logger(__name__).debug(f'Cross-validation: Train from {min(idx_train)} to {max(idx_train)} / '
                                               f'Test from {min(idx_test)} to {max(idx_test)}.')

        price_data_train = PriceDataMirror(price_data=price_data, idx_filter=idx_train)
        price_data_test = PriceDataMirror(price_data=price_data, idx_filter=idx_test)

        strategy.initialize(price_data=price_data_train, classifier_parameters=classifier_parameters)

        profit = evaluate_performance(strategy=strategy, price_data=price_data_test)
        performance_per_fold_detailed.append(profit.performance_detailed)

    performance_per_fold = [sum(p) for p in performance_per_fold_detailed]
    performance_detailed = list(itertools.chain(*performance_per_fold_detailed))
    signals = len(performance_detailed)
    performance = round(sum(performance_detailed), 2)

    profitable_signals = len(list(filter(lambda x: x > 0, performance_detailed)))
    win_rate = round(profitable_signals / signals, 2) if signals > 0 else np.NAN
    pps = round(performance / signals * 100, 2) if signals > 0 else np.NAN
    logging.get_aif_logger(__name__).debug(f'Total cross-validation performance: {round(performance * 100, 2)}% with'
                                           f' {signals} signals (Win-rate: {round(win_rate * 100, 2)}% / '
                                           f'PPS: {pps}%)')

    if signals > 0:
        pps = round(performance / signals, 3)
    else:
        pps = 0.0

    return StrategyPerformance(win_rate=win_rate, pps=pps, performance_detailed=performance_per_fold)


def evaluate_performance(strategy: Strategy, price_data: PriceData) -> StrategyPerformance:
    """This method evaluates the performance for the complete price_data and assumes, that the strategy is already
    initialized. The method is used to evaluate the performance per fold in cross-validation, but also for a complete
    backtest of price_data."""
    last_exit_idx = datetime.min
    performance_detailed = []
    price_data_df = strategy._get_data_with_signals(price_data)

    for idx in price_data_df[price_data_df[ENTER_TRADE_COLUMN]].index:
        if idx > last_exit_idx:  # Only enter a trade, after the last one was exited
            p, exit_idx = _get_profit_for_signal(strategy=strategy, price_data_df=price_data_df, idx=idx,
                                                 max_leverage=price_data.asset_information.max_leverage,
                                                 fees_per_trade=price_data.asset_information.fees_market_order)

            if exit_idx is not None:
                last_exit_idx = exit_idx
                performance_detailed.append(p)
                logging.get_aif_logger(__name__).trace(f'Performance of trade from {idx} to {exit_idx}: '
                                                       f'{round(p * 100, 2)}%')
        else:
            logging.get_aif_logger(__name__).trace(f'Skipping entry signal at {idx}. Last trade still active.')

    signals = len(performance_detailed)
    performance = round(sum(performance_detailed), 2)

    profitable_signals = len(list(filter(lambda x: x > 0, performance_detailed)))
    win_rate = round(profitable_signals / signals, 2) if signals > 0 else np.NAN

    if signals > 0:
        pps = round(performance / signals, 3)
    else:
        pps = 0.0

    logging.get_aif_logger(__name__).debug(f'Performance of fold: {round(performance * 100, 2)}% with'
                                           f' {signals} signals (Win-rate: {round(win_rate * 100, 2)}% / '
                                           f'PPS: {round(pps * 100, 2)}%)')

    return StrategyPerformance(win_rate=win_rate, pps=pps, performance_detailed=list(performance_detailed))


def _get_profit_for_signal(strategy: Strategy, price_data_df: pd.DataFrame, idx: pd.Timestamp, max_leverage: int,
                           fees_per_trade: float) -> (float, Optional[pd.Timestamp]):
    """Method to calculate the profit for a trading signal that occurred in price_data_df at index idx."""
    price_data_until_signal_df = price_data_df.loc[:idx].copy(deep=False)
    sl_price = strategy.risk_control.get_sl_price(price_data_df=price_data_until_signal_df,
                                                  trading_type=strategy.trading_type)
    tp_price = strategy.risk_control.get_tp_price(price_data_df=price_data_until_signal_df,
                                                  trading_type=strategy.trading_type)

    price_data_since_signal_df = price_data_df.loc[idx:].copy(deep=False)
    exit_signal = get_exit_for_entry_signal(price_data_for_signal_df=price_data_since_signal_df,
                                            sl_price=sl_price, tp_price=tp_price, trading_type=strategy.trading_type)

    if exit_signal is not None:  # Can happen for the last entry signal, without exit signal
        entry_price = price_data_until_signal_df.iloc[-1, :]['Close']
        leverage = strategy.risk_control.get_leverage_for_prices(entry_price_planned=entry_price, sl_price=sl_price,
                                                                 trading_type=strategy.trading_type,
                                                                 max_leverage=max_leverage)

        p = _get_profit_for_trade(entry_price=entry_price, exit_price=exit_signal.exit_price,
                                  trading_type=strategy.trading_type, leverage=leverage, fees_per_trade=fees_per_trade)

        return p, exit_signal.idx
    else:
        logging.get_aif_logger(__name__).trace(f'No exit signal found for entry signal at {idx})')
        return 0, None


def _get_profit_for_trade(entry_price, exit_price: float, trading_type: TradingType, leverage: int,
                          fees_per_trade: float) -> float:
    """Method for calculating the profit for a trade with a given entry and exit price, a trading_type, the used
    leverage and the fees for the trade."""
    if trading_type == TradingType.LONG:
        profit_abs = exit_price - entry_price
    else:
        profit_abs = entry_price - exit_price

    profit_leveraged = (profit_abs / entry_price) * leverage
    fees = fees_per_trade * leverage

    return profit_leveraged - fees
