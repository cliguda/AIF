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
from typing import Callable, Optional, Union

import pandas as pd

from aif.common.config import settings
from aif.strategies.strategy_trading_type import TradingType


class TradeRiskControl:
    """The class provides Take-Profit and Stop-Loss information for a strategy. Both can be provided by a relative
    value (e.g. 0.05 for 5% TP or SL), a string (e.g. Last_Low) or a method that calculates a TP/SL based on the
    price data until the trading signal appeared. Furthermore the theoretical leverage can be calculated, that
    liquidates a trade on hitting sl (e.g. if sl=0.05, the theoretical leverage would be 20)."""

    def __init__(self,
                 tp: Optional[Union[float, str, Callable[[pd.DataFrame], float]]],
                 sl: Union[float, str, Callable[[pd.DataFrame], float]]):
        self.tp = tp  # Optional, because a strategy can use an exit strategy without a fixed tp
        self.sl = sl  # Currently SL is needed, for calculating the leverage.

    def get_tp_price(self, price_data_df: pd.DataFrame, trading_type: TradingType) -> Optional[float]:
        if isinstance(self.tp, float):
            entering_price_planned = price_data_df.iloc[-1, :]['Close']
            if trading_type == TradingType.LONG:
                tp_price = (1 + self.tp) * entering_price_planned
            else:
                tp_price = (1 - self.tp) * entering_price_planned
        elif isinstance(self.tp, Callable):
            tp_price = self.tp(price_data_df)
        elif isinstance(self.tp, str):
            tp_df = price_data_df.iloc[-1:, :].eval(f'_tp = {self.tp}')
            tp_price = tp_df.iloc[0]['_tp']
        else:
            tp_price = None

        if tp_price is not None:
            return round(tp_price, 3)
        else:
            return tp_price

    def get_sl_price(self, price_data_df: pd.DataFrame, trading_type: TradingType) -> Optional[float]:
        if isinstance(self.sl, float):
            entering_price_planned = price_data_df.iloc[-1]['Close']
            if trading_type == TradingType.LONG:
                sl_price = (1 - self.sl) * entering_price_planned
            else:
                sl_price = (1 + self.sl) * entering_price_planned
        elif isinstance(self.sl, Callable):
            sl_price = self.sl(price_data_df)
        elif isinstance(self.sl, str):
            sl_df = price_data_df.iloc[-1:, :].eval(f'_sl = {self.sl}')
            sl_price = sl_df.iloc[0]['_sl']
        else:
            sl_price = None

        if sl_price is not None:
            return round(sl_price, 3)
        else:
            return sl_price

    def get_leverage_from_data(self, price_data_df: pd.DataFrame, trading_type: TradingType, max_leverage: int) -> int:
        """Returns the theoretical leverage to liquidate a trade when hitting sl."""
        sl_price = self.get_sl_price(price_data_df, trading_type)
        entry_price_planned = price_data_df.iloc[-1, :]['Close']

        return self.get_leverage_for_prices(entry_price_planned=entry_price_planned, sl_price=sl_price,
                                            trading_type=trading_type, max_leverage=max_leverage)

    @staticmethod
    def get_leverage_for_prices(entry_price_planned: float, sl_price: Optional[float],
                                trading_type: TradingType, max_leverage: int) -> int:
        """This method is used internally but also for backtesting and returns the theoretical leverage to liquidate
        a trade when hitting sl. This method is more low-level then get_leverage_from_data, since it deals with
        concrete prices and not a DataFrame any more."""
        if sl_price is None:
            raise ValueError('Leverage can not be calculated without SL.')

        if trading_type == TradingType.LONG:
            leverage = math.floor(1 / ((entry_price_planned - sl_price) / entry_price_planned))
        else:
            leverage = math.floor(1 / ((sl_price - entry_price_planned) / entry_price_planned))

        leverage_adjusted = math.floor((1 - settings.trading.leverage_reduction) * leverage)
        return min(math.floor(leverage_adjusted), max_leverage)

    def __str__(self):
        return f'tp: {self.tp} / sl: {self.sl}'
