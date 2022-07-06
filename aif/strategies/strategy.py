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

from dataclasses import dataclass
from typing import Callable, Optional, Union

import pandas as pd
from pandas import DataFrame

from aif.common import logging as logging
from aif.data_manangement.definitions import Asset, Context
from aif.data_manangement.price_data import ENTER_TRADE_COLUMN, EXIT_TRADE_COLUMN, PriceData
from aif.strategies.classifier import Classifier
from aif.strategies.prep_command import CommandDescription
from aif.strategies.strategy_trading_type import TradingType
from aif.strategies.trade_risk_control import TradeRiskControl


@dataclass(frozen=True)
class OrderInformation:
    """
    The class contains all information to place an order and is returned, when a strategy found a trading signal.
    """
    from_strategy: Strategy
    asset: Asset
    trading_type: TradingType
    pps: float
    entering_price_planned: float
    leverage: int
    tp_price: Optional[float] = None
    sl_price: Optional[float] = None

    def __post_init__(self):
        if self.tp_price is None and self.from_strategy.exit_signal is None:
            raise ValueError('Neither tp_price not exit strategy are provided.')
        if not self._validate_leverage():
            raise ValueError('Incorrect leverage.')

    def _validate_leverage(self):
        # Checking for correct SL/Leverage settings.
        if self.sl_price is not None:
            if self.trading_type == TradingType.LONG:
                lvg_by_sl = 1 / ((self.entering_price_planned - self.sl_price) / self.entering_price_planned)
            else:
                lvg_by_sl = 1 / ((self.sl_price - self.entering_price_planned) / self.entering_price_planned)

            if lvg_by_sl < 0.99 * self.leverage:  # Check for over-leveraging (0.99 to avoid rounding errors).
                logging.get_aif_logger(__name__).warning(
                    'Leverage too high. Order would get liquidated before hitting sl!')
                return False

        return True

    def __str__(self):
        return f'Order for {self.asset.name} from {self.from_strategy} with {self.leverage} leverage ' \
               f'(Entry price: {self.entering_price_planned} / TP: {self.tp_price} / SL: {self.sl_price}).'


@dataclass
class StrategyPerformance:
    """Stores the performance of a strategy."""
    win_rate: float
    pps: float
    performance_detailed: list[float]

    def __repr__(self):
        x = [str(round(p * 100, 2)) + '%' for p in self.performance_detailed]
        performance_per_fold = ' / '.join(x)
        s = f'Win-rate: {round(self.win_rate * 100, 2)}% / PPS: {round(self.pps * 100, 2)}% / Performance per fold: ' \
            f'{performance_per_fold}'

        return s


class Strategy:
    """The main Strategy class (see library for some examples)."""
    def __init__(self, name: str, trading_type: TradingType,
                 preprocessor: list[CommandDescription],
                 entry_signal: Union[str, Classifier],
                 exit_signal: Optional[Union[str, Classifier]],
                 risk_control: TradeRiskControl,
                 convert_data_for_classifier: bool,
                 prepare_classifier_data: Optional[Callable[[PriceData], pd.Series]] = None):
        self.name = name
        self.trading_type = trading_type
        self.preprocessor = preprocessor
        self.entry_signal = entry_signal
        self.exit_signal = exit_signal
        self.risk_control = risk_control
        self.convert_data_for_classifier = convert_data_for_classifier
        self.prepare_classifier_data = prepare_classifier_data

        self.init_for_context = None
        self.max_leverage = None
        self.strategy_performance = None

    def initialize(self, price_data: PriceData, max_leverage: int, skip_fitting: bool = False,
                   classifier_parameters: Optional[dict] = None) -> None:
        """Before a strategy can be applied, the strategy must be initialized for a certain PriceData. For some tests
        and experiments the fitting is not always necessary and can be skipped."""
        self.init_for_context = Context(asset=price_data.asset, timeframe=price_data.timeframe)
        self.max_leverage = max_leverage

        """If the strategy contains a classifier, the classifier needs to be trained."""
        if skip_fitting:
            logging.get_aif_logger(__name__).debug('Skip fitting! Classifiers will not work.')
        elif isinstance(self.entry_signal, Classifier) or isinstance(self.exit_signal, Classifier):
            price_data_df: DataFrame = price_data.get_price_data(convert=self.convert_data_for_classifier).dropna()

            if self.prepare_classifier_data is not None:
                y = self.prepare_classifier_data(price_data)
                y = y[price_data_df.index]
            else:
                y = None

            if isinstance(self.entry_signal, Classifier):
                if classifier_parameters is not None:
                    self.entry_signal.set_params(**classifier_parameters)
                self.entry_signal.fit(X=price_data_df, y=y)

            if isinstance(self.exit_signal, Classifier):
                if classifier_parameters is not None:
                    self.exit_signal.set_params(**classifier_parameters)
                self.exit_signal.fit(X=price_data_df, y=y)

    def set_performance(self, strategy_performance: StrategyPerformance):
        self.strategy_performance = strategy_performance

    def get_performance(self) -> StrategyPerformance:
        return self.strategy_performance

    def __str__(self) -> str:
        return f'{self.name} ({self.trading_type.name})'

    def apply_entry_strategy(self, price_data: PriceData) -> Optional[OrderInformation]:
        """Applies the strategy to price_data. An OrderInformation will be returned, if the last row contains an
        entry signal, specified by the entry_signal rule or classifier. If the strategy was not initialized, an
        exception is raised."""
        order_information = None

        price_data_df = self._get_data_with_signals(price_data)
        entering_signal = bool(price_data_df.iloc[-1][ENTER_TRADE_COLUMN])

        # Create order information in case of an entry signal
        if entering_signal:
            # To determine entry-price and tp/sl, unconverted data is needed.
            price_data_df = price_data.get_price_data(convert=False)
            entering_price_planned = price_data_df.iloc[-1, :]['Close']
            tp_price = self.risk_control.get_tp_price(price_data_df, self.trading_type)
            sl_price = self.risk_control.get_sl_price(price_data_df, self.trading_type)
            leverage = self.risk_control.get_leverage_from_data(price_data_df, self.trading_type,
                                                                max_leverage=self.max_leverage)

            order_information = OrderInformation(from_strategy=self, asset=price_data.asset,
                                                 trading_type=self.trading_type,
                                                 pps=self.strategy_performance.pps,
                                                 entering_price_planned=entering_price_planned,
                                                 leverage=leverage, tp_price=tp_price, sl_price=sl_price)

        return order_information

    def apply_exit_strategy(self, price_data: PriceData) -> bool:
        """Applies the rule or classifier of exit_signal to price_data. True is returned, if the last row contains an
        exit signal.  If the strategy was not initialized, an exception is raised."""
        price_data_df = self._get_data_with_signals(price_data)
        exit_signal = bool(price_data_df.iloc[-1][EXIT_TRADE_COLUMN])

        return exit_signal

    def _get_data_with_signals(self, price_data: PriceData) -> pd.DataFrame:
        """
        This method is used internally but also for backtesting (therefore the model_performance can still be missing).
        The returned DataFrame will be unconverted!
        """
        if self.init_for_context is None:
            raise RuntimeError('Strategy was not correctly initialized.')
        elif (self.init_for_context.asset != price_data.asset) or (
                self.init_for_context.timeframe != price_data.timeframe):
            raise RuntimeError('Strategy was initialized for a different context (asset/timeframe),')

        # 1) Get Dataframe
        price_data_df = price_data.get_price_data(convert=self.convert_data_for_classifier).dropna().copy()

        # 2) Preprocess data
        for cmd_spec in self.preprocessor:
            cmd_spec.apply_command(price_data_df)

        # 3) Get entry signals
        if isinstance(self.entry_signal, str):
            price_data_df.eval(f'{ENTER_TRADE_COLUMN} = {self.entry_signal}', inplace=True)
        else:
            price_data_df.loc[:, ENTER_TRADE_COLUMN] = self.entry_signal.predict(price_data_df).astype(bool)

        # 4) Get exit signals
        if isinstance(self.exit_signal, str):
            price_data_df.eval(f'{EXIT_TRADE_COLUMN} = {self.exit_signal}', inplace=True)
        elif isinstance(self.exit_signal, Classifier):
            price_data_df.loc[:, EXIT_TRADE_COLUMN] = self.exit_signal.predict(price_data_df).astype(bool)
        else:
            price_data_df.loc[:, EXIT_TRADE_COLUMN] = False

        # Merge signals with unconverted data
        price_data_unconverted_df = price_data.get_price_data(convert=False).dropna()
        price_data_unconverted_df.loc[:, ENTER_TRADE_COLUMN] = price_data_df[ENTER_TRADE_COLUMN]
        price_data_unconverted_df.loc[:, EXIT_TRADE_COLUMN] = price_data_df[EXIT_TRADE_COLUMN]

        return price_data_unconverted_df
