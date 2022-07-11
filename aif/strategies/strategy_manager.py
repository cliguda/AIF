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

from typing import Optional

import aif.common.logging as logging
from aif.common.config import settings
from aif.data_manangement.definitions import Asset, Context, Timeframe
from aif.data_manangement.price_data import PriceData
from aif.strategies import backtest
from aif.strategies.strategy import OrderInformation, Strategy, StrategyPerformance


class StrategyManager:
    """Contains all strategies for all assets and keeps track of possible exit strategies, when a trade was opened."""

    def __init__(self):
        self.current_strategies: dict[Context, list[Strategy]] = {}
        self.all_strategies: dict[Context, list[Strategy]] = {}  # accepted as well as rejected strategies
        self.exit_strategies: dict[Context, Strategy] = {}

    def get_all_contexts(self) -> list[Context]:
        return list(self.all_strategies.keys())

    def reevaluate_all_strategies(self, price_data: PriceData) -> None:
        """This methods evaluates all strategies that have been added with add_strategy"""
        logging.get_aif_logger(__name__).debug(f'Start reevaluation for {price_data.context}')
        self.current_strategies[price_data.context] = []    # Reset current strategies

        for strategy in self.all_strategies[price_data.context]:
            self._add_entry_strategy(strategy=strategy, price_data=price_data)

    def add_strategy(self, strategy: Strategy, price_data: PriceData,
                     strategy_performance: Optional[StrategyPerformance] = None) -> bool:
        """Adds a strategy to the StrategyManager. The strategy is only accepted, if the performance of the strategy
        fulfills the requirements provided in the settings.strategies. The performance can be provided (e.g. for testing
        or if the performance has been calculated beforehand), otherwise the strategy will be cross validated."""
        # TODO: Classifier Parameter

        # Add strategy to all_strategies, for later reevaluation
        if price_data.context not in self.all_strategies.keys():
            self.all_strategies[price_data.context] = []

        self.all_strategies[price_data.context].append(strategy)

        return self._add_entry_strategy(strategy=strategy, price_data=price_data,
                                        strategy_performance=strategy_performance)

    def add_exit_strategy(self, strategy: Strategy) -> None:
        """Adds an exit strategy, generally after an order was successfully placed."""
        context = Context(asset=strategy.init_for_context.asset, timeframe=strategy.init_for_context.timeframe)

        if context in self.exit_strategies.keys():
            logging.get_aif_logger(__name__).error('Try to add an exit strategy, but exit strategy already exists.')
            logging.get_aif_logger(__name__).error(f'New exit strategy: {strategy}')
            logging.get_aif_logger(__name__).error(f'Existing exit strategy: {self.exit_strategies[context]}')
            raise RuntimeError(f'Exit strategy conflict.')

        self.exit_strategies[context] = strategy
        logging.get_aif_logger(__name__).info(f'Added exit-strategy {strategy} for {context}')

    def remove_exit_strategy(self, asset: Asset, timeframe: Timeframe) -> None:
        """Removes an exit strategy, generally after the exit strategy was successfully applied."""
        context = Context(asset=asset, timeframe=timeframe)

        if context in self.exit_strategies.keys():
            self.exit_strategies.pop(context)
        else:
            logging.get_aif_logger(__name__).warning(f'No exit strategy found for {asset.name} on {timeframe.name}')

    def apply(self, price_data: PriceData) -> Optional[OrderInformation]:
        """Applies all strategies to price_data. If more then one strategy found an entry signal, the order from the
        strategy with the highest pps is returned."""
        context = Context(asset=price_data.asset, timeframe=price_data.timeframe)

        if context not in self.current_strategies.keys():
            logging.get_aif_logger(__name__).info(f'No strategies for context {context}.')
            return None

        best_order: Optional[OrderInformation] = None

        for strategy in self.current_strategies[context]:
            logging.get_aif_logger(__name__).debug(f'Applying strategy {strategy} to {context}.')
            order_information = strategy.apply_entry_strategy(price_data)
            logging.get_aif_logger(__name__).debug(f'Tradingsignal for strategy: {order_information}')

            if order_information is not None and (best_order is None or best_order.pps < order_information.pps):
                best_order = order_information

        return best_order

    def apply_exit_strategies(self, price_data: PriceData) -> bool:
        """Applies an exit-strategies to price_data (if one is existing)."""
        context = Context(asset=price_data.asset, timeframe=price_data.timeframe)

        if context in self.exit_strategies.keys():
            exit_strategy = self.exit_strategies[context]

            logging.get_aif_logger(__name__).debug(f'Applying exit-strategy for {exit_strategy} to {context}.')
            exit_signal = exit_strategy.apply_exit_strategy(price_data)
            logging.get_aif_logger(__name__).debug(f'Result: {exit_signal}')

            return exit_signal
        else:
            return False

    def _add_entry_strategy(self, strategy: Strategy, price_data: PriceData,
                            strategy_performance: Optional[StrategyPerformance] = None) -> bool:
        """This method is used from the public method add_entry_strategy, as well as in the reevaluation phase."""

        if strategy_performance is None:
            performance = backtest.cross_validate_strategy(strategy=strategy, price_data=price_data)
        else:
            performance = strategy_performance

        strategy.initialize(price_data=price_data)
        strategy.set_performance(performance)

        if (strategy.get_performance().win_rate < settings.strategies.threshold_strategy_winrate) or \
                (strategy.get_performance().pps < settings.strategies.threshold_strategy_pps) or \
                (len([p for p in strategy.get_performance().performance_detailed if
                      p < 0]) > settings.strategies.allowed_negative_folds):
            logging.get_aif_logger(__name__).info(
                f'Rejected strategy {str(strategy)} for {price_data.asset.name} on {price_data.timeframe.name} '
                f'(Performance: {strategy.get_performance()})')
            return False

        context = price_data.context
        if context not in self.current_strategies.keys():
            self.current_strategies[context] = []

        self.current_strategies[context].append(strategy)
        logging.get_aif_logger(__name__).info(
            f'Added strategy {str(strategy)} for {str(context)} (Performance: {strategy.get_performance()})')
        return True
