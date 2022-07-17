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

import time

import schedule

import aif.common.logging as logging
import aif.data_preparation.ta as ta
from aif.bot.order_management.order_status import OrderStatus
from aif.bot.order_management.portfolio_manager import PortfolioManager
from aif.common.config import settings
from aif.data_manangement.data_provider import DataProvider
from aif.data_manangement.definitions import Context
from aif.data_manangement.price_data import PriceData, PriceDataComplete
from aif.strategies.strategy_manager import StrategyManager


class Bot:
    """This class is the main bot functionality. All relevant initialized submodules are passed to the bot, to iterate
    over all strategies in predefined intervals and take out the relevant actions."""

    def __init__(self, price_data_list: dict[Context, PriceData], strategy_manager: StrategyManager, dp: DataProvider,
                 pm: PortfolioManager):
        self.price_data_list = price_data_list
        self.strategy_manager = strategy_manager

        self.dp = dp
        self.pm = pm

    def run(self):
        """This method will not return and continue to iterate forever."""
        number_of_strategies = sum([len(s) for s in self.strategy_manager.current_strategies.values()])

        if number_of_strategies == 0:
            logging.get_aif_logger(__name__).info('No strategies are available, so I have nothing todo...')
            return

        logging.get_aif_logger(__name__).info(f'Starting bot with {number_of_strategies} strategies')

        schedule.every().hour.at(settings.bot.run_hourly_at).do(self._bot_job)
        schedule.every().day.at("02:15").do(self._update_data_job)
        schedule.every().day.at("02:30").do(self._reevaluate_strategies_job)

        logging.get_aif_logger(__name__).info('Bot started. Start looping....')

        while True:
            schedule.run_pending()
            time.sleep(1)

    def _update_data_job(self) -> None:
        """The method updates all price data."""
        logging.get_aif_logger(__name__).info('Bot-Status: Start updating price data...')
        for context, price_data in self.price_data_list.items():
            try:  # Not the best to put the complete block in a try/except, but it avoids the total crash.
                logging.get_aif_logger(__name__).debug(f'Updating {context}')
                indicator_conf = price_data.get_indicator_configuration()
                aggregations = price_data.aggregations
                asset_information = price_data.asset_information

                self.dp.update_historical_data(asset=context.asset, timeframe=context.timeframe)
                price_data_new_tf = self.dp.get_historical_data(asset=context.asset, timeframe=context.timeframe)
                price_data_new = PriceDataComplete.create_from_timeframe(price_data_tf=price_data_new_tf,
                                                                         aggregations=aggregations,
                                                                         asset_information=asset_information)
                ta.add_indicators(price_data_new, indicator_conf)

                self.price_data_list[context] = price_data_new
                logging.get_aif_logger(__name__).debug(f'Update completed for {context} (Max. date: '
                                                       f'{max(price_data_new_tf.price_data_df.index)})')
            except Exception as e:
                logging.get_aif_logger(__name__).error(
                    f'Something went really wrong while updating {price_data.context}: {e}')

        logging.get_aif_logger(__name__).info('Bot-Status: Update completed.')

    def _reevaluate_strategies_job(self) -> None:
        """The method reevaluates all strategies."""
        logging.get_aif_logger(__name__).info('Bot-Status: Start reevaluating strategies...')
        for price_data in self.price_data_list.values():
            try:  # Not the best to put the complete block in a try/except, but it avoids the total crash.
                self.strategy_manager.reevaluate_all_strategies(price_data=price_data)
            except Exception as e:
                logging.get_aif_logger(__name__).error(
                    f'Something went really wrong while reevaluating strategies for {price_data.context}: {e}')

        number_of_strategies = sum([len(s) for s in self.strategy_manager.current_strategies.values()])
        logging.get_aif_logger(__name__).info(
            f'Bot-Status: Reevaluation completed with {number_of_strategies} active strategies.')

    def _bot_job(self) -> None:
        """The method for iterating over all strategies and applying them to the given data once."""
        logging.get_aif_logger(__name__).info('Bot-Status: Start iteration...')
        for price_data in self.price_data_list.values():
            try:  # Not the best to put the complete block in a try/except, but it avoids the total crash.
                logging.get_aif_logger(__name__).info(
                    f'Processing {price_data.asset.name} on {price_data.timeframe.name}...')
                # Update data and indicators
                indicator_conf = price_data.get_indicator_configuration()
                price_data = self.dp.get_updated_price_data(price_data, use_lookback_window=True)

                ta.add_indicators(price_data, indicator_conf)

                # Apply exit strategies if available
                self._apply_exit_strategies_for_price_data(price_data)

                # Apply all strategies
                self._apply_entry_strategies_for_price_data(price_data)
            except Exception as e:
                logging.get_aif_logger(__name__).error(
                    f'Something went really wrong while applying strategies for {price_data.context}: {e}')

        logging.get_aif_logger(__name__).info('Bot-Status: Iteration completed...taking a nap now.')

    def _apply_exit_strategies_for_price_data(self, price_data: PriceData) -> None:
        """When an order was placed successfully by a strategy, the strategy can add an exit strategy for that trade.
        Before an exit strategy is applied, we check for an existing trade, because trade could have been closed by
        TP/SL. If a position is still open, the exit strategy is applied to identify a possible exit signal. When
        an exit signal is found, the exit strategy is removed from the strategy manager, since the trade is closed
        now. """

        # Check for trade, if an exit strategy exists.
        context = Context(asset=price_data.asset, timeframe=price_data.timeframe)
        if context not in self.strategy_manager.exit_strategies.keys():
            return  # No exit strategy available

        active_positions = self.pm.get_active_positions(asset=price_data.asset)
        if not any([p.position_size > 0 for p in active_positions]):
            logging.get_aif_logger(__name__).info(
                'No active positions for active exit strategy. Exit strategy will be removed!')
            self.strategy_manager.remove_exit_strategy(asset=price_data.asset, timeframe=price_data.timeframe)
        else:
            logging.get_aif_logger(__name__).debug(
                'Exit strategy for active position found. Check for exit signal.')

        exit_signal = self.strategy_manager.apply_exit_strategies(price_data)
        if exit_signal:
            logging.get_aif_logger(__name__).info(
                f'Exit Signal for {price_data.asset.name} on {price_data.timeframe.name}')

            order_status = self.pm.exit_trade(price_data.asset)

            if order_status == OrderStatus.ACCEPTED:
                self.strategy_manager.remove_exit_strategy(asset=price_data.asset, timeframe=price_data.timeframe)
            else:
                logging.get_aif_logger(__name__).warning(f'Could NOT place exit-order. Exit strategy still active.')

    def _apply_entry_strategies_for_price_data(self, price_data: PriceData) -> None:
        """Apply all strategies to price data and placing the best order (if one or more trading signals were found).
        If an order is successfully placed, the corresponding exit-strategy will be added (if available for the winning
        strategy)."""
        logging.get_aif_logger(__name__).debug(
            f'Applying all strategies for context {price_data.asset.name} on {price_data.timeframe.name}')
        # Apply strategies
        order = self.strategy_manager.apply(price_data)

        # Placing the order
        if order is None:
            logging.get_aif_logger(__name__).info(
                f'No tradingsignal found for {price_data.asset} on {price_data.timeframe}.')
        else:
            logging.get_aif_logger(__name__).info(f'Tradingsignal with order: {order}')
            order_status = self.pm.place_order(order)

            if order_status == OrderStatus.ACCEPTED and order.from_strategy.exit_signal is not None:
                self.strategy_manager.add_exit_strategy(order.from_strategy)
                logging.get_aif_logger(__name__).info(f'Added exit strategy for {order.from_strategy}')
