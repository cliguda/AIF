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

import aif.common.logging as logging
import aif.data_preparation.ta as ta
import aif.strategies.library.ema_stochastic as ema_stochastic_strategy
import aif.strategies.library.heikin_ashi as heikin_ashi_strategy
from aif.bot.bot import Bot
from aif.bot.order_management.portfolio_manager import PortfolioManager
from aif.common.license import get_license_notice
from aif.data_manangement.data_provider import DataProvider
from aif.data_manangement.definitions import Asset, Context, Timeframe
from aif.data_manangement.price_data import PriceDataComplete
from aif.data_preparation.indicator_config import PriceDataConfiguration
from aif.strategies.strategy_manager import StrategyManager

"""
Main script to start the bot. By default the live mode is disabled (settings.toml: trading -> live_mode). If live mode 
is activated, API keys for the exchanges must be provided (in secrets.toml). In default mode, trading signals are
logged to the console and to log_alert_filename (see settings).

-------------------------------------------------------------------------------------------
WARNING: The program, all results and the live mode are for education and fun purpose only! 
         READ THE DISCLAIMER BEFORE TAKING ANY ACTIONS!
-------------------------------------------------------------------------------------------

Notes:
- The bot will run every hour and apply all relevant strategies to the price data. (Currently only hourly strategies
  are supported, but this will be extended in future).
- If a trading signal is found and the bot runs in live mode, an order will be placed.
- If the strategy responsible for the order contains an exit strategy, the exit strategy will be applied in future
  iterations, until the trade is closed.

Configurations:
    PARAM_CONTEXT: Defining assets and timeframes
    PARAM_STRATEGIES: All strategies are applied to all contexts (and possible rejected, depending on the 
                      defined thresholds in settings -> strategies).
    PARAM_AGGREGATIONS: If strategies need higher timeframes (e.g. 4-hourly or daily), the aggregation levels need to
                        be specified here.
"""

# Define all assets
PARAM_CONTEXT = [
    Context(Asset.BTCUSD, Timeframe.HOURLY),
    Context(Asset.ETHUSD, Timeframe.HOURLY),
    Context(Asset.BNBUSD, Timeframe.HOURLY),
    Context(Asset.XRPUSD, Timeframe.HOURLY),
    Context(Asset.ADAUSD, Timeframe.HOURLY),
    Context(Asset.SOLUSD, Timeframe.HOURLY),
]

# Define all strategies to use
PARAM_STRATEGIES = [
    ema_stochastic_strategy.get_long_strategy_configuration,
    ema_stochastic_strategy.get_short_strategy_configuration,
    heikin_ashi_strategy.get_long_strategy_configuration,
    heikin_ashi_strategy.get_short_strategy_configuration,
]

# Define aggregation levels
PARAM_AGGREGATIONS = [Timeframe.FOURHOURLY]


def main():
    print(get_license_notice())
    dp = DataProvider()
    pm = PortfolioManager()

    # Merging the configuration for all strategies
    price_data_conf = PriceDataConfiguration()
    for s in PARAM_STRATEGIES:
        strategy_conf = s()
        price_data_conf.merge(strategy_conf.price_data_configuration)

    logging.get_aif_logger(__name__).info('Status: Update, load and prepare data....')

    asset_information = pm.get_all_asset_information()  # Get meta-information for assets

    price_data_all: dict[Context, PriceDataComplete] = {}
    for context in PARAM_CONTEXT:
        logging.get_aif_logger(__name__).info(f'Load and prepare data for {context.asset} on {context.timeframe}')
        dp.update_historical_data(context.asset, context.timeframe)

        # Get data
        price_data_tf = dp.get_historical_data(context.asset, context.timeframe)
        price_data = PriceDataComplete.create_from_timeframe(price_data_tf, aggregations=PARAM_AGGREGATIONS,
                                                             asset_information=asset_information[context.asset])
        price_data_all[context] = price_data

        ta.add_indicators(price_data_all[context], price_data_conf.configurations)

    # Setup strategies
    sm = StrategyManager()

    logging.get_aif_logger(__name__).info('Status: Create strategies....')
    for s in PARAM_STRATEGIES:
        for context, price_data in price_data_all.items():
            strategy = s().strategy
            sm.add_strategy(strategy=strategy, price_data=price_data)

    # Run strategies every hour
    bot = Bot(price_data_list=price_data_all, strategy_manager=sm, dp=dp, pm=pm)
    logging.get_aif_logger(__name__).info('Status: Setup completed. Now starting the bot...')
    bot.run()


if __name__ == "__main__":
    main()
