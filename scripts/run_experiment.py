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
import aif.strategies.library.rsi_stochastic_macd as library_strategy
from aif.bot.order_management.portfolio_information import ExchangeAssetInformation
from aif.bot.order_management.portfolio_manager import PortfolioManager
from aif.common.license import get_license_notice
from aif.data_manangement.data_provider import DataProvider
from aif.data_manangement.definitions import Asset, Context, Timeframe
from aif.data_manangement.price_data import PriceDataComplete
from aif.strategies import backtest
from aif.strategies.strategy import StrategyPerformance
from aif.strategies.strategy_definitions import StrategyConfiguration

"""
This script can be used to evaluate a concrete strategy. Therefore the strategy is backtested with the complete 
dataset as well as cross-validated for understanding the current performance of a strategy. 
No orders are generated, this script if for evaluation only.

-------------------------------------------------------------------------------------------
WARNING: The program, all results and the live mode are for education and fun purpose only! 
         READ THE DISCLAIMER BEFORE TAKING ANY ACTIONS!
-------------------------------------------------------------------------------------------

Configurations:
    The asset, timeframe and strategy can be configured by PARAM_ASSET, PARAM_TIMEFRAME and PARAM_STRATEGY.
"""

PARAM_CONTEXT = [
    Context(Asset.BTCUSD, Timeframe.HOURLY),
    Context(Asset.ETHUSD, Timeframe.HOURLY),
    Context(Asset.BNBUSD, Timeframe.HOURLY),
    Context(Asset.XRPUSD, Timeframe.HOURLY),
    Context(Asset.ADAUSD, Timeframe.HOURLY),
    Context(Asset.SOLUSD, Timeframe.HOURLY),
    Context(Asset.DOGEUSD, Timeframe.HOURLY),
    Context(Asset.DOTUSD, Timeframe.HOURLY),
    Context(Asset.MATICUSD, Timeframe.HOURLY),
    Context(Asset.LINKUSD, Timeframe.HOURLY),
]

PARAM_STRATEGY = library_strategy.get_long_strategy_configuration


def get_performance(strategy_conf: StrategyConfiguration, context: Context,
                    dp: DataProvider, asset_information: dict[Asset, ExchangeAssetInformation]) \
        -> (StrategyPerformance, StrategyPerformance):
    logging.get_aif_logger(__name__).info(f'Evaluate strategy for {context}:')
    # Get data
    price_data_tf = dp.get_historical_data(context.asset, context.timeframe)
    price_data = PriceDataComplete.create_from_timeframe(price_data_tf,
                                                         aggregations=[Timeframe.FOURHOURLY, Timeframe.DAILY,
                                                                       Timeframe.WEEKLY],
                                                         asset_information=asset_information[context.asset])

    # Prepare PriceData
    ta.add_indicators(price_data, strategy_conf.price_data_configuration.configurations)

    # Build and evaluate strategy
    strategy = strategy_conf.strategy
    strategy.initialize(price_data=price_data, skip_fitting=True)

    # Run complete backtest - only possible for rule based strategies
    if type(strategy.entry_signal) == str and strategy.exit_signal is None or type(strategy.exit_signal) == str:
        performance_complete = backtest.evaluate_performance(strategy=strategy, price_data=price_data)
        logging.get_aif_logger(__name__).info(f'Total performance of strategy: '
                                              f'Winrate: {round(performance_complete.win_rate * 100, 2)}% / '
                                              f'PPS: {round(performance_complete.pps * 100, 2)}%')
    else:
        performance_complete = None

    # For cross validation the strategy will be reinitialized for every fold.
    performance_cross_validation = backtest.cross_validate_strategy(strategy=strategy, price_data=price_data)

    logging.get_aif_logger(__name__).info(f'CV performance of strategy: {performance_cross_validation}')

    return performance_complete, performance_cross_validation


def main():
    print(get_license_notice())
    dp = DataProvider(initialize=False)

    # Get max leverage for asset
    pm = PortfolioManager()
    asset_information = pm.get_all_asset_information()

    logging.get_aif_logger(__name__).info('Start evaluating strategy...')
    strategy_performance: dict[Context, (StrategyPerformance, StrategyPerformance)] = {}
    for context in PARAM_CONTEXT:
        p = get_performance(strategy_conf=PARAM_STRATEGY(), context=context, dp=dp, asset_information=asset_information)
        strategy_performance[context] = p

    logging.get_aif_logger(__name__).info('Results:')
    for context, performance in strategy_performance.items():
        logging.get_aif_logger(__name__).info(f'{context} - Backtesting result: {performance[0]}')
        logging.get_aif_logger(__name__).info(f'{context} - Cross-Validation result: {performance[1]}')


if __name__ == "__main__":
    main()
