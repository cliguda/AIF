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

import aif.data_preparation.ta as ta
import aif.strategies.library.macd_ema as library_strategy
from aif.bot.order_management.portfolio_manager import PortfolioManager
from aif.common.license import get_license_notice
from aif.common.ml.price_data_split import PriceDataSplit
from aif.data_manangement.data_provider import DataProvider
from aif.data_manangement.definitions import Asset, Timeframe
from aif.data_manangement.price_data import PriceDataComplete, PriceDataMirror
from aif.data_preparation.indicator_config import PriceDataConfiguration
from aif.plot.plot_price_data import PlotPriceData
from aif.strategies.strategy_definitions import StrategyConfiguration

"""
Script to plot OHLC data, trades and deeper analysis of the relation between different indicators and profitable and
unprofitable trades.

Usage: Only set the PARAM_ variables and run the script. Different plots will be opened within different tabs of the 
browser.
"""

PARAM_ASSET = Asset.BTCUSD
PARAM_TIMEFRAME = Timeframe.HOURLY
PARAM_STRATEGY = library_strategy.get_long_strategy_configuration    # NOTE: WITHOUT (), it's just the function!

PARAM_INDICATORS_TO_PLOT = ['EMA_200']      # E.G. EMA, everything that can be plotted with the OHLC data
PARAM_OSCILLATORS_TO_PLOT = []              # E.G. RSI, everything that should be plotted below the OHLC data.

PARAM_MARK_HIGH_LOWS = False                # Marking local highs and lows. NOTE: Takes a time....

"""Plots all indicators regarding to the outcome of the trades."""
PARAM_EVAL_INDICATORS_FOR_SIGNAL = True

""" Adds some additional indicators on different timeframes to the price_data. This can be useful, to analyze the 
outcome of trades regarding to different indicators."""
PARAM_ADD_DEFAULT_INDICATORS = True


def main():
    print(get_license_notice())

    dp = DataProvider(initialize=False)

    # Get asset information first
    pm = PortfolioManager()
    asset_information = pm.get_asset_information(asset=PARAM_ASSET)

    # Get data
    price_data_tf = dp.get_historical_data(PARAM_ASSET, PARAM_TIMEFRAME)
    price_data = PriceDataComplete.create_from_timeframe(price_data_tf, aggregations=[Timeframe.FOURHOURLY,
                                                                                      Timeframe.DAILY,
                                                                                      Timeframe.WEEKLY],
                                                         asset_information=asset_information)
    # Prepare
    strategy_conf: StrategyConfiguration = PARAM_STRATEGY()

    price_data_conf = PriceDataConfiguration()
    price_data_conf.merge(strategy_conf.price_data_configuration)
    if PARAM_ADD_DEFAULT_INDICATORS:
        price_data_conf.merge(ta.get_default_configuration())

    ta.add_indicators(price_data, price_data_conf.configurations)

    strategy = strategy_conf.strategy
    strategy.initialize(price_data=price_data, skip_fitting=True)

    # Filter just current data for plotting
    cv = PriceDataSplit(timeframe=PARAM_TIMEFRAME)

    price_data_df = price_data.get_price_data(convert=False)
    price_data_current = PriceDataMirror(price_data=price_data,
                                         idx_filter=cv.get_all_test_data(price_data_df))

    # Plot data and trades of strategy
    ppd = PlotPriceData()

    if PARAM_MARK_HIGH_LOWS and 'LastLow' in [i.indicator for i in price_data_conf.configurations.get(PARAM_TIMEFRAME)]:
        ppd.add_lows()

    if PARAM_MARK_HIGH_LOWS and 'LastHigh' in [i.indicator for i in price_data_conf.configurations.get(PARAM_TIMEFRAME)]:
        ppd.add_highs()

    ppd.add_indicator(PARAM_INDICATORS_TO_PLOT)
    ppd.add_oscillator(PARAM_OSCILLATORS_TO_PLOT)
    ppd.add_trades_of_strategy(strategy)

    if PARAM_EVAL_INDICATORS_FOR_SIGNAL:
        ppd.add_indicators_for_signal_evaluation()

    ppd.plot(price_data_current, max_leverage=asset_information.max_leverage, price_data_indicator_analysis=price_data)


if __name__ == '__main__':
    main()
