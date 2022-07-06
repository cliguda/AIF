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

import inspect

import aif.data_preparation.indicators as ind
from aif.data_manangement.price_data import PriceDataComplete, Timeframe
from aif.data_preparation.indicator_config import IndicatorConfiguration


def add_indicators(price_data: PriceDataComplete, conf: dict[Timeframe, list[IndicatorConfiguration]]) -> None:
    """
    Adds indicators on the different timeframes, based on the provided configuration.
    """
    indicators = {
        i.__name__: i for i in
        list(filter(lambda i: issubclass(i, ind.Indicator), [i[1] for i in inspect.getmembers(ind, inspect.isclass)]))
    }

    for tf in conf.keys():
        price_data_tf = price_data.get_price_data_for_timeframe(tf)
        for c in conf.get(tf):
            indicator: ind.Indicator = indicators.get(c.indicator, None)

            if indicator is None:
                raise ValueError(f'Indicator {c.indicator} not found.')

            if c.args is None:
                indicator.add_indicator(price_data_tf, window=c.window)
            else:
                indicator.add_indicator(price_data_tf, window=c.window, **c.args)


def get_default_configuration() -> dict[Timeframe, list[IndicatorConfiguration]]:
    """Get a simple default configuration that can be used for experimenting."""
    indicator_conf = {
        Timeframe.HOURLY: [
            IndicatorConfiguration('EMA', 20, None),
            IndicatorConfiguration('EMA', 55, None),
            IndicatorConfiguration('EMA', 200, None),
            IndicatorConfiguration('RSI', 14, None),
            IndicatorConfiguration('Stochastic', 14, None),
            IndicatorConfiguration('BollingerBands', 20, None),
            IndicatorConfiguration('LastLow', 10, None),
            IndicatorConfiguration('LastHigh', 10, None),
        ],
        Timeframe.FOURHOURLY: [
            IndicatorConfiguration('EMA', 20, None),
            IndicatorConfiguration('EMA', 55, None),
            IndicatorConfiguration('EMA', 200, None),
            IndicatorConfiguration('RSI', 14, None),
            IndicatorConfiguration('Stochastic', 14, None),
            IndicatorConfiguration('BollingerBands', 20, None),
            IndicatorConfiguration('LastLow', 10, None),
            IndicatorConfiguration('LastHigh', 10, None),
        ],
        Timeframe.DAILY: [
            IndicatorConfiguration('EMA', 20, None),
            IndicatorConfiguration('EMA', 55, None),
            IndicatorConfiguration('EMA', 200, None),
            IndicatorConfiguration('RSI', 14, None),
            IndicatorConfiguration('Stochastic', 14, None),
            IndicatorConfiguration('BollingerBands', 20, None),
        ],
        Timeframe.WEEKLY: [
            IndicatorConfiguration('EMA', 20, None),
            IndicatorConfiguration('RSI', 14, None),
            IndicatorConfiguration('Stochastic', 14, None),
            IndicatorConfiguration('BollingerBands', 21, None),
        ]
    }

    return indicator_conf
