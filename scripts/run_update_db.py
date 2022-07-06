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
from aif.common.license import get_license_notice

"""
This script updates the historical data for the given assets.

Notes:
    The run_bot script starts by updating the historical data as well. This script is only necessary, for running
    experiments, plotting results, etc. on up-to-date historical data.

Configurations:
    The assets and timeframes can be configured by PARAM_ASSET and PARAM_TIMEFRAME. Not all timeframes may work,
    depending on the API.
"""

import aif.common.logging as logging
from aif.data_manangement.data_provider import DataProvider
from aif.data_manangement.definitions import Asset, Context, Timeframe

PARAM_UPDATE = [
    Context(Asset.BTCUSD, Timeframe.HOURLY),
    Context(Asset.ETHUSD, Timeframe.HOURLY),
]


def main():
    print(get_license_notice())
    dp = DataProvider()
    for context in PARAM_UPDATE:
        logging.get_aif_logger(__name__).info(f'Updating database for {context.asset.name} on {context.timeframe.name}.')
        dp.update_historical_data(context.asset, context.timeframe)


if __name__ == '__main__':
    main()
