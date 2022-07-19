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
from aif.bot.order_management.exchanges.exchange_bybit import ExchangeBybit
from aif.bot.order_management.order_status import OrderStatus
from aif.strategies.strategy import OrderInformation
from aif.bot.order_management.portfolio_information import ExchangeAssetInformation, PositionInformation, WalletBalance
from aif.common.config import settings
from aif.data_manangement.definitions import Asset


class ExchangeException(Exception):

    def __init__(self, msg):
        self.msg = msg

    def __repr__(self):
        return self.msg


class PortfolioManager:
    """The PortfolioManager is the go to place, for all orders and balances. All requests are forwarded to the
    corresponding exchanges. All exceptions are caught and an ExchangeException is raised instead, to make exception
    handling easier and to avoid a system crash, when an error occurs."""

    def __init__(self):
        exchange_bybit = ExchangeBybit()
        self.exchanges = {
            Asset.BTCUSD: exchange_bybit,
            Asset.ETHUSD: exchange_bybit,
            Asset.BNBUSD: exchange_bybit,
            Asset.XRPUSD: exchange_bybit,
            Asset.ADAUSD: exchange_bybit,
            Asset.SOLUSD: exchange_bybit,
        }

        self.live_mode = settings.trading.live_mode
        if self.live_mode:
            logging.get_aif_logger(__name__).info('WARNING: Starting in live mode. Orders will be placed.')
        else:
            logging.get_aif_logger(__name__).info('Starting in simulation mode. Orders will NOT be placed.')

    def get_equity_for_asset(self, asset: Asset) -> WalletBalance:
        try:
            return self.exchanges.get(asset).get_equity()
        except Exception as e:
            raise ExchangeException(str(e))

    def get_active_positions(self, asset: Asset) -> list[PositionInformation]:
        try:
            return self.exchanges.get(asset).get_active_positions(asset)
        except Exception as e:
            raise ExchangeException(str(e))

    def get_all_asset_information(self) -> dict[Asset, ExchangeAssetInformation]:
        asset_information: dict[Asset, ExchangeAssetInformation] = {}
        for asset in self.exchanges.keys():
            asset_information[asset] = self.get_asset_information(asset)

        return asset_information

    def get_asset_information(self, asset: Asset) -> ExchangeAssetInformation:
        try:
            asset_information = self.exchanges[asset].get_asset_information(asset)
            if asset_information is None:
                asset_information = ExchangeAssetInformation(max_leverage=settings.trading.default_max_leverage,
                                                             fees_market_order=settings.trading.default_fees)
            return asset_information
        except Exception as e:
            raise ExchangeException(str(e))

    def place_order(self, order: OrderInformation) -> OrderStatus:
        if self.live_mode:
            try:
                # Checking limit of open orders
                open_positions = self.get_number_of_open_positions()
                if open_positions >= settings.trading.max_open_positions:
                    logging.get_aif_logger(__name__).warning(f'Max. number of open positions. Skipping order: {order}')
                    return OrderStatus.MAX_OPEN_POSITIONS

                logging.get_aif_logger(__name__).info(f'Placing order: {order}')
                order_status = self.exchanges.get(order.asset).place_order(order)

                if order_status == OrderStatus.ACCEPTED:
                    logging.get_aif_logger(__name__).action(
                        f'Order was placed. Order: {order}')
                else:
                    logging.get_aif_logger(__name__).warning(
                        f'Order was NOT placed. Status: {order_status.name} / Order: {order}')

                return order_status
            except Exception as e:
                raise ExchangeException(str(e))
        else:
            logging.get_aif_logger(__name__).info(f'Simulation mode - Order: {order}')
            return OrderStatus.SIMULATION_ONLY

    def exit_trade(self, asset: Asset) -> OrderStatus:
        """Closes all trades for asset."""
        if self.live_mode:
            try:
                logging.get_aif_logger(__name__).info(f'Exit trade for {asset}.')
                order_status = self.exchanges.get(asset).exit_trade(asset)
                logging.get_aif_logger(__name__).action(
                    f'Closing order for asset {asset.name} was placed. Status: {order_status.name}')
                return order_status
            except Exception as e:
                raise ExchangeException(str(e))
        else:
            logging.get_aif_logger(__name__).info(f'Simulation mode - Exit Signal for {asset}')
            return OrderStatus.SIMULATION_ONLY

    def get_number_of_open_positions(self) -> int:
        positions = [p for asset in self.exchanges.keys() for p in self.get_active_positions(asset)]
        open_positions = sum([p.position_size != 0 for p in positions])

        return open_positions
