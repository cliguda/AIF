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

import pandas as pd
from pybit import usdt_perpetual
from pybit.exceptions import InvalidRequestError

from aif.bot.order_management.exchanges.exchange import Exchange
from aif.bot.order_management.order_status import OrderStatus
from aif.bot.order_management.portfolio_information import ExchangeAssetInformation, PositionInformation, WalletBalance
from aif.common import logging
from aif.common.config import settings
from aif.data_manangement.definitions import Asset
from aif.strategies.strategy import OrderInformation
from aif.strategies.strategy_trading_type import TradingType

_BASE_CURRENCY = 'USDT'


class ExchangeBybit(Exchange):
    """Direct connection to ByBit. WARNING: The API can change without warning and some function calls may fail."""

    def __init__(self):
        if len(settings.bybit.api_key) > 0 and len(settings.bybit.api_secret) > 0:
            logging.get_aif_logger(__name__).info('Setting up bybit client.')

            self.client = usdt_perpetual.HTTP(
                endpoint='https://api.bybit.com',
                api_key=settings.bybit.api_key,
                api_secret=settings.bybit.api_secret
            )

            self.asset_information = self._get_asset_information_for_init()
        else:
            self.client = None
            self.asset_information = {}

    def get_equity(self) -> WalletBalance:

        req_result = self.client.get_wallet_balance(coin=_BASE_CURRENCY)['result']

        equity = req_result.get(_BASE_CURRENCY).get('equity')
        available_balance = req_result.get(_BASE_CURRENCY).get('available_balance')

        return WalletBalance(equity=equity, available_balance=available_balance)

    def get_asset_information(self, asset: Asset) -> Optional[ExchangeAssetInformation]:
        """Some information like maximal leverage and fees per trade depend an the asset and/or exchange."""
        return self.asset_information.get(asset, None)

    def get_active_positions(self, asset: Asset) -> list[PositionInformation]:
        positions = self.client.my_position(symbol=self._get_symbol_name(asset))['result']

        active_positions = []
        for position in positions:
            position_size = position['size']
            trading_type = None
            if position['side'] == 'Buy':
                trading_type = TradingType.LONG
            elif position['side'] == 'Sell':
                trading_type = TradingType.SHORT
            leverage = float(position['leverage'])

            sl = float(position['stop_loss'])
            tp = float(position['take_profit'])

            active_positions.append(
                PositionInformation(position_size=position_size, trading_type=trading_type, leverage=leverage, sl=sl,
                                    tp=tp))

        return active_positions

    def place_order(self, order: OrderInformation) -> OrderStatus:
        # Check for an existing position
        current_positions = self.get_active_positions(order.asset)
        total_position_size = sum([position.position_size for position in current_positions])
        if total_position_size > 0:
            return OrderStatus.EXISTING_ORDER

        # Check for existing conditional orders
        conditional_orders = len(self.client.get_conditional_order(symbol=self._get_symbol_name(order.asset),
                                                                   stop_order_status='Untriggered')['result']['data'])
        if conditional_orders > 0:
            logging.get_aif_logger(__name__).action(f'Close untriggered conditional order for {order.asset.name}')
            self.client.cancel_all_conditional_orders(symbol=self._get_symbol_name(order.asset))

        # Set new leverage
        leverage_accepted = False
        try:
            req_result = self.client.set_leverage(symbol=self._get_symbol_name(order.asset),
                                                  buy_leverage=str(order.leverage),
                                                  sell_leverage=str(order.leverage))
            if isinstance(req_result, dict):
                leverage_accepted = req_result['ret_msg'] in ['OK', 'leverage not modified']
        except InvalidRequestError as e:
            leverage_accepted = (e.message == 'leverage not modified')

        if not leverage_accepted:
            return OrderStatus.LEVERAGE_FAILED

        # Calculate position size
        wallet_balance = self.get_equity()
        position_size = wallet_balance.equity * settings.trading.size_per_trade

        # Place order
        if position_size > wallet_balance.available_balance:
            return OrderStatus.AVAILABLE_BALANCE_INSUFFICIENT

        quantity = round((position_size * order.leverage) / order.entering_price_planned, 4)
        req_side = 'Buy' if order.trading_type == TradingType.LONG else 'Sell'

        if not order.conditional:
            req_result = self.client.place_active_order(side=req_side,
                                                        symbol=self._get_symbol_name(order.asset),
                                                        qty=quantity,
                                                        take_profit=order.tp_price,
                                                        stop_loss=order.sl_price,
                                                        order_type='Market',
                                                        reduce_only=False,
                                                        time_in_force='GoodTillCancel',
                                                        close_on_trigger=False)
        else:
            current_price = self._get_last_price(order.asset)
            req_result = self.client.place_conditional_order(side=req_side,
                                                             symbol=self._get_symbol_name(order.asset),
                                                             qty=quantity,
                                                             take_profit=order.tp_price,
                                                             stop_loss=order.sl_price,
                                                             order_type='Market',
                                                             trigger_by='MarkPrice',
                                                             stop_px=order.entering_price_planned,
                                                             base_price=current_price,
                                                             reduce_only=False,
                                                             time_in_force='GoodTillCancel',
                                                             close_on_trigger=False)

        order_accepted = (req_result['ret_code'] == 0)
        if not order_accepted:
            req_msg = req_result['ret_msg']
            logging.get_aif_logger(__name__).error(f'Order was not accepted: {req_msg}')
            return OrderStatus.FAILED_AT_REQUEST
        else:
            return OrderStatus.ACCEPTED

    def exit_trade(self, asset: Asset) -> OrderStatus:
        # Check for an existing position
        current_positions = self.get_active_positions(asset)
        total_position_size = sum([position.position_size for position in current_positions])
        if total_position_size == 0:
            return OrderStatus.NO_TRADE_TO_CLOSE

        for position in current_positions:
            if position.position_size > 0:
                # place an inverse order to close an existing trade
                req_side = 'Sell' if position.trading_type == TradingType.LONG else 'Buy'

                req_result = self.client.place_active_order(side=req_side,
                                                            symbol=self._get_symbol_name(asset),
                                                            qty=position.position_size,
                                                            order_type='Market',
                                                            reduce_only=True, time_in_force='GoodTillCancel',
                                                            close_on_trigger=False)

                req_msg = req_result['ret_msg']
                order_accepted = req_msg == 'OK'
                if not order_accepted:
                    logging.get_aif_logger(__name__).error(f'Closing order was not accepted: {req_msg}')
                    return OrderStatus.FAILED_AT_REQUEST
                else:
                    return OrderStatus.ACCEPTED

    def _get_last_price(self, asset: Asset) -> float:
        req_result = self.client.latest_information_for_symbol(symbol=self._get_symbol_name(asset))
        last_price = req_result['result'][0]['last_price']

        return float(last_price)

    def _get_asset_information_for_init(self) -> dict[Asset, ExchangeAssetInformation]:
        """This method should only be called by the init method once."""
        req_result = self.client.query_symbol()["result"]
        symbol_information = pd.DataFrame(req_result).set_index('name')

        symbol_information = symbol_information[symbol_information['quote_currency'] == 'USDT']
        lvg = symbol_information[symbol_information['quote_currency'] == 'USDT']['leverage_filter']. \
            apply(lambda r: r['max_leverage']).to_dict()
        fees = symbol_information[symbol_information['quote_currency'] == 'USDT']['taker_fee'].to_dict()

        asset_information: dict[Asset, ExchangeAssetInformation] = {}
        for sym in symbol_information.index:
            asset = self._get_asset(sym)
            if asset is not None:
                asset_max_lvg = lvg[sym]
                asset_fees = float(fees[sym])
                asset_information[asset] = ExchangeAssetInformation(max_leverage=asset_max_lvg,
                                                                    fees_market_order=asset_fees)

        return asset_information

    @staticmethod
    def _get_symbol_name(asset: Asset) -> str:
        """The linear contracts on bybit are on USDT, therefore we need to change the name of the asset."""
        if asset.name.endswith('USD'):
            return asset.name + 'T'
        else:
            raise ValueError('Only supporting linear contracts (Assetname ending with USD)')

    @staticmethod
    def _get_asset(symbol_name: str) -> Optional[Asset]:
        """The linear contracts on bybit are on USDT, therefore we need to change the name to get the correct asset."""
        sym = symbol_name
        if sym.endswith('USDT'):
            sym = sym.replace('USDT', 'USD')

        if sym in Asset.__dict__['_member_names_']:
            return Asset[sym]
        else:
            return None
