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

from datetime import datetime
from typing import Optional, Union

import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp

# Data to plot
import aif.common.logging as logging
from aif.data_manangement.price_data import ENTER_TRADE_COLUMN, PriceData
from aif.plot.layout import update_layout
from aif.strategies import backtest
from aif.strategies.strategy import Strategy
from aif.strategies.strategy_helper import get_exit_for_entry_signal
from aif.strategies.strategy_trading_type import TradingType

_MAX_INDICATORS = 3
_MAX_OSCILLATORS = 3
_INDICATOR_COLORS = ['blue', 'orange', 'green']

_MAX_ROWS_EVAL = 3
_MAX_COLUMN_EVAL = 3

COLUMN_NAME_SIGNAL_PROFITABLE = 'Signal_Profitable'


class PlotPriceData:
    """Main class for plotting price data, indicators and evaluation of trading signals. Add everything that should be
    plotted, and call plot() at the end."""

    def __init__(self):
        self.indicators: list[str] = []
        self.oscillators: list[str] = []
        self.strategy: Optional[Strategy] = None
        self.indicator_signal_evaluation: bool = False
        self.mark_highs: bool = False
        self.mark_lows: bool = False

    def add_highs(self):
        """Mark all local highs. Needs the indicator LastHigh."""
        self.mark_highs = True

    def add_lows(self):
        """Mark all local lows. Needs the indicator LastLow."""
        self.mark_lows = True

    def add_indicator(self, indicator: Union[str, list[str]]):
        """Adds one or more indicators that should be printed with the price data."""
        if isinstance(indicator, str):
            indicator_names = [indicator]
        else:
            indicator_names = indicator

        for i in indicator_names:
            if len(self.indicators) < _MAX_INDICATORS:
                self.indicators.append(i)
            else:
                raise RuntimeError('Too many indicators added.')

    def add_oscillator(self, oscillator: Union[str, list[str]]):
        """Adds one or more oscillators that should be printed below the price data"""
        if isinstance(oscillator, str):
            oscillator_names = [oscillator]
        else:
            oscillator_names = oscillator

        for o in oscillator_names:
            if len(self.oscillators) < _MAX_OSCILLATORS:
                self.oscillators.append(o)
            else:
                raise RuntimeError('Too many oscillator added.')

    def add_trades_of_strategy(self, strategy: Strategy):
        """Add a rule for which all trades are printed with the price data. Also this rule is used for
           add_indicators_for_signal_evaluation
        """
        self.strategy = strategy

    def add_indicators_for_signal_evaluation(self):
        """Adds features, that are evaluated for all trading signals (by add_indicators_for_signal_evaluation)."""
        self.indicator_signal_evaluation = True

    def plot(self, price_data: PriceData, max_leverage: int, price_data_indicator_analysis: Optional[PriceData] = None,
             ohlc_prefix: str = ''):
        """Plots everything for the gives price data."""
        price_data_df = price_data.get_price_data(convert=False)

        fig = self._setup_price_figure(asset_name=price_data.asset.name, timeframe_name=price_data.timeframe.name)

        self._add_ohlc_to_fig(price_data_df, fig, ohlc_prefix)

        if self.mark_highs:
            self._add_marks(price_data_df, fig, name='High')

        if self.mark_lows:
            self._add_marks(price_data_df, fig, name='Low')

        self._add_indicators_to_fig(price_data_df, fig)

        self._add_oscillators_to_fig(price_data_df, fig)

        if self.strategy is not None:
            self._add_trades_for_strategy_to_fig(price_data, fig)

        fig.show()

        if self.indicator_signal_evaluation:
            price_data_analysis = price_data_indicator_analysis if price_data_indicator_analysis is not None \
                else price_data
            self._plot_signal_eval_figure(price_data=price_data_analysis, max_leverage=max_leverage)

    def _setup_price_figure(self, asset_name: str, timeframe_name: str):
        """Setup the main plot for price data-"""
        rows = 1 + len(self.oscillators)

        if rows == 1:
            fig = go.Figure()
        else:
            if rows == 2:
                row_heights = [0.7, 0.25]
                vertical_spacing = 0.05
            elif rows == 3:
                row_heights = [0.6, 0.18, 0.18]
                vertical_spacing = 0.02
            elif rows == 4:
                row_heights = [0.55, 0.14, 0.14, 0.14]
                vertical_spacing = 0.01
            else:
                raise RuntimeError('Cannot create subfigure. Too many rows.')

            fig = sp.make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=vertical_spacing,
                                   row_heights=row_heights)

        title = f'{asset_name} / {timeframe_name.capitalize()}'
        if self.strategy is not None:
            title = title + ' with Trading Signals'

        update_layout(fig, title=title, y_axis_prefix='$')

        return fig

    @staticmethod
    def _add_ohlc_to_fig(price_data_df: pd.DataFrame, fig, ohlc_prefix: str):
        # OHLC data
        fig.add_trace(go.Candlestick(x=price_data_df.index,
                                     open=price_data_df[f'{ohlc_prefix}Open'],
                                     high=price_data_df[f'{ohlc_prefix}High'],
                                     low=price_data_df[f'{ohlc_prefix}Low'],
                                     close=price_data_df[f'{ohlc_prefix}Close'],
                                     name=f'{ohlc_prefix}OHLC',
                                     line=dict(width=1),
                                     opacity=1,
                                     increasing={'fillcolor': '#24A06B', 'line_color': '#2EC886'},
                                     decreasing={'fillcolor': '#CC2E3C', 'line_color': '#FF3A4C'},
                                     ))

    @staticmethod
    def _add_marks(price_data_df: pd.DataFrame, fig, name: str):
        """Method is used to identify highs and lows. Therefore the columns Low/High and Last_Low/Last_High
        are needed."""
        # Identifying changes
        price_data_df.loc[:, f'_last_{name.lower()}_shift'] = price_data_df[f'Last_{name.capitalize()}'].shift(1)
        price_data_df.loc[price_data_df[f'_last_{name.lower()}_shift'] !=
                          price_data_df[f'Last_{name.capitalize()}'], f'_is_{name.lower()}'] = True
        price_data_df[f'_is_{name.lower()}'] = price_data_df[f'_is_{name.lower()}'].fillna(False)

        # Mark all changing-points
        for i, idx in enumerate(price_data_df[price_data_df[f'_is_{name.lower()}']].index):
            logging.get_aif_logger(__name__).debug(f'Marked {name.lower()} at {idx}')
            if i == 0:
                from_idx = idx
            else:
                from_idx = max(price_data_df.loc[price_data_df.index < idx].index)
            to_idx = min(price_data_df.loc[price_data_df.index > idx].index)
            y_value = price_data_df.loc[idx, f'{name.capitalize()}']
            fig.add_shape(type="line",
                          x0=from_idx, y0=y_value, x1=to_idx,
                          y1=y_value,
                          line=dict(color="grey", width=2)
                          )

        price_data_df.drop(columns=[f'_last_{name.lower()}_shift', f'_is_{name.lower()}'], inplace=True)

    def _add_indicators_to_fig(self, price_data_df: pd.DataFrame, fig):
        # Add indicators
        for idx, indicator in enumerate(self.indicators):
            fig.add_trace(go.Scatter(x=price_data_df.index,
                                     y=price_data_df[indicator],
                                     opacity=0.7,
                                     line=dict(color=_INDICATOR_COLORS[idx], width=1),
                                     name=indicator))

    def _add_oscillators_to_fig(self, price_data_df: pd.DataFrame, fig, lower_line: Optional[int] = 20,
                                upper_band: Optional[int] = 80):

        # Add oscillators
        for idx, oscillator in enumerate(self.oscillators):
            fig.add_trace(go.Scatter(x=price_data_df.index,
                                     y=price_data_df[oscillator],
                                     opacity=0.7,
                                     line=dict(color='orange', width=2),
                                     name=oscillator), row=idx + 2, col=1)
            fig['layout'][f'yaxis{idx + 2}'].update(title_text=oscillator, title_font_size=12)

            if lower_line is not None:
                fig.add_shape(type="line", x0=min(price_data_df.index), y0=lower_line, x1=max(price_data_df.index),
                              y1=lower_line, line=dict(color="grey", dash='dot', width=1), row=idx + 2, col=1)

            if upper_band is not None:
                fig.add_shape(type="line", x0=min(price_data_df.index), y0=upper_band, x1=max(price_data_df.index),
                              y1=upper_band, line=dict(color="grey", dash='dot', width=1), row=idx + 2, col=1)

    def _add_trades_for_strategy_to_fig(self, price_data, fig):
        """Add all trades to the ohlc data."""
        last_exit_idx = datetime.min

        price_data_df = self.strategy._get_data_with_signals(price_data)

        for idx in price_data_df[price_data_df[ENTER_TRADE_COLUMN]].index:
            if idx <= last_exit_idx:
                continue

            price_data_to_time_df = price_data_df[:idx].copy(deep=True)
            price_data_for_signal_df = price_data_df[idx:].copy(deep=True)

            tp_price = self.strategy.risk_control.get_tp_price(price_data_to_time_df, self.strategy.trading_type)
            sl_price = self.strategy.risk_control.get_sl_price(price_data_to_time_df, self.strategy.trading_type)

            exit_signal = get_exit_for_entry_signal(price_data_for_signal_df, tp_price=tp_price,
                                                    sl_price=sl_price, trading_type=self.strategy.trading_type)

            if exit_signal is not None:
                logging.get_aif_logger(__name__).info(f'Adding trade from {idx} to {exit_signal.idx}')

                enter_price = price_data_to_time_df.iloc[-1]['Close']
                exit_price = exit_signal.exit_price

                if sl_price is not None:
                    fig.add_shape(type="line",
                                  x0=idx, y0=sl_price, x1=exit_signal.idx,
                                  y1=sl_price,
                                  line=dict(color="red", width=2)
                                  )

                if tp_price is not None:
                    fig.add_shape(type="line",
                                  x0=idx, y0=tp_price, x1=exit_signal.idx,
                                  y1=tp_price,
                                  line=dict(color="green", width=2)
                                  )

                if self.strategy.trading_type == TradingType.LONG:
                    if exit_price > enter_price:
                        color = 'green'
                    else:
                        color = 'red'
                else:
                    if exit_price < enter_price:
                        color = 'green'
                    else:
                        color = 'red'

                fig.add_shape(type='rect', x0=idx, y0=enter_price, x1=exit_signal.idx, y1=exit_price,
                              fillcolor=color,
                              opacity=0.5)

                last_exit_idx = exit_signal.idx

    def _plot_signal_eval_figure(self, price_data: PriceData, max_leverage: int):
        """Evaluate all features for all trading signals, for a better understanding their impact on positive
        and negative trades.
        """
        if self.strategy is None:
            raise RuntimeError('No rule was provided')

        price_data_unconv_df = self._mark_profit_loss(price_data=price_data, max_leverage=max_leverage)
        price_data_conv_df = price_data.get_price_data(convert=True)

        indicators_to_evaluate = price_data_conv_df.columns

        plots_per_page = _MAX_ROWS_EVAL * _MAX_COLUMN_EVAL
        pages = len(indicators_to_evaluate) // plots_per_page
        if len(indicators_to_evaluate) % plots_per_page > 0:
            pages += 1

        for page in range(0, pages):
            offset = page * plots_per_page
            fig = sp.make_subplots(
                rows=_MAX_ROWS_EVAL, cols=_MAX_COLUMN_EVAL,
                subplot_titles=indicators_to_evaluate[(0 + offset):(plots_per_page + offset)],
                vertical_spacing=0.1, y_title='test'
            )
            title = f'Indicators per Tradingsignal - ({page + 1} / {pages})'
            update_layout(fig, title=title)

            remaining_columns = len(indicators_to_evaluate) - offset
            for i in range(0, min(plots_per_page, remaining_columns)):
                logging.get_aif_logger(__name__).debug(
                    f'Plot for {indicators_to_evaluate[i + offset]} on page {page + 1} in row '
                    f'{i // 2 + 1} column {i % 2 + 1}.')
                fig.add_trace(
                    go.Scatter(x=price_data_unconv_df[COLUMN_NAME_SIGNAL_PROFITABLE],
                               y=price_data_conv_df[indicators_to_evaluate[i + offset]],
                               mode='markers', showlegend=True),
                    row=(i // 3 + 1),
                    col=(i % 3 + 1)
                )

            fig.show()

    def _mark_profit_loss(self, price_data, max_leverage: int) -> pd.DataFrame:
        """Mark a positive trade as true and a negative one as false, for evaluating different indicators for the
        trades.
        """
        last_exit_idx = datetime.min

        price_data_df = self.strategy._get_data_with_signals(price_data)
        for idx in price_data_df[price_data_df[ENTER_TRADE_COLUMN]].index:
            if idx <= last_exit_idx:
                continue

            p, exit_idx = backtest._get_profit_for_signal(strategy=self.strategy, price_data_df=price_data_df, idx=idx,
                                                          max_leverage=max_leverage, fees_per_trade=0.0)

            logging.get_aif_logger(__name__).info(f'Marking trade from {idx} with profit {round(p * 100, 2)}%')
            if p > 0:
                price_data_df.loc[idx, COLUMN_NAME_SIGNAL_PROFITABLE] = 'Profitable'
            else:
                price_data_df.loc[idx, COLUMN_NAME_SIGNAL_PROFITABLE] = 'Not profitable'

            if exit_idx is not None:
                last_exit_idx = exit_idx

        return price_data_df
