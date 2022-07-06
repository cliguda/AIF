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

from functools import partial
from typing import Optional, Union

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from aif import settings
from aif.data_preparation.indicator_config import PriceDataConfiguration
from aif.strategies.strategy_definitions import StrategyConfiguration
from aif.strategies.strategy_trading_type import TradingType
from aif.strategies.classifier import Classifier
from aif.strategies.library.tpsl_classifier_preperation import mark_tpsl_signals
from aif.strategies.strategy import Strategy
from aif.strategies.trade_risk_control import TradeRiskControl


def get_knn_strategy_configuration(tp: float, sl: float, trading_type: TradingType, n_neighbors: int = 1) -> \
        StrategyConfiguration:
    """Return a strategy that learns to classify signals, that hit tp before hitting sl.
    The PriceDataConfiguration is empty, since a classifier learns, whatever is available.
    For the KNN classifier, the weights are depending on the columns of the price_data Dataframe and must be
    provided, when the strategy is initialized:
    wd = WeightedDistance(weights=np.ones(len(price_data.get_price_data(convert=True).columns)))
    strategy.initialize(..., classifier_parameters={'model__metric': wd})
    """
    classifier = Pipeline([
        ('scalar', StandardScaler()),
        ('model', KNeighborsClassifier(n_neighbors=n_neighbors))
    ])

    return StrategyConfiguration(price_data_configuration=PriceDataConfiguration(),
                                 strategy=_get_strategy_configuration(tp=tp, sl=sl, trading_type=trading_type,
                                                                      classifier=classifier))


def get_rf_strategy_configuration(tp: float, sl: float, trading_type: TradingType, n_estimators: int = 2000,
                                  max_depth: Optional[int] = None, min_samples_leaf: int = 4,
                                  max_features: Optional[Union[str, int, float]] = None,
                                  class_weight: Optional[Union[str, dict]] = None) -> \
        (PriceDataConfiguration, Strategy):
    """Return a strategy that learns to classify signals, that hit tp before hitting sl.
    The PriceDataConfiguration is empty, since a classifier learns, whatever is available.
    """

    rf = RandomForestClassifier(n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_leaf=min_samples_leaf,
                                max_features=max_features,
                                class_weight=class_weight,
                                random_state=settings.common.random_seed)

    classifier = Pipeline([
        ('model', rf)
    ])

    return StrategyConfiguration(price_data_configuration=PriceDataConfiguration(),
                                 strategy=_get_strategy_configuration(tp=tp, sl=sl, trading_type=trading_type,
                                                                      classifier=classifier))


def get_svm_strategy_configuration(tp: float, sl: float, trading_type: TradingType, c: float = 0.2,
                                   kernel: str = 'poly', degree: int = 3) -> \
        (PriceDataConfiguration, Strategy):
    """Return a strategy that learns to classify signals, that hit tp before hitting sl.
    The PriceDataConfiguration is empty, since a classifier learns, whatever is available.
    """
    svm = SVC(C=c, kernel=kernel, degree=degree, random_state=settings.common.random_seed)
    classifier = Pipeline([
        ('scalar', StandardScaler()),
        ('model', svm)
    ])

    return StrategyConfiguration(price_data_configuration=PriceDataConfiguration(),
                                 strategy=_get_strategy_configuration(tp=tp, sl=sl, trading_type=trading_type,
                                                                      classifier=classifier))


def _get_strategy_configuration(tp: float, sl: float, trading_type: TradingType, classifier: Classifier) -> Strategy:
    risk_control = TradeRiskControl(tp=tp, sl=sl)

    s = Strategy(name='TPSL Classifier',
                 trading_type=trading_type,
                 preprocessor=[],
                 entry_signal=classifier,
                 exit_signal=None,
                 risk_control=risk_control,
                 convert_data_for_classifier=True,
                 prepare_classifier_data=partial(mark_tpsl_signals,
                                                 tp_threshold=risk_control.tp,
                                                 sl_threshold=risk_control.sl,
                                                 trading_type=trading_type)
                 )
    return s
