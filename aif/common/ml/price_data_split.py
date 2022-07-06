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

import numpy as np
from sklearn.model_selection import BaseCrossValidator

import aif.common.logging as logging
from aif.common.config import settings
from aif.data_manangement.definitions import Timeframe


class PriceDataSplit(BaseCrossValidator):
    """
    A custom train/test splitter. The training data contains the last fold_size * fold rows,
    for each fold in folds.
    """

    def __init__(self, timeframe: Timeframe, validation_phase=False):
        if validation_phase:
            self.offset = settings.evaluation.cv_folds_testing * \
                          settings['evaluation'][f'cv_fold_size_{timeframe.name.lower()}_testing']
            self.folds = settings.evaluation.cv_folds_validation
            self.fold_size = settings['evaluation'][f'cv_fold_size_{timeframe.name.lower()}_validation']
        else:
            self.offset = 0
            self.folds = settings.evaluation.cv_folds_testing
            self.fold_size = settings['evaluation'][f'cv_fold_size_{timeframe.name.lower()}_testing']
        logging.get_aif_logger(__name__).debug(
            f'Initialized PriceDataSplit with {self.folds} folds and {self.fold_size} entries per fold.')

    def split(self, X, y=None, groups=None) -> (np.ndarray, np.ndarray):
        # For validation we only use data that is not used in the later testing phase.

        if self.offset > 0:
            X = X.iloc[:-self.offset]

        for i in range(self.folds, 0, -1):
            idx_train = np.arange(len(X) - i * self.fold_size)
            idx_test = np.arange(len(X) - i * self.fold_size,
                                 (len(X) - i * self.fold_size) + self.fold_size)

            logging.get_aif_logger(__name__).debug(f'Splitting data: Train data from {min(X.iloc[idx_train].index)} to '
                                                   f' {max(X.iloc[idx_train].index)} / Test data from '
                                                   f'{min(X.iloc[idx_test].index)} to {max(X.iloc[idx_test].index)}')
            yield idx_train, idx_test

    def get_n_splits(self, X=None, y=None, groups=None):
        # Arguments are just for signature reasons
        return self.folds

    def _iter_test_indices(self, X=None, y=None, groups=None):
        raise NotImplementedError('TODO')

    def get_all_test_data(self, X) -> np.ndarray:
        idx_test = np.arange(len(X) - self.folds * self.fold_size, len(X))
        logging.get_aif_logger(__name__).debug(f'Using all test-data in one fold: {min(idx_test)}-{max(idx_test)}')

        return idx_test
